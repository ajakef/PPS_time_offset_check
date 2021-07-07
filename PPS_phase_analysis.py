import numpy as np
from numpy.fft import fft, fftshift, fftfreq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import obspy

def pps_phase_analysis_single(tr, rad_exclude = 4, padded_length = 200, freq_lim = None):
    """Compare a recording of a single PPS step to an idealized PPS step to determine subsample time error.
    Parameters:

    tr: obspy.Trace showing a single PPS step where the second begins on the middle sample
    rad_exclude: number of points before and after the middle sample to exclude from calculating the
    high and low levels because of the anti-aliasing filter
    padded_length: number of samples contained in the padded time series
    freq_lim: highest frequency used to estimate time lag (default 0.2 * tr.stats.sampling_rate)

    Value: dict with following elements:
    freq: numpy.array containing frequencies for spectra
    amp: numpy.array containing recorded-to-ideal transfer function (amplitude)
    phase: numpy.array containing recorded-to-ideal transfer function (phase, radians)
    time: time lag spectrum (seconds)
    offset: calculated time offset (seconds)

    """
    if freq_lim is None:
        freq_lim = 0.2 * tr.stats.sampling_rate
    rad_original = (tr.count() - 1) // 2 # number of samples before and after the middle point
    tr.data = tr.data.astype(float) 
    high_rough = np.quantile(tr.data, 0.95) # approximate "high" level of PPS signal
    low_rough = np.quantile(tr.data, 0.05) # approximate "low" level of PPS signal
    mid_rough = 0.5 * (high_rough + low_rough)
    high = np.mean(tr.data[np.where(tr.data > mid_rough)[0][rad_exclude:]]) # better "high" level
    low = np.mean(tr.data[np.where(tr.data < mid_rough)[0][:rad_exclude]]) # better "low" level
    
    ## data currently ranges from low to high; rescale from -1 to 1
    tr.data -= low # tr now ranges from 0 to (high - low)
    tr.data /= (high - low) # tr now ranges from 0 to 1
    tr.data = 2 * (tr.data - 0.5) # tr now ranges from -1 to 1

    ## Pad data so it starts with a step up, a long high, the same shaped step down, and a long low.
    padded_data = np.concatenate([tr.data, 1 + np.zeros(padded_length//2 - tr.count())]) 
    padded_data = np.concatenate([padded_data, -padded_data])
    # Shift so first sample is middle of step. This allows easy comparison to ideal signal.
    padded_data = np.concatenate([padded_data[rad_original:], padded_data[:rad_original]]) 

    ## Define ideal square wave for comparison.
    ideal = np.ones(padded_length)
    ideal[(padded_length//2):] = -1
    ideal[0] = 0
    ideal[padded_length//2] = 0

    ## A square wave's Fourier transform has a lot of zeros; omit them from upcoming calculations.
    keep = np.abs(fftshift(fft(ideal))) > 1e-9

    ## Calculate amplitude and phase relationship between observed and ideal signal.
    spec_deconvolved = fftshift(fft(padded_data) / waterlevel(fft(ideal)))[keep]
    freq = fftshift(fftfreq(padded_length, tr.stats.delta))[keep]
    amp = np.abs(spec_deconvolved)
    phase = np.unwrap(np.angle(spec_deconvolved))
    phase -= phase.mean() # time series are real, so spectrum must be hermitian

    ## Calculate the overall time offset and the time offset as a function of frequency.
    time = phase / (2 * np.pi * freq)
    offset = np.polyfit(freq[np.abs(freq) < freq_lim], phase[np.abs(freq) < freq_lim], 1)[0] / (2 * np.pi)
    return {'freq':freq, 'amp':amp, 'phase':phase, 'time':time, 'offset':offset}


def pps_phase_analysis(tr, rad = 0.08):
    """
    Calculate sub-sample timing errors from a trace consisting of pulse-per-second recordings.

    Parameters:

    tr: obspy.Trace trimmed to include PPS pulses all the way from beginning to end.
    rad: number of samples before and after PPS to analyze (depends on the pulse width of your GPS)

    Value: dict with following elements:
    time_series: list of zoomed-in views of PPS steps
    freq: numpy.array containing frequencies for spectra
    amp: list of recorded-to-ideal transfer functions (amplitude)
    phase: list of recorded-to-ideal transfer functions (phase, radians)
    time: list of time lag spectra (seconds)
    offset: numpy.array of calculated time offsets
    """
    eps = 1e-4

    ## Calculate times of PPS steps.
    t1 = 1 + np.ceil(tr.stats.starttime.timestamp)
    t2 = np.floor(tr.stats.endtime.timestamp)
    times = np.arange(t1, t2)

    ## Declare output lists and loop through the PPS pulses.
    time_series = []
    amp = []
    phase = []
    time = []
    offset = []
    for midtime in times:
        ## Extract a single PPS's time window from the full data.
        tr_win = tr.slice(obspy.UTCDateTime(midtime - rad - eps),
                          obspy.UTCDateTime(midtime + rad + eps))

        ## Calculate time offset information for this pulse, and save results.
        win_output = pps_phase_analysis_single(tr_win)
        time_series.append(tr_win.data)
        amp.append(win_output['amp'])
        phase.append(win_output['phase'])
        time.append(win_output['time'])
        offset.append(win_output['offset'])
        
    freq = win_output['freq'] # frequencies corresponding to output spectra

    return {'time_series':time_series, 'freq':freq, 'amp':amp, 'phase':phase, 'time':time, 'offset':np.array(offset)}
                   
def resample_trace(tr, offset):
    """Resample a trace to correct a timing offset.

    Parameters:
    
    tr: obspy.Trace including the data to resample
    offset: timing offset to correct (float, in seconds)

    Value: obspy.Trace that has been resampled.
    """
    tr_new = tr.copy()
    t = np.arange(tr.count()) * tr.stats.delta
    interp_function = interp1d(t, tr.data, 'cubic', bounds_error = False)
    tr_new.data = interp_function(t - offset)
    return tr_new

def waterlevel(x):
    eps = 1e-6
    x[abs(x) < eps] += eps
    return x

def plot_offset(offset_data):
    """ Plot time offsets as histograms and time series

    Parameters:
    offset_data: result of PPS_phase_analysis()
    """
    plt.subplot(2,1,1)
    plt.hist(offset_data['offset'])
    plt.xlabel('Time offset (seconds)')
    plt.title('Histogram of Time Offsets')
    plt.tight_layout()

    plt.subplot(2,1,2)
    t = np.arange(len(offset_data['time_series'][0]))
    t = t - (len(t)-1)/2
    for ts in offset_data['time_series']:
        plt.plot(t, ts)
    plt.axhline(0)
    plt.axvline(0) # 17 -> 16 -> 8: 8 samples before, 8 samples after
    plt.xlabel('Time after start of second (samples)')
    plt.title('Overlay of PPS Steps (%d Pulses)' % len(offset_data['time_series']))
    plt.tight_layout()
