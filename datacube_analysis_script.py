#%%
from PPS_phase_analysis import *
import matplotlib.pyplot as plt

#%% Read a datacube trace containing recordings of a pulse-per-second signal
##  cube2mseed   Release: GIPPtools-2020.352
##  firmware version:  V2.0T
##  Datacube AD8
##  PPS Source: Adafruit Ultimate GPS Breakout

tr = obspy.read('mseed/AD8_p0_PPS.mseed')[0]

#%% Calculate the sub-sample time offset for this trace
output = pps_phase_analysis(tr, 0.08)
plt.figure(1)
plot_offset(output)

#%% Resample the trace to correct the offset, then recalculate to make sure it worked
tr_resampled = resample_trace(tr, output['offset'].mean())
output_resampled = pps_phase_analysis(tr_resampled)
plt.figure(2)
plot_offset(output_resampled)


