#%%
from PPS_phase_analysis import *
import matplotlib.pyplot as plt

#%% Read a Gem trace containing recordings of a pulse-per-second signal
##  gemlog 1.3.11
##  firmware version:  0.962
##  Gem #188
##  PPS Source: Adafruit Ultimate GPS Breakout

tr = obspy.read('mseed/2018-11-27T09_10_04..188..HDF.mseed')[0]
#tr = obspy.read('mseed/2018-11-27T09_30_27..188..HDF.mseed')[0]

#%% Calculate the sub-sample time offset for this trace
output = pps_phase_analysis(tr, 0.08)
plt.figure(1)
plot_offset(output)

#%% Resample the trace to correct the offset, then recalculate to make sure it worked
tr_resampled = resample_trace(tr, output['offset'].mean())
output_resampled = pps_phase_analysis(tr_resampled)
plt.figure(2)
plot_offset(output_resampled)


