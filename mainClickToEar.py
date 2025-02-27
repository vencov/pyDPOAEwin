#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:01:42 2023

@author: audiobunka
"""



fsamp = 96000; bufsize = 4096; 


micGain = 40  # gain of the probe microphone
#ear_t = 'R' # which ear

# parameters of evoking stimuli

Nclicks = 1000

import numpy as np
import matplotlib.pyplot as plt
from UserModules.pyDPOAEmodule import generateClickTrain, RMEplayrec, getSClat, changeSampleRate
from UserModules.pyUtilities import butter_highpass_filter

plt.close('all')
changeSampleRate(fsamp,bufsize,SC=10)
latency_SC = getSClat(fsamp,bufsize,SC=10)


cfname = 'clickInBP_OK400_4000.mat'
Nclicks = 800
yupr = generateClickTrain(cfname, Nclicks)
clicktrainmat01 = np.vstack((yupr,np.zeros_like(yupr),yupr)).T 
import time
Amp = 0.1

recordedclicktrain01 = RMEplayrec(Amp*clicktrainmat01,fsamp,SC=10,buffersize=4096)



cutoff = 300 # cutoff frequency for high pass filter to filter out low frequency noise
recordedclicktrain01f = butter_highpass_filter(recordedclicktrain01[:,0], cutoff, fsamp, 3)
Npulse = 2048


fig,ax = plt.subplots()
ax.plot(recordedclicktrain01f)

#%% average in the time domain
rct01f = recordedclicktrain01f[latency_SC:]

rct01f = np.reshape(rct01f[:Nclicks*Npulse],(Npulse,Nclicks),order='F')

mrct01f = np.mean(rct01f,1)

y_dev = rct01f-mrct01f[:,np.newaxis]

# Example threshold
threshold = 0.01  # Adjust this based on your criteria

# Compute the maximum absolute deviation for each frame
max_dev_per_frame = np.max(np.abs(y_dev), axis=0)

# Identify frames where the deviation exceeds the threshold
frames_to_skip = np.where(max_dev_per_frame > threshold)[0]

print("Frames to skip:", frames_to_skip)


# Select only the columns that are NOT in frames_to_skip
valid_columns = np.setdiff1d(np.arange(rct01f[:, :].shape[1]), frames_to_skip)

# Compute the mean only for the selected columns
mrct01f = np.mean(rct01f[:, :][:, valid_columns], axis=1)

# calibrate
MG = 40

mrct01fPa = mrct01f/(0.003*10**(MG/20))
tx = np.linspace(0,Npulse*1/fsamp,Npulse)
fig,ax = plt.subplots()
ax.plot(tx,mrct01fPa)
ax.set_ylabel('Amplitude (Pa)')
ax.set_xlabel('Time (s)')
maxPa = max(mrct01fPa)
minPa = min(mrct01fPa)


print(f"max value: {maxPa}")
print(f"min value: {minPa}")

print(20*np.log10(0.5*(max(mrct01fPa)-min(mrct01fPa))/(np.sqrt(2)*2e-5)), 'peSPL')

plt.show()
