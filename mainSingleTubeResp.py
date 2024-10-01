# -*- coding: utf-8 -*-
"""
Measure responses in a single tube

Created on Wed Sep 25 11:17:10 2024

@author: audiobunka
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
from UserModules.pyDPOAEmodule import sendChirpToEar, getSClat, processChirpResp

plt.close('all')

fsamp = 44100
MicGain = 40
AmpChirp = 0.04
Nsamp = 2048
#fsamp=44100,MicGain=40,Nchirps=300,buffersize=2048,latency_SC=8236):


pathfolder = 'Calibration_files/FAV24'
filename = 'short_tube_27.7'   # here set the file name where the responses are saved
TL = 27.7
diameter = 8.3

buffersize = 2048
latSC = getSClat(fsamp,buffersize)

Hinear1, Hinear2, fxinear, y1, y2, y1all, y2all = sendChirpToEar(AmpChirp=AmpChirp,fsamp=fsamp,MicGain=40,Nchirps=300,buffersize=2048,latency_SC=latSC)
#(AmpChirp=AmpChirp,fsamp=fsamp,MicGain=MicGain,Nchirps=300,buffersize=buffersize,latency_SC=lat_SC)

# convert recorded responses to pascals

MicGain = 40 # dB gain set on the OAE probe
y1 = y1/(0.003*10**(MicGain/20))
y2 = y2/(0.003*10**(MicGain/20))

y1all = y1all/(0.003*10**(MicGain/20))
y2all = y2all/(0.003*10**(MicGain/20))


#caldata = {'Hinear1':Hinear1,'Hinear2':Hinear2,'fxinear':fxinear,'y1':y1,'y2':y2,'AmpChirp':AmpChirp,'Nsamp':Nsamp,'TL':TL}
caldata = {'y1all':y1all,'y2all':y2all,'AmpChirp':AmpChirp,'Nsamp':Nsamp,'TL':TL,'fsamp':fsamp,'latSC':latSC, "diameter": diameter}


savemat(pathfolder + '/' + filename + '.mat', caldata)

    #
        
fig,ax = plt.subplots()
ax.semilogx(fxinear,np.abs(Hinear1))
ax.semilogx(fxinear,np.abs(Hinear2))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude of the transfer function (-)')
ax.set_title('Transfer function for tube with length {} cm'.format(TL))

fig,ax = plt.subplots()
ax.plot(y1)
ax.plot(y2)
ax.set_xlabel('Samples (-)')
ax.set_ylabel('Amplitude (Pascals)')
ax.set_title('Averaged time domain response for tube length {} cm'.format(TL))

fig,ax = plt.subplots()
ax.plot(y1all)
ax.plot(y2all)
ax.set_xlabel('Samples (-)')
ax.set_ylabel('Amplitude (Pascals)')
ax.set_title('Overall recorded responses for tube length {} cm'.format(TL))




    
# input("Press Enter to continue...")
