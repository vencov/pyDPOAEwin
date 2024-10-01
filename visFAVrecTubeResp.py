# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:46:46 2024

@author: audiobunka
"""

from scipy.io import loadmat
from UserModules.pyDPOAEmodule import processChirpResp
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
dataLLT = loadmat('Calibration_files/FAV24/long_tube_NAN.mat')
dataW = loadmat('Calibration_files/FAV24/long_tube_62.mat')



y1LLT = dataLLT['y1all'][0]
y2LLT = dataLLT['y2all'][0]
Nsamp = dataLLT['Nsamp'][0][0]
fsamp = dataLLT['fsamp'][0][0]
latSC = dataLLT['latSC'][0][0]
Nchirps = 300

y1W = dataW['y1all'][0]
y2W = dataW['y2all'][0]


# responses in the long lossy tube
chR1LLT, chR2LLT, fx = processChirpResp(y1LLT,y2LLT,latSC,fsamp,Nsamp,Nchirps)

# responses in the long lossy tube
chR1W, chR2W, fx = processChirpResp(y1W,y2W,latSC,fsamp,Nsamp,Nchirps)



fig,ax = plt.subplots()
ax.plot(fx,np.abs(chR1W/chR1LLT))
ax.plot(fx,np.abs(chR2W/chR2LLT))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Normalized response amplitude (-)')
