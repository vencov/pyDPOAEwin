# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 14:35:20 2025

Code for chirp train response in the ear canal for the FAV course. 
The code loads previosly saved response for long-lossy tube

@author: audiobunka
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat, loadmat
from UserModules.pyDPOAEmodule import sendChirpToEar, giveRforTScalibration, getSClat, changeSampleRate, calcPressuresFPL
from matplotlib.ticker import ScalarFormatter
fsamp = 44100
MicGain = 0
lat_SC=8236  # 44100 Hz 2048 buffersize
lat_SC=8205  # 44100 Hz 2048 buffersize
#lat_SC=8356  # 44100 Hz 2048 buffersize
lat_SC=10284
AmpChirp = 0.02
buffersize = 2048

changeSampleRate(fsamp,buffersize,SC=10)
lat_SC = getSClat(fsamp,buffersize)

Hinear1, Hinear2, fxinear, y1, y2, y1all, y2all = sendChirpToEar(AmpChirp=AmpChirp,fsamp=fsamp,MicGain=MicGain,Nchirps=300,buffersize=buffersize,latency_SC=lat_SC)
#xxx, Hinear1, Hinear2, fxinear = sendChirpToEar(fsamp,MicGain)

# choose SPL (0) or FPL (1) calibration
#pathfolder = 'Calibration_files/Files'  # path for saving calib file


Pecs1 = Hinear1*AmpChirp  # pressure in Pascals evoked by the first speaker
Pecs2 = Hinear2*AmpChirp  # pressure in Pascals     evoked by the second speaker

THpar = loadmat('Calibration_files/Files/THpar.mat')
#THpar = loadmat('Calibration_files/Files/THpar_071725vPH.mat')
Psrc1 = THpar['Psrc1'][0]/2
Psrc2 = THpar['Psrc2'][0]/2
Zsrc1 = THpar['Zsrc1'][0]
Zsrc2 = THpar['Zsrc2'][0]
fxTS = THpar['fxTS'][0]

Zec1 = Zsrc1*Pecs1/(Psrc1 - Pecs1) # impedance of ear canal
Zec2 = Zsrc2*Pecs2/(Psrc2 - Pecs2) # impedance of ear canal

R1,Z01 = giveRforTScalibration(Zec1,fxTS)
R2,Z02 = giveRforTScalibration(Zec2,fxTS)

Hinear1 = (Pecs1 / (1 + R1))/AmpChirp
Hinear2 = (Pecs2 / (1 + R2))/AmpChirp


# save data into a filename
pathfolder = 'Results/FAV/cv01_group02/'

subjN = 'fav6'  # subject name
filename = 'InEarData_' + subjN
caldata = {'Hinear1':Hinear1,'Hinear2':Hinear2,'fxinear':fxinear,'AmpChirp':AmpChirp,'Psrc1':Psrc1,'Psrc2':Psrc2,'Zsrc1':Zsrc1,'Zsrc2':Zsrc2,'Z01':Z01,'Z02':Z02,'fxTS':fxTS,'Zec1':Zec1,'Zec2':Zec2,'lat_SC':lat_SC,
     'Pecs1':Pecs1,'Pecs2':Pecs2,'y1all':y1all}
    

savemat(pathfolder + '/' + filename + '.mat', caldata)


#    fig,(ax1,ax2) = plt.subplots(1,2)
#    ax1.semilogx(fxinear/1e3,np.abs(Hinear1))
#    ax1.semilogx(fxinear/1e3,np.abs(Hinear2))
#    ax1.set_xlabel('Frequency (kHz)')
#    ax1.set_ylabel('|H(f)| (-)')
#    ax1.set_xlim((0.1, 13))
#    ax1.set_title('Absolute value of the transfer function for each speaker')
#    ax1.legend(('Speaker 1','Speaker 2'))
    
    #fig,ax = plt.subplots()
#    ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(R1)))
#    ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(R2)))
#    ax2.set_xlabel('Frequency (kHz)')
#    ax2.set_ylabel('|R| (dB re 1)')
#    ax2.set_xlim((0.1, 13))
#    ax2.set_title('Absolute value of the reflection coeficient for each speaker')
#    ax2.legend(('Speaker 1','Speaker 2'))


# compare figures:
Pfor1,Prev1,Rscr1,Pifw1 = calcPressuresFPL(Pecs1, R1, Zsrc1, Z01)
Pfor2,Prev2,Rscr2,Pifw2 = calcPressuresFPL(Pecs2, R2, Zsrc2, Z02)

refP = np.sqrt(2)*2e-5  # reference pressure
fig,(ax1,ax2) = plt.subplots(2,1, figsize=(7,8))

# --- First speaker ---
ax1.semilogx(fxTS/1e3,20*np.log10(np.abs(Pecs1/refP)),'k',label="Total pressure near the source")
ax1.semilogx(fxTS/1e3,20*np.log10(np.abs(Pfor1/refP)),'r--',label="Forward waves")
ax1.semilogx(fxTS/1e3,20*np.log10(np.abs(Prev1/refP)),'b--',label="Reverse waves")
ax1.semilogx(fxTS/1e3,20*np.log10(np.abs(Pifw1/refP)),'r',label="Initial outgoing pressure wave")
ax1.set_xlabel('Frequency (kHz)')
ax1.set_ylabel('|P| (dB SPL)')
ax1.set_title("First speaker")
ax1.set_xlim((0.3, 13))
ax1.legend()  # legend for first subplot

# Format x-axis ticks in kHz, not scientific notation
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.set_xticks([0.5, 1, 2, 4, 8, 12])
ax1.set_xticklabels(['0.5', '1', '2', '4', '8', '12'])

# --- Second speaker ---
ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(Pecs2/refP)),'k',label="Total pressure near the source")
ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(Pfor2/refP)),'r--',label="Forward waves")
ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(Prev2/refP)),'b--',label="Reverse waves")
ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(Pifw2/refP)),'r',label="Initial outgoing pressure wave")
ax2.set_xlabel('Frequency (kHz)')
ax2.set_ylabel('|P| (dB SPL)')
ax2.set_title("Second speaker")
ax2.set_xlim((0.3, 13))
ax2.legend()  # legend for second subplot

# Format x-axis ticks in kHz, not scientific notation
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.set_xticks([0.5, 1, 2, 4, 8, 12])
ax2.set_xticklabels(['0.5', '1', '2', '4', '8', '12'])

plt.tight_layout()



PecsLLT1 = Hinear1*AmpChirp  # pressure in Pascals evoked by the first speaker
PecsLLT2 = Hinear2*AmpChirp  # pressure in Pascals evoked by the second speaker


from scipy.io import loadmat

# --- Load data from the .mat file ---
filename = 'InLLTData'
data = loadmat(pathfolder + '/' + filename + '.mat')


PecsLLT1 = data['PecsLLT1'][0]
PecsLLT2 = data['PecsLLT2'][0]

# Now all variables are loaded and ready to use



fig,(ax1,ax2) = plt.subplots(2,1, figsize=(7,8))

# --- First speaker ---
ax1.semilogx(fxTS/1e3,20*np.log10(np.abs(Pecs1/PecsLLT1)),'k',label="Total pressure near the source")
ax1.semilogx(fxTS/1e3,20*np.log10(np.abs(Pfor1/PecsLLT1)),'r--',label="Forward waves")
ax1.semilogx(fxTS/1e3,20*np.log10(np.abs(Prev1/PecsLLT1)),'b--',label="Reverse waves")
ax1.semilogx(fxTS/1e3,20*np.log10(np.abs(Pifw1/PecsLLT1)),'r',label="Initial outgoing pressure wave")
ax1.set_xlabel('Frequency (kHz)')
ax1.set_ylabel('|P/P_{LLT}| (dB)')
ax1.set_title("First speaker, relative to LLT response")
ax1.set_xlim((0.3, 13))
ax1.legend()  # legend for first subplot

# Format x-axis ticks in kHz, not scientific notation
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.set_xticks([0.5, 1, 2, 4, 8, 12])
ax1.set_xticklabels(['0.5', '1', '2', '4', '8', '12'])

# --- Second speaker ---
ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(Pecs2/PecsLLT2)),'k',label="Total pressure near the source")
ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(Pfor2/PecsLLT2)),'r--',label="Forward waves")
ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(Prev2/PecsLLT2)),'b--',label="Reverse waves")
ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(Pifw2/PecsLLT2)),'r',label="Initial outgoing pressure wave")
ax2.set_xlabel('Frequency (kHz)')
ax2.set_ylabel('|P/P_{LLT}| (dB)')
ax2.set_title("Second speaker, relative to LLT response")
ax2.set_xlim((0.3, 13))
ax2.legend()  # legend for second subplot

# Format x-axis ticks in kHz, not scientific notation
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.set_xticks([0.5, 1, 2, 4, 8, 12])
ax2.set_xticklabels(['0.5', '1', '2', '4', '8', '12'])

plt.tight_layout()


            
        
plt.show()
