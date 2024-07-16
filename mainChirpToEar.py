import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat, loadmat
from UserModules.pyDPOAEmodule import sendChirpToEar, giveRforTScalibration, getSClat, changeSampleRate

fsamp = 44100
MicGain = 40
lat_SC=8236  # 44100 Hz 2048 buffersize
lat_SC=8205  # 44100 Hz 2048 buffersize
#lat_SC=8356  # 44100 Hz 2048 buffersize
lat_SC=10284
AmpChirp = 0.04
buffersize = 2048

changeSampleRate(fsamp,buffersize,SC=10)
lat_SC = getSClat(fsamp,buffersize)

Hinear1, Hinear2, fxinear, y1, y2 = sendChirpToEar(AmpChirp=AmpChirp,fsamp=fsamp,MicGain=MicGain,Nchirps=300,buffersize=buffersize,latency_SC=lat_SC)
#xxx, Hinear1, Hinear2, fxinear = sendChirpToEar(fsamp,MicGain)

# choose SPL (0) or FPL (1) calibration
FPL = 1
pathfolder = 'Calibration_files/Files'  # path for saving calib file

if not os.path.exists(pathfolder):
    os.makedirs(pathfolder)


if FPL:
    Pecs1 = Hinear1*AmpChirp  # pressure in Pascals evoked by the first speaker
    Pecs2 = Hinear2*AmpChirp  # pressure in Pascals     evoked by the second speaker

    THpar = loadmat('Calibration_files/Files/THpar.mat')
    Psrc1 = THpar['Psrc1'][0]
    Psrc2 = THpar['Psrc2'][0]
    Zsrc1 = THpar['Zsrc1'][0]
    Zsrc2 = THpar['Zsrc2'][0]
    fxTS = THpar['fxTS'][0]
    
    Zec1 = Zsrc1*Pecs1/(Psrc1 - Pecs1) # impedance of ear canal
    Zec2 = Zsrc2*Pecs2/(Psrc2 - Pecs2) # impedance of ear canal

    R1,Z01 = giveRforTScalibration(Zec1,fxTS)
    R2,Z02 = giveRforTScalibration(Zec2,fxTS)

    Hinear1 = (Pecs1 / (1 + R1))/AmpChirp
    Hinear2 = (Pecs2 / (1 + R2))/AmpChirp
    


    filename = 'InEarCalData'
    caldata = {'Hinear1':Hinear1,'Hinear2':Hinear2,'fxinear':fxinear,'AmpChirp':AmpChirp,
    'Psrc1':Psrc1,'Psrc2':Psrc2,'Zsrc1':Zsrc1,'Zsrc2':Zsrc2,'Z01':Z01,'Z02':Z02,'fxTS':fxTS,'Zec1':Zec1,'Zec2':Zec2,'lat_SC':lat_SC,
     'Pecs1':Pecs1,'Pecs2':Pecs2}
    
    savemat(pathfolder + '/' + filename + '.mat', caldata)


    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.semilogx(fxinear/1e3,np.abs(Hinear1))
    ax1.semilogx(fxinear/1e3,np.abs(Hinear2))
    ax1.set_xlabel('Frequency (kHz)')
    ax1.set_ylabel('|H(f)| (-)')
    ax1.set_xlim((0.1, 13))
    ax1.set_title('Absolute value of the transfer function for each speaker')
    ax1.legend(('Speaker 1','Speaker 2'))
    
    #fig,ax = plt.subplots()
    ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(R1)))
    ax2.semilogx(fxTS/1e3,20*np.log10(np.abs(R2)))
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('|R| (dB re 1)')
    ax2.set_xlim((0.1, 13))
    ax2.set_title('Absolute value of the reflection coeficient for each speaker')
    ax2.legend(('Speaker 1','Speaker 2'))


    plt.show()
else:
    
    Pecs1 = Hinear1*AmpChirp  # pressure in Pascals evoked by the first speaker
    Pecs2 = Hinear2*AmpChirp  # pressure in Pascals evoked by the second speaker

    
    filename = 'InEarCalData'
    caldata = {'Hinear1':Hinear1,'Hinear2':Hinear2,'fxinear':fxinear,'AmpChirp':AmpChirp,
    'lat_SC':lat_SC,'Pecs1':Pecs1,'Pecs2':Pecs2}
    
    savemat(pathfolder + '/' + filename + '.mat', caldata)
    
    
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.semilogx(fxinear/1e3,np.abs(Hinear1))
    ax1.semilogx(fxinear/1e3,np.abs(Hinear2))
    ax1.set_xlabel('Frequency (kHz)')
    ax1.set_ylabel('|H(f)| (-)')
    ax1.set_xlim((0.1, 13))
    ax1.set_title('Absolute value of the transfer function for each speaker')
    ax1.legend(('Speaker 1','Speaker 2'))
    
    pREF = np.sqrt(2)*2e-5
    ax2.semilogx(fxinear/1e3,20*np.log10(np.abs(Pecs1)/pREF))
    ax2.semilogx(fxinear/1e3,20*np.log10(np.abs(Pecs2)/pREF))
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('|Pec| (dB SPL)')
    ax2.set_xlim((0.1, 13))
    ax2.set_title('Absolute value of the pressure in the ear canal')
    ax2.legend(('Speaker 1','Speaker 2'))
