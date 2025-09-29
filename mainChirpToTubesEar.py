import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
from UserModules.pyDPOAEmodule import sendChirpToEar, getSClat

fsamp = 44100
MicGain = 40
AmpChirp = 0.04
Nsamp = 2048
#fsamp=44100,MicGain=40,Nchirps=300,buffersize=2048,latency_SC=8236):


pathfolder = 'Calibration_files/Files'
#if not os.path.exists(pathfolder):
#    os.makedirs(pathfolder)
#filename = 'TScalTube01_080324_v2'
#filename = 'TScalTube02_080324'
#filename = 'TScalTube03_080324'
#filename = 'TScalTube04_080324'
#filename = 'TScalTube05_080324'
#filename = 'Syringe03'
#filename = 'TScalLLS_080324'
#TL = 72 # Tube01 tube length in mm
#TL = 62 # Tube02 tube length in mm
#TL = 54 # Tube03 tube length in mm
#TL = 37 # Tube04 tube length in mm
#TL = 19 # Tube05 tube length in mm
#TL = 'Sp03'
#TL = 'LLS'
TLlist = [72,62,55,37,26,'LLT']
TLlist = [28.9,34.0,40.1,69.1,82.7,'LLT']
FLlist = ['TScalTube05_071725vPH','TScalTube04_071725vPH','TScalTube03_071725vPH','TScalTube02_071725vPH','TScalTube01_071725vPH','TScalLLS_071725vPH']
latSC = 4169
#FLlist = ['TScalLLT_071725vPH']
#TLlist = ['LLT']

for i in range(len(TLlist)):
    buffersize = 2048
    latSC = getSClat(fsamp,buffersize)

    Hinear1, Hinear2, fxinear, y1, y2, ry1, ry2 = sendChirpToEar(AmpChirp=AmpChirp,fsamp=fsamp,MicGain=40,Nchirps=300,buffersize=2048,latency_SC=latSC)
    #Hinear1, Hinear2, fxinear, y_mean1, y_mean2, recordedchirp1, recordedchirp2
    #(AmpChirp=AmpChirp,fsamp=fsamp,MicGain=MicGain,Nchirps=300,buffersize=buffersize,latency_SC=lat_SC)
    TL = TLlist[i]

    caldata = {'Hinear1':Hinear1,'Hinear2':Hinear2,'fxinear':fxinear,'y1':y1,'y2':y2,'AmpChirp':AmpChirp,'Nsamp':Nsamp,'TL':TL}

    filename = FLlist[i]

    savemat(pathfolder + '/' + filename + '.mat', caldata)

    #
    
    fig,ax = plt.subplots()
    ax.semilogx(fxinear,np.abs(Hinear1))
    ax.semilogx(fxinear,np.abs(Hinear2))
    #plt.show()

    fig,ax = plt.subplots()
    ax.plot(y1)
    ax.plot(y2)
    plt.show()
    
    input("Press Enter to continue...")
