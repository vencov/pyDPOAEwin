# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:46:20 2024

@author: audiobunka
"""

# the main code for DPOAE steady state with LF biasing

import numpy as np
from scipy.io import savemat

import datetime
from UserModules.pyDPOAEmodule import makeTwoPureTones, calcDPstst, ValToPaSimple, getSClat, makeLFPureTone
from UserModules.pyRMEsd import RMEplayrecBias
from UserModules.pyUtilities import butter_highpass_filter
#from matplotlib.pyplot import Figure
import matplotlib.pyplot as plt

plt.close('all')

fsamp = 44100 # sampling freuqency
micGain = 0  # gain of the probe microphone
ear_t = 'L' # which ear
Twin = 100e-3 # window length for averaging in the time domain
T = 3  # tone duration
ch1 = 1 # first tone goes to the channel 1
ch2 = 2 # second tone goes to the channel 2
buffersize = 2048
latency_SC = getSClat(fsamp,buffersize)



fBias = 80
#LBias = 90
phiBias = 0





def get_time() -> str:
    # to get current time
    now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return now_time

t = get_time() # current date and time
# data intialization
    
# generate stimuli    
#s1,s2 = generateDPOAEstimulus(f2f1, fsamp, f2b, f2e, r, L1, L2)    

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


# measurement phase
counter = 0
SpdpVect = np.array([])
SpNVect = np.array([])


LBias = 90
     
lfBias = makeLFPureTone(fBias,LBias,phiBias,T,fsamp,fade=(4410,4410))


sigin = np.column_stack((0*lfBias, 0*lfBias, 0*lfBias, 0*lfBias, 0*lfBias, 0*lfBias, lfBias))            

# based on data acquisition approach, make a matrix or not?        
#s = np.vstack([s1,s2,s1+s2]).T  # make matrix where the first column

recsigR = RMEplayrecBias(sigin,fsamp,SC=10,buffersize=2048) # send signal to the sound card
    

Nskip = 10e3  # skip some samples in the onset
Nwin = 2*fsamp
recsig = recsigR[int(latency_SC+Nskip):int((latency_SC+Nskip)+Nwin),0]    #calculate frequency response
recsig /= 0.003*10**(micGain/20)
Thresh = 4e-5
#fig,ax = plt.subplots()
#ax.plot(recSig)
#plt.show()
#fig,ax = plt.subplots()
#ax.plot(np.abs(np.fft.rfft(recSig)))
#plt.show()


dpspect = 2*np.fft.rfft(recsig)/len(recsig)

fx = np.arange(len(recsig))/len(recsig)*fsamp

pREF = 2e-5*np.sqrt(2)
fig,ax = plt.subplots()
ax.plot(fx[:int(len(recsig)//2+1)],20*np.log10(np.abs(dpspect)/pREF))


differences = np.abs(fx - fdp)



# Find the index with the minimum difference
closest_index = np.argmin(differences)

# Get the value in fx that is closest to fdp
closest_value = fx[closest_index]
idxFdp = int(np.where(fx==closest_value)[0]) # find the index equal to freq value    

differences = np.abs(fxI - fdp)

# Find the index with the minimum difference
closest_index = np.argmin(differences)

# Get the value in fx that is closest to fdp
closest_value = fxI[closest_index]
fig,ax = plt.subplots()
ax.plot(fx[:int(len(fx)//2)+1],20*np.log10(np.abs(Spcalib)))
plt.show()
idxFdpISp = int(np.where(fxI==closest_value)[0]) # find the index equal to freq value    

SpDP = dpspect[idxFdp]
print(abs(SpDP))
# get noise bins

