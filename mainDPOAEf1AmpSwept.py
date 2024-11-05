# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:35:12 2024

@author: audiobunka
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:57:34 2024

@author: audiobunka
"""

import numpy as np
from scipy.io import savemat

import datetime
from UserModules.pyDPOAEmodule import RMEplayrec, makeTwoPureTones, calcDPstst, ValToPaSimple, getSClat, makeAmpSweptPureTone, makePureTone
from UserModules.pyUtilities import butter_highpass_filter
#from matplotlib.pyplot import Figure
import matplotlib.pyplot as plt
plt.close('all')

fsamp = 44100 # sampling freuqency
micGain = 40  # gain of the probe microphone
ear_t = 'L' # which ear
Twin = 100e-3 # window length for averaging in the time domain

ch1 = 1 # first tone goes to the channel 1
ch2 = 2 # second tone goes to the channel 2
buffersize = 2048
latency_SC = getSClat(fsamp,buffersize)


# parameters of evoking stimuli

fstep = 10
f2b = 1200  # f2 frequency
f2f1 = 1.2  # f2/f1 ratio
f1val = f2b/f2f1  # f1value
f1valR = np.round(f1val / 10) * 10 # round to the 10 Hz step for fft() calculation in 100 ms long windows



L1b = 25   # starting intensity in dB
L1e = 65   # end intensity in dB 

Lrate = 4/10   # ampl. swept rate seconds/10 dB

T = (L1e-L1b)*Lrate # tone duration

stepL1 = 5  # intensity step in dB

L2vect = [20,30,40,50]   # intensity of f2 tone
phi1 = 0
phi2 = 0


save_path = 'Results/fav02/'
subj_name = 's002'


def get_time() -> str:
    # to get current time
    now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return now_time

t = get_time() # current date and time
# data intialization
    
# generate stimuli    
#s1,s2 = generateDPOAEstimulus(f2f1, fsamp, f2b, f2e, r, L1, L2)    

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)


# measurement phase
counter = 0
#SpdpVect = np.full_like(Lvect, np.nan, dtype=np.complex128)
#SpNVect = np.full_like(L1vect, np.nan, dtype=np.complex128)

try:
    for i in range(len(L2vect)):
        L2 = L2vect[i]
        
        swTone = makeAmpSweptPureTone(f1valR,L1b,L1e,phi1,T,fsamp,ch1,fade=(4410,4410))
        pTone = makePureTone(f2b,L2,phi2,T,fsamp,ch2,fade=(4410,4410))
        if ch1==1 and ch2 == 2:
            twtones = np.column_stack((swTone, pTone, swTone+pTone))
        elif ch1==2 and ch2==1:
            twtones = np.column_stack((pTone, swTone, swTone+pTone))
        # based on data acquisition approach, make a matrix or not?        
        #s = np.vstack([s1,s2,s1+s2]).T  # make matrix where the first column
        
        recsig = RMEplayrec(twtones,fsamp,SC=10,buffersize=buffersize) # send signal to the sound card
        counter += 1
        print('Rep: {}'.format(counter))    
        # every recorded response is saved, so first make a dictionary with needed data
        data = {"recsig": recsig,"fsamp":fsamp,"f2f1":f2f1,"f2":f2b,"f1":f1valR,"L1b":L1b,"L1e":L1e,"L2":L2,"latency_SC":latency_SC,"Lrate":Lrate,"T":T}  # dictionary
            

        file_name = 'stampswDPOAE_' + subj_name + '_' + t[2:] + '_' + 'F2' + '_' + str(f2b) + 'F1' + '_' + str(f1valR) + 'L1b' + '_' + str(L1b) + 'dB' + 'L1e' + '_' + str(L1e) + 'dB' '_' + 'L2' + '_' + str(L2) + 'dB' + '_' + 'Lrate' + str(int(10*Lrate)) + ear_t
            
        savemat(save_path + '/' + file_name + '.mat', data)
        # now do processing to show the result to the experimenter
            #    
        

except KeyboardInterrupt:
    plt.show()
    pass

plt.show()