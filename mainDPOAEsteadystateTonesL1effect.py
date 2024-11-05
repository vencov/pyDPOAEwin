# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:57:34 2024

@author: audiobunka
"""

import numpy as np
from scipy.io import savemat

import datetime
from UserModules.pyDPOAEmodule import RMEplayrec, makeTwoPureTones, calcDPstst, ValToPaSimple, getSClat
from UserModules.pyUtilities import butter_highpass_filter
#from matplotlib.pyplot import Figure
import matplotlib.pyplot as plt
plt.close('all')

fsamp = 44100 # sampling freuqency
micGain = 40  # gain of the probe microphone
ear_t = 'L' # which ear
Twin = 100e-3 # window length for averaging in the time domain
T = 4  # tone duration
ch1 = 1 # first tone goes to the channel 1
ch2 = 2 # second tone goes to the channel 2
buffersize = 2048
latency_SC = getSClat(fsamp,buffersize)


# parameters of evoking stimuli

fstep = 10
f2b = 5000  # f2 frequency
f2f1 = 1.2  # f2/f1 ratio
f1val = f2b/f2f1  # f1value
f1valR = np.round(f1val / 10) * 10 # round to the 10 Hz step for fft() calculation in 100 ms long windows



L1b = 35   # starting intensity in dB
L1e = 65   # end intensity in dB 
stepL1 = 5  # intensity step in dB
L1vect = np.arange(L1b,L1e+stepL1,stepL1)
L2 = 40   # intensity of f2 tone
phi1 = 0
phi2 = 0


save_path = 'Results/pokus/'
subj_name = 's00p'


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
SpdpVect = np.full_like(L1vect, np.nan, dtype=np.complex128)
SpNVect = np.full_like(L1vect, np.nan, dtype=np.complex128)

try:
    for i in range(len(L1vect)):
        L1 = L1vect[i]
        twtones = makeTwoPureTones(f1valR,f2b,L1,L2,phi1,phi2,T,fsamp,ch1,ch2,fade=(4410,4410))

        # based on data acquisition approach, make a matrix or not?        
        #s = np.vstack([s1,s2,s1+s2]).T  # make matrix where the first column
        
        recsig = RMEplayrec(twtones,fsamp,SC=10,buffersize=2048) # send signal to the sound card
        counter += 1
        print('Rep: {}'.format(counter))    
        # every recorded response is saved, so first make a dictionary with needed data
        data = {"recsig": recsig,"fsamp":fsamp,"f2f1":f2f1,"f2":f2b,"f1":f1valR,"L1":L1,"L2":L2,"latency_SC":latency_SC}  # dictionary
            

        file_name = 'ststDPOAE_' + subj_name + '_' + t[2:] + '_' + 'F2' + '_' + str(f2b) + 'F1' + '_' + str(f1valR) + 'L1' + '_' + str(L1) + 'dB' + '_' + 'L2' + '_' + str(L2) + 'dB' + '_' +  ear_t
            
        savemat(save_path + '/' + file_name + '.mat', data)
        # now do processing to show the result to the experimenter
            #    
        cut_off = 200 # cut of frequency of the high pass filter
        #recSig = butter_highpass_filter(recsig[:,0], cut_off, fsamp, order=5)
        recSig = recsig[:,0]
        f2 = f2b    
        f1 = f1valR

        fdp = 2*f1-f2

        recSig = recSig[latency_SC:]    #calculate frequency response
        Thresh = 4e-5   # threshold for signal removal
        #fig,ax = plt.subplots()
        #ax.plot(recSig)
        #plt.show()
        #fig,ax = plt.subplots()
        #ax.plot(np.abs(np.fft.rfft(recSig)))
        #plt.show()
        mean_oae, selected, Spcalib, fxI = calcDPstst(recSig,f2f1,f2,f1,fsamp,Twin,T,micGain,Thresh)
        
        
        dpspect = 2*np.fft.rfft(mean_oae)/len(mean_oae)

        fx = np.arange(len(mean_oae))*fsamp/len(mean_oae)

        differences = np.abs(fx - fdp)

        # Find the index with the minimum difference
        closest_index = np.argmin(differences)

        # Get the value in fx that is closest to fdp
        closest_value = fx[closest_index]
        idxFdp = int(np.where(fx==closest_value)[0]) # find the index equal to freq value    
        
        differences = np.abs(fx - fdp)

        # Find the index with the minimum difference
        closest_index = np.argmin(differences)

        # Get the value in fx that is closest to fdp
        closest_value = fx[closest_index]

        #fig,ax = plt.subplots()
        ax1.clear()
        ax1.plot(fx[:int(len(fx)//2)+1]/1000,20*np.log10(np.abs(Spcalib)/(np.sqrt(2)*2e-5)))
        ax1.set_xlabel('Frequency f_2 (kHz)')        
        ax1.set_ylabel('Ampl. (dB SPL)')        
        idxFdpISp = int(np.where(fx==closest_value)[0]) # find the index equal to freq value    

        SpDP = dpspect[idxFdp]
        #print(abs(SpDP))
        # get noise bins

        fxnoise = np.concatenate((fx[idxFdp-4:idxFdp-1],fx[idxFdp+2:idxFdp+5]))
        dpnoise = np.concatenate((dpspect[idxFdp-4:idxFdp-1],dpspect[idxFdp+2:idxFdp+5]))
        #(MicCalFN,Val,fxV,MicGain):
        
        SpnoisePa = ValToPaSimple(dpnoise,micGain) # convert to Pa
        Spn = np.mean(abs(SpnoisePa))  # mean value across the noise bins 
        print(Spn)

        #(MicCalFN,Val,fxV,MicGain):
        SpdpPa = ValToPaSimple(SpDP,micGain) # co
        print(SpdpPa)
        SpdpVect[i] = SpdpPa
        SpNVect[i] = Spn
            
        ax2.clear()
        ax3.clear()                
        fx2 = f2f1*fxI/(2-f2f1)
        ax2.plot(L1vect,20*np.log10(abs(SpdpVect)/(np.sqrt(2)*2e-5)),'o-')
        ax2.plot(L1vect,20*np.log10(abs(SpNVect)/(np.sqrt(2)*2e-5)),'+:')
        
        #ax1.plot(fx2,20*np.log10(abs(DPOAEcalibNL)/(np.sqrt(2)*2e-5)))
        #    ax1.plot(fx2,20*np.log10(abs(DPOAEcalibCR)/(np.sqrt(2)*2e-5)))
        #    ax1.plot(fx2,20*np.log10(abs(NFLOORcalib)/(np.sqrt(2)*2e-5)),':')
        #    ax1.plot(fx2,20*np.log10(abs(NFLOORcalibNL)/(np.sqrt(2)*2e-5)),':')
        #ax1.set_ylim((-30,30))
        ax2.set_xlim((L1b-stepL1,L1e+stepL1))
        ax2.set_xlabel('L1 (dB)')        
        ax2.set_ylabel('DP amplitude (dB SPL)')        
        ax2.legend('DPOAE','Noise floor')
        ax3.plot(L1vect,np.angle(SpdpVect)/(2*np.pi))
        
        #ax2.set_ylim((-30,70))
        ax3.set_xlim((L1b-stepL1,L1e+stepL1))
        ax3.set_xlabel('L1 (dB)')        
        ax3.set_ylabel('Phase (cycles)')                    

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.0001) #Note this correction


except KeyboardInterrupt:
    plt.show()
    pass

plt.show()