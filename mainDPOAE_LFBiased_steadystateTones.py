# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:41:50 2024

@author: audiobunka
"""

# the main code for DPOAE steady state with LF biasing

import numpy as np
from scipy.io import savemat

import datetime
from UserModules.pyDPOAEmodule import makeTwoPureTones, calcDPststBias, ValToPaSimple, getSClat, makeLFPureTone
from UserModules.pyRMEsd import RMEplayrecBias
from UserModules.pyUtilities import butter_highpass_filter
#from matplotlib.pyplot import Figure
import matplotlib.pyplot as plt

plt.close('all')

fsamp = 44100 # sampling freuqency
micGain = 0  # gain of the probe microphone
ear_t = 'L' # which ear
Nwin = 44100   # window duration for analysis
T = 20  # tone duration
ch1 = 1 # first tone goes to the channel 1
ch2 = 2 # second tone goes to the channel 2
buffersize = 2048
latency_SC = getSClat(fsamp,buffersize)


# parameters of evoking stimuli

fstep = 10
f2b = 2400  # f1 start frequency
f2e = 2400 # f2 end frequency
f2f1 = 1.2  # f2/f1 ratio
L1 = 65   # intensity of f1 tone
L2 = 55   # intensity of f2 tone
phi1 = 0
phi2 = 0

fBias = 32
LBias = 85
phiBias = 0


f2val = np.arange(f2b,f2e+fstep,fstep) # make a numpy array with f2 values
f1val = f2val/f2f1  # f1values
f1valR = np.round(f1val / 10) * 10 # round to the 10 Hz step for fft() calculation in 100 ms long windows
f1valR = [int(val) for val in f1valR]  # convert to integers


save_path = 'Results/s004/LFbias/'
subj_name = 's004'


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

Nrep = 
try:
    for i in range(Nrep):
    
        twtones = makeTwoPureTones(f1valR[i],f2val[i],L1,L2,phi1,phi2,T,fsamp,ch1,ch2,fade=(4410,4410))
        lfBias = makeLFPureTone(fBias,LBias,phiBias,T,fsamp,fade=(4410,4410))

        sigin = np.column_stack((twtones, 0*lfBias, 0*lfBias, lfBias))            

        # based on data acquisition approach, make a matrix or not?        
        #s = np.vstack([s1,s2,s1+s2]).T  # make matrix where the first column
        
        recsig = RMEplayrecBias(sigin,fsamp,SC=10,buffersize=2048) # send signal to the sound card
        counter += 1
        print('Rep: {}'.format(counter))    
        # every recorded response is saved, so first make a dictionary with needed data
        data = {"recsig": recsig,"fsamp":fsamp,"f2f1":f2f1,"f2":f2val[i],"f1":f1valR[i],"L1":L1,"L2":L2,'micGain':micGain,'latency_SC':latency_SC,'fBias':fBias,'Lbias':Lbias}  # dictionary
            

        file_name = 'LFbststDPOAE_' + subj_name + '_' + t[2:] + '_' + 'F2' + '_' + str(f2val[i]) + 'F1' + '_' + str(f1valR[i]) + 'L1' + '_' + str(L1) + 'dB' + '_' + 'L2' + '_' + str(L2) + 'dB' + '_' + 'f2f1' + '_' + str(int(f2f1 * 100)) + '_fbias' str(fBias) + 'Hz_Lbias' + str(LBias) + 'dB_' + ear_t
            
        savemat(save_path + '/' + file_name + '.mat', data)
        # now do processing to show the result to the experimenter
            #    
        cut_off = 200 # cut of frequency of the high pass filter
        recSig = butter_highpass_filter(recsig[:,0], cut_off, fsamp, order=5)
        
        f2 = f2val[i]    
        f1 = f1valR[i]

        fdp = 2*f1-f2

        recSig = recSig[latency_SC:]    #calculate frequency response
        Thresh = 4e-1
        #fig,ax = plt.subplots()
        #ax.plot(recSig)
        #plt.show()
        #fig,ax = plt.subplots()
        #ax.plot(np.abs(np.fft.rfft(recSig)))
        #plt.show()
        mean_oae, selected, Spcalib, fxI = calcDPststBias(recSig,f2f1,f2,f1,fsamp,Nwin,micGain,Thresh)
        
        
        dpspect = 2*np.fft.rfft(mean_oae)/len(mean_oae)

        fx = np.arange(len(mean_oae))*fsamp/len(mean_oae)

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
        ax.plot(fx[:int(len(fx)//2)+1],20*np.log10(np.abs(Spcalib)/(np.sqrt(2)*2e-5)))


        # indices of nearest FFT bins to desired freqs
        freqs_interest = [fdp + m*fBias for m in range(-3, 4)]
        idx_interest = [np.argmin(np.abs(fx - f)) for f in freqs_interest]

        Sp_filt = np.zeros_like(Spcalib, dtype=complex)
        for idx in idx_interest:
            Sp_filt[idx] = Spcalib[idx]
            
            
        N = len(Spcalib)
        for idx in idx_interest:
            if idx != 0 and idx != N//2:  # skip DC and Nyquist
                Sp_filt[-idx] = np.conj(Spcalib[idx])            
    
        
        sig_filt = np.fft.ifft(Sp_filt)
        sig_filt = np.real(sig_filt)  # enforce real
        
        from scipy.signal import hilbert

        analytic_sig = hilbert(sig_filt)
        envelope = np.abs(analytic_sig)

        plt.show()



    

        
        idxFdpISp = int(np.where(fxI==closest_value)[0]) # find the index equal to freq value    




        SpDP = dpspect[idxFdp]
        print(abs(SpDP))
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
        
        
        
#        SpdpVect = np.concatenate((SpdpVect,np.array(SpdpPa)))
#        SpNVect = np.concatenate((SpNVect,np.array([Spn])))
            
 #       ax1.clear()
 #       ax2.clear()                
  #      fx2 = f2f1*fxI/(2-f2f1)
  #      ax1.plot(f2val[:len(SpdpVect)],20*np.log10(abs(SpdpVect)/(np.sqrt(2)*2e-5)),'o-')
   #     ax1.plot(f2val[:len(SpNVect)],20*np.log10(abs(SpNVect)/(np.sqrt(2)*2e-5)),'+:')
        #ax1.plot(fx2,20*np.log10(abs(DPOAEcalibNL)/(np.sqrt(2)*2e-5)))
        #    ax1.plot(fx2,20*np.log10(abs(DPOAEcalibCR)/(np.sqrt(2)*2e-5)))
        #    ax1.plot(fx2,20*np.log10(abs(NFLOORcalib)/(np.sqrt(2)*2e-5)),':')
        #    ax1.plot(fx2,20*np.log10(abs(NFLOORcalibNL)/(np.sqrt(2)*2e-5)),':')
        #ax1.set_ylim((-30,30))
  #      ax1.set_xlim((5,100))
  #      ax1.set_xlabel('Frequency $f_2$ (Hz)')        
  #      ax1.set_ylabel('DP amplitude (dB SPL)')        
  #      ax1.legend('DPOAE','Noise floor')
  #      ax2.plot(fxI,20*np.log10(abs(Spcalib)/(np.sqrt(2)*2e-5)))
  #      ax2.plot(fxI[idxFdpISp],20*np.log10(abs(Spcalib[idxFdpISp])/(np.sqrt(2)*2e-5)),'x')
        #ax2.set_ylim((-30,70))
  #      ax2.set_xlim([5,100])
  #      ax2.set_xlabel('Frequency (Hz)')        
  #      ax2.set_ylabel('Amplitude (dB SPL)')                    

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.0001) #Note this correction


except KeyboardInterrupt:
    plt.show()
    pass

plt.show()