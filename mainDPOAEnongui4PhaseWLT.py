# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:17:04 2024

@author: audiobunka
"""

import numpy as np
from scipy.io import savemat, loadmat

import datetime
from UserModules.pyDPOAEmodule import RMEplayrec, generateDPOAEstimulusPhase, getSClat, mother_wavelet2, wavelet_filterDPOAE, calcDPgramTD, changeSampleRate
from UserModules.pyUtilities import butter_highpass_filter
import matplotlib.pyplot as plt
# parameters of evoking stimuli

f2b = 8000  # f1 start frequency
f2e = 500 # f2 end frequency
f2f1 = 1.2  # f2/f1 ratio
#L1 = 53    # intensity of f1 tone
L2 = 45   # intensity of f2 tone
L1 = int(0.4*L2+39)
#L1 = 55
r = 2   # sweep rate in octaves per second
fsamp = 44100; lat_SC=8236; bufsize = 2048  # 44100 Hz 2048 buffersize# sampling freuqency
fsamp = 96000; lat_SC= 16448; bufsize = 4096
#fsamp = 96000; lat_SC= 12352; bufsize = 4096
fsamp = 96000; lat_SC= 16436; bufsize = 4096
fsamp = 96000; lat_SC= 20532; bufsize = 4096
#fsamp = 96000; lat_SC= 20544; bufsize = 4096

changeSampleRate(fsamp,bufsize,SC=10)
lat_SC = getSClat(fsamp,bufsize,SC=10)
micGain = 40
ear_t = 'R' # which ear

plt.close('all')

#save_path = 'Results/s003'
#subj_name = 's003'
save_path = 'Results/fav01/'
subj_name = 's001'


def get_time() -> str:
    # to get current time
    now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return now_time


# data initialization
    
# generate stimuli    
s1p1,s2p1 = generateDPOAEstimulusPhase(f2f1, fsamp, f2b, f2e, 0,0, r, L1, L2)    

s1p2,s2p2 = generateDPOAEstimulusPhase(f2f1, fsamp, f2b, f2e, np.pi/2, np.pi, r, L1, L2)    

s1p3,s2p3 = generateDPOAEstimulusPhase(f2f1, fsamp, f2b, f2e, np.pi, 2*np.pi, r, L1, L2)    

s1p4,s2p4 = generateDPOAEstimulusPhase(f2f1, fsamp, f2b, f2e, 3*np.pi/2, 3*np.pi, r, L1, L2)    

# based on data acquisition approach, make a matrix or not?        
sp1 = np.vstack([s1p1,s2p1,s1p1+s2p1]).T  # make matrix where the first column
sp2 = np.vstack([s1p2,s2p2,s1p2+s2p2]).T  # make matrix where the first column
sp3 = np.vstack([s1p3,s2p3,s1p3+s2p3]).T  # make matrix where the first column
sp4 = np.vstack([s1p4,s2p4,s1p4+s2p4]).T  # make matrix where the first column
'''
    if self.ui.get_address() == "":
        save_path = '/home/audiobunka/Experiments/pyDPOAE/Results/pokus'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            save_path = self.ui.get_address()
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        t = get_time()
'''




if f2e>f2b:
    numofoct = np.log2(f2e/f2b)  # number of octaves for upward sweep
else:
    numofoct = np.log2(f2b/f2e)  # number of octaves for downward sweep

T = numofoct/r   # time duration of the sweep for the given sweep rate
        
t_ = get_time() # current date and time


# load calibration data and save them to results
calib_data = loadmat('Calibration_files/Files/InEarCalData.mat')

file_name = 'calib_data_' + subj_name + '_' + t_[2:] + '_' + ear_t
savemat(save_path + '/' + file_name + '.mat', calib_data)

# measurement phase
counter = 0

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)



try:
    while True:
        counter += 1
        print('Rep: {}'.format(counter))    
        recsigp1 = RMEplayrec(sp1,fsamp,SC=10,buffersize=bufsize) # send signal to the sound card
        #time.sleep(1) 
        recsigp2 = RMEplayrec(sp2,fsamp,SC=10,buffersize=bufsize) # send signal to the sound card
        #time.sleep(1) 
        recsigp3 = RMEplayrec(sp3,fsamp,SC=10,buffersize=bufsize) # send signal to the sound card
        #time.sleep(1) 
        recsigp4 = RMEplayrec(sp4,fsamp,SC=10,buffersize=bufsize) # send signal to the sound card
        if counter<10:  # to add 0 before the counter number
            counterSTR = '0' + str(counter)
        else:
            counterSTR = str(counter)    
        # every recorded response is saved, so first make a dictionary with needed data
        data = {"recsigp1": recsigp1,"recsigp2": recsigp2,"recsigp3": recsigp3,"recsigp4": recsigp4,"fsamp":fsamp,"f2f1":f2f1,"f2b":f2b,"f2e":f2e,"r":r,"L1":L1,"L2":L2,"lat_SC":lat_SC}  # dictionary
        file_name = 'p4swDPOAE_' + subj_name + '_' + t_[2:] + '_' + 'F2b' + '_' + str(f2b) + 'HzF2a' + '_' + str(f2e) + 'HzL1' + '_' + str(L1) + 'dB' + '_' + 'L2' + '_' + str(L2) + 'dB' + '_' + 'f2f1' + '_' + str(f2f1 * 100) + '_' + 'Oct' + '_' + str(r * 10) + '_' + ear_t + '_' + counterSTR
        savemat(save_path + '/' + file_name + '.mat', data)

        # now do processing to show the result to the experimenter
        #    
        cut_off = 200 # cut of frequency of the high pass filter
        recSigp1 = butter_highpass_filter(recsigp1[:,0], cut_off, fsamp, order=5)

        recsigp1 = np.reshape(recSigp1,(-1,1)) # reshape to a matrix with 1 column

        recSigp2 = butter_highpass_filter(recsigp2[:,0], cut_off, fsamp, order=5)

        recsigp2 = np.reshape(recSigp2,(-1,1)) # reshape to a matrix with 1 column

        recSigp3 = butter_highpass_filter(recsigp3[:,0], cut_off, fsamp, order=5)

        recsigp3 = np.reshape(recSigp3,(-1,1)) # reshape to a matrix with 1 column

        recSigp4 = butter_highpass_filter(recsigp4[:,0], cut_off, fsamp, order=5)

        recsigp4 = np.reshape(recSigp4,(-1,1)) # reshape to a matrix with 1 column

        #recsig = (recsigp1 + recsigp2 + recsigp3 + recsigp4)/4

        

        if counter == 1:
            ricMat1 = recsigp1[lat_SC:,0]
            ricMat2 = recsigp2[lat_SC:,0]
            ricMat3 = recsigp3[lat_SC:,0]
            ricMat4 = recsigp4[lat_SC:,0]
            
        elif counter > 1 and counter < 3:
            ricMat1 = np.c_[ricMat1, recsigp1[lat_SC:,0]]
            ricMat2 = np.c_[ricMat2, recsigp2[lat_SC:,0]]
            ricMat3 = np.c_[ricMat3, recsigp3[lat_SC:,0]]
            ricMat4 = np.c_[ricMat4, recsigp4[lat_SC:,0]]
            
        else:
            ricMat1 = np.c_[ricMat1, recsigp1[lat_SC:,0]]
            ricMat2 = np.c_[ricMat2, recsigp2[lat_SC:,0]]
            ricMat3 = np.c_[ricMat3, recsigp3[lat_SC:,0]]                
            ricMat4 = np.c_[ricMat4, recsigp4[lat_SC:,0]]                
                

            recMean1 = np.median(ricMat1,1)
            recMean2 = np.median(ricMat2,1)
            recMean3 = np.median(ricMat3,1)
            recMean4 = np.median(ricMat4,1)

            recMean1 = np.reshape(recMean1, (-1,1))
            recMean2 = np.reshape(recMean2, (-1,1))
            recMean3 = np.reshape(recMean3, (-1,1))
            recMean4 = np.reshape(recMean4, (-1,1))
            recMat1 = np.array(ricMat1)
            recMat2 = np.array(ricMat2)
            recMat3 = np.array(ricMat3)
            recMat4 = np.array(ricMat4)
            
            noiseM1 = recMat1 - recMean1
            noiseM2 = recMat2 - recMean2
            noiseM3 = recMat3 - recMean3
            noiseM4 = recMat4 - recMean4

            Nsamp = len(recMean1) # number of samples
            print(Nsamp)

            sigma1 = np.sqrt(1/(Nsamp*counter)*np.sum(np.sum(noiseM1**2,1)))
            sigma2 = np.sqrt(1/(Nsamp*counter)*np.sum(np.sum(noiseM2**2,1)))
            sigma3 = np.sqrt(1/(Nsamp*counter)*np.sum(np.sum(noiseM3**2,1)))
            sigma4 = np.sqrt(1/(Nsamp*counter)*np.sum(np.sum(noiseM4**2,1)))
            Theta1 = 8*sigma1 # estimation of the threshold for sample removal
            Theta2 = 8*sigma2 # estimation of the threshold for sample removal
            Theta3 = 8*sigma3 # estimation of the threshold for sample removal
            Theta4 = 8*sigma4 # estimation of the threshold for sample removal

            # now we have to make NAN where the noise was to large, but we have to do it in the wave
            # which 
            '''
            recMat1[np.abs(noiseM1)>Theta1] = np.nan 
            recMat1[np.abs(noiseM2)>Theta2] = np.nan
            recMat1[np.abs(noiseM3)>Theta3] = np.nan
            recMat1[np.abs(noiseM4)>Theta4] = np.nan
        
            recMat2[np.abs(noiseM1)>Theta1] = np.nan 
            recMat2[np.abs(noiseM2)>Theta2] = np.nan
            recMat2[np.abs(noiseM3)>Theta3] = np.nan
            recMat2[np.abs(noiseM4)>Theta4] = np.nan
        
            recMat3[np.abs(noiseM1)>Theta1] = np.nan 
            recMat3[np.abs(noiseM2)>Theta2] = np.nan
            recMat3[np.abs(noiseM3)>Theta3] = np.nan
            recMat3[np.abs(noiseM4)>Theta4] = np.nan
                                
            recMat4[np.abs(noiseM1)>Theta1] = np.nan 
            recMat4[np.abs(noiseM2)>Theta2] = np.nan
            recMat4[np.abs(noiseM3)>Theta3] = np.nan
            recMat4[np.abs(noiseM4)>Theta4] = np.nan
            
            noiseM1[np.abs(noiseM1)>Theta1] = np.nan
            noiseM1[np.abs(noiseM2)>Theta2] = np.nan
            noiseM1[np.abs(noiseM3)>Theta3] = np.nan
            noiseM1[np.abs(noiseM4)>Theta4] = np.nan

            noiseM2[np.abs(noiseM1)>Theta1] = np.nan
            noiseM2[np.abs(noiseM2)>Theta2] = np.nan
            noiseM2[np.abs(noiseM3)>Theta3] = np.nan
            noiseM2[np.abs(noiseM4)>Theta4] = np.nan

            noiseM3[np.abs(noiseM1)>Theta1] = np.nan
            noiseM3[np.abs(noiseM2)>Theta2] = np.nan
            noiseM3[np.abs(noiseM3)>Theta3] = np.nan
            noiseM3[np.abs(noiseM4)>Theta4] = np.nan

            noiseM4[np.abs(noiseM1)>Theta1] = np.nan
            noiseM4[np.abs(noiseM2)>Theta2] = np.nan
            noiseM4[np.abs(noiseM3)>Theta3] = np.nan
            noiseM4[np.abs(noiseM4)>Theta4] = np.nan


            oaeDS = (np.nanmean(recMat1,1) + np.nanmean(recMat2,1) + np.nanmean(recMat3,1) + np.nanmean(recMat4,1))/4  # exclude samples set to NaN (noisy samples)
            nfloorDS = (np.nanmean(noiseM1,1) + np.nanmean(noiseM2,1) + np.nanmean(noiseM3,1) + np.nanmean(noiseM4,1))/4
            
            '''
            
            Bl01chosen = (sum(np.abs(noiseM1)>Theta1))==0
            Bl02chosen = (sum(np.abs(noiseM2)>Theta2))==0
            Bl03chosen = (sum(np.abs(noiseM3)>Theta3))==0
            Bl04chosen = (sum(np.abs(noiseM4)>Theta4))==0
            
            N01chosen = sum(Bl01chosen)
            N02chosen = sum(Bl02chosen)
            N03chosen = sum(Bl03chosen)
            N04chosen = sum(Bl04chosen)
            
            
            recMat1Mean = np.mean(recMat1[:,Bl01chosen],1)
            recMat2Mean = np.mean(recMat2[:,Bl02chosen],1)
            recMat3Mean = np.mean(recMat3[:,Bl03chosen],1)
            recMat4Mean = np.mean(recMat4[:,Bl04chosen],1)
            
            recNoise1Mean = np.mean(noiseM1[:,Bl01chosen],1)
            recNoise2Mean = np.mean(noiseM2[:,Bl02chosen],1)
            recNoise3Mean = np.mean(noiseM3[:,Bl03chosen],1)
            recNoise4Mean = np.mean(noiseM4[:,Bl04chosen],1)
            
            oaeDS = (recMat1Mean+recMat2Mean+recMat3Mean+recMat4Mean)/4  # exclude samples set to NaN (noisy samples)
            nfloorDS = (recNoise1Mean+recNoise2Mean+recNoise3Mean+recNoise4Mean)/4  # exclude samples set to NaN (noisy samples)
            
            nfilt = 2**14
            #print(oaeDS)
            #calculate frequency response
            #[DPOAEcalib, DPOAEcalibNL, DPOAEcalibCR, NFLOORcalib, NFLOORcalibNL, fxI] = calcDPgramFAV_HS(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,r,nfilt,micGain)
            #[DPOAEcalib, DPOAEcalibNL, DPOAEcalibCR, NFLOORcalib, NFLOORcalibNL, fxI] = calcDPgramMatrix(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,r,nfilt,micGain)

                                    
            hm, hn,  hmN, hN = calcDPgramTD(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,r,nfilt,micGain)

            f2max = 8000  # maximal f2 frequency
    
            #t = np.arange(0,0.1,1/fs)
            t = np.arange(0,len(hm)/fsamp,1/fsamp)
            dt = 1/fsamp
            Nt = len(t)
            fx = np.arange(Nt)*fsamp/Nt   # frequency axis
            df = fx[1]-fx[0]

            Nw = Nt//2  # number of wavelet filters
            Nwmax = int(f2max//df)  # maximal wavelet which is usefull for the given frequency range


            vlnky = mother_wavelet2(Nw,Nt,df,dt)


            tDPOAEmwf, tNLmwf, cwLIN = wavelet_filterDPOAE(hm, vlnky,fx,fsamp)
            tDPOAENmwf,tNLNmwf,cwNLIN = wavelet_filterDPOAE(hmN, vlnky,fx,fsamp)  # noise floor

            #SLINmwf = 2*np.fft.rfft(np.concatenate((tLINmwf,np.zeros(int(2**15)))))/Nw  # calculate spectrum

            Nfft = 2**15
            fxx = np.arange(int(Nfft))*fsamp/(int(Nfft))
            nfilt = 2**12    
            SDPOAEmwf = np.fft.rfft(np.fft.fftshift(tDPOAEmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum
            SDPOAENmwf = np.fft.rfft(np.fft.fftshift(tDPOAENmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum
            SNLmwf = np.fft.rfft(np.fft.fftshift(tNLmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum
            SNLNmwf = np.fft.rfft(np.fft.fftshift(tNLNmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum


            
            fxx = np.arange(int(2**15))*fsamp/(int(2**15))



            ax1.clear()
            ax2.clear()                
            fx2 = f2f1*fxx/(2-f2f1)
            

            pREF = np.sqrt(2)*2e-5

            ax1.plot(fx2[:int(len(fx2)//2)+1],20*np.log10(np.abs(SDPOAEmwf)/pREF),color='C0',linewidth=1.2)
            ax1.plot(fx2[:int(len(fx2)//2)+1],20*np.log10(np.abs(SNLmwf)/pREF),color='C1',linewidth=1.2)
            ax1.plot(fx2[:int(len(fx2)//2)+1],20*np.log10(np.abs(SDPOAENmwf)/pREF),':',color='C0',linewidth=1.0)
            ax1.plot(fx2[:int(len(fx2)//2)+1],20*np.log10(np.abs(SNLNmwf)/pREF),':',color='C1',linewidth=1.0)


            ax1.set_xlim([500,8000])
            ax1.set_ylim([-40,20])

            ax1.set_ylabel('Amplitude (dB SPL)')
            ax1.legend(('DPOAE','NL comp.','CR comp.','noise floor'))

            DPphaseU = np.copy(np.angle(SDPOAENmwf)[~np.isnan(SDPOAENmwf)])
            DPphaseU = np.unwrap(DPphaseU)

            DPphaseNLU = np.copy(np.angle(SNLmwf)[~np.isnan(SNLmwf)])
            DPphaseNLU = np.unwrap(DPphaseNLU)
            fx2sub = fx2[:int(len(fx2)//2)+1]
            
            cycle = 2*np.pi
            ax2.plot(fx2sub[~np.isnan(SDPOAEmwf)],DPphaseU/cycle,color='C0',linewidth=1.2)
            ax2.plot(fx2sub[~np.isnan(SNLmwf)],DPphaseNLU/cycle,color='C1',linewidth=1.2)
            

            
            
            if f2b<f2e:
                ax1.set_xlim((f2b,f2e))
                ax2.set_xlim((f2b,f2e))
            else:
                ax1.set_xlim((f2e,f2b))            
                ax2.set_xlim((f2e,f2b))
            
            
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.0001) #Note this correction
            
    
except KeyboardInterrupt:
    plt.show()
    pass
            


        
    
