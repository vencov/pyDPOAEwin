# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:41:23 2024

@author: audiobunka
"""


from scipy.io import loadmat
import os
from UserModules.pyUtilities import butter_highpass_filter 
import numpy as np
import matplotlib.pyplot as plt

import UserModules.pyDPOAEmodule as pDP
    
plt.close('all')
# for s056
path = "Results/s004/speedeffect/"
subj_name = 's004'

# left ear
#ear = 'left'
#deL_r2 = '24_02_14_14_13_59'
#deL_r4 = '24_02_14_14_15_13' # left ear, L2=30
#deL_r8 = '24_02_14_14_16_10'
#deL_r12 = '24_02_14_14_17_04'


path = "Results/s003/rate/"
#subj_name = 's003'

# left ear
ear = 'left'
deL_r2 = '24_03_20_12_59_37'
deL_r4 = '24_03_20_13_00_47' # left ear, L2=30
deL_r6 = '24_03_20_13_01_36'
deL_r8 = '24_03_20_13_02_24'
deL_r10 = '24_03_20_13_03_16'


f2f1 = 1.2

fsamp = 96000; lat = 16448

subjD = {}


subjD['s055L_L2_30'] = ['Results/s055/sweep_rate/', '24_06_04_14_06_37_F2b', '24_06_04_14_08_26_F2b', '24_06_04_14_10_17_F2b']
subjD['s055L_L2_55'] = ['Results/s055/sweep_rate/', '24_06_04_14_00_59_F2b', '24_06_04_14_02_32_F2b', '24_06_04_14_03_51_F2b']
subjD['s055R_L2_30'] = ['Results/s055/sweep_rate/', 'p4swDPOAE_s055_24_06_04_16_54_22_F2b_8000HzF2a_500HzL1_51dB_L2_30dB_f2f1_120.0_Oct_20_R_', 'p4swDPOAE_s055_24_06_04_16_56_29_F2b_8000HzF2a_500HzL1_51dB_L2_30dB_f2f1_120.0_Oct_80_R_', 'p4swDPOAE_s055_24_06_04_16_58_25_F2b_8000HzF2a_500HzL1_51dB_L2_30dB_f2f1_120.0_Oct_120_R_', 'p4swDPOAE_s055_24_06_04_14_17_53_F2b_8000HzF2a_500HzL1_61dB_L2_55dB_f2f1_120.0_Oct_20_R_', 'p4swDPOAE_s055_24_06_04_14_19_42_F2b_8000HzF2a_500HzL1_61dB_L2_55dB_f2f1_120.0_Oct_80_R_', 'p4swDPOAE_s055_24_06_04_14_21_18_F2b_8000HzF2a_500HzL1_61dB_L2_55dB_f2f1_120.0_Oct_120_R_', 'p4swDPOAE_s055_24_06_04_16_48_24_F2b_8000HzF2a_500HzL1_61dB_L2_55dB_f2f1_120.0_Oct_20_R_', 'p4swDPOAE_s055_24_06_04_16_50_29_F2b_8000HzF2a_500HzL1_61dB_L2_55dB_f2f1_120.0_Oct_80_R_', 'p4swDPOAE_s055_24_06_04_16_51_50_F2b_8000HzF2a_500HzL1_61dB_L2_55dB_f2f1_120.0_Oct_120_R_']

subjD['L2_55_dB_L'] = ['Results/s055/sweep_rate/', '24_06_04_14_00_59_F2b_8000Hz', '24_06_04_14_02_32_F2b_8000Hz', '24_06_04_14_03_51_F2b_8000Hz']
subjD['L2_30_dB_L'] = ['Results/s055/sweep_rate/', '24_06_04_14_06_37_F2b_8000Hz', '24_06_04_14_08_26_F2b_8000Hz', '24_06_04_14_10_17_F2b_8000Hz']
subjD['L2_55_dB_R'] = ['Results/s055/sweep_rate/', '24_06_04_14_17_53_F2b_8000Hz', '24_06_04_14_19_42_F2b_8000Hz', '24_06_04_14_21_18_F2b_8000Hz', '24_06_04_16_48_24_F2b_8000Hz', '24_06_04_16_50_29_F2b_8000Hz', '24_06_04_16_51_50_F2b_8000Hz']
subjD['L2_30_dB_R'] =  ['Results/s055/sweep_rate/', '24_06_04_16_54_22_F2b_8000Hz', '24_06_04_16_56_29_F2b_8000Hz', '24_06_04_16_58_25_F2b_8000Hz']

subjD['L2_50_dB_L'] = ['Results/s089/R/', '24_07_01_11_27_00_F2b_8000Hz', '24_07_01_11_28_39_F2b_8000Hz', '24_07_01_11_30_23_F2b_8000Hz', '24_07_01_11_31_37_F2b_8000Hz']
subjD['L2_55_dB_L'] = ['Results/s089/R/', '24_07_01_11_33_17_F2b_8000Hz', '24_07_01_11_34_49_F2b_8000Hz', '24_07_01_11_36_08_F2b_8000Hz', '24_07_01_11_37_15_F2b_8000Hz']

subjN = 'L2_55_dB_L'
#subjN = 's055L_L2_55'

def mother_wavelet2(Nw,Nt,df,dt):
    vlnky = np.zeros((Nt,Nw))
    tx = (np.arange(Nt)-Nw)*dt
    for k in range(Nw):
        vlnky[:,k] = np.cos(2*np.pi*(k+1)*df*tx)*1/(1+(0.075*(k+1)*2*np.pi*df*tx)**4)
    return vlnky


def wavelet_filterDPOAE(signal, wavelet_basis,fx):
    """
    Perform wavelet filtering on a time-domain signal.

    Parameters:
    - signal: np.array, time-domain signal
    - wavelet_basis: np.array, wavelet basis functions
    
    Returns:
    - filtered_signal: np.array, filtered signal
    """
    # Compute the length of the signal
    N = len(signal)
    
    # Compute the length of the wavelet basis
    M = wavelet_basis.shape[0]
    
    # Compute the number of wavelet basis functions
    NW = wavelet_basis.shape[1]
    
    # Initialize filtered signal
    filtered_signalDPOAE = np.zeros(N)  # prealocate memory for overall filtered DPOAE
    filtered_signalNL = np.zeros(N) # prealocate memory for zero-latency component (SL component)
    coefwtiDPOAE = np.zeros_like(wavelet_basis)
    # Perform wavelet filtering
    
    
    
    for k in range(20,400):
        # Compute the Fourier transform of the wavelet basis function
        wbtf = 2*np.fft.fft(wavelet_basis[:, k])/len(wavelet_basis[:, k])
        
        # Compute the Fourier transform of the signal
        signalf = 2*np.fft.fft(signal)/len(signal)
        
        # Perform convolution in the frequency domain
        coewtf = wbtf * signalf
        
        # Use roex windows to cut chosen latencies
        
        Nrw = 20 # order of the roex win
        Nall = len(signalf)  # number of samples in the entire window
        
        Trwc01 = 0.0005 # the rwc time constant for negative time (contant at 0.5 ms)
        awltSh = 0.015 # time constant for long latency components
        awltZL = 0.005 # time constant for zero-latency components
        bwlt = 0.8  # exponent
        TrwcSh = awltSh*(fx[k]/1000)**(-bwlt)  # rwc time const for positive time (frequency dependent vis Moleti 2012)
        TrwcZL = awltZL*(fx[k]/1000)**(-bwlt)  # rwc time const for positive time (frequency dependent vis Moleti 2012)
        #NsampShift = int(fsamp*TrwcSh)  # number of samples  for shifting the roexwin
        
        rwC = pDP.roexwin(Nall,Nrw,fsamp,Trwc01,TrwcSh) #  not shifted window
        rwCNL = pDP.roexwin(Nall,Nrw,fsamp,Trwc01,TrwcZL) #  not shifted window
        #rwC = roexwin(Nall,Nrw,fsamp,TrwcSh-TrwcZL,TrwcSh-TrwcZL) # 
        #rwC = np.roll(rwC,NsampShift)
        
        
        # Compute the inverse Fourier transform
        coew = np.fft.ifft(coewtf)*len(signal)/2
        coewDPOAE = coew*np.fft.fftshift(rwC)
        coewNL = coew*np.fft.fftshift(rwCNL)
        coefwtiDPOAE[:,k] = np.fft.fftshift(coewDPOAE)
        
        
        # Add the filtered signal to the overall result
        filtered_signalDPOAE += coewDPOAE.real
        filtered_signalNL += coewNL.real
    
    return filtered_signalDPOAE, filtered_signalNL, coefwtiDPOAE




def estimateFreqResp(oaeDS,f2f1,f2b,f2e,octpersec,GainMic):



    #f2f1 = 1.2
    #f2b = 8000
    #f2e = 500
    
    nfloorDS  = np.zeros_like(oaeDS)
    nfilt = 2**12
    #GainMic = 40
    
    #(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tau02,tshift):
    
    if f2b<f2e:
        T = np.log2(f2e/f2b)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        
    else:
        T = np.log2(f2b/f2e)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        
    #L1sw =  T/np.log(f2e/f2b) 
    #pDP.fceDPOAEinwinSS(oaeDS,Nsamp,f2b/rF,L1sw,rF,fsamp,0.01,0.02,0)
    #oaeDS = np.concatenate((oaeDS[:int(T*fsamp)],np.zeros((Nsamp4-int(T*fsamp),))))
    #hmfftlen = 2**14
    
    #DPgram, DPgramNL, DPgramCR, NF, NFnl, fx = pDP.calcDPgramFAV_HS(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    #DPgram, DPgramNL, DPgramCR, NF, NFnl, fx = pDP.calcDPgram(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    #hm, h, hmN, hN = pDP.calcDPgramTD(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    fxF1, HF1calcTFatF = pDP.calcTFatF(oaeDS,f2b/f2f1,f2e/f2f1,T,fsamp)
    fxF2, HF2calcTFatF = pDP.calcTFatF(oaeDS,f2b,f2e,T,fsamp)
        
    #f2max = 8000
    
    
    return fxF1, HF1calcTFatF, fxF2, HF2calcTFatF


def estimateDRgram(oaeDS,f2f1,f2b,f2e,octpersec,GainMic):



    #f2f1 = 1.2
    #f2b = 8000
    #f2e = 500
    
    nfloorDS  = np.zeros_like(oaeDS)
    nfilt = 2**12
    #GainMic = 40
    
    #(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tau02,tshift):
    
    if f2b<f2e:
        T = np.log2(f2e/f2b)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        
    else:
        T = np.log2(f2b/f2e)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        
    #L1sw =  T/np.log(f2e/f2b) 
    #pDP.fceDPOAEinwinSS(oaeDS,Nsamp,f2b/rF,L1sw,rF,fsamp,0.01,0.02,0)
    #oaeDS = np.concatenate((oaeDS[:int(T*fsamp)],np.zeros((Nsamp4-int(T*fsamp),))))
    #hmfftlen = 2**14
    
    #DPgram, DPgramNL, DPgramCR, NF, NFnl, fx = pDP.calcDPgramFAV_HS(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    #DPgram, DPgramNL, DPgramCR, NF, NFnl, fx = pDP.calcDPgram(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    hm, h, hmN, hN = pDP.calcDPgramTD(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    
    f2max = 8000
    
    hm_50lin = hm
        
    fs = fsamp
    #t = np.arange(0,0.1,1/fs)
    t = np.arange(0,len(hm_50lin)/fs,1/fs)
    # Finding signal by adding three different signals
    #signal = np.cos(2 * np.pi * 1000 * t)# + np.real(np.exp(-7 * (t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
    
    
    dt = 1/fs
    Nt = len(t)
    fx = np.arange(Nt)*fs/Nt   # frequency axis
    df = fx[1]-fx[0]

    print(df)
    print(Nt)
    Nw = Nt//2  # number of wavelet filters
    Nwmax = int(f2max//df)  # maximal wavelet which is usefull for the given frequency range
    
    
    vlnky = mother_wavelet2(Nw,Nt,df,dt)


    tDPOAEmwf, tNLmwf, cwLIN = wavelet_filterDPOAE(hm_50lin, vlnky,fx)
    tDPOAENmwf,tNLNmwf,cwNLIN = wavelet_filterDPOAE(hmN, vlnky,fx)  # noise floor
    
    #SLINmwf = 2*np.fft.rfft(np.concatenate((tLINmwf,np.zeros(int(2**15)))))/Nw  # calculate spectrum
    
    Nfft = 2**15
    fxx = np.arange(int(Nfft))*fsamp/(int(Nfft))
    
    SDPOAEmwf = np.fft.rfft(np.fft.fftshift(tDPOAEmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum
    SDPOAENmwf = np.fft.rfft(np.fft.fftshift(tDPOAENmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum
    SNLmwf = np.fft.rfft(np.fft.fftshift(tNLmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum
    SNLNmwf = np.fft.rfft(np.fft.fftshift(tNLNmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum
    
    DPgr = {'DPgr':SDPOAEmwf,'DPgrN':SDPOAENmwf,'NLgr':SNLmwf,'NLgrN':SNLNmwf,'fxx':fxx}
    return DPgr



def getDPgram(path,DatePat):

    dir_list = os.listdir(path)
    
    n = 0
    DPgramsDict = {}
    
    for k in range(len(dir_list)):
        
        if  DatePat in dir_list[k]:
            data = loadmat(path+ dir_list[k])
            #lat = 16448
            rateOct = data['r'][0][0]
            lat = data['lat_SC'][0][0]
            print([path+ dir_list[k]])
            print(f"SC latency: {lat}")
            octpersec = data['r'][0][0]
            f2b = data['f2b'][0][0]
            f2e = data['f2e'][0][0]
            f2f1 = data['f2f1'][0][0]
            L1dB = data['L1'][0][0]
            L2dB = data['L2'][0][0]
            fsamp = data['fsamp'][0][0]
            GainMic = 40
            
            recSig1 = data['recsigp1'][:,0]
            recSig2 = data['recsigp2'][:,0]
            recSig3 = data['recsigp3'][:,0]
            recSig4 = data['recsigp4'][:,0]
            #print(np.shape(recSig))
            recSig1 = butter_highpass_filter(recSig1, 200, fsamp, order=5)
            recSig1 = np.reshape(recSig1,(-1,1)) # reshape to a matrix with 1 column
            recSig2 = butter_highpass_filter(recSig2, 200, fsamp, order=5)
            recSig2 = np.reshape(recSig2,(-1,1)) # reshape to a matrix with 1 column
            recSig3 = butter_highpass_filter(recSig3, 200, fsamp, order=5)
            recSig3 = np.reshape(recSig3,(-1,1)) # reshape to a matrix with 1 column
            recSig4 = butter_highpass_filter(recSig4, 200, fsamp, order=5)
            recSig4 = np.reshape(recSig4,(-1,1)) # reshape to a matrix with 1 column
            if n == 0:
                recMat1 = recSig1[lat:,0]
                recMat2 = recSig2[lat:,0]
                recMat3 = recSig3[lat:,0]
                recMat4 = recSig4[lat:,0]
                
                # calcualte DP-gram and estimate noise
                
                #oaeDS = (recMat1+recMat2+recMat3+recMat4)/4  # exclude samples set to NaN (noisy samples)
                
                #fxF1_1,HxF1_1, fxF2_1,HxF2_1 = estimateFreqResp(recSig1[lat:,0], f2f1, f2b, f2e, octpersec, GainMic)
                #fxF1_2,HxF1_2, fxF2_2,HxF2_2 = estimateFreqResp(recSig2[lat:,0], f2f1, f2b, f2e, octpersec, GainMic)
                #fxF1_3,HxF1_3, fxF2_3,HxF2_3 = estimateFreqResp(recSig3[lat:,0], f2f1, f2b, f2e, octpersec, GainMic)
                #fxF1_4,HxF1_4, fxF2_4,HxF2_4 = estimateFreqResp(recSig4[lat:,0], f2f1, f2b, f2e, octpersec, GainMic)
                
                
                #DPgrR = estimateDRgram(oaeDS,f2f1,f2b,f2e,octpersec,GainMic)
                
#                DPgramsDict[str(n)] = DPgrR
                
                
            else:
                
                #fxF1_1,HxF1_1, fxF2_1,HxF2_1 = estimateFreqResp(recSig1[lat:,0], f2f1, f2b, f2e, octpersec, GainMic)
                #fxF1_2,HxF1_2, fxF2_2,HxF2_2 = estimateFreqResp(recSig2[lat:,0], f2f1, f2b, f2e, octpersec, GainMic)
                #fxF1_3,HxF1_3, fxF2_3,HxF2_3 = estimateFreqResp(recSig3[lat:,0], f2f1, f2b, f2e, octpersec, GainMic)
                #fxF1_4,HxF1_4, fxF2_4,HxF2_4 = estimateFreqResp(recSig4[lat:,0], f2f1, f2b, f2e, octpersec, GainMic)
                
                
                recMat1 = np.c_[recMat1, recSig1[lat:,0]]  # add to make a matrix with columns for every run
                recMat2 = np.c_[recMat2, recSig2[lat:,0]]  # add to make a matrix with columns for every run
                recMat3 = np.c_[recMat3, recSig3[lat:,0]]  # add to make a matrix with columns for every run
                recMat4 = np.c_[recMat4, recSig4[lat:,0]]  # add to make a matrix with columns for every run
           
    
                #oaeDS = (np.nanmean(recMat1,1)+np.nanmean(recMat2,1)+np.nanmean(recMat3,1)+np.nanmean(recMat4,1))/4  # exclude samples set to NaN (noisy samples)
                
                #DPgrR = estimateDRgram(oaeDS,f2f1,f2b,f2e,octpersec,GainMic)
                
                #DPgramsDict[str(n)] = DPgrR

            n += 1
            
    recMean1 = np.median(recMat1,1)  # median across rows
    recMean1 = np.reshape(recMean1,(-1,1))
    recMean2 = np.median(recMat2,1)  # median across rows
    recMean2 = np.reshape(recMean2,(-1,1))
    recMean3 = np.median(recMat3,1)  # median across rows
    recMean3 = np.reshape(recMean3,(-1,1))
    recMean4 = np.median(recMat4,1)  # median across rows
    recMean4 = np.reshape(recMean4,(-1,1))
    
    # 2. calculate noise matrix
    noiseM1 = recMat1 - recMean1
    noiseM2 = recMat2 - recMean2
    noiseM3 = recMat3 - recMean3
    noiseM4 = recMat4 - recMean4
    Nsamp = len(recMean1) # number of samples


    sigma1 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM1**2,1)))
    sigma2 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM2**2,1)))
    sigma3 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM3**2,1)))
    sigma4 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM4**2,1)))
    Nt = 8
    Theta1 = Nt*sigma1 # estimation of the threshold for sample removal
    Theta2 = Nt*sigma2 # estimation of the threshold for sample removal
    Theta3 = Nt*sigma3 # estimation of the threshold for sample removal
    Theta4 = Nt*sigma4 # estimation of the threshold for sample removal
    
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
            
    return DPgramsDict, rateOct



DPgr = {}

for i in range(1,len(subjD[subjN])):

    DPgrD, rateOct = getDPgram(subjD[subjN][0], subjD[subjN][i])
    DPgr[str(rateOct)] = DPgrD
    
    #DPgrD10 = getDPgram(path, deL_r10)

#%%



fig,ax = plt.subplots()
pREF = np.sqrt(2)*2e-5

Nopak = 4  # nuber of presentation

fxx2 = DPgr['2'][str(Nopak)]['fxx']
fxx8 = DPgr['8'][str(Nopak)]['fxx']

ax.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgr['2'][str(Nopak)]['DPgr'])/pREF),color='C3')
ax.plot(fxx8[:int(len(fxx8)//2)+1],20*np.log10(np.abs(DPgr['8'][str(Nopak)]['DPgr'])/pREF),color='C4')
#ax.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD2['0']['NLgr'])/pREF),color='C0')
#ax.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD2['0']['NLgrN'])/pREF),":",color='C0')
#ax.plot(fxx10[:int(len(fxx10)//2)+1],20*np.log10(np.abs(DPgrD10['0']['NLgr'])/pREF),color='C1')
#ax.plot(fxx10[:int(len(fxx10)//2)+1],20*np.log10(np.abs(DPgrD10['0']['NLgrN'])/pREF),":",color='C1')
ax.set_xlim([400,5000])
ax.set_ylim([-50,20])
ax.legend(('2oct/sec, DP-gram','10oct/sec, DP-gram','2oct/sec, DP-gram zero latency comp.','2oct/sec noise floor zero latency comp.','10oct/sec, DP-gram zero latency comp.','10oct/sec noise floor zero latency comp.'))
ax.set_ylabel('Amplitude (dB SPL)')
ax.set_xlabel('Frequency $f_{dp}$(Hz)')




'''
fig,(ax1,ax2) = plt.subplots(2,1)
pREF = np.sqrt(2)*2e-5
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['0']['DPgr'])/pREF),color='C0')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['0']['DPgrN'])/pREF),":",color='C0')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['1']['DPgr'])/pREF),color='C1')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['1']['DPgrN'])/pREF),":",color='C1')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['2']['DPgr'])/pREF),color='C2')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['2']['DPgrN'])/pREF),":",color='C2')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['3']['DPgr'])/pREF),color='C3')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['3']['DPgrN'])/pREF),":",color='C3')


fig,(ax1,ax2) = plt.subplots(2,1)
pREF = np.sqrt(2)*2e-5


fxx2 = DPgrD['0']['fxx']
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['0']['NLgr'])/pREF),color='C0')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['0']['NLgrN'])/pREF),":",color='C0')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['1']['NLgr'])/pREF),color='C1')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['1']['NLgrN'])/pREF),":",color='C1')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['2']['NLgr'])/pREF),color='C2')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['2']['NLgrN'])/pREF),":",color='C2')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['3']['NLgr'])/pREF),color='C3')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['3']['NLgrN'])/pREF),":",color='C3')



fig, ax = plt.subplots()
ax.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(DPgrD['0']['NLgr'])/pREF),color='C0')


'''
Nmax = 12  # maximum number of repetitions
r = 2



S2,S2N, SNL2, SNL2N,  fxx2, t2, t2N, tNL2, tNL2N = getDPgram(path, deL_r2, fsamp, lat, r,Nmax)

r = 4

S4,S4N, SNL4, SNL4N,  fxx4, t4, t4N, tNL4, tNL4N = getDPgram(path, deL_r4, fsamp, lat, r,Nmax)

#r = 6

#S6,S6N, SNL6, SNL6N,  fxx6, t6, t6N, tNL6, tNL6N = getDPgram(path, deL_r6, fsamp, lat, r,Nmax)

r = 8

S8,S8N, SNL8, SNL8N,  fxx8, t8, t8N, tNL8, tNL8N  = getDPgram(path, deL_r8, fsamp, lat, r,Nmax)

r = 10

S10,S10N, SNL10, SNL10N,  fxx10, t10, t10N, tNL10, tNL10N = getDPgram(path, deL_r10, fsamp, lat, r,Nmax)

#%% visualization


fig,(ax1,ax2) = plt.subplots(2,1)
pREF = np.sqrt(2)*2e-5

ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(S2)/pREF))
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(SNL2)/pREF))
ax1.plot(fxx2[:int(len(fxx8)//2)+1],20*np.log10(np.abs(S8)/pREF))
ax1.plot(fxx2[:int(len(fxx8)//2)+1],20*np.log10(np.abs(SNL8)/pREF))

ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(S2N)/pREF),':')
ax1.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(SNL2N)/pREF),':')



ax1.set_xlim([400,5000])
ax1.set_ylim([-40,20])

cycle = 2*np.pi
ax2.plot(fxx2[:int(len(fxx2)//2)+1],np.unwrap(np.angle(S2))/cycle)
ax2.plot(fxx2[:int(len(fxx2)//2)+1],np.unwrap(np.angle(SNL2))/cycle)
ax2.plot(fxx2[:int(len(fxx10)//2)+1],np.unwrap(np.angle(S8))/cycle)
ax2.plot(fxx2[:int(len(fxx10)//2)+1],np.unwrap(np.angle(SNL8))/cycle)

ax2.set_xlim([400,5000])
ax2.set_ylim([-10,2])

#S6,S6N, SNL6, SNL6N,  fxx6, t6, t6N, tNL6, tNL6N = getDPgram(path, deL_r6, fsamp, lat, 6)
#S8,S8N, SNL8, SNL8N,  fxx8, t8, t8N, tNL8, tNL8N  = getDPgram(path, deL_r8, fsamp, lat, 8)
#S10,S10N, SNL10, SNL10N,  fxx10, t10, t10N, tNL10, tNL10N = getDPgram(path, deL_r10, fsamp, lat, 10)

#fig,ax = plt.subplots()
#pREF = np.sqrt(2)*2e-5

#ax.plot(fxx2[:int(len(fxx2)//2)+1],20*np.log10(np.abs(S2)/pREF))

'''
'''
#DPgramr2, DPgramNLr2, DPgramCRr2, NFr2, NFnlr2, fxr2 = getDPgram(path, deL_r2, fsamp, lat, r)
r = 4
DPgramr4, DPgramNLr4, DPgramCRr4, NFr4, NFnlr4, fxr4 = getDPgram(path, deL_r4, fsamp, lat, r)

r = 8
DPgramr8, DPgramNLr8, DPgramCRr8, NFr8, NFnlr8, fxr8 = getDPgram(path, deL_r8, fsamp, lat, r)


r = 10
DPgramr12, DPgramNLr12, DPgramCRr12, NFr12, NFnlr12, fxr12 = getDPgram(path, deL_r10, fsamp, lat, r)


f2xL2_30 = f2f1*fxr2/(2-f2f1)
f2xr4 = f2f1*fxr4/(2-f2f1)
f2xr8 = f2f1*fxr8/(2-f2f1)
f2xr12 = f2f1*fxr12/(2-f2f1)

plt.close('all')
f2min = 500
f2max = 8000
fig,(ax1,ax2) = plt.subplots(2,1)

LWv = np.linspace(1,1,9)

cycle = np.pi*2 
#ax1.plot(fx,20*np.log10(np.abs(DPgram[:,-1])/(np.sqrt(2)*2e-5)))
ax1.plot(f2xL2_30,20*np.log10(np.abs(DPgramr2)/(np.sqrt(2)*2e-5)),color='C0',linewidth=LWv[0])
ax1.plot(f2xL2_30,20*np.log10(np.abs(DPgramNLr2)/(np.sqrt(2)*2e-5)),color='C1',linewidth=LWv[0])
ax1.plot(f2xr4,20*np.log10(np.abs(DPgramr4)/(np.sqrt(2)*2e-5)),color='C0',linewidth=LWv[0])
ax1.plot(f2xr4,20*np.log10(np.abs(DPgramNLr4)/(np.sqrt(2)*2e-5)),color='C1',linewidth=LWv[0])
ax1.plot(f2xr8,20*np.log10(np.abs(DPgramr8)/(np.sqrt(2)*2e-5)),color='C0',linewidth=LWv[0])
ax1.plot(f2xr8,20*np.log10(np.abs(DPgramNLr8)/(np.sqrt(2)*2e-5)),color='C1',linewidth=LWv[0])
ax1.plot(f2xr12,20*np.log10(np.abs(DPgramr12)/(np.sqrt(2)*2e-5)),color='C0',linewidth=LWv[0])
ax1.plot(f2xr12,20*np.log10(np.abs(DPgramNLr12)/(np.sqrt(2)*2e-5)),color='C1',linewidth=LWv[0])


#%%



f2xL2_30 = f2f1*fxL2_30/(2-f2f1)
f2xL2_35 = f2f1*fxL2_35/(2-f2f1)
f2xL2_40 = f2f1*fxL2_40/(2-f2f1)
f2xL2_45 = f2f1*fxL2_45/(2-f2f1)
f2xL2_50 = f2f1*fxL2_50/(2-f2f1)
f2xL2_55 = f2f1*fxL2_55/(2-f2f1)
f2xL2_60 = f2f1*fxL2_60/(2-f2f1)
f2xL2_65 = f2f1*fxL2_65/(2-f2f1)
f2xL2_70 = f2f1*fxL2_70/(2-f2f1)

plt.close('all')
f2min = 500
f2max = 8000
fig,(ax1,ax2) = plt.subplots(2,1)

LWv = np.linspace(1,1,9)

cycle = np.pi*2 
#ax1.plot(fx,20*np.log10(np.abs(DPgram[:,-1])/(np.sqrt(2)*2e-5)))
ax1.plot(f2xL2_30,20*np.log10(np.abs(DPgramL2_30)/(np.sqrt(2)*2e-5)),color='C0',linewidth=LWv[0])
ax1.plot(f2xL2_35,20*np.log10(np.abs(DPgramL2_35)/(np.sqrt(2)*2e-5)),color='C1',linewidth=LWv[1])
ax1.plot(f2xL2_40,20*np.log10(np.abs(DPgramL2_40)/(np.sqrt(2)*2e-5)),color='C2',linewidth=LWv[2])
ax1.plot(f2xL2_45,20*np.log10(np.abs(DPgramL2_45)/(np.sqrt(2)*2e-5)),color='C3',linewidth=LWv[3])
ax1.plot(f2xL2_50,20*np.log10(np.abs(DPgramL2_50)/(np.sqrt(2)*2e-5)),color='C4',linewidth=LWv[4])
ax1.plot(f2xL2_55,20*np.log10(np.abs(DPgramL2_55)/(np.sqrt(2)*2e-5)),color='C5',linewidth=LWv[5])
ax1.plot(f2xL2_60,20*np.log10(np.abs(DPgramL2_60)/(np.sqrt(2)*2e-5)),color='C6',linewidth=LWv[6])
ax1.plot(f2xL2_65,20*np.log10(np.abs(DPgramL2_65)/(np.sqrt(2)*2e-5)),color='C7',linewidth=LWv[7])
ax1.plot(f2xL2_70,20*np.log10(np.abs(DPgramL2_70)/(np.sqrt(2)*2e-5)),color='C8',linewidth=LWv[8])
ax1.plot(f2xL2_30,20*np.log10(np.abs(NFL2_30)/(np.sqrt(2)*2e-5)),linestyle=':',color='C0',linewidth=LWv[0])
ax1.plot(f2xL2_35,20*np.log10(np.abs(NFL2_35)/(np.sqrt(2)*2e-5)),linestyle=':',color='C1',linewidth=LWv[1])
ax1.plot(f2xL2_40,20*np.log10(np.abs(NFL2_40)/(np.sqrt(2)*2e-5)),linestyle=':',color='C2',linewidth=LWv[2])
ax1.plot(f2xL2_45,20*np.log10(np.abs(NFL2_45)/(np.sqrt(2)*2e-5)),linestyle=':',color='C3',linewidth=LWv[3])
ax1.plot(f2xL2_50,20*np.log10(np.abs(NFL2_50)/(np.sqrt(2)*2e-5)),linestyle=':',color='C4',linewidth=LWv[4])
ax1.plot(f2xL2_55,20*np.log10(np.abs(NFL2_55)/(np.sqrt(2)*2e-5)),linestyle=':',color='C5',linewidth=LWv[5])
ax1.plot(f2xL2_60,20*np.log10(np.abs(NFL2_60)/(np.sqrt(2)*2e-5)),linestyle=':',color='C6',linewidth=LWv[6])
ax1.plot(f2xL2_65,20*np.log10(np.abs(NFL2_65)/(np.sqrt(2)*2e-5)),linestyle=':',color='C7',linewidth=LWv[7])
ax1.plot(f2xL2_70,20*np.log10(np.abs(NFL2_70)/(np.sqrt(2)*2e-5)),linestyle=':',color='C8',linewidth=LWv[8])
#ax1.plot(fx,20*np.log10(np.abs(DPgramCR)/(np.sqrt(2)*2e-5)))
#ax1.plot(fx,20*np.log10(np.abs(NF)/(np.sqrt(2)*2e-5)),':')
ax1.set_xlim([f2min,f2max])
ax1.set_ylim([-40,30])
ax1.set_ylabel('Amplitude (dB SPL)')
ax1.set_title('DP-gram for scissor paradigm, '+ subj_name + ', ' + ear + ' ear')
ax1.legend(('$L_2$ = 30','35','40','45','50','55', '60', '65', '70 dB SPL','noise floor'))
#ax1.legend(('DPOAE','NL comp.','CR comp.','noise floor'))
if not np.isnan(DPgramL2_30).any():
    ax2.plot(f2xL2_30,np.unwrap(np.angle(DPgramL2_30))/cycle,color='C0',linewidth=LWv[0])
ax2.plot(f2xL2_35,np.unwrap(np.angle(DPgramL2_35))/cycle,color='C1',linewidth=LWv[1])
ax2.plot(f2xL2_40,np.unwrap(np.angle(DPgramL2_40))/cycle,color='C2',linewidth=LWv[2])
ax2.plot(f2xL2_45,np.unwrap(np.angle(DPgramL2_45))/cycle,color='C3',linewidth=LWv[3])
ax2.plot(f2xL2_50,np.unwrap(np.angle(DPgramL2_50))/cycle,color='C4',linewidth=LWv[4])
ax2.plot(f2xL2_55,np.unwrap(np.angle(DPgramL2_55))/cycle,color='C5',linewidth=LWv[5])
ax2.plot(f2xL2_60,np.unwrap(np.angle(DPgramL2_60))/cycle,color='C6',linewidth=LWv[6])
if not np.isnan(DPgramL2_65).any():
    ax2.plot(f2xL2_60,np.unwrap(np.angle(DPgramL2_65))/cycle,color='C7',linewidth=LWv[7])
if not np.isnan(DPgramL2_70).any():
    ax2.plot(f2xL2_70,np.unwrap(np.angle(DPgramL2_70))/cycle,color='C8',linewidth=LWv[8])

#ax2.plot(fx,np.unwrap(np.angle(DPgramNL))/cycle)
#ax2.plot(fx,np.unwrap(np.angle(DPgramCR))/cycle)
ax2.set_xlim([f2min,f2max])
ax2.set_ylim([-40,10])
ax2.set_xlabel('Frequency $f_2 (Hz)')
ax2.set_ylabel('Phase (cycles)')
plt.show()




fig,(ax1,ax2) = plt.subplots(2,1)

cycle = np.pi*2 
#ax1.plot(fx,20*np.log10(np.abs(DPgram[:,-1])/(np.sqrt(2)*2e-5)))
ax1.plot(f2xL2_30,20*np.log10(np.abs(DPgramNLL2_30)/(np.sqrt(2)*2e-5)),color='C0',linewidth=LWv[0])
ax1.plot(f2xL2_35,20*np.log10(np.abs(DPgramNLL2_35)/(np.sqrt(2)*2e-5)),color='C1',linewidth=LWv[1])
ax1.plot(f2xL2_40,20*np.log10(np.abs(DPgramNLL2_40)/(np.sqrt(2)*2e-5)),color='C2',linewidth=LWv[2])
ax1.plot(f2xL2_45,20*np.log10(np.abs(DPgramNLL2_45)/(np.sqrt(2)*2e-5)),color='C3',linewidth=LWv[3])
ax1.plot(f2xL2_50,20*np.log10(np.abs(DPgramNLL2_50)/(np.sqrt(2)*2e-5)),color='C4',linewidth=LWv[4])
ax1.plot(f2xL2_55,20*np.log10(np.abs(DPgramNLL2_55)/(np.sqrt(2)*2e-5)),color='C5',linewidth=LWv[5])
ax1.plot(f2xL2_60,20*np.log10(np.abs(DPgramNLL2_60)/(np.sqrt(2)*2e-5)),color='C6',linewidth=LWv[6])
ax1.plot(f2xL2_65,20*np.log10(np.abs(DPgramNLL2_65)/(np.sqrt(2)*2e-5)),color='C7',linewidth=LWv[7])
ax1.plot(f2xL2_70,20*np.log10(np.abs(DPgramNLL2_70)/(np.sqrt(2)*2e-5)),color='C8',linewidth=LWv[8])

ax1.plot(f2xL2_30,20*np.log10(np.abs(NFnlL2_30)/(np.sqrt(2)*2e-5)),linestyle=':',color='C0',linewidth=LWv[0])
ax1.plot(f2xL2_35,20*np.log10(np.abs(NFnlL2_35)/(np.sqrt(2)*2e-5)),linestyle=':',color='C1',linewidth=LWv[1])
ax1.plot(f2xL2_40,20*np.log10(np.abs(NFnlL2_40)/(np.sqrt(2)*2e-5)),linestyle=':',color='C2',linewidth=LWv[2])
ax1.plot(f2xL2_45,20*np.log10(np.abs(NFnlL2_45)/(np.sqrt(2)*2e-5)),linestyle=':',color='C3',linewidth=LWv[3])
ax1.plot(f2xL2_50,20*np.log10(np.abs(NFnlL2_50)/(np.sqrt(2)*2e-5)),linestyle=':',color='C4',linewidth=LWv[4])
ax1.plot(f2xL2_55,20*np.log10(np.abs(NFnlL2_55)/(np.sqrt(2)*2e-5)),linestyle=':',color='C5',linewidth=LWv[5])
ax1.plot(f2xL2_60,20*np.log10(np.abs(NFnlL2_60)/(np.sqrt(2)*2e-5)),linestyle=':',color='C6',linewidth=LWv[6])
ax1.plot(f2xL2_65,20*np.log10(np.abs(NFnlL2_65)/(np.sqrt(2)*2e-5)),linestyle=':',color='C7',linewidth=LWv[7])
ax1.plot(f2xL2_70,20*np.log10(np.abs(NFnlL2_70)/(np.sqrt(2)*2e-5)),linestyle=':',color='C8',linewidth=LWv[8])
#ax1.plot(fx,20*np.log10(np.abs(DPgramCR)/(np.sqrt(2)*2e-5)))
#ax1.plot(fx,20*np.log10(np.abs(NF)/(np.sqrt(2)*2e-5)),':')
ax1.set_xlim([f2min,f2max])
ax1.set_ylim([-40,30])
ax1.set_ylabel('Amplitude (dB SPL)')
ax1.set_title('NL component of DP-gram for scissor paradigm, '+ subj_name + ', ' + ear + ' ear')
ax1.legend(('$L_2$ = 30','35','40','45','50','55', '60', '65', '70 dB SPL','noise floor'))
#ax1.legend(('DPOAE','NL comp.','CR comp.','noise floor'))
ax2.plot(f2x,np.unwrap(np.angle(DPgramNLL2_30))/cycle,color='C0',linewidth=LWv[0])
ax2.plot(f2x,np.unwrap(np.angle(DPgramNLL2_35))/cycle,color='C1',linewidth=LWv[0])
ax2.plot(f2x,np.unwrap(np.angle(DPgramNLL2_40))/cycle,color='C2',linewidth=LWv[0])
ax2.plot(f2x,np.unwrap(np.angle(DPgramNLL2_45))/cycle,color='C3',linewidth=LWv[0])
ax2.plot(f2x,np.unwrap(np.angle(DPgramNLL2_50))/cycle,color='C4',linewidth=LWv[0])
ax2.plot(f2x,np.unwrap(np.angle(DPgramNLL2_55))/cycle,color='C5',linewidth=LWv[0])
#ax2.plot(fx,np.unwrap(np.angle(DPgramNL))/cycle)
#ax2.plot(fx,np.unwrap(np.angle(DPgramCR))/cycle)
ax2.set_xlim([f2min,f2max])
ax2.set_ylim([-4,2])
ax2.set_xlabel('Frequency $f_2$ (Hz)')
ax2.set_ylabel('Phase (cycles)')
ax2.legend(('$L_2$ = 30','35','40','45','50','55 dB SPL','noise floor'))
plt.show()



fig,(ax1,ax2) = plt.subplots(2,1)


cycle = np.pi*2 
#ax1.plot(fx,20*np.log10(np.abs(DPgram[:,-1])/(np.sqrt(2)*2e-5)))
ax1.plot(f2x,20*np.log10(np.abs(DPgramCRL2_30)/(np.sqrt(2)*2e-5)),color='C0',linewidth=LWv[0])
ax1.plot(f2x,20*np.log10(np.abs(DPgramCRL2_35)/(np.sqrt(2)*2e-5)),color='C1',linewidth=LWv[1])
ax1.plot(f2x,20*np.log10(np.abs(DPgramCRL2_40)/(np.sqrt(2)*2e-5)),color='C2',linewidth=LWv[2])
ax1.plot(f2x,20*np.log10(np.abs(DPgramCRL2_45)/(np.sqrt(2)*2e-5)),color='C3',linewidth=LWv[3])
ax1.plot(f2x,20*np.log10(np.abs(DPgramCRL2_50)/(np.sqrt(2)*2e-5)),color='C4',linewidth=LWv[4])
ax1.plot(f2x,20*np.log10(np.abs(DPgramCRL2_55)/(np.sqrt(2)*2e-5)),color='C5',linewidth=LWv[5])
ax1.plot(f2x,20*np.log10(np.abs(NFL2_30)/(np.sqrt(2)*2e-5)),linestyle=':',color='C0',linewidth=LWv[0])
ax1.plot(f2x,20*np.log10(np.abs(NFL2_35)/(np.sqrt(2)*2e-5)),linestyle=':',color='C1',linewidth=LWv[1])
ax1.plot(f2x,20*np.log10(np.abs(NFL2_40)/(np.sqrt(2)*2e-5)),linestyle=':',color='C2',linewidth=LWv[2])
ax1.plot(f2x,20*np.log10(np.abs(NFL2_45)/(np.sqrt(2)*2e-5)),linestyle=':',color='C3',linewidth=LWv[3])
ax1.plot(f2x,20*np.log10(np.abs(NFL2_50)/(np.sqrt(2)*2e-5)),linestyle=':',color='C4',linewidth=LWv[4])
ax1.plot(f2x,20*np.log10(np.abs(NFL2_55)/(np.sqrt(2)*2e-5)),linestyle=':',color='C5',linewidth=LWv[5])
#ax1.plot(fx,20*np.log10(np.abs(DPgramNL)/(np.sqrt(2)*2e-5)))
#ax1.plot(fx,20*np.log10(np.abs(DPgramCR)/(np.sqrt(2)*2e-5)))
#ax1.plot(fx,20*np.log10(np.abs(NF)/(np.sqrt(2)*2e-5)),':')
ax1.set_xlim([f2min,f2max])
ax1.set_ylim([-40,30])
ax1.set_ylabel('Amplitude (dB SPL)')
ax1.set_title('CR component of DP-gram for scissor paradigm, '+ subj_name + ', ' + ear + ' ear')

#ax1.legend(('DPOAE','NL comp.','CR comp.','noise floor'))
ax2.plot(f2x,np.unwrap(np.angle(DPgramCRL2_30))/cycle,color='C0',linewidth=LWv[0])
ax2.plot(f2x,np.unwrap(np.angle(DPgramCRL2_35))/cycle,color='C1',linewidth=LWv[1])
ax2.plot(f2x,np.unwrap(np.angle(DPgramCRL2_40))/cycle,color='C2',linewidth=LWv[2])
ax2.plot(f2x,np.unwrap(np.angle(DPgramCRL2_45))/cycle,color='C3',linewidth=LWv[3])
ax2.plot(f2x,np.unwrap(np.angle(DPgramCRL2_50))/cycle,color='C4',linewidth=LWv[4])
ax2.plot(f2x,np.unwrap(np.angle(DPgramCRL2_55))/cycle,color='C5',linewidth=LWv[5])

#ax2.plot(fx,np.unwrap(np.angle(DPgramNL))/cycle)
#ax2.plot(fx,np.unwrap(np.angle(DPgramCR))/cycle)
ax2.set_xlim([f2min,f2max])
ax2.set_ylim([-40,10])
ax2.set_xlabel('Frequency $f_2$ (Hz)')
ax2.set_ylabel('Phase (cycles)')
ax2.legend(('$L_2$ = 30','35','40','45','50','55 dB SPL','noise floor'))
plt.show()





# plot IO for several frequencies

CF = [1500,2500,4000,6000]
CFidx = np.zeros_like(CF)

for i in range(len(CF)):
    
    value_to_find = CF[i]

    # Calculate the absolute differences and find the index with the minimum difference
    CFidx[i] = np.abs(np.array(f2xL2_50) - value_to_find).argmin()

DPioCF01NL = [DPgramNLL2_35[CFidx[0]], DPgramNLL2_40[CFidx[0]], DPgramNLL2_45[CFidx[0]], DPgramNLL2_50[CFidx[0]], DPgramNLL2_55[CFidx[0]], DPgramNLL2_60[CFidx[0]], DPgramNLL2_65[CFidx[0]]]
DPioCF02NL = [DPgramNLL2_35[CFidx[1]], DPgramNLL2_40[CFidx[1]], DPgramNLL2_45[CFidx[1]], DPgramNLL2_50[CFidx[1]], DPgramNLL2_55[CFidx[1]], DPgramNLL2_60[CFidx[1]], DPgramNLL2_65[CFidx[1]]]

L2s = np.arange(35,70,5)
DPioCF01 = [DPgramL2_30[CFidx[0]], DPgramL2_35[CFidx[0]], DPgramL2_40[CFidx[0]], DPgramL2_45[CFidx[0]], DPgramL2_50[CFidx[0]], DPgramL2_55[CFidx[0]]]
DPioCF01NL = [DPgramNLL2_30[CFidx[0]], DPgramNLL2_35[CFidx[0]], DPgramNLL2_40[CFidx[0]], DPgramNLL2_45[CFidx[0]], DPgramNLL2_50[CFidx[0]], DPgramNLL2_55[CFidx[0]]]
DPioCF01CR = [DPgramCRL2_30[CFidx[0]], DPgramCRL2_35[CFidx[0]], DPgramCRL2_40[CFidx[0]], DPgramCRL2_45[CFidx[0]], DPgramCRL2_50[CFidx[0]], DPgramCRL2_55[CFidx[0]]]
NFioCF01 = [NFL2_30[CFidx[0]], NFL2_35[CFidx[0]], NFL2_40[CFidx[0]], NFL2_45[CFidx[0]], NFL2_50[CFidx[0]], NFL2_55[CFidx[0]]]

DPioCF02 = [DPgramL2_30[CFidx[1]], DPgramL2_35[CFidx[1]], DPgramL2_40[CFidx[1]], DPgramL2_45[CFidx[1]], DPgramL2_50[CFidx[1]], DPgramL2_55[CFidx[1]]]
DPioCF02NL = [DPgramNLL2_30[CFidx[1]], DPgramNLL2_35[CFidx[1]], DPgramNLL2_40[CFidx[1]], DPgramNLL2_45[CFidx[1]], DPgramNLL2_50[CFidx[1]], DPgramNLL2_55[CFidx[1]]]
DPioCF02CR = [DPgramCRL2_30[CFidx[1]], DPgramCRL2_35[CFidx[1]], DPgramCRL2_40[CFidx[1]], DPgramCRL2_45[CFidx[1]], DPgramCRL2_50[CFidx[1]], DPgramCRL2_55[CFidx[1]]]
NFioCF02 = [NFL2_30[CFidx[1]], NFL2_35[CFidx[1]], NFL2_40[CFidx[1]], NFL2_45[CFidx[1]], NFL2_50[CFidx[1]], NFL2_55[CFidx[1]]]


DPioCF03 = [DPgramL2_30[CFidx[2]], DPgramL2_35[CFidx[2]], DPgramL2_40[CFidx[2]], DPgramL2_45[CFidx[2]], DPgramL2_50[CFidx[2]], DPgramL2_55[CFidx[2]]]
DPioCF03NL = [DPgramNLL2_30[CFidx[2]], DPgramNLL2_35[CFidx[2]], DPgramNLL2_40[CFidx[2]], DPgramNLL2_45[CFidx[2]], DPgramNLL2_50[CFidx[2]], DPgramNLL2_55[CFidx[2]]]
DPioCF03CR = [DPgramCRL2_30[CFidx[2]], DPgramCRL2_35[CFidx[2]], DPgramCRL2_40[CFidx[2]], DPgramCRL2_45[CFidx[2]], DPgramCRL2_50[CFidx[2]], DPgramCRL2_55[CFidx[2]]]
NFioCF03 = [NFL2_30[CFidx[2]], NFL2_35[CFidx[2]], NFL2_40[CFidx[2]], NFL2_45[CFidx[2]], NFL2_50[CFidx[2]], NFL2_55[CFidx[2]]]

DPioCF04 = [DPgramL2_30[CFidx[3]], DPgramL2_35[CFidx[3]], DPgramL2_40[CFidx[3]], DPgramL2_45[CFidx[3]], DPgramL2_50[CFidx[3]], DPgramL2_55[CFidx[3]]]
DPioCF04NL = [DPgramNLL2_30[CFidx[3]], DPgramNLL2_35[CFidx[3]], DPgramNLL2_40[CFidx[3]], DPgramNLL2_45[CFidx[3]], DPgramNLL2_50[CFidx[3]], DPgramNLL2_55[CFidx[3]]]
DPioCF04CR = [DPgramCRL2_30[CFidx[3]], DPgramCRL2_35[CFidx[3]], DPgramCRL2_40[CFidx[3]], DPgramCRL2_45[CFidx[3]], DPgramCRL2_50[CFidx[3]], DPgramCRL2_55[CFidx[3]]]
NFioCF04 = [NFL2_30[CFidx[3]], NFL2_35[CFidx[3]], NFL2_40[CFidx[3]], NFL2_45[CFidx[3]], NFL2_50[CFidx[3]], NFL2_55[CFidx[3]]]
              


fig,ax = plt.subplots()
#ax.plot(L2s,20*np.log10(np.abs(DPioCF01)/(np.sqrt(2)*2e-5)))
ax.plot(L2s,20*np.log10(np.abs(DPioCF01NL)/(np.sqrt(2)*2e-5)))
ax.plot(L2s,20*np.log10(np.abs(DPioCF02NL)/(np.sqrt(2)*2e-5)))
#ax.plot(L2s,20*np.log10(np.abs(DPioCF01CR)/(np.sqrt(2)*2e-5)))
#ax.plot(L2s,20*np.log10(np.abs(NFioCF01)/(np.sqrt(2)*2e-5)),color='C0',linestyle=':')
ax.set_ylim(-30,20)
ax.set_xlim(25,60)
ax.set_title('DPOAE IO, $f_2 = 1.5$ kHz, '+ subj_name + ', ' + ear + ' ear')
ax.legend(('DP-gram','NL comp.','CR comp.','noise floor'))
ax.set_ylabel('Amplitude (dB SPL)')
ax.set_xlabel('$L_2$ (dB SPL)')


fig,ax = plt.subplots()
ax.plot(L2s,20*np.log10(np.abs(DPioCF02)/(np.sqrt(2)*2e-5)))
ax.plot(L2s,20*np.log10(np.abs(DPioCF02NL)/(np.sqrt(2)*2e-5)))
ax.plot(L2s,20*np.log10(np.abs(DPioCF02CR)/(np.sqrt(2)*2e-5)))
ax.plot(L2s,20*np.log10(np.abs(NFioCF02)/(np.sqrt(2)*2e-5)),color='C0',linestyle=':')
ax.set_ylim(-30,20)
ax.set_xlim(25,60)
ax.set_title('DPOAE IO, $f_2 = 2.5$ kHz, '+ subj_name + ', ' + ear + ' ear')
ax.legend(('DP-gram','NL comp.','CR comp.','noise floor'))
ax.set_ylabel('Amplitude (dB SPL)')
ax.set_xlabel('$L_2$ (dB SPL)')



fig,ax = plt.subplots()
ax.plot(L2s,20*np.log10(np.abs(DPioCF03)/(np.sqrt(2)*2e-5)))
ax.plot(L2s,20*np.log10(np.abs(DPioCF03NL)/(np.sqrt(2)*2e-5)))
ax.plot(L2s,20*np.log10(np.abs(DPioCF03CR)/(np.sqrt(2)*2e-5)))
ax.plot(L2s,20*np.log10(np.abs(NFioCF03)/(np.sqrt(2)*2e-5)),color='C0',linestyle=':')
ax.set_ylim(-30,20)
ax.set_xlim(25,60)
ax.set_title('DPOAE IO, $f_2 = 4$ kHz, '+ subj_name + ', ' + ear + ' ear')
ax.legend(('DP-gram','NL comp.','CR comp.','noise floor'))
ax.set_ylabel('Amplitude (dB SPL)')
ax.set_xlabel('$L_2$ (dB SPL)')


fig,ax = plt.subplots()
ax.plot(L2s,20*np.log10(np.abs(DPioCF04)/(np.sqrt(2)*2e-5)))
ax.plot(L2s,20*np.log10(np.abs(DPioCF04NL)/(np.sqrt(2)*2e-5)))
ax.plot(L2s,20*np.log10(np.abs(DPioCF04CR)/(np.sqrt(2)*2e-5)))
ax.plot(L2s,20*np.log10(np.abs(NFioCF04)/(np.sqrt(2)*2e-5)),color='C0',linestyle=':')
ax.set_ylim(-30,20)
ax.set_xlim(25,60)
ax.set_title('DPOAE IO, $f_2 = 6$ kHz, '+ subj_name + ', ' + ear + ' ear')
ax.legend(('DP-gram','NL comp.','CR comp.','noise floor'))
ax.set_ylabel('Amplitude (dB SPL)')
ax.set_xlabel('$L_2$ (dB SPL)')

