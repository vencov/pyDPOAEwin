# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:18:20 2024


script for swept sine rate comparison



@author: VV
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

#subjD['L2_55_dB_L'] = ['Results/s055/sweep_rate/', '24_06_04_14_00_59_F2b_8000Hz', '24_06_04_14_02_32_F2b_8000Hz', '24_06_04_14_03_51_F2b_8000Hz']
#subjD['L2_30_dB_L'] = ['Results/s055/sweep_rate/', '24_06_04_14_06_37_F2b_8000Hz', '24_06_04_14_08_26_F2b_8000Hz', '24_06_04_14_10_17_F2b_8000Hz']
#subjD['L2_55_dB_R'] = ['Results/s055/sweep_rate/', '24_06_04_14_17_53_F2b_8000Hz', '24_06_04_14_19_42_F2b_8000Hz', '24_06_04_14_21_18_F2b_8000Hz', '24_06_04_16_48_24_F2b_8000Hz', '24_06_04_16_50_29_F2b_8000Hz', '24_06_04_16_51_50_F2b_8000Hz']
#subjD['L2_30_dB_R'] =  ['Results/s055/sweep_rate/', '24_06_04_16_54_22_F2b_8000Hz', '24_06_04_16_56_29_F2b_8000Hz', '24_06_04_16_58_25_F2b_8000Hz']

#subjD['L2_50_dB_L'] = ['Results/s089/R/', '24_07_01_11_27_00_F2b_8000Hz', '24_07_01_11_28_39_F2b_8000Hz', '24_07_01_11_30_23_F2b_8000Hz', '24_07_01_11_31_37_F2b_8000Hz']
#subjD['L2_55_dB_L'] = ['Results/s089/R/', '24_07_01_11_33_17_F2b_8000Hz', '24_07_01_11_34_49_F2b_8000Hz', '24_07_01_11_36_08_F2b_8000Hz', '24_07_01_11_37_15_F2b_8000Hz']


subjD['s089L_L2_50'] = ['Results/s089/R/', '24_07_01_11_27_00_F2b_8000Hz', '24_07_01_11_28_39_F2b_8000Hz', '24_07_01_11_30_23_F2b_8000Hz', '24_07_01_11_31_37_F2b_8000Hz']
subjD['s089L_L2_55'] =  ['Results/s089/R/', '24_07_01_11_33_17_F2b_8000Hz', '24_07_01_11_34_49_F2b_8000Hz', '24_07_01_11_36_08_F2b_8000Hz', '24_07_01_11_37_15_F2b_8000Hz']
subjD['s089L_L2_65'] = ['Results/s089/R/', '24_07_04_11_16_56_F2b_8000Hz']

#subjN = 'L2_55_dB_L'
subjN = 's055L_L2_30'

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
    
    
    
    for k in range(14,400):
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
    #np.shape(np.mean(recMat1[np.sum(np.abs(noiseM1)>Theta1],1))
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
    
    
    oaeDS = (recMat1Mean+recMat2Mean+recMat3Mean+recMat4Mean)/4  # exclude samples set to NaN (noisy samples)
    DPgrR = estimateDRgram(oaeDS,f2f1,f2b,f2e,octpersec,GainMic)
    
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
    '''
        
    return DPgrR, rateOct, [N01chosen, N02chosen, N03chosen, N04chosen]



DPgr = {}

for i in range(1,len(subjD[subjN])):

    DPgrD, rateOct, Nchosen = getDPgram(subjD[subjN][0], subjD[subjN][i])
    DPgr[str(rateOct)] = DPgrD
    DPgr[str(rateOct)+'ch'] = Nchosen
    
    #DPgrD10 = getDPgram(path, deL_r10)

#%%



plt.close('all')


pREF = np.sqrt(2)*2e-5

Nopak = 4  # nuber of presentation

fxx2 = DPgr['2']['fxx']

f2f1 = 1.2
f2xx = f2f1*fxx2/(2-f2f1)  # convert fdp to f2
cycle = np.pi*2

if 1==1:  # s055L NL
    
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))  # 2 rows, 1 column
    
    # Plot amplitude data on ax1
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['2']['NLgr']) / pREF), color='C0')
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['8']['NLgr']) / pREF), color='C1', alpha=0.7)
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['12']['NLgr']) / pREF), color='C2', alpha=0.7)
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['2']['NLgrN']) / pREF), color='C0', linestyle=':')
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['8']['NLgrN']) / pREF), color='C1', linestyle=':', alpha=0.7)
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['12']['NLgrN']) / pREF), color='C2', linestyle=':', alpha=0.7)
    ax1.set_xlim([0.5, 8])
    ax1.set_ylim([-30, 20])
    ax1.set_ylabel('Amplitude (dB SPL)')
    ax1.set_xticklabels([])  # Remove x-tick labels for ax1
    ax1.legend(['2 oct/sec,' + str(np.sum(DPgr['2ch'])) + ' rep', '8 oct/sec, ' + str(np.sum(DPgr['8ch'])) + ' rep', '12 oct/sec, ' + str(np.sum(DPgr['12ch'])) + ' rep', 'NF 2 oct/sec', 'NF 8 oct/sec', 'NF 12 oct/sec'], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot phase data on ax2
    idxUnwr = np.where(f2xx >= 500)[0][0]
    ax2.plot(f2xx[idxUnwr:len(f2xx) // 2 + 1] / 1000, np.unwrap(np.angle(DPgr['2']['NLgr'][idxUnwr:])) / cycle, color='C0')
    ax2.plot(f2xx[idxUnwr:len(f2xx) // 2 + 1] / 1000, np.unwrap(np.angle(DPgr['8']['NLgr'][idxUnwr:])) / cycle , color='C1', alpha=0.7)
    ax2.plot(f2xx[idxUnwr:len(f2xx) // 2 + 1] / 1000, np.unwrap(np.angle(DPgr['12']['NLgr'][idxUnwr:])) / cycle, color='C2', alpha=0.7)
    ax2.set_xlim([0.5, 8])
    ax2.set_ylim([-4, 0])
    ax2.set_ylabel('Phase (cycles)')
    ax2.set_xlabel('Frequency $f_{2}$ (kHz)')
    
    # Customize the ticks to be inside the plot
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', direction='in', length=3)  # 'in' sets ticks inside
    
    
    
    # Adjust the y-axis label positions to align them
    label_y_pos = 0.5
    ax1.yaxis.set_label_coords(-0.1, label_y_pos)
    ax2.yaxis.set_label_coords(-0.1, label_y_pos)
    
    
    # Adjust spacing between the panels with tight_layout
    plt.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.5)
    
    
    
    # Display the plot
    plt.show()
    
    
    fileName = 'DPgrNL' + subjN + '.jpg' 
    fileFolder = 'Figures/'
    plt.savefig(fileFolder + fileName, format='jpg', dpi=300, bbox_inches='tight')



if 1==1:  # s055L All
    
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))  # 2 rows, 1 column
    
    # Plot amplitude data on ax1
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['2']['DPgr']) / pREF), color='C0')
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['8']['DPgr']) / pREF), color='C1', alpha=0.7)
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['12']['DPgr']) / pREF), color='C2', alpha=0.7)
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['2']['DPgrN']) / pREF), color='C0', linestyle=':')
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['8']['DPgrN']) / pREF), color='C1', linestyle=':', alpha=0.7)
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['12']['DPgrN']) / pREF), color='C2', linestyle=':', alpha=0.7)
    ax1.set_xlim([0.5, 8])
    ax1.set_ylim([-30, 20])
    ax1.set_ylabel('Amplitude (dB SPL)')
    ax1.set_xticklabels([])  # Remove x-tick labels for ax1
    ax1.legend(['2 oct/sec,' + str(np.sum(DPgr['2ch'])) + ' rep', '8 oct/sec, ' + str(np.sum(DPgr['8ch'])) + ' rep', '12 oct/sec, ' + str(np.sum(DPgr['12ch'])) + ' rep', 'NF 2 oct/sec', 'NF 8 oct/sec', 'NF 12 oct/sec'], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot phase data on ax2
    idxUnwr = np.where(f2xx >= 500)[0][0]
    ax2.plot(f2xx[idxUnwr:len(f2xx) // 2 + 1] / 1000, np.unwrap(np.angle(DPgr['2']['DPgr'][idxUnwr:])) / cycle, color='C0')
    ax2.plot(f2xx[idxUnwr:len(f2xx) // 2 + 1] / 1000, np.unwrap(np.angle(DPgr['8']['DPgr'][idxUnwr:])) / cycle , color='C1', alpha=0.7)
    ax2.plot(f2xx[idxUnwr:len(f2xx) // 2 + 1] / 1000, np.unwrap(np.angle(DPgr['12']['DPgr'][idxUnwr:])) / cycle, color='C2', alpha=0.7)
    ax2.set_xlim([0.5, 8])
    ax2.set_ylim([-4, 0])
    ax2.set_ylabel('Phase (cycles)')
    ax2.set_xlabel('Frequency $f_{2}$ (kHz)')
    
    # Customize the ticks to be inside the plot
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', direction='in', length=3)  # 'in' sets ticks inside
    
    
    
    # Adjust the y-axis label positions to align them
    label_y_pos = 0.5
    ax1.yaxis.set_label_coords(-0.1, label_y_pos)
    ax2.yaxis.set_label_coords(-0.1, label_y_pos)
    
    
    # Adjust spacing between the panels with tight_layout
    plt.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.5)
    
    
    
    # Display the plot
    plt.show()
    
    
    fileName = 'DPgr' + subjN + '.jpg' 
    fileFolder = 'Figures/'
    plt.savefig(fileFolder + fileName, format='jpg', dpi=300, bbox_inches='tight')


if 1==0:  # s089L
    
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))  # 2 rows, 1 column
    
    # Plot amplitude data on ax1
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['2']['NLgr']) / pREF), color='C0')
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['4']['NLgr']) / pREF), color='C1', alpha=0.7)
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['8']['NLgr']) / pREF), color='C2', alpha=0.7)
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['2']['NLgrN']) / pREF), color='C0', linestyle=':')
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['4']['NLgrN']) / pREF), color='C1', linestyle=':', alpha=0.7)
    ax1.plot(f2xx[:len(f2xx) // 2 + 1] / 1000, 20 * np.log10(np.abs(DPgr['8']['NLgrN']) / pREF), color='C2', linestyle=':', alpha=0.7)
    ax1.set_xlim([0.5, 8])
    ax1.set_ylim([-30, 20])
    ax1.set_ylabel('Amplitude (dB SPL)')
    ax1.set_xticklabels([])  # Remove x-tick labels for ax1
    ax1.legend(['2 oct/sec, 20 rep', '4 oct/sec, 23 rep', '8 oct/sec, 19 rep', 'NF 2 oct/sec', 'NF 4 oct/sec', 'NF 8 oct/sec'], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot phase data on ax2
    idxUnwr = np.where(f2xx >= 500)[0][0]
    ax2.plot(f2xx[idxUnwr:len(f2xx) // 2 + 1] / 1000, np.unwrap(np.angle(DPgr['2']['NLgr'][idxUnwr:])) / cycle, color='C0')
    ax2.plot(f2xx[idxUnwr:len(f2xx) // 2 + 1] / 1000, np.unwrap(np.angle(DPgr['4']['NLgr'][idxUnwr:])) / cycle , color='C1', alpha=0.7)
    ax2.plot(f2xx[idxUnwr:len(f2xx) // 2 + 1] / 1000, np.unwrap(np.angle(DPgr['8']['NLgr'][idxUnwr:])) / cycle, color='C2', alpha=0.7)
    ax2.set_xlim([0.5, 8])
    ax2.set_ylim([-4, 0])
    ax2.set_ylabel('Phase (cycles)')
    ax2.set_xlabel('Frequency $f_{2}$ (kHz)')
    
    # Customize the ticks to be inside the plot
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', direction='in', length=3)  # 'in' sets ticks inside
    
    
    
    # Adjust the y-axis label positions to align them
    label_y_pos = 0.5
    ax1.yaxis.set_label_coords(-0.1, label_y_pos)
    ax2.yaxis.set_label_coords(-0.1, label_y_pos)
    
    
    # Adjust spacing between the panels with tight_layout
    plt.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.5)
    
    
    
    # Display the plot
    plt.show()
    
    plt.savefig('DPgr_sweeprate_s089L_55dB.jpg', format='jpg', dpi=300, bbox_inches='tight')




#%%

def InfoOnData(data_dict):
    

    import re
  
    
    # Regular expression to match keys that start with a number and end with 'ch'
    pattern = re.compile(r'^(\d+)ch$')
    
    # List to hold the table rows
    table = []
    
    # Iterate through the dictionary keys and values
    for key, value_list in data_dict.items():
        match = pattern.match(key)
        if match:
            # Extract the integer from the key
            integer_part = int(match.group(1))
            # Create a row starting with the integer, followed by the values in the list
            row = [integer_part] + value_list
            # Append the row to the table
            table.append(row)
            
    
    
    # Define the headers
    header_main = "r (oct/sec)"
    header_phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
    header_full = [header_main] + header_phases
    
    # Print the headers
    print(f"{header_full[0]:<15} {header_full[1]:<10} {header_full[2]:<10} {header_full[3]:<10} {header_full[4]:<10}")
    
    # Print the table rows
    for row in table:
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10}")

InfoOnData(DPgr)
    