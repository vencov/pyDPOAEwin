# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:23:04 2024

@author: audiobunka
"""




from scipy.io import loadmat
import os
from UserModules.pyUtilities import butter_highpass_filter 
import numpy as np
import matplotlib.pyplot as plt

import UserModules.pyDPOAEmodule as pDP
    
plt.close('all')


f2f1 = 1.2

fsamp = 96000
    
subjD = {}

subjD['s004R'] = ['Results/TW/subject4_4phase/aa/', '24_05_21_12_25_26_F2b_8000Hz', '24_05_21_12_27_01_F2b_8000Hz', '24_05_21_12_28_32_F2b_8000Hz', '24_05_21_12_30_03_F2b_8000Hz']

subjD['s004R2'] = ['Results/TW/subject4_4phase/', '24_05_21_12_51_16_F2b_8000Hz', '24_05_21_12_52_57_F2b_8000Hz', '24_05_21_12_54_45_F2b_8000Hz', '24_05_21_12_56_19_F2b_8000Hz']

subjD['s011R'] = ['Results/TW/s011_4phase/', '24_05_21_13_30_14_F2b_8000Hz', '24_05_21_13_31_47_F2b_8000Hz', '24_05_21_13_33_42_F2b_8000Hz', '24_05_21_13_35_43_F2b_8000Hz', '24_05_21_13_37_29_F2b_8000Hz', '24_05_21_13_39_34_F2b_8000Hz', '24_05_21_13_40_02_F2b_8000Hz']

#subjD['s005L'] = ['Results/TW/subject5_4phase/', '24_07_16_08_51_10_F2b_8000Hz', '24_07_16_08_53_18_F2b_8000Hz', '24_07_16_08_54_49_F2b_8000Hz', '24_07_16_08_56_20_F2b_8000Hz', '24_07_16_08_58_31_F2b_8000Hz', '24_07_16_12_48_10_F2b_8000Hz', '24_07_16_12_50_23_F2b_8000Hz', '24_07_16_12_51_52_F2b_8000Hz', '24_07_16_12_53_23_F2b_8000Hz', '24_07_16_12_55_28_F2b_8000Hz']
#subjD['s005R'] =  ['Results/TW/subject5_4phase/', '24_07_16_09_01_28_F2b_8000Hz', '24_07_16_09_03_37_F2b_8000Hz', '24_07_16_09_05_10_F2b_8000Hz', '24_07_16_09_06_39_F2b_8000Hz', '24_07_16_09_08_09_F2b_8000Hz', '24_07_16_12_58_25_F2b_8000Hz', '24_07_16_12_59_56_F2b_8000Hz', '24_07_16_13_01_40_F2b_8000Hz', '24_07_16_13_03_21_F2b_8000Hz', '24_07_16_13_04_39_F2b_8000Hz', '24_07_16_13_05_21_F2b_8000Hz']

subjD['s005La'] =  ['Results/TW/subject5_4phase/after/', '24_07_16_12_48_10_F2b_8000Hz', '24_07_16_12_50_23_F2b_8000Hz', '24_07_16_12_51_52_F2b_8000Hz', '24_07_16_12_53_23_F2b_8000Hz', '24_07_16_12_55_28_F2b_8000Hz']
subjD['s005Ra'] = ['Results/TW/subject5_4phase/after/', '24_07_16_12_58_25_F2b_8000Hz', '24_07_16_12_59_56_F2b_8000Hz', '24_07_16_13_01_40_F2b_8000Hz', '24_07_16_13_03_21_F2b_8000Hz', '24_07_16_13_04_39_F2b_8000Hz', '24_07_16_13_05_21_F2b_8000Hz']

subjD['s005Lb'] =  ['Results/TW/subject5_4phase/before/', '24_07_16_08_51_10_F2b_8000Hz', '24_07_16_08_53_18_F2b_8000Hz', '24_07_16_08_54_49_F2b_8000Hz', '24_07_16_08_56_20_F2b_8000Hz', '24_07_16_08_58_31_F2b_8000Hz']
subjD['s005Rb'] =  ['Results/TW/subject5_4phase/before/', '24_07_16_09_01_28_F2b_8000Hz', '24_07_16_09_03_37_F2b_8000Hz', '24_07_16_09_05_10_F2b_8000Hz', '24_07_16_09_06_39_F2b_8000Hz', '24_07_16_09_08_09_F2b_8000Hz']

subjN = 's005Ra'

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
            #lat = data['lat_SC'][0][0]
            lat = 4180
            print([path+ dir_list[k]])
            print(f"SC latency: {lat}")
            octpersec = data['r'][0][0]
            f2b = data['f2b'][0][0]
            f2e = data['f2e'][0][0]
            f2f1 = data['f2f1'][0][0]
            L1dB = data['L1'][0][0]
            L2dB = data['L2'][0][0]
            fsamp = data['fsamp'][0][0]
            GainMic = 0
            
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
        
    return DPgrR, rateOct, L2dB, f2f1, [N01chosen, N02chosen, N03chosen, N04chosen]



DPgr = {}
L2list = []
f2f1list = []
for i in range(1,len(subjD[subjN])):

    DPgrD, rateOct, L2, f2f1, Nchosen = getDPgram(subjD[subjN][0], subjD[subjN][i])
    DPgr[str(L2)] = DPgrD
    DPgr[str(L2)+'ch'] = Nchosen
    L2list.append(L2)  # list of L2 values
    f2f1list.append(f2f1)
    #DPgrD10 = getDPgram(path, deL_r10)


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
    header_main = "L2 (dB FPL)"
    header_phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
    header_full = [header_main] + header_phases
    
    # Print the headers
    print(f"{header_full[0]:<15} {header_full[1]:<10} {header_full[2]:<10} {header_full[3]:<10} {header_full[4]:<10}")
    
    # Print the table rows
    for row in table:
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10}")

InfoOnData(DPgr)



#%%


fig,(ax1,ax2) = plt.subplots(2,1)
pREF = np.sqrt(2)*2e-5

Nopak = 4  # nuber of presentation

fxx = DPgr['65']['fxx']
f2xx = f2f1list[-1]*fxx[:int(len(fxx)//2)+1]/(2-f2f1)
cList = ['C1','C3','C2','C4','C5','C6','C7','C8','C9','C10','C11']
#fxx8 = DPgr['12']['fxx']
for i in range(len(L2list)):
#ax.plot(fxx[:int(len(fxx)//2)+1],20*np.log10(np.abs(DPgr['30']['NLgr'])/pREF),color='C1')
    ax1.plot(f2xx,20*np.log10(np.abs(DPgr[str(L2list[i])]['NLgr'])/pREF),color=cList[i],label=str(L2list[i]))
    ax1.plot(f2xx,20*np.log10(np.abs(DPgr[str(L2list[i])]['NLgrN'])/pREF),':',color=cList[i],label='_nolegend_')

ax1.set_xlim([500,8000])
ax1.set_ylim([-40,20])
ax1.legend()
ax1.set_ylabel('Amplitude (dB SPL)')

cycle = 2*np.pi
F2start = 700
idx1 = np.where(f2xx>=F2start)[0][0]  # freq index for unwraping
for i in range(len(L2list)):
#ax.plot(fxx[:int(len(fxx)//2)+1],20*np.log10(np.abs(DPgr['30']['NLgr'])/pREF),color='C1')
    ax2.plot(f2xx[idx1:],np.unwrap(np.angle(DPgr[str(L2list[i])]['NLgr'][idx1:]))/cycle,color=cList[i],label=str(L2list[i]))
    
ax2.set_ylabel('Phase (cycles)')


ax2.set_xlim([500,8000])
ax2.set_ylim([-5,1])
ax2.set_xlabel('Frequency $f_{2}$ (kHz)')

# Convert x-ticks to kHz
ax1.set_xticks([1000, 2000, 3000, 4000,5000,6000,7000,8000])
ax1.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])  # Update x-tick labels to kHz
ax2.set_xticks([1000, 2000, 3000, 4000,5000,6000,7000,8000])
ax2.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])  # Update x-tick labels to kHz

#ax2.set_xticks([500, 1000, 2000, 4000, 8000])
#ax2.set_xticklabels([0.5, 1, 2, 4, 8])  # Update x-tick labels to kHz

# Add subject name and ear information to the bottom-right corner
if subjN[-1] == 'L':
    ear = 'left ear'
elif subjN[-1] == 'R':
    ear = 'right ear'
subject_name = subjN[:-1]  # Exclude the last character from subject name
text_to_display = f'{subject_name} ({ear})'

# Increase font sizes by approximately 50%
label_fontsize = 16  # Adjust as necessary
legend_fontsize = 12
text_fontsize = 13
# Add the text in the bottom-right corner of the plot
ax2.text(0.3, 0.2, text_to_display, transform=ax2.transAxes, 
        fontsize=text_fontsize, verticalalignment='top', horizontalalignment='right')


# Save the second plot as well (optional)
plt.savefig('Figures/DPgrams/dpgr' + subjN + '.png', format='png', dpi=300)  # Save the second graph


#%% visualization


import numpy as np

CF = [1000,1500,2000,3000]
CFidx = np.zeros_like(CF)

DPioNL = []
GAdpNL = []
NOxNL = []  # Background noise array
for i in range(len(CF)):
    
    CFidx[i] = np.where(f2xx>=CF[i])[0][0]

    IOx = []
    GAx = []
    NOx = []  # Background noise array
    for j in range(len(L2list)):
        IOx.append(20*np.log10(np.abs(DPgr[str(L2list[j])]['NLgr'][CFidx[i]])/pREF))
        GAx.append(IOx[j]-L2list[j])
        # Extract background noise and convert to dB
        NOx.append(20 * np.log10(np.abs(DPgr[str(L2list[j])]['NLgrN'][CFidx[i]])/pREF))

    DPioNL.append(IOx)
    GAdpNL.append(GAx)
    NOxNL.append(NOx)

data_line = []
noise_lines = []
fig, ax = plt.subplots()
data_line.append(ax.plot(L2list, DPioNL[0], label=r'${\it f}_2$ = 1 kHz'))
data_line.append(ax.plot(L2list, DPioNL[1], label=r'${\it f}_2$ = 1.5 kHz'))
data_line.append(ax.plot(L2list, DPioNL[2], label=r'${\it f}_2$ = 2 kHz'))
data_line.append(ax.plot(L2list, DPioNL[3], label=r'${\it f}_2$ = 3 kHz'))
noise_lines.append(ax.plot(L2list, NOxNL[0], color=data_line[0][0].get_color(), 
                               linestyle=':', linewidth=0.5, label="_nolegend_"))
noise_lines.append(ax.plot(L2list, NOxNL[1], color=data_line[1][0].get_color(), 
                               linestyle=':', linewidth=0.5, label="_nolegend_"))
noise_lines.append(ax.plot(L2list, NOxNL[2], color=data_line[2][0].get_color(), 
                               linestyle=':', linewidth=0.5, label="_nolegend_"))                   
noise_lines.append(ax.plot(L2list, NOxNL[3], color=data_line[3][0].get_color(), 
                               linestyle=':', linewidth=0.5, label="_nolegend_"))                   

ax.tick_params(axis='both', direction='in')

# Convert L2list to a NumPy array for element-wise operations
L2array = np.array(L2list)

# Plot a gray dotted line with slope 1, shifted 35 dB down
ax.plot(L2array, L2array - 35, color='gray', linestyle='--', linewidth=1)

# Set x and y limits to the specified values
ax.set_xlim([20, 70])  # X-axis limits from 20 dB to 70 dB
ax.set_ylim([-20, 20])  # Y-axis limits from -20 dB to 20 dB

# Increase font sizes by approximately 50%
label_fontsize = 16  # Adjust as necessary
legend_fontsize = 12
text_fontsize = 13


# Labels and legend with increased font size
ax.set_ylabel('Amplitude (dB SPL)', fontsize=label_fontsize)
ax.set_xlabel('$L_2$ (dB SPL)', fontsize=label_fontsize)
ax.legend(fontsize=legend_fontsize)

# Add subject name and ear information to the bottom-right corner
if subjN[-1] == 'L':
    ear = 'left ear'
elif subjN[-1] == 'R':
    ear = 'right ear'
subject_name = subjN[:-1]  # Exclude the last character from subject name
text_to_display = f'{subject_name} ({ear})'

# Add the text in the bottom-right corner of the plot
ax.text(0.3, 0.05, text_to_display, transform=ax.transAxes, 
        fontsize=text_fontsize, verticalalignment='top', horizontalalignment='right')

# Increase tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=label_fontsize)

# Adjust layout to fit everything
plt.tight_layout()

# Second plot (optional)
#fig, ax = plt.subplots()
#ax.plot(L2list, GAdpNL[0])
#ax.plot(L2list, GAdpNL[1])
#ax.plot(L2list, GAdpNL[2])
#ax.plot(L2list, GAdpNL[3])

# Optionally, you can also add the shifted line here if relevant
# ax.plot(L2array, L2array - 35, color='gray', linestyle='--', linewidth=1)

# Set x and y limits for the second plot (if needed)
# ax.set_xlim([20, 70])  # X-axis limits for the second plot
# ax.set_ylim([-20, 20])  # Y-axis limits for the second plot

# Adjust layout for the second plot
plt.tight_layout()

#%% fitting


from numpy.polynomial.polynomial import Polynomial
from scipy.io import savemat
import numpy as np
from numpy.polynomial import Polynomial

def fit_polynomial(L2, y_data, degree=4, max_slope_limit=50):
    """
    Fit a polynomial to the given data and calculate key estimates.

    Parameters:
    - L2: array-like, input x-axis values (L2 levels)
    - y_data: array-like, input y-axis values (measured amplitudes)
    - degree: int, the degree of the polynomial to fit (default is 4)
    - max_slope_limit: float, the maximum allowable slope (default is 50 dB)

    Returns:
    - p: Polynomial, the fitted polynomial object
    - max_slope: float, maximum slope of the fitted curve (capped at max_slope_limit)
    - L2_at_max_slope: float, L2 level at maximum slope
    - OAE_level_at_max_slope: float, OAE level at maximum slope
    - L2_half_slope: float, L2 level where slope equals 1/2
    - OAE_level_half_slope: float, OAE level at slope 1/2
    - L2_half_max_slope: float, L2 level where slope equals max_slope/2 (above max slope)
    - OAE_level_half_max_slope: float, OAE level at max_slope/2
    """

    # Fit the polynomial
    p = Polynomial.fit(L2, y_data, deg=degree)

    # Generate fitted values
    x_fit = np.linspace(np.min(L2), np.max(L2), 100)
    y_fit = p(x_fit)

    

    # Calculate slopes numerically for the fitted data
    dy = np.gradient(y_fit, x_fit)

    # Find the maximum slope and its corresponding L2 level
    max_slope_index = np.argmax(dy[:70])
    max_slope = dy[max_slope_index]

    # Cap the maximum slope at the specified limit
    if max_slope > max_slope_limit:
        max_slope = max_slope_limit

    L2_at_max_slope = x_fit[max_slope_index]
    OAE_level_at_max_slope = y_fit[max_slope_index]

    # Calculate the target slopes (1/2 and max_slope/2)
    slope_half = 1 / 2
    slope_half_max = max_slope / 2

    # Find the first point where the slope is below or equal to 1/2, after the max slope
    indices_above_max_slope = np.where(x_fit > L2_at_max_slope)[0]
    half_slope_index = np.where(dy[indices_above_max_slope] <= slope_half)[0]

    if len(half_slope_index) > 0:
        L2_half_slope = x_fit[indices_above_max_slope[half_slope_index[0]]]
        OAE_level_half_slope = y_fit[indices_above_max_slope[half_slope_index[0]]]
    else:
        L2_half_slope = None
        OAE_level_half_slope = None

    # Find the first point where the slope is below or equal to max_slope/2, after the max slope
    half_max_slope_index = np.where(dy[indices_above_max_slope] <= slope_half_max)[0]
    if len(half_max_slope_index) > 0:
        L2_half_max_slope = x_fit[indices_above_max_slope[half_max_slope_index[0]]]
        OAE_level_half_max_slope = y_fit[indices_above_max_slope[half_max_slope_index[0]]]
    else:
        L2_half_max_slope = None
        OAE_level_half_max_slope = None

    return (p, max_slope, L2_at_max_slope, OAE_level_at_max_slope,
            L2_half_slope, OAE_level_half_slope,
            L2_half_max_slope, OAE_level_half_max_slope)


# Example usage

L2 = np.array(L2list)

# Create a dictionary to hold all estimated results
estimated_results = {}
for i in range(4):  # Loop through each dataset index
   
    y_data = DPioNL[i]
    # Call the fitting function
    fit_results = fit_polynomial(L2, y_data, degree=4)
    
    x_fit = np.linspace(np.min(L2), np.max(L2), 100)  # Smooth curve for the fit
    y_fit = fit_results[0](x_fit)
    
    # Plot fitted curve using the same color as the data but exclude from legend
    ax.plot(x_fit, y_fit, color=data_line[i][0].get_color(), linestyle='--', linewidth=1, 
            label="_nolegend_")  # No label in the legend for the fit

    # Extract key points from the fit results
    L2_at_max_slope = fit_results[2]
    OAE_level_at_max_slope = fit_results[3]
    L2_half_slope = fit_results[4]  # Slope of 1/2
    OAE_level_half_slope = fit_results[5]
    L2_half_max_slope = fit_results[6]  # Slope of max_slope / 2
    OAE_level_half_max_slope = fit_results[7]

    # Plot the point where the slope is maximum but exclude from legend
    ax.plot(L2_at_max_slope, OAE_level_at_max_slope, 'o', color=data_line[i][0].get_color(), 
            markersize=8, label="_nolegend_")  # Circle marker, no legend

    # Plot the point where the slope is 1/2 but exclude from legend
    if L2_half_slope is not None:
        ax.plot(L2_half_slope, OAE_level_half_slope, 's', color=data_line[i][0].get_color(), 
                markersize=8, label="_nolegend_")  # Square marker, no legend

    # Plot the point where the slope is max_slope/2 but exclude from legend
    if L2_half_max_slope is not None:
        ax.plot(L2_half_max_slope, OAE_level_half_max_slope, '^', color=data_line[i][0].get_color(), 
                markersize=8, label="_nolegend_")  # Triangle marker, no legend

    # Store results in the dictionary
    estimated_results[f'fit_results_{i}'] = {
        'fitted_polynomial': fit_results[0],
        'max_slope': fit_results[1],
        'L2_at_max_slope': fit_results[2],
        'OAE_level_at_max_slope': fit_results[3],
        'L2_half_slope': fit_results[4],
        'OAE_level_half_slope': fit_results[5],
        'L2_half_max_slope': fit_results[6],
        'OAE_level_half_max_slope': fit_results[7],
    }

# Show legend only for data
ax.legend()


# Save the second plot as well (optional)
plt.savefig('Figures/DPgrams/io' + subjN + '.png', format='png', dpi=300)  # Save the second graph

# Save the results to a .mat file
filename = f'estData{subjN}.mat'
savemat(filename, estimated_results)

print(f'Saved estimated results to {filename}')


