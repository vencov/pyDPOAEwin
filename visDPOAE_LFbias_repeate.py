# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:38:12 2025

@author: audiobunka
"""

import os
from scipy.io import loadmat
import numpy as np
from UserModules.pyUtilities import butter_highpass_filter, butter_lowpass_filter
import matplotlib.pyplot as plt
path = "Results/s151/"
subj_name = 's151'


del1 = '25_09_29_17_13_22Lbbn20dBFPL'
del1 = '25_09_29_17_26_46Lbbn10'
del1 = 'SOAE_bbn_pokus_25_09_29_17_33_11Lbbn15dBFPL'
del1 = '25_10_02_13_31_12_F2_2400'
del1 = 's001_25_10_02_14_56_12_F2_2400'
del1 = 's004_25_10_02_18_11_26_F2'
del1 = 'LFbststDPOAE_s004_25_10_03_14_51_51_F2_2400F1_2000L1_65dB_L2_60dB_f2f1_120_fbias96Hz_Lbias0dB_R_1'
#del1 = '25_10_02_18_22_55_F2_2400F1_2000'
#del1 = '25_10_03_16_35_28_F2_2400F1_2000'
#del1 = '25_10_03_14_56_22_F2_2400F1_2000L1_65dB_L2_60dB'
#del1 = '25_10_03_14_54_30_F2_2400F1_2000L1_65dB_L2_60dB'
#del1 = '25_10_03_16_40_26_F2_2400F1_2000L1_75dB_L2_65dB_f2f1_120_fbias96Hz_Lbias83dB_R'
#del1 = 's001_25_10_06_11_10_25_F2_2400F1_2000L1_60dB_L2_50dB_f2f1_120_fbias96Hz_Lbias85dB_R'
#del1 = 's001_25_10_06_11_08_44_F2_2400F1_2000L1_60dB_L2_50dB_f2f1_120_fbias96Hz_Lbias85dB_R'
#del1 = 's001_25_10_06_11_06_41_F2_2400F1_2000L1_55dB_L2_40dB_f2f1_120_fbias96Hz_Lbias0dB'
#del1 = 's001_25_10_06_11_07_27_F2_2400F1_2000L1_60dB_L2_50dB_f2f1_120_fbias96Hz_Lbias0dB_R'
del1 = 's151_25_10_06_12_13_55_F2_2400F1_2000L1_57dB_L2_45dB_f2f1_120_fbias96Hz_Lbias0dB_R'
#del1 = 's151_25_10_06_12_14_31_F2_2400F1_2000L1_57dB_L2_45dB_f2f1_120_fbias96Hz_Lbias85dB_R'
#del1 = 's151_25_10_06_12_16_03_F2_2400F1_2000L1_57dB_L2_45dB_f2f1_120_fbias96Hz_Lbias88dB_R'
#del1 = 's001_25_10_06_11_11_54_F2_2400F1_2000L1_60dB_L2_50dB_f2f1_120_fbias96Hz_Lbias90dB_R'
#del1 = '25_10_03_16_55_45_F2_2400F1_2000L1_75dB_L2_65dB_f2f1_120_fbias96Hz_Lbias86dB_R'
#del1 = '25_09_29_17_53_33Lbbn20'
fsamp = 44100;
plt.close('all')
Nwin = 11025
avg = 0


def getDPgram(path,DatePat,fsamp):

    dir_list = os.listdir(path)
    result = None    
    n = 0
    SPall = 0
    all_frames = []  # list to collect frames
    for k in range(len(dir_list)):
        if  DatePat in dir_list[k]:
            data = loadmat(path+ dir_list[k])
            lat = data['latency_SC'][0][0]
            print([path+ dir_list[k]])
            print(f"SC latency: {lat}")
            
            sig_outR = data['recsig']
            cutoff = 300 # cutoff frequency for high pass filter to filter out low frequency noise
            sig_outR1 = butter_highpass_filter(sig_outR[:,0], cutoff, fsamp, 3)

# convert to dB SPL (apply the calibration curve for the probe microphone)
            MG = 0  # gain on the probe
            
            sig_out1RS = sig_outR1[lat+Nwin:420000]/(0.003*10**(MG/20))  # remove the sound card latency

            Pbias = butter_lowpass_filter(sig_outR1[lat+Nwin:420000]/(0.003*10**(MG/20)), 150, fsamp, order=5)


            total_cols = len(sig_out1RS) // Nwin
            
            reshaped = np.reshape(sig_out1RS[:total_cols*Nwin], (Nwin, total_cols),order='F')
            Pbreshaped = np.reshape(Pbias[:total_cols*Nwin], (Nwin, total_cols),order='F')
            # concatenate
            if result is None:
                result = reshaped
                resultPb = Pbreshaped
            else:
                result = np.concatenate((result, reshaped), axis=1)  # concatenate horizontally (columns)
                resultPb = np.concatenate((resultPb, Pbreshaped), axis=1)  # concatenate horizontally (columns)
                
            
    return result, resultPb
    
    
    
res, Pb = getDPgram(path,del1,fsamp)        

avg_signal = np.mean(res, axis=1)  # shape = (Nwin,)
Pbavg = np.mean(Pb, axis=1)  # shape = (Nwin,)

fig,ax = plt.subplots()
ax.plot(avg_signal)

fig,ax = plt.subplots()

fx = np.fft.fftfreq(len(avg_signal),1/fsamp)

Spcalib=np.fft.fft(avg_signal)

ax.plot(fx,20*np.log10(abs(Spcalib)))


f1 = 2000
f2 = 2400
fdp = 2*f1-f2
fBias = 96
# indices of nearest FFT bins to desired freqs
freqs_interest = [fdp + m*fBias for m in range(-2, 4)]
idx_interest = [np.argmin(np.abs(fx - f)) for f in freqs_interest]



Sp_filt = np.zeros_like(Spcalib, dtype=complex)
for idx in idx_interest:
    Sp_filt[idx] = Spcalib[idx]

ax.plot(fx,20*np.log10(np.abs(Sp_filt)),'rx')
    
N = len(Spcalib)
for idx in idx_interest:
    if idx != 0 and idx != N//2:  # skip DC and Nyquist
        Sp_filt[-idx] = np.conj(Spcalib[idx])            


sig_filt = np.fft.ifft(Sp_filt)
sig_filt = np.real(sig_filt)  # enforce real

from scipy.signal import hilbert

analytic_sig = hilbert(sig_filt)
envelope = np.abs(analytic_sig)

fig,(ax1,ax2)=plt.subplots(2,1)
ax1.plot(envelope[2000:3000])
diff = 140
ax2.plot(Pbavg[2000+diff:3000+diff])

fig,(ax1,ax2)=plt.subplots(2,1)
ax1.plot(envelope)
diff = 140
ax2.plot(Pbavg)



#%%

fig,ax = plt.subplots()
ax.plot(Pbavg[2000+diff:3000+diff],envelope[2000:3000])



