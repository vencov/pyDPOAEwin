# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 13:46:05 2025

@author: audiobunka
"""


from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from UserModules.pyUtilities import butter_highpass_filter, butter_lowpass_filter
from UserModules.pyDPOAEmodule import calcDPststBias
plt.close('all')

foldername = 'Results/pokus/'
filename = 'LFbststDPOAE_lFbias_25_09_22_17_48_47_F2_2400F1_2000L1_65dB_L2_55dB_f2f1_120_Lbias85dB_L.mat'
filename = 'LFbststDPOAE_lFbias_25_09_22_17_50_07_F2_2400F1_2000L1_65dB_L2_55dB_f2f1_120_Lbias88dB_L.mat'
filename = 'LFbststDPOAE_lFbias_25_09_22_17_43_13_F2_2400F1_2000L1_65dB_L2_55dB_f2f1_120_Lbias70dB_L.mat'
filename = 'LFbststDPOAE_lFbias_25_09_22_17_40_20_F2_2400F1_2000L1_65dB_L2_55dB_f2f1_120_Lbias90dB_L.mat'
filename = 'LFbststDPOAE_lFbias_25_09_22_17_41_45_F2_2400F1_2000L1_65dB_L2_55dB_f2f1_120_Lbias80dB_L.mat'

data = loadmat(foldername  + filename)

# Unpack into variables
for key, value in data.items():
    if not key.startswith('__'):  # skip MATLAB metadata keys
        # If it's a numpy array with exactly one element, convert to scalar
        if isinstance(value, np.ndarray) and value.size == 1:
            globals()[key] = value.item()
        else:
            globals()[key] = value


latency_SC = 4169

cut_off = 300 # cut of frequency of the high pass filter
recSig = butter_highpass_filter(recsig[:,0], cut_off, fsamp, order=5)

Pbias = butter_lowpass_filter(recsig[:,0], 100, fsamp, order=5)



fdp = 2*f1-f2

recSig = recSig[latency_SC:]    #calculate frequency response
Thresh = 4e-1
fig,ax = plt.subplots()
ax.plot(recSig)
plt.show()
#fig,ax = plt.subplots()
#ax.plot(np.abs(np.fft.rfft(recSig)))
#plt.show()
fBias = 96
Nwin = 44100
Nwin = 22050    
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
freqs_interest = [fdp + m*fBias for m in range(-2, 3)]
idx_interest = [np.argmin(np.abs(fx - f)) for f in freqs_interest]



Sp_filt = np.zeros_like(Spcalib, dtype=complex)
for idx in idx_interest:
    Sp_filt[idx] = Spcalib[idx]

ax.plot(fx[:int(len(fx)//2)+1],20*np.log10(np.abs(Sp_filt)/(np.sqrt(2)*2e-5)),'rx')
    
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
ax1.plot(envelope[:2000])
ax2.plot(Pbias[latency_SC+10000:latency_SC+12000])



plt.show()
