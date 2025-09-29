# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:20:35 2025

@author: audiobunka
"""

from scipy.io import loadmat
import os
from UserModules.pyUtilities import butter_highpass_filter 
import numpy as np
import matplotlib.pyplot as plt

import UserModules.pyDPOAEmodule as pDP
    
#plt.close('all')


f2f1 = 1.2

fsamp = 96000;
    
subjD = {}



subjN = 's139L'


subjN_L = 's139L'
subjN_R = 's139R'

subjD['s001L'] = ['Results/s001/GACR/ticho/', '24_12_18_11_02_18_F2b_8000Hz']
subjD['s001R'] = ['Results/s001/GACR/ticho/', '24_12_18_11_00_01_F2b_8000Hz']


subjN_L = 's001L'
subjN_R = 's001R'


def getSOAE(path,DatePat):

    dir_list = os.listdir(path)
    
    n = 0
    DPgramsDict = {}
    
    for k in range(len(dir_list)):
        
        if  DatePat in dir_list[k]:
            data = loadmat(path+ dir_list[k])
            #lat = 16448
            
            lat = data['lat_SC'][0][0]
            print([path+ dir_list[k]])
            print(f"SC latency: {lat}")
            
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
                
                
                
                max_len = max(recSig1.shape[0], recSig2.shape[0], recSig3.shape[0], recSig4.shape[0])
                max_mat = recMat1.shape[0]
                # Function to pad a matrix with zeros
                def pad_matrix(matrix, max_len):
                    rows, cols = matrix.shape
                    if rows < max_len:
                        pad = np.zeros((max_len - rows, cols))
                        return np.vstack((matrix, pad))
                    return matrix
                
                # Apply padding
                recSig1 = pad_matrix(recSig1, max_len)
                recSig2 = pad_matrix(recSig2, max_len)
                recSig3 = pad_matrix(recSig3, max_len)
                recSig4 = pad_matrix(recSig4, max_len)
                
                
                
                recMat1 = np.c_[recMat1, recSig1[lat:max_mat+lat,0]]  # add to make a matrix with columns for every run
                recMat2 = np.c_[recMat2, recSig2[lat:max_mat+lat,0]]  # add to make a matrix with columns for every run
                recMat3 = np.c_[recMat3, recSig3[lat:max_mat+lat,0]]  # add to make a matrix with columns for every run
                recMat4 = np.c_[recMat4, recSig4[lat:max_mat+lat,0]]  # add to make a matrix with columns for every run
           
    
                #oaeDS = (np.nanmean(recMat1,1)+np.nanmean(recMat2,1)+np.nanmean(recMat3,1)+np.nanmean(recMat4,1))/4  # exclude samples set to NaN (noisy samples)
                
                #DPgrR = estimateDRgram(oaeDS,f2f1,f2b,f2e,octpersec,GainMic)
                
                #DPgramsDict[str(n)] = DPgrR

            n += 1
            
    vec1 = recMat1.reshape(-1, order='F')  # 'F' for Fortran-style (column-major) order
    vec2 = recMat2.reshape(-1, order='F')
    vec3 = recMat3.reshape(-1, order='F')
    vec4 = recMat4.reshape(-1, order='F')

    # Concatenate all vectors into a single 1D vector
    final_vec = np.concatenate([vec1, vec2, vec3, vec4])
        
    return final_vec



DPgr_L = {}
L2list_L = []
f2f1list_L = []
DPgr_R = {}
L2list_R = []
f2f1list_R = []

for i in range(1,len(subjD[subjN_L])):

    DPgrD_L = getSOAE(subjD[subjN_L][0], subjD[subjN_L][i])

for i in range(1,len(subjD[subjN_R])):

    DPgrD_R = getSOAE(subjD[subjN_R][0], subjD[subjN_R][i])
        
    


Nsamp = int(fsamp) # one second long frames in which we calculate absolute value of spectrum
Nframes = int(np.floor(len(DPgrD_L)/Nsamp))


Ntotal = Nframes*Nsamp # total length of used samples

sig_out1SLs = DPgrD_L[:Ntotal]  # Limit to integer number of frames
sig_out1SRs = DPgrD_R[:Ntotal]  # Limit to integer number of frames

#SRSsel = rfft(RSsel2)
SpL = np.zeros(int(Nsamp/2)+1)
SpR = np.zeros(int(Nsamp/2)+1)
for i in range(len(sig_out1SLs)//Nsamp):
    SpL += np.abs(np.fft.rfft(sig_out1SLs[i*Nsamp:(i+1)*Nsamp])/Nsamp)
    SpR += np.abs(np.fft.rfft(sig_out1SRs[i*Nsamp:(i+1)*Nsamp])/Nsamp)

SpL /= Nframes
SpR /= Nframes

fx = np.linspace(0,fsamp/2,len(SpL),endpoint=False)  # frequency axis

#fxI = np.linspace(0, 10, num=41, endpoint=True) # interpolated freq axis
fxI = fx# np.arange(0,20e3)
SpIL = SpL #interp1d(fx,SpL,kind='cubic')
#SpIL = SpIL(fxI)
SpIR = SpR  #interp1d(fx,SpR,kind='cubic')
#SpIR = SpIR(fxI)

# convert to dB SPL (apply the calibration curve for the probe microphone)
MG = 40  # gain on the probe
SpILc = SpIL/(0.003*10**(MG/20))  # convert to Pascals
SpIRc = SpIR/(0.003*10**(MG/20))  # convert to Pascals

subj_name = 's079'


# Convert width to inches (since matplotlib uses inches for figure size)
width_cm = 20
width_inch = width_cm / 2.54
setFS = 14 # fontsize
# Define the aspect ratio (keeping it the same as before, 1:1)
aspect_ratio = 0.8

# Set height accordingly
height_inch = width_inch * aspect_ratio



# Create the figure with the new size
fig, ax = plt.subplots(figsize=(width_inch, height_inch))

# Plot the data
ax.plot(fxI, 20 * np.log10(SpILc / (np.sqrt(2) * 2e-5)))
ax.plot(fxI, 20 * np.log10(SpIRc / (np.sqrt(2) * 2e-5)))

# Set x-axis ticks at linear intervals (1kHz, 2kHz, etc.)
ax.set_xticks([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])

# Change the font size of the tick labels
ax.tick_params(axis='both', labelsize=setFS-1)  # Set font size for both x and y tick labels

# Set x-axis to linear scale and limits
ax.set_xlim([300, 8000])
ax.set_ylim([-40, 20])

# Set labels with font size
ax.set_xlabel('Frequency (kHz)', fontsize=setFS)
ax.set_ylabel('Sound pressure level (dB)', fontsize=setFS)

# Move tick marks inside the graph
ax.tick_params(axis='both', direction='in')

# Set legend and remove title
ax.legend(('Left ear', 'Right ear'), fontsize=setFS)

# Add subject name in the bottom left corner
