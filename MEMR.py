#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from UserModules.pyRMEsd import RMEplayrecBias
from UserModules.pyUtilities import butter_highpass_filter
from UserModules.pyDPOAEmodule import create_noise, generateClickTrain, getSClat, changeSampleRate

def modulate_noise_aplitude(noise,fsamp):
    n = len(noise)
    half = n//2
    ramp_up = np.linspace(0,1,half)
    ramp_down = np.linspace(1,0,n-half)
    amp_env = np.concatenate((ramp_up, ramp_down))
    modulated_noise = noise * amp_env
    
    return modulated_noise

fsamp = 44100
mod_len = 4
bufsize = 4096
micGain = 40
Npulse = 2048
channelN = 5  




plt.close('all')
changeSampleRate(fsamp,bufsize,SC=10)
latency_SC = getSClat(fsamp,bufsize,SC=10)


#---------------------------------------------- Noise creation ---------------------------------------------------------

noise_samples, n_reps, H, n_loops = create_noise(mod_len, fsamp)



# Generating noise
#noise = np.random.normal(0, 1, noise_samples * n_reps * n_loops)
noise = np.random.normal(0, 1, noise_samples * 2)



# Opakování 16x
noise_train = np.tile(H[:,0]*noise, 16)

Nnoisetr = len(noise_train)

Nclicks = Nnoisetr//Npulse


cfname = 'clickInBP_OK400_4000_44k1.mat'
yupr = generateClickTrain(cfname, Nclicks)

len_diff = len(noise_train) - len(yupr)
yupr_train = np.pad(yupr, (0,len_diff), 'constant', constant_values = 0)


H_resized = np.resize(H.flatten(), len(noise))
noise = noise * H_resized
noise = np.resize(noise, len(yupr))

"""
clickIndx = np.where(yupr_train > 0.01)[0]
holeDur1 = 0.003
holeDur2 = 0.012
holeN1 = round(fsamp * holeDur1)
holeN2 = round(fsamp * holeDur2)


Mask = np.ones_like(noise_train)
for idx in clickIndx:
    start_idx = max(0, idx - holeN1 + 1)
    end_idx = min(len(noise_train), idx + holeN2)
    Mask[start_idx:end_idx] = 0

noise_train *= Mask
"""
noise_train = noise_train * 1/10
#-----------------------------------------------------------------------------------------------------------------------

#1print("Playing noise...")
##sd.play(Noise, fsamp)
#sd.wait()
#print("Done!")

# Combination to play: clicks and noise separated to different channels

if channelN == 5:
    clicktrain_with_noise = np.vstack((yupr_train,  np.zeros_like(yupr_train),   np.zeros_like(yupr_train),  np.zeros_like(yupr_train),  noise_train, np.zeros_like(yupr_train))).T
elif channelN == 6:
    clicktrain_with_noise = np.vstack((yupr_train,  np.zeros_like(yupr_train),   np.zeros_like(yupr_train),  np.zeros_like(yupr_train),  np.zeros_like(yupr_train), noise_train)).T

Amp = 0.1
recorded_clicktrain = RMEplayrecBias(Amp * clicktrain_with_noise, fsamp, SC=10, buffersize=4096)
#recorded_clicktrain = Amp * clicktrain_with_noise + np.random.normal(0, 0.01, clicktrain_with_noise.shape)

# Filtration of the signal
cutoff = 300
recorded_clicktrain_filtered = butter_highpass_filter(recorded_clicktrain[:, 0], cutoff, fsamp, 3)




#--------------------------average in the time domain------------------------------------------------------------------
reshaped_data = recorded_clicktrain_filtered[latency_SC:]

reshaped_data = np.reshape(reshaped_data[:Nclicks*Npulse],(Npulse,Nclicks),order='F')
averaged_data = np.mean(reshaped_data, axis=1)


averaged_data_pa = averaged_data / (0.003 * 10**(micGain / 20))
tx = np.linspace(0, Npulse * 1 / fsamp, Npulse)
plt.figure()
plt.plot(tx, averaged_data_pa)
plt.xlabel('Čas (s)')
plt.ylabel('Amplituda (Pa)')
plt.title('Klikový signál (pravé ucho)')
plt.show()

noise_time = np.linspace(0, len(noise_train) / fsamp, len(noise_train))
plt.figure()
plt.plot(noise_time, noise_train, color='orange')
plt.xlabel('Čas (s)')
plt.ylabel('Amplituda')
plt.title('Šum (levé ucho)')

maxPa = max(averaged_data_pa)
minPa = min(averaged_data_pa)
peSPL = 20 * np.log10(0.5 * (maxPa - minPa) / (np.sqrt(2) * 2e-5))

print(f"Maximální hodnota: {maxPa}")
print(f"Minimální hodnota: {minPa}")
print(f"Úroveň šumu (peSPL): {peSPL}")

plt.show()