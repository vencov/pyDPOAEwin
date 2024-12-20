# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:37:52 2024

@author: audiobunka
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from UserModules.pyDPOAEmodule import create_noise

# load RESPL for headphones
data = loadmat('Calibration_files/HDA300.mat')
ear = 'L'

Hleft = data['HtrLeft'].flatten()
Hright = data['HtrRight'].flatten()
fxL = data['fxL'].flatten()
fxR = data['fxR'].flatten()

fsamp = 44100
mod_len = 4
noise_samples, n_reps, H, n_loops = create_noise(mod_len, fsamp)

# Generating noise
#noise = np.random.normal(0, 1, noise_samples * n_reps * n_loops)
noise = np.random.normal(0, 1, noise_samples * 2)

noiserfft = np.fft.rfft(noise)

Nnrfft = len(noiserfft)
fx =np.arange(2*Nnrfft-2)/(2*Nnrfft-1)
fx = 44100*fx

hifreq = 14000
lofreq = 200

# Calculate bandwidth and RMS noise
bw = hifreq - lofreq


nptslofreq = np.where(fx >= lofreq)[0][0]
nptshifreq = np.where(fx >= hifreq)[0][0]

# Zero out undesired frequency bands
noiserfft[:nptslofreq] = 0.0
noiserfft[nptshifreq:] = 0.0

Hrint = np.interp(fx[nptslofreq:nptshifreq],fxL,Hleft)

noiseSc = noiserfft[nptslofreq:nptshifreq]/np.abs(Hrint)

 # 10**(Ltone/20)*numpy.sqrt(2)*2e-5
stim1 = np.real(np.fft.irfft(noiseSc))

fig,ax = plt.subplots()
ax.plot(stim1)