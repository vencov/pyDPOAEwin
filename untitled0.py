# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:28:18 2025

@author: audiobunka
"""

from UserModules.pyDPOAEmodule import makeChirpTrain
import numpy as np
import matplotlib.pyplot as plt

fsamp = 44100
plotflag = 0  # plot responses? for debuging purposes set to 1
f1 = 0  # start frequency
f2 = fsamp/2 # stop frequency
buffersize = 2048
Nsamp = buffersize  # number of samples in the chirp
Nchirps = 4
chirptrain, chirpIn = makeChirpTrain(f1,f2,Nsamp,fsamp,Nchirps)

plt.close('all')
import matplotlib.pyplot as plt

# Base font size increased by 30%
base_fs = 14
fs = int(base_fs * 1.3)  # ~13

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharey=True)

# --- Upper panel: single chirp ---
ax1.plot(chirpIn, color="b")
ax1.set_title("Single Chirp", fontsize=fs)
ax1.set_ylabel("Amplitude", fontsize=fs)
ax1.tick_params(axis='both', labelsize=fs)

# --- Lower panel: chirp train ---
ax2.plot(chirptrain, color="r")
ax2.set_title("Chirp Train (4 chirps)", fontsize=fs)
ax2.set_xlabel("Samples", fontsize=fs)
ax2.set_ylabel("Amplitude", fontsize=fs)
ax2.tick_params(axis='both', labelsize=fs)


plt.tight_layout()

# --- Export figure to PNG ---
plt.savefig("Figures/FAV/chirp_figure.png", dpi=500)  # saves as 300 dpi PNG


plt.show()
