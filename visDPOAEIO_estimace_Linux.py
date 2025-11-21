# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:05:58 2025

vis estimated DPOAE i/o for linux computer

@author: audiobunka
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

plt.close('all')

FolderName = 'EstimaceLinux/'
data = loadmat(FolderName + 'DPioCS_results_s081L.mat')


L2ar = data['L2array'].flatten() # L2values

DPio = data['DPioNL'] # DPiofce at CF values

CFs = data['CF'].flatten()



fig,ax = plt.subplots()
ax.plot(L2ar,DPio[3,:])