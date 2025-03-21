# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:47:35 2025

program for loading manually detected hearing thresholds



@author: audiobunka
"""

import numpy as np
import matplotlib.pyplot as plt

def load_audiogram_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    data = {}
    ear = None
    
    for line in lines:
        line = line.strip()
        if line in ['L:', 'R:']:
            ear = line[0]  # 'L' or 'R'
            data[ear] = []
        elif ear:
            data[ear].append(list(map(float, line.split())))
    
    return data

def plot_audiogram(data):
    plt.figure(figsize=(6, 4))
    
    markers = {'L': 'o-', 'R': 's-'}  # Different markers for left and right ears
    colors = {'L': 'blue', 'R': 'red'}
    labels = {'L': 'Left Ear', 'R': 'Right Ear'}
    
    for ear in data:
        freqs = np.array(data[ear][0])
        hl_values = np.array(data[ear][1])
        plt.plot(freqs, hl_values, markers[ear], color=colors[ear], label=labels[ear])
    
    plt.ylim((-10,120))
    plt.gca().invert_yaxis()  # Reverse y-axis
    plt.xscale('log')  # Logarithmic scale for frequency
    plt.xticks([0.125, 0.25, 0.5, 1, 2, 6, 4, 8], ['0.125', '0.25', '0.5', '1', '2', '4', '6', '8'], fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Frequency (kHz)', fontsize=16)
    plt.ylabel('Hearing Level (dB HL)', fontsize=16)
    plt.title('Audiogram', fontsize=16)
        
    plt.legend(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()



subjN = 's139'
FolderName = 'Results/' + subjN + '/HT/'
FileName = subjN + 'HTm.txt'
# Usage example

data = load_audiogram_data(FolderName + FileName)
plot_audiogram(data)
