#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:08:05 2025

@author: vencov
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_audiogram(freq_left, audiogram_left, freq_right, audiogram_right,
                    freq_bone_left=None, audiogram_bone_left=None,
                    freq_bone_right=None, audiogram_bone_right=None):
    plt.figure(figsize=(6, 4))

    # Audiogram is plotted upside down, so invert the y-axis
    plt.gca().invert_yaxis()

    # Function to plot multiple measurements
    def plot_multiple(freq_list, audiogram_list, color, marker, linestyle, label):
        if freq_list is not None and audiogram_list is not None:
            for i, (freq, audiogram) in enumerate(zip(freq_list, audiogram_list)):
                plt.plot(freq, audiogram, linestyle, color=color, marker=marker, 
                         label=label if i == 0 else "_nolegend_")  # Only label first line

    # Plot left ear (blue, 'x' marker)
    plot_multiple(freq_left, audiogram_left, 'b', 'x', '-', 'Left Ear')

    # Plot right ear (red, 'o' marker)
    plot_multiple(freq_right, audiogram_right, 'r', 'o', '-', 'Right Ear')

    # Plot left ear bone conduction (blue, '^' marker, dashed line)
    plot_multiple(freq_bone_left, audiogram_bone_left, 'b', '^', '--', 'Left Ear (Bone)')

    # Plot right ear bone conduction (red, 's' marker, dashed line)
    plot_multiple(freq_bone_right, audiogram_bone_right, 'r', 's', '--', 'Right Ear (Bone)')

    # Formatting
    plt.ylim(-10, 80)
    plt.gca().invert_yaxis()
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Hearing Level (dB HL)')
    plt.title('Audiogram')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    default_xticks = np.array([0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8])
    plt.xticks(default_xticks, labels=[str(f) for f in default_xticks])
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.savefig("Figures/s139.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


# Example usage
freq_left = [np.array([0.25, 0.5, 1, 2, 4, 6,8 ]) ]
audiogram_left = [np.array([25, 35, 45, 45, 40, 50, 40]) 
                  ]


freq_right = [np.array([0.25, 0.5, 1, 2,  4, 6,8])]# Right ear frequencies in kHz
audiogram_right = [np.array([20, 35, 50, 45, 40, 45, 35])]  # Example right ear values




#freq_left = None
#audiogram_left = None

freq_bone_left = None
audiogram_bone_left = None

freq_bone_right = None
audiogram_bone_right = None

plot_audiogram(freq_left, audiogram_left, freq_right, audiogram_right,
               freq_bone_left, audiogram_bone_left,
               freq_bone_right, audiogram_bone_right)
