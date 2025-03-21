import matplotlib.pyplot as plt
import numpy as np

def plot_audiogram(freq_left, audiogram_left, freq_right, audiogram_right,
                    freq_bone_left=None, audiogram_bone_left=None,
                    freq_bone_right=None, audiogram_bone_right=None):
    FontS = 16  # Font size variable for easy adjustments
    
    plt.figure(figsize=(6, 4))
    plt.gca().invert_yaxis()  # Audiogram is plotted upside down

    def plot_multiple(freq_list, audiogram_list, color, marker, linestyle, label):
        if freq_list is not None and audiogram_list is not None:
            for i, (freq, audiogram) in enumerate(zip(freq_list, audiogram_list)):
                plt.plot(freq, audiogram, linestyle, color=color, marker=marker, 
                         label=label if i == 0 else "_nolegend_")

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
    plt.xlabel('Frequency (kHz)', fontsize=FontS)
    plt.ylabel('Hearing Level (dB HL)', fontsize=FontS)
    plt.title('Audiogram', fontsize=FontS)
    plt.xscale('log')
    
    # Define xticks and format labels
    default_xticks = np.array([0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8])
    xtick_labels = [str(int(f)) if f.is_integer() else str(f) for f in default_xticks]
    plt.xticks(default_xticks, labels=xtick_labels, fontsize=FontS)
    plt.yticks(fontsize=FontS)
    
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(fontsize=FontS)
    plt.gca().invert_yaxis()
    plt.savefig("Figures/s139.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

# Example usage
freq_left = [np.array([0.25, 0.5, 1, 2, 4, 6, 8])]
audiogram_left = [np.array([25, 35, 45, 45, 40, 50, 40])]

freq_right = [np.array([0.25, 0.5, 1, 2, 4, 6, 8])]
audiogram_right = [np.array([20, 35, 50, 45, 40, 45, 35])]

freq_bone_left = None
audiogram_bone_left = None
freq_bone_right = None
audiogram_bone_right = None

plot_audiogram(freq_left, audiogram_left, freq_right, audiogram_right,
               freq_bone_left, audiogram_bone_left,
               freq_bone_right, audiogram_bone_right)
