# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 19:15:20 2025

@author: vacla
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:41:23 2024

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


subjD['s084L'] = ['Results/s084/', '24_06_04_14_51_55_F2b_8000Hz', '24_06_04_14_53_19_F2b_8000Hz', '24_06_04_14_55_29_F2b_8000Hz', '24_06_04_14_57_04_F2b_8000Hz', '24_06_04_14_59_03_F2b_8000Hz', '24_06_04_15_00_36_F2b_8000Hz', '24_06_04_15_02_12_F2b_8000Hz', '24_06_04_15_03_49_F2b_8000Hz', '24_06_04_15_05_37_F2b_8000Hz']
subjD['s084R'] = ['Results/s084/', '24_06_04_15_43_12_F2b_8000Hz', '24_06_04_15_44_49_F2b_8000Hz', '24_06_04_15_46_11_F2b_8000Hz', '24_06_04_15_47_36_F2b_8000Hz', '24_06_04_15_49_17_F2b_8000Hz', '24_06_04_15_51_05_F2b_8000Hz', '24_06_04_15_52_52_F2b_8000Hz', '24_06_04_15_54_39_F2b_8000Hz', '24_06_04_15_56_25_F2b_8000Hz']

subjD['s086L'] = ['Results/s086/', '24_06_20_13_06_20_F2b_8000Hz', '24_06_20_13_08_06_F2b_8000Hz', '24_06_20_13_11_09_F2b_8000Hz', '24_06_20_13_13_02_F2b_8000Hz', '24_06_20_13_16_10_F2b_8000Hz', '24_06_20_13_17_52_F2b_8000Hz', '24_06_20_13_19_37_F2b_8000Hz', '24_06_20_13_21_16_F2b_8000Hz', '24_06_20_13_22_54_F2b_8000Hz']
subjD['s086R'] =   ['Results/s086/', '24_06_20_12_49_29_F2b_8000Hz', '24_06_20_12_50_53_F2b_8000Hz', '24_06_20_12_52_05_F2b_8000Hz', '24_06_20_12_53_45_F2b_8000Hz', '24_06_20_12_55_38_F2b_8000Hz', '24_06_20_12_57_05_F2b_8000Hz', '24_06_20_12_58_57_F2b_8000Hz', '24_06_20_13_00_48_F2b_8000Hz', '24_06_20_13_02_40_F2b_8000Hz']

subjD['s087L'] =  ['Results/s087/', '24_06_20_14_42_18_F2b_8000Hz', '24_06_20_14_44_16_F2b_8000Hz', '24_06_20_14_46_10_F2b_8000Hz', '24_06_20_14_48_00_F2b_8000Hz', '24_06_20_14_49_53_F2b_8000Hz', '24_06_20_14_51_56_F2b_8000Hz', '24_06_20_14_54_55_F2b_8000Hz', '24_06_20_14_57_07_F2b_8000Hz', '24_06_20_14_58_56_F2b_8000Hz']
subjD['s087R'] =  ['Results/s087/', '24_06_20_14_15_58_F2b_8000Hz', '24_06_20_14_18_28_F2b_8000Hz', '24_06_20_14_20_29_F2b_8000Hz', '24_06_20_14_23_08_F2b_8000Hz', '24_06_20_14_25_05_F2b_8000Hz', '24_06_20_14_27_09_F2b_8000Hz', '24_06_20_14_29_32_F2b_8000Hz', '24_06_20_14_31_38_F2b_8000Hz', '24_06_20_14_33_40_F2b_8000Hz', '24_06_20_14_35_31_F2b_8000Hz', '24_06_20_14_37_27_F2b_8000Hz']

subjD['s088L'] =  ['Results/s088/', '24_06_20_16_13_15_F2b_8000Hz', '24_06_20_16_14_35_F2b_8000Hz', '24_06_20_16_15_57_F2b_8000Hz', '24_06_20_16_17_25_F2b_8000Hz', '24_06_20_16_18_48_F2b_8000Hz', '24_06_20_16_20_08_F2b_8000Hz', '24_06_20_16_21_34_F2b_8000Hz', '24_06_20_16_22_57_F2b_8000Hz', '24_06_20_16_24_28_F2b_8000Hz']
subjD['s088R'] =  ['Results/s088/', '24_06_20_16_40_30_F2b_8000Hz', '24_06_20_16_41_49_F2b_8000Hz', '24_06_20_16_43_17_F2b_8000Hz', '24_06_20_16_44_45_F2b_8000Hz', '24_06_20_16_46_10_F2b_8000Hz', '24_06_20_16_47_34_F2b_8000Hz', '24_06_20_16_48_56_F2b_8000Hz', '24_06_20_16_50_20_F2b_8000Hz', '24_06_20_16_51_42_F2b_8000Hz']

subjD['s089L'] = ['Results/s089/', '24_07_01_10_59_59_F2b_8000Hz', '24_07_01_11_01_21_F2b_8000Hz', '24_07_01_11_02_34_F2b_8000Hz', '24_07_01_11_03_41_F2b_8000Hz', '24_07_01_11_05_18_F2b_8000Hz', '24_07_01_11_06_38_F2b_8000Hz', '24_07_01_11_07_55_F2b_8000Hz']
subjD['s089R'] = ['Results/s089/', '24_07_11_11_15_18_F2b_8000Hz', '24_07_11_11_17_10_F2b_8000Hz', '24_07_11_11_19_01_F2b_8000Hz', '24_07_11_11_20_26_F2b_8000Hz', '24_07_11_11_22_01_F2b_8000Hz', '24_07_11_11_23_45_F2b_8000Hz', '24_07_11_11_25_24_F2b_8000Hz', '24_07_11_11_27_03_F2b_8000Hz', '24_07_11_11_28_29_F2b_8000Hz']

subjD['s091L'] = ['Results/s091/', '24_07_11_13_33_01_F2b_8000Hz', '24_07_11_13_34_57_F2b_8000Hz', '24_07_11_13_36_53_F2b_8000Hz', '24_07_11_13_38_33_F2b_8000Hz', '24_07_11_13_40_43_F2b_8000Hz', '24_07_11_13_42_04_F2b_8000Hz', '24_07_11_13_43_39_F2b_8000Hz', '24_07_11_13_44_59_F2b_8000Hz', '24_07_11_13_46_07_F2b_8000Hz', '24_07_11_13_47_27_F2b_8000Hz']
subjD['s091R'] = ['Results/s091/', '24_07_11_14_01_22_F2b_8000Hz', '24_07_11_14_02_44_F2b_8000Hz', '24_07_11_14_04_30_F2b_8000Hz', '24_07_11_14_05_46_F2b_8000Hz', '24_07_11_14_07_26_F2b_8000Hz', '24_07_11_14_09_27_F2b_8000Hz', '24_07_11_14_11_03_F2b_8000Hz', '24_07_11_14_12_39_F2b_8000Hz', '24_07_11_14_14_28_F2b_8000Hz', '24_07_11_14_15_55_F2b_8000Hz']


subjD['s999L'] = ['Results/s999/', '24_11_05_11_26_07_F2b_8000Hz', '24_11_05_11_27_07_F2b_8000Hz', '24_11_05_11_28_06_F2b_8000Hz', '24_11_05_11_28_50_F2b_8000Hz', '24_11_05_11_29_51_F2b_8000Hz', '24_11_05_11_30_35_F2b_8000Hz', '24_11_05_11_31_19_F2b_8000Hz', '24_11_05_11_32_03_F2b_8000Hz', '24_11_05_11_33_03_F2b_8000Hz']
subjD['s999R'] = ['Results/s999/', '24_11_05_11_48_01_F2b_8000Hz', '24_11_05_11_49_33_F2b_8000Hz', '24_11_05_11_50_48_F2b_8000Hz', '24_11_05_11_52_21_F2b_8000Hz', '24_11_05_11_53_53_F2b_8000Hz', '24_11_05_11_55_41_F2b_8000Hz']

subjD['s999Radd'] = ['Results/s999/','24_11_05_11_57_02_F2b_8000Hz']


subjD['s998L'] = ['Results/s998/', '24_11_08_08_02_11_F2b_8000Hz', '24_11_08_08_02_57_F2b_8000Hz', '24_11_08_08_03_57_F2b_8000Hz', '24_11_08_08_04_59_F2b_8000Hz', '24_11_08_08_05_59_F2b_8000Hz', '24_11_08_08_07_16_F2b_8000Hz', '24_11_08_08_08_49_F2b_8000Hz', '24_11_08_08_10_10_F2b_8000Hz', '24_11_08_08_11_43_F2b_8000Hz']


subjD['s998R'] = ['Results/s998/', '24_11_08_08_31_05_F2b_8000Hz', '24_11_08_08_32_54_F2b_8000Hz', '24_11_08_08_35_00_F2b_8000Hz', '24_11_08_08_37_04_F2b_8000Hz']

# additional measurements for L2 = 60 and L1 = 70, and L2 = 55 and L1 = 70
subjD['s998Radd'] = ['Results/s998/', '24_11_08_08_39_39_F2b_8000Hz', '24_11_08_08_41_05_F2b_8000Hz']

subjD['s997L'] = ['Results/s997/', '24_11_19_11_55_31_F2b_8000Hz', '24_11_19_11_56_51_F2b_8000Hz', '24_11_19_11_58_09_F2b_8000Hz', '24_11_19_11_59_19_F2b_8000Hz', '24_11_19_12_00_34_F2b_8000Hz', '24_11_19_12_01_38_F2b_8000Hz', '24_11_19_12_03_12_F2b_8000Hz', '24_11_19_12_04_29_F2b_8000Hz', '24_11_19_12_06_01_F2b_8000Hz']
subjD['s997R'] = ['Results/s997/', '24_11_19_12_20_05_F2b_8000Hz', '24_11_19_12_21_36_F2b_8000Hz', '24_11_19_12_22_52_F2b_8000Hz', '24_11_19_12_24_09_F2b_8000Hz', '24_11_19_12_25_25_F2b_8000Hz', '24_11_19_12_27_13_F2b_8000Hz', '24_11_19_12_28_29_F2b_8000Hz', '24_11_19_12_29_45_F2b_8000Hz']

subjD['s100L'] = ['Results/s100/', '24_12_04_15_59_59_F2b_8000Hz', '24_12_04_16_01_30_F2b_8000Hz', '24_12_04_16_03_03_F2b_8000Hz', '24_12_04_16_04_34_F2b_8000Hz', '24_12_04_16_05_52_F2b_8000Hz', '24_12_04_16_07_24_F2b_8000Hz', '24_12_04_16_08_56_F2b_8000Hz', '24_12_04_16_10_27_F2b_8000Hz']
subjD['s100R'] = ['Results/s100/', '24_12_04_15_36_48_F2b_8000Hz', '24_12_04_15_38_47_F2b_8000Hz', '24_12_04_15_40_02_F2b_8000Hz', '24_12_04_15_41_17_F2b_8000Hz', '24_12_04_15_42_32_F2b_8000Hz', '24_12_04_15_44_03_F2b_8000Hz', '24_12_04_15_45_18_F2b_8000Hz', '24_12_04_15_46_49_F2b_8000Hz', '24_12_04_15_48_20_F2b_8000Hz']

subjD['s092L'] =  ['Results/s092/', '24_12_05_11_24_38_F2b_8000Hz', '24_12_05_11_25_23_F2b_8000Hz', '24_12_05_11_26_07_F2b_8000Hz', '24_12_05_11_26_51_F2b_8000Hz', '24_12_05_11_27_36_F2b_8000Hz', '24_12_05_11_28_20_F2b_8000Hz', '24_12_05_11_29_20_F2b_8000Hz', '24_12_05_11_30_20_F2b_8000Hz', '24_12_05_11_31_20_F2b_8000Hz', '24_12_05_11_32_23_F2b_8000Hz', '24_12_05_11_33_23_F2b_8000Hz']
subjD['s092R'] =  ['Results/s092/', '24_12_05_11_44_49_F2b_8000Hz', '24_12_05_11_45_48_F2b_8000Hz', '24_12_05_11_46_32_F2b_8000Hz', '24_12_05_11_47_22_F2b_8000Hz', '24_12_05_11_48_23_F2b_8000Hz', '24_12_05_11_49_08_F2b_8000Hz', '24_12_05_11_50_08_F2b_8000Hz', '24_12_05_11_51_09_F2b_8000Hz', '24_12_05_11_52_10_F2b_8000Hz']

subjD['s093L'] = ['Results/s093/', '24_12_06_13_25_51_F2b_8000Hz', '24_12_06_13_26_35_F2b_8000Hz', '24_12_06_13_27_35_F2b_8000Hz', '24_12_06_13_28_35_F2b_8000Hz', '24_12_06_13_29_52_F2b_8000Hz', '24_12_06_13_31_24_F2b_8000Hz', '24_12_06_13_32_41_F2b_8000Hz', '24_12_06_13_34_12_F2b_8000Hz']
subjD['s093R'] = ['Results/s093/', '24_12_06_13_50_25_F2b_8000Hz', '24_12_06_13_51_25_F2b_8000Hz', '24_12_06_13_52_40_F2b_8000Hz', '24_12_06_13_53_41_F2b_8000Hz', '24_12_06_13_54_58_F2b_8000Hz', '24_12_06_13_56_13_F2b_8000Hz', '24_12_06_13_57_13_F2b_8000Hz', '24_12_06_13_58_29_F2b_8000Hz']


subjD['s094L'] = ['Results/s094/', '24_12_06_14_38_26_F2b_8000Hz', '24_12_06_14_40_47_F2b_8000Hz', '24_12_06_14_43_23_F2b_8000Hz', '24_12_06_14_45_45_F2b_8000Hz']
subjD['s094R'] = ['Results/s094/', '24_12_06_15_03_09_F2b_8000Hz', '24_12_06_15_04_41_F2b_8000Hz', '24_12_06_15_06_14_F2b_8000Hz', '24_12_06_15_07_30_F2b_8000Hz', '24_12_06_15_09_01_F2b_8000Hz', '24_12_06_15_10_48_F2b_8000Hz', '24_12_06_15_12_20_F2b_8000Hz']

subjD['s095L'] =  ['Results/s095/', '24_12_09_09_49_40_F2b_8000Hz', '24_12_09_09_50_34_F2b_8000Hz', '24_12_09_09_51_48_F2b_8000Hz', '24_12_09_09_52_42_F2b_8000Hz', '24_12_09_09_53_37_F2b_8000Hz', '24_12_09_09_54_52_F2b_8000Hz', '24_12_09_09_56_26_F2b_8000Hz', '24_12_09_09_57_20_F2b_8000Hz']
subjD['s095R'] = ['Results/s095/', '24_12_09_09_25_05_F2b_8000Hz', '24_12_09_09_26_20_F2b_8000Hz', '24_12_09_09_27_34_F2b_8000Hz', '24_12_09_09_28_49_F2b_8000Hz', '24_12_09_09_30_25_F2b_8000Hz', '24_12_09_09_32_01_F2b_8000Hz', '24_12_09_09_33_57_F2b_8000Hz', '24_12_09_09_35_32_F2b_8000Hz']


subjD['s996L'] = ['Results/s996/', '24_12_13_11_33_35_F2b_8000Hz', '24_12_13_11_35_09_F2b_8000Hz', '24_12_13_11_36_24_F2b_8000Hz', '24_12_13_11_37_38_F2b_8000Hz', '24_12_13_11_39_13_F2b_8000Hz', '24_12_13_11_40_48_F2b_8000Hz', '24_12_13_11_42_43_F2b_8000Hz', '24_12_13_11_44_39_F2b_8000Hz']
subjD['s996R'] = ['Results/s996/', '24_12_13_12_01_48_F2b_8000Hz', '24_12_13_12_03_02_F2b_8000Hz', '24_12_13_12_03_57_F2b_8000Hz', '24_12_13_12_05_11_F2b_8000Hz', '24_12_13_12_06_26_F2b_8000Hz', '24_12_13_12_07_41_F2b_8000Hz', '24_12_13_12_09_16_F2b_8000Hz', '24_12_13_12_10_30_F2b_8000Hz']


subjD['s995Ladd'] = ['Results/s995/','24_12_20_11_39_46_F2b_8000Hz', '24_12_20_11_41_22_F2b_8000Hz', '24_12_20_11_42_59_F2b_8000Hz']
subjD['s995L'] = ['Results/s995/', '24_12_20_11_48_32_F2b_8000Hz', '24_12_20_11_49_47_F2b_8000Hz', '24_12_20_11_51_03_F2b_8000Hz', '24_12_20_11_52_39_F2b_8000Hz']


subjD['s995R'] = ['Results/s995/', '24_12_20_12_05_30_F2b_8000Hz', '24_12_20_12_06_45_F2b_8000Hz', '24_12_20_12_08_00_F2b_8000Hz', '24_12_20_12_09_16_F2b_8000Hz', '24_12_20_12_10_31_F2b_8000Hz', '24_12_20_12_11_46_F2b_8000Hz', '24_12_20_12_12_40_F2b_8000Hz', '24_12_20_12_13_58_F2b_8000Hz', '24_12_20_12_14_53_F2b_8000Hz', '24_12_20_12_16_30_F2b_8000Hz']



subjD['s097L'] = ['Results/s097/', '24_12_10_09_33_01_F2b_8000Hz', '24_12_10_09_34_15_F2b_8000Hz', '24_12_10_09_35_29_F2b_8000Hz', '24_12_10_09_36_44_F2b_8000Hz', '24_12_10_09_37_58_F2b_8000Hz', '24_12_10_09_38_57_F2b_8000Hz', '24_12_10_09_40_12_F2b_8000Hz', '24_12_10_09_41_41_F2b_8000Hz']
subjD['s097R'] = ['Results/s097/', '24_12_10_09_55_12_F2b_8000Hz', '24_12_10_09_56_27_F2b_8000Hz', '24_12_10_09_57_41_F2b_8000Hz', '24_12_10_09_59_11_F2b_8000Hz', '24_12_10_10_00_41_F2b_8000Hz', '24_12_10_10_01_40_F2b_8000Hz', '24_12_10_10_02_55_F2b_8000Hz', '24_12_10_10_04_09_F2b_8000Hz', '24_12_10_10_05_24_F2b_8000Hz']



subjD['s995L2s'] =  ['Results/s995/', '25_02_07_13_29_24_F2b_8000Hz', '25_02_07_13_31_10_F2b_8000Hz', '25_02_07_13_32_26_F2b_8000Hz']
subjD['s995R2s'] = ['Results/s995/', '25_02_07_14_00_07_F2b_8000Hz', '25_02_07_14_01_38_F2b_8000Hz', '25_02_07_14_02_46_F2b_8000Hz', '25_02_07_14_03_45_F2b_8000Hz', '25_02_07_14_05_01_F2b_8000Hz', '25_02_07_14_06_01_F2b_8000Hz', '25_02_07_14_07_17_F2b_8000Hz', '25_02_07_14_08_33_F2b_8000Hz']

 
subjD['s995L2_10'] = ['Results/s995/','25_02_07_13_39_05_F2b_8000Hz', '_25_02_07_13_40_21_F2b_8000Hz','25_02_07_13_41_54_F2b_8000Hz']

subjD['s140L'] = ['Results/s140/', '25_02_13_16_53_28_F2b_8000Hz', '25_02_13_16_54_44_F2b_8000Hz', '25_02_13_16_55_51_F2b_8000Hz', '25_02_13_16_57_07_F2b_8000Hz', '25_02_13_16_58_39_F2b_8000Hz', '25_02_13_16_59_54_F2b_8000Hz', '25_02_13_17_01_26_F2b_8000Hz', '25_02_13_17_02_42_F2b_8000Hz']
subjD['s140R'] = ['Results/s140/', '25_02_13_16_24_56_F2b_8000Hz', '25_02_13_16_26_10_F2b_8000Hz', '25_02_13_16_27_27_F2b_8000Hz', '25_02_13_16_28_27_F2b_8000Hz', '25_02_13_16_29_42_F2b_8000Hz', '25_02_13_16_30_42_F2b_8000Hz', '25_02_13_16_32_14_F2b_8000Hz', '25_02_13_16_33_46_F2b_8000Hz']


subjD['s120L'] = ['Results/s120/', '25_02_19_11_22_51_F2b_8000Hz', '25_02_19_11_23_50_F2b_8000Hz', '25_02_19_11_24_49_F2b_8000Hz', '25_02_19_11_25_48_F2b_8000Hz', '25_02_19_11_26_46_F2b_8000Hz', '25_02_19_11_27_45_F2b_8000Hz', '25_02_19_11_28_44_F2b_8000Hz', '25_02_19_11_29_59_F2b_8000Hz', '25_02_19_11_31_16_F2b_8000Hz']
subjD['s120R'] = ['Results/s120/', '25_02_19_11_41_44_F2b_8000Hz', '25_02_19_11_42_28_F2b_8000Hz', '25_02_19_11_43_43_F2b_8000Hz', '25_02_19_11_44_27_F2b_8000Hz', '25_02_19_11_45_25_F2b_8000Hz', '25_02_19_11_46_58_F2b_8000Hz', '25_02_19_11_48_12_F2b_8000Hz', '25_02_19_11_49_42_F2b_8000Hz', '25_02_19_11_50_57_F2b_8000Hz']

subjN_L = 's120L'
subjN_R = 's120R'




#subjN = 's055L_L2_55'

def mother_wavelet2(Nw,Nt,df,dt):
    vlnky = np.zeros((Nt,Nw))
    tx = (np.arange(Nt)-Nw)*dt
    for k in range(Nw):
        vlnky[:,k] = np.cos(2*np.pi*(k+1)*df*tx)*1/(1+(0.075*(k+1)*2*np.pi*df*tx)**4)
    return vlnky


def wavelet_filterDPOAE(signal, wavelet_basis,fx):
    """
    Perform wavelet filtering on a time-domain signal.

    Parameters:
    - signal: np.array, time-domain signal
    - wavelet_basis: np.array, wavelet basis functions
    
    Returns:
    - filtered_signal: np.array, filtered signal
    """
    # Compute the length of the signal
    N = len(signal)
    
    # Compute the length of the wavelet basis
    M = wavelet_basis.shape[0]
    
    # Compute the number of wavelet basis functions
    NW = wavelet_basis.shape[1]
    
    # Initialize filtered signal
    filtered_signalDPOAE = np.zeros(N)  # prealocate memory for overall filtered DPOAE
    filtered_signalNL = np.zeros(N) # prealocate memory for zero-latency component (SL component)
    coefwtiDPOAE = np.zeros_like(wavelet_basis)
    # Perform wavelet filtering
    
    
    
    for k in range(5,400):
        # Compute the Fourier transform of the wavelet basis function
        wbtf = 2*np.fft.fft(wavelet_basis[:, k])/len(wavelet_basis[:, k])
        
        # Compute the Fourier transform of the signal
        signalf = 2*np.fft.fft(signal)/len(signal)
        
        # Perform convolution in the frequency domain
        coewtf = wbtf * signalf
        
        # Use roex windows to cut chosen latencies
        
        Nrw = 20 # order of the roex win
        Nall = len(signalf)  # number of samples in the entire window
        
        Trwc01 = 0.0005 # the rwc time constant for negative time (contant at 0.5 ms)
        awltSh = 0.015 # time constant for long latency components
        awltZL = 0.005 # time constant for zero-latency components
        bwlt = 0.8  # exponent
        TrwcSh = awltSh*(fx[k]/1000)**(-bwlt)  # rwc time const for positive time (frequency dependent vis Moleti 2012)
        TrwcZL = awltZL*(fx[k]/1000)**(-bwlt)  # rwc time const for positive time (frequency dependent vis Moleti 2012)
        #NsampShift = int(fsamp*TrwcSh)  # number of samples  for shifting the roexwin
        
        rwC = pDP.roexwin(Nall,Nrw,fsamp,Trwc01,TrwcSh) #  not shifted window
        rwCNL = pDP.roexwin(Nall,Nrw,fsamp,Trwc01,TrwcZL) #  not shifted window
        #rwC = roexwin(Nall,Nrw,fsamp,TrwcSh-TrwcZL,TrwcSh-TrwcZL) # 
        #rwC = np.roll(rwC,NsampShift)
        
        
        # Compute the inverse Fourier transform
        coew = np.fft.ifft(coewtf)*len(signal)/2
        coewDPOAE = coew*np.fft.fftshift(rwC)
        coewNL = coew*np.fft.fftshift(rwCNL)
        coefwtiDPOAE[:,k] = np.fft.fftshift(coewDPOAE)
        
        
        # Add the filtered signal to the overall result
        filtered_signalDPOAE += coewDPOAE.real
        filtered_signalNL += coewNL.real
    
    return filtered_signalDPOAE, filtered_signalNL, coefwtiDPOAE




def estimateFreqResp(oaeDS,f2f1,f2b,f2e,octpersec,GainMic):



    #f2f1 = 1.2
    #f2b = 8000
    #f2e = 500
    
    nfloorDS  = np.zeros_like(oaeDS)
    nfilt = 2**12
    #GainMic = 40
    
    #(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tau02,tshift):
    
    if f2b<f2e:
        T = np.log2(f2e/f2b)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        
    else:
        T = np.log2(f2b/f2e)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        
    #L1sw =  T/np.log(f2e/f2b) 
    #pDP.fceDPOAEinwinSS(oaeDS,Nsamp,f2b/rF,L1sw,rF,fsamp,0.01,0.02,0)
    #oaeDS = np.concatenate((oaeDS[:int(T*fsamp)],np.zeros((Nsamp4-int(T*fsamp),))))
    #hmfftlen = 2**14
    
    #DPgram, DPgramNL, DPgramCR, NF, NFnl, fx = pDP.calcDPgramFAV_HS(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    #DPgram, DPgramNL, DPgramCR, NF, NFnl, fx = pDP.calcDPgram(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    #hm, h, hmN, hN = pDP.calcDPgramTD(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    fxF1, HF1calcTFatF = pDP.calcTFatF(oaeDS,f2b/f2f1,f2e/f2f1,T,fsamp)
    fxF2, HF2calcTFatF = pDP.calcTFatF(oaeDS,f2b,f2e,T,fsamp)
        
    #f2max = 8000
    
    
    return fxF1, HF1calcTFatF, fxF2, HF2calcTFatF


def estimateDRgram(oaeDS,f2f1,f2b,f2e,octpersec,GainMic):



    #f2f1 = 1.2
    #f2b = 8000
    #f2e = 500
    
    nfloorDS  = np.zeros_like(oaeDS)
    nfilt = 2**12
    #GainMic = 40
    
    #(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tau02,tshift):
    
    if f2b<f2e:
        T = np.log2(f2e/f2b)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        
    else:
        T = np.log2(f2b/f2e)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        
    #L1sw =  T/np.log(f2e/f2b) 
    #pDP.fceDPOAEinwinSS(oaeDS,Nsamp,f2b/rF,L1sw,rF,fsamp,0.01,0.02,0)
    #oaeDS = np.concatenate((oaeDS[:int(T*fsamp)],np.zeros((Nsamp4-int(T*fsamp),))))
    #hmfftlen = 2**14
    
    #DPgram, DPgramNL, DPgramCR, NF, NFnl, fx = pDP.calcDPgramFAV_HS(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    #DPgram, DPgramNL, DPgramCR, NF, NFnl, fx = pDP.calcDPgram(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    hm, h, hmN, hN = pDP.calcDPgramTD(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,GainMic)
    
    f2max = 8000
    
    hm_50lin = hm
        
    fs = fsamp
    #t = np.arange(0,0.1,1/fs)
    t = np.arange(0,len(hm_50lin)/fs,1/fs)
    # Finding signal by adding three different signals
    #signal = np.cos(2 * np.pi * 1000 * t)# + np.real(np.exp(-7 * (t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
    
    
    dt = 1/fs
    Nt = len(t)
    fx = np.arange(Nt)*fs/Nt   # frequency axis
    df = fx[1]-fx[0]

    print(df)
    print(Nt)
    Nw = Nt//2  # number of wavelet filters
    Nwmax = int(f2max//df)  # maximal wavelet which is usefull for the given frequency range
    
    
    vlnky = mother_wavelet2(Nw,Nt,df,dt)


    tDPOAEmwf, tNLmwf, cwLIN = wavelet_filterDPOAE(hm_50lin, vlnky,fx)
    tDPOAENmwf,tNLNmwf,cwNLIN = wavelet_filterDPOAE(hmN, vlnky,fx)  # noise floor
    
    #SLINmwf = 2*np.fft.rfft(np.concatenate((tLINmwf,np.zeros(int(2**15)))))/Nw  # calculate spectrum
    
    Nfft = 2**15
    fxx = np.arange(int(Nfft))*fsamp/(int(Nfft))
    
    SDPOAEmwf = np.fft.rfft(np.fft.fftshift(tDPOAEmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum
    SDPOAENmwf = np.fft.rfft(np.fft.fftshift(tDPOAENmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum
    SNLmwf = np.fft.rfft(np.fft.fftshift(tNLmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum
    SNLNmwf = np.fft.rfft(np.fft.fftshift(tNLNmwf),int(Nfft))*np.exp(1j*2*np.pi*fxx[:int(len(fxx)//2)+1]*(nfilt/2)*1/fsamp)  # calculate spectrum
    
    DPgr = {'DPgr':SDPOAEmwf,'DPgrN':SDPOAENmwf,'NLgr':SNLmwf,'NLgrN':SNLNmwf,'fxx':fxx}
    return DPgr



def getDPgram(path,DatePat):

    dir_list = os.listdir(path)
    
    n = 0
    DPgramsDict = {}
    
    for k in range(len(dir_list)):
        
        if  DatePat in dir_list[k]:
            data = loadmat(path+ dir_list[k])
            #lat = 16448
            rateOct = data['r'][0][0]
            lat = data['lat_SC'][0][0]
            print([path+ dir_list[k]])
            print(f"SC latency: {lat}")
            octpersec = data['r'][0][0]
            f2b = data['f2b'][0][0]
            f2e = data['f2e'][0][0]
            f2f1 = data['f2f1'][0][0]
            L1dB = data['L1'][0][0]
            L2dB = data['L2'][0][0]
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
                
                
                recMat1 = np.c_[recMat1, recSig1[lat:,0]]  # add to make a matrix with columns for every run
                recMat2 = np.c_[recMat2, recSig2[lat:,0]]  # add to make a matrix with columns for every run
                recMat3 = np.c_[recMat3, recSig3[lat:,0]]  # add to make a matrix with columns for every run
                recMat4 = np.c_[recMat4, recSig4[lat:,0]]  # add to make a matrix with columns for every run
           
    
                #oaeDS = (np.nanmean(recMat1,1)+np.nanmean(recMat2,1)+np.nanmean(recMat3,1)+np.nanmean(recMat4,1))/4  # exclude samples set to NaN (noisy samples)
                
                #DPgrR = estimateDRgram(oaeDS,f2f1,f2b,f2e,octpersec,GainMic)
                
                #DPgramsDict[str(n)] = DPgrR

            n += 1
            
    recMean1 = np.median(recMat1,1)  # median across rows
    recMean1 = np.reshape(recMean1,(-1,1))
    recMean2 = np.median(recMat2,1)  # median across rows
    recMean2 = np.reshape(recMean2,(-1,1))
    recMean3 = np.median(recMat3,1)  # median across rows
    recMean3 = np.reshape(recMean3,(-1,1))
    recMean4 = np.median(recMat4,1)  # median across rows
    recMean4 = np.reshape(recMean4,(-1,1))
    
    # 2. calculate noise matrix
    noiseM1 = recMat1 - recMean1
    noiseM2 = recMat2 - recMean2
    noiseM3 = recMat3 - recMean3
    noiseM4 = recMat4 - recMean4
    Nsamp = len(recMean1) # number of samples


    sigma1 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM1**2,1)))
    sigma2 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM2**2,1)))
    sigma3 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM3**2,1)))
    sigma4 = np.sqrt(1/(Nsamp*n)*np.sum(np.sum(noiseM4**2,1)))
    Nt = 12
    Theta1 = Nt*sigma1 # estimation of the threshold for sample removal
    Theta2 = Nt*sigma2 # estimation of the threshold for sample removal
    Theta3 = Nt*sigma3 # estimation of the threshold for sample removal
    Theta4 = Nt*sigma4 # estimation of the threshold for sample removal
    #np.shape(np.mean(recMat1[np.sum(np.abs(noiseM1)>Theta1],1))
    Bl01chosen = (sum(np.abs(noiseM1)>Theta1))==0
    Bl02chosen = (sum(np.abs(noiseM2)>Theta2))==0
    Bl03chosen = (sum(np.abs(noiseM3)>Theta3))==0
    Bl04chosen = (sum(np.abs(noiseM4)>Theta4))==0
    
    N01chosen = sum(Bl01chosen)
    N02chosen = sum(Bl02chosen)
    N03chosen = sum(Bl03chosen)
    N04chosen = sum(Bl04chosen)
    
    
    recMat1Mean = np.mean(recMat1[:,Bl01chosen],1)
    recMat2Mean = np.mean(recMat2[:,Bl02chosen],1)
    recMat3Mean = np.mean(recMat3[:,Bl03chosen],1)
    recMat4Mean = np.mean(recMat4[:,Bl04chosen],1)
    
    
    oaeDS = (recMat1Mean+recMat2Mean+recMat3Mean+recMat4Mean)/4  # exclude samples set to NaN (noisy samples)
    DPgrR = estimateDRgram(oaeDS,f2f1,f2b,f2e,octpersec,GainMic)
    
    '''
    recMat1[np.abs(noiseM1)>Theta1] = np.nan
    recMat1[np.abs(noiseM2)>Theta2] = np.nan
    recMat1[np.abs(noiseM3)>Theta3] = np.nan
    recMat1[np.abs(noiseM4)>Theta4] = np.nan
    
    recMat2[np.abs(noiseM1)>Theta1] = np.nan
    recMat2[np.abs(noiseM2)>Theta2] = np.nan
    recMat2[np.abs(noiseM3)>Theta3] = np.nan
    recMat2[np.abs(noiseM4)>Theta4] = np.nan
    
    recMat3[np.abs(noiseM1)>Theta1] = np.nan
    recMat3[np.abs(noiseM2)>Theta2] = np.nan
    recMat3[np.abs(noiseM3)>Theta3] = np.nan
    recMat3[np.abs(noiseM4)>Theta4] = np.nan
    
    recMat4[np.abs(noiseM1)>Theta1] = np.nan
    recMat4[np.abs(noiseM2)>Theta2] = np.nan
    recMat4[np.abs(noiseM3)>Theta3] = np.nan
    recMat4[np.abs(noiseM4)>Theta4] = np.nan
    
    noiseM1[np.abs(noiseM1)>Theta1] = np.nan
    noiseM1[np.abs(noiseM2)>Theta2] = np.nan
    noiseM1[np.abs(noiseM3)>Theta3] = np.nan
    noiseM1[np.abs(noiseM4)>Theta4] = np.nan
    
    noiseM2[np.abs(noiseM1)>Theta1] = np.nan
    noiseM2[np.abs(noiseM2)>Theta2] = np.nan
    noiseM2[np.abs(noiseM3)>Theta3] = np.nan
    noiseM2[np.abs(noiseM4)>Theta4] = np.nan
    
    noiseM3[np.abs(noiseM1)>Theta1] = np.nan
    noiseM3[np.abs(noiseM2)>Theta2] = np.nan
    noiseM3[np.abs(noiseM3)>Theta3] = np.nan
    noiseM3[np.abs(noiseM4)>Theta4] = np.nan
    
    noiseM4[np.abs(noiseM1)>Theta1] = np.nan
    noiseM4[np.abs(noiseM2)>Theta2] = np.nan
    noiseM4[np.abs(noiseM3)>Theta3] = np.nan
    noiseM4[np.abs(noiseM4)>Theta4] = np.nan
    '''
        
    return DPgrR, rateOct, L2dB, f2f1, [N01chosen, N02chosen, N03chosen, N04chosen]



DPgr_L = {}
L2list_L = []
f2f1list_L = []
DPgr_R = {}
L2list_R = []
f2f1list_R = []

for i in range(1,len(subjD[subjN_L])):

    DPgrD_L, rateOct_L, L2_L, f2f1_L, Nchosen_L = getDPgram(subjD[subjN_L][0], subjD[subjN_L][i])
    DPgr_L[str(L2_L)] = DPgrD_L
    DPgr_L[str(L2_L)+'ch'] = Nchosen_L
    L2list_L.append(L2_L)  # list of L2 values
    f2f1list_L.append(f2f1_L)
    #DPgrD10 = getDPgram(path, deL_r10)
for i in range(1,len(subjD[subjN_R])):
    DPgrD_R, rateOct_R, L2_R, f2f1_R, Nchosen_R = getDPgram(subjD[subjN_R][0], subjD[subjN_R][i])
    DPgr_R[str(L2_R)] = DPgrD_R
    DPgr_R[str(L2_R)+'ch'] = Nchosen_R
    L2list_R.append(L2_R)  # list of L2 values
    f2f1list_R.append(f2f1_R)


def InfoOnData(data_dict):
    

    import re
  
    
    # Regular expression to match keys that start with a number and end with 'ch'
    pattern = re.compile(r'^(\d+)ch$')
    
    # List to hold the table rows
    table = []
    
    # Iterate through the dictionary keys and values
    for key, value_list in data_dict.items():
        match = pattern.match(key)
        if match:
            # Extract the integer from the key
            integer_part = int(match.group(1))
            # Create a row starting with the integer, followed by the values in the list
            row = [integer_part] + value_list
            # Append the row to the table
            table.append(row)
    
    
    # Define the headers
    header_main = "L2 (dB FPL)"
    header_phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
    header_full = [header_main] + header_phases
    
    # Print the headers
    print(f"{header_full[0]:<15} {header_full[1]:<10} {header_full[2]:<10} {header_full[3]:<10} {header_full[4]:<10}")
    
    # Print the table rows
    for row in table:
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10}")

InfoOnData(DPgr_L)



#%%
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
pREF = np.sqrt(2) * 2e-5

Nopak = 4  # number of presentations

fxx_L = DPgr_L['55']['fxx']
fxx_R = DPgr_R['55']['fxx']
f2xx_L = f2f1list_L[-1] * fxx_L[: int(len(fxx_L) // 2) + 1] / (2 - f2f1)
f2xx_R = f2f1list_R[-1] * fxx_R[: int(len(fxx_R) // 2) + 1] / (2 - f2f1)
f2xx_L /= 1000
f2xx_R /= 1000

cList = ['C1', 'C3', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']

L2mensi = L2list_L if len(L2list_L)<=len(L2list_R) else L2list_R

for i in range(len(L2mensi)):
    # Left ear data (included in legend)
    ax1.plot(
        f2xx_L,
        20 * np.log10(np.abs(DPgr_L[str(L2mensi[i])]['NLgr']) / pREF),
        color=cList[i],
        label=str(L2mensi[i])
    )

    # Left ear noise (not in legend)
    ax1.plot(
        f2xx_L,
        20 * np.log10(np.abs(DPgr_L[str(L2mensi[i])]['NLgrN']) / pREF),
        ':',
        color=cList[i],
        label='_nolegend_'
    )

    # Right ear data (not in legend)
    ax1.plot(
        f2xx_R,
        20 * np.log10(np.abs(DPgr_R[str(L2mensi[i])]['NLgr']) / pREF),
        '--',
        color=cList[i],
        label='_nolegend_'
    )

    # Right ear noise (not in legend)
    ax1.plot(
        f2xx_R,
        20 * np.log10(np.abs(DPgr_R[str(L2mensi[i])]['NLgrN']) / pREF),
        ':',
        color=cList[i],
        label='_nolegend_'
    )

ax1.set_xlim([0.5, 8])
ax1.set_ylim([-30, 20])
ax1.set_ylabel('Amplitude (dB SPL)', fontsize=14)

# Set x-tick lines inward & increase font size
ax1.tick_params(axis='x', direction='in', labelsize=12)
ax1.tick_params(axis='y', direction='in', labelsize=12)

cycle = 2 * np.pi
F2start = 700 / 1000
idx1 = np.where(f2xx_L >= F2start)[0][0]  # Frequency index for unwrapping

for i in range(len(L2mensi)):
    # Left ear phase (included in legend)
    ax2.plot(
        f2xx_L[idx1:],
        np.unwrap(np.angle(DPgr_L[str(L2mensi[i])]['NLgr'][idx1:])) / cycle,
        color=cList[i],
        label=str(L2mensi[i])
    )

    # Right ear phase (not in legend)
    ax2.plot(
        f2xx_R[idx1:],
        np.unwrap(np.angle(DPgr_R[str(L2mensi[i])]['NLgr'][idx1:])) / cycle,
        '--',
        color=cList[i],
        label='_nolegend_'
    )

ax2.set_xlim([0.5, 8])
ax2.set_ylim([-5, 1])
ax2.set_xlabel(r'Frequency $f_{\rm 2}$ (kHz)', fontsize=14)
ax2.set_ylabel('Phase (cycles)', fontsize=14)

# Set x-tick lines inward & increase font size
ax2.tick_params(axis='x', direction='in', labelsize=12)
ax2.tick_params(axis='y', direction='in',labelsize=12)

# Place only one legend outside the plot (to the right) with italic L and subscript 2
ax1.legend(
    title=r"$\mathit{L}_2$ dB FPL",  # L2 in italic with subscript
    title_fontsize=14,
    fontsize=12,
    loc='center left',
    bbox_to_anchor=(1, 0.5)
)

import matplotlib.lines as mlines

# Custom legend handles
left_ear_line = mlines.Line2D([], [], color='black', linestyle='-', label='Left ear')
right_ear_line = mlines.Line2D([], [], color='black', linestyle='--', label='Right ear')

# Create L2 labels without duplicating line styles
l2_handles = [mlines.Line2D([], [], color=cList[i], linestyle='-', label=str(L2mensi[i])) for i in range(len(L2mensi))]

# First legend: Ear distinction
legend1 = ax1.legend(handles=[left_ear_line, right_ear_line], loc='upper right', title="")

# Second legend: L2 values
legend2 = ax1.legend(handles=l2_handles, title=r"$\mathit{L}_2$ dB FPL", title_fontsize=14, fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))

# Add the first legend back to the plot
ax1.add_artist(legend1)

ax2.text(0.05, 0.05, subjN_L[:-1], transform=ax2.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='left')

plt.show()

plt.savefig(f'Figures/DPgrams/DPgrBothEars{subjN_L}.png', format='png', dpi=300)


#%% fitting

import numpy as np
import matplotlib.pyplot as plt

# Define center frequencies
CF = [2000, 3000, 4000, 7000]
CFidx = np.zeros_like(CF)

DPioNL_L = []
GAdpNL_L = []
NOxNL_L = []
NOxNL_R = []
DPioNL_R = []
GAdpNL_R = []

# Find indices of CF in f2xx_R
for i in range(len(CF)):
    CFidx[i] = np.where(f2xx_R >= 0.001*CF[i])[0][0]

    IOx_L = []
    GAx_L = []
    NOx_L = []
    for j in range(len(L2list_L)):
        IOx_L.append(20 * np.log10(np.abs(DPgr_L[str(L2list_L[j])]['NLgr'][CFidx[i]]) / pREF))
        GAx_L.append(IOx_L[j] - L2list_L[j])
        NOx_L.append(20 * np.log10(np.abs(DPgr_L[str(L2list_L[j])]['NLgrN'][CFidx[i]])/pREF))
    DPioNL_L.append(IOx_L)
    GAdpNL_L.append(GAx_L)
    NOxNL_L.append(NOx_L)

    IOx_R = []
    GAx_R = []
    NOx_R = []
    for j in range(len(L2list_R)):
        IOx_R.append(20 * np.log10(np.abs(DPgr_R[str(L2list_R[j])]['NLgr'][CFidx[i]]) / pREF))
        GAx_R.append(IOx_R[j] - L2list_R[j])
        NOx_R.append(20 * np.log10(np.abs(DPgr_R[str(L2list_R[j])]['NLgrN'][CFidx[i]])/pREF))

    DPioNL_R.append(IOx_R)
    GAdpNL_R.append(GAx_R)
    NOxNL_R.append(NOx_R)

# Create figure
fig, ax = plt.subplots(figsize=(7, 5))

# Store colors for consistency
colors = []

# Plot left ear data (solid lines)
for i in range(len(CF)):
    line, = ax.plot(
        L2list_L, DPioNL_L[i], label=r'${\it f}_2$ = ' + str(CF[i] / 1000) + ' kHz'
    )
    colors.append(line.get_color())  # Store color for right ear
    ax.plot(L2list_L, NOxNL_L[i], color=colors[i],
                               linestyle=':', linewidth=0.5, label="_nolegend_")



# Plot right ear data (dashed lines with same colors)
for i in range(len(CF)):
    ax.plot(
        L2list_R, DPioNL_R[i], linestyle='--', color=colors[i], label='_nolegend_'
    )
    ax.plot(L2list_R, NOxNL_R[i], color=colors[i],
                               linestyle=':', linewidth=0.5, label="_nolegend_")


# Convert L2list to NumPy array for element-wise operations
L2array = np.array(L2list_L)

# Reference line (gray dashed line with slope 1, shifted 35 dB down)
ax.plot(L2array, L2array - 35, color='gray', linestyle='--', linewidth=1)

# Set x and y limits
ax.set_xlim([20, 70])
ax.set_ylim([-20, 20])

# Increase font sizes
label_fontsize = 16
legend_fontsize = 12
text_fontsize = 13

# Labels and legend
ax.set_ylabel('Amplitude (dB SPL)', fontsize=label_fontsize)
ax.set_xlabel(r'$L_2$ (dB FPL)', fontsize=label_fontsize)
ax.legend(fontsize=legend_fontsize)

# Add subject name and ear information to the bottom-right corner
ear = "left ear" if subjN_L[-1] == 'L' else "right ear"
subject_name = subjN_L[:-1]
text_to_display = f'{subject_name}'

ax.text(
    0.1, 0.05, text_to_display, transform=ax.transAxes,
    fontsize=text_fontsize, verticalalignment='top', horizontalalignment='right'
)

# Increase tick label font sizes and set both x and y tick marks inward
ax.tick_params(axis='both', which='major', labelsize=label_fontsize, direction='in')

# Adjust layout and show plot
plt.tight_layout()
plt.show()


import matplotlib.lines as mlines

# Custom legend handles for left and right ear
left_ear_line = mlines.Line2D([], [], color='black', linestyle='-', label='Left ear')
right_ear_line = mlines.Line2D([], [], color='black', linestyle='--', label='Right ear')

# Create L2 labels (without duplicating line styles)
l2_handles = [
    mlines.Line2D([], [], color=colors[i], linestyle='-', label=r'${\it f}_2$ = ' + str(int(CF[i] / 1000)))
    for i in range(len(CF))
]

# First legend: Ear distinction (top right)
legend1 = ax.legend(handles=[left_ear_line, right_ear_line], loc='upper right', title="")

# Second legend: Frequency labels (side)
legend2 = ax.legend(handles=l2_handles, title=r"$\mathit{f}_2$ (kHz)", title_fontsize=14, fontsize=12, loc='upper left')

# Add the first legend back to the plot
ax.add_artist(legend1)

# Subject name & ear information (bottom left)
text_to_display = ''
ax.text(
    0.02, 0.05, text_to_display, transform=ax.transAxes,
    fontsize=legend_fontsize, verticalalignment='bottom', horizontalalignment='left'
)



# Save the figure
plt.savefig(f'Figures/DPgrams/ioBothEars{subjN_L}.png', format='png', dpi=300)

#%%

# Convert L2list to a NumPy array for element-wise operations
L2array_L = np.array(L2list_L)
L2array_R = np.array(L2list_R)



# Define your desired range
L2_min, L2_max = 30, 60

# Initialize lists to store results for each x
DPio_selected_L, NFio_selected_L, cumsum_results_L, nDPio_results_L = [], [], [], []
DPio_selected_R, NFio_selected_R, cumsum_results_R, nDPio_results_R = [], [], [], []

# Loop over all x in DPioNL
for x in range(len(DPioNL_L)):
    DPio = np.array(DPioNL_L[x])
    NFio = np.array(NOxNL_L[x])

    # Select elements where L2array is within the desired range
    mask = (L2array_L >= L2_min) & (L2array_L <= L2_max)
    DPio, NFio = DPio[mask], NFio[mask]

    # Normalize DPio values
    nDPio = (10**(DPio / 20)) / np.max(10**(DPio / 20))

    # Compute the valid mask for cumulative sum condition
    valid_mask1 = (DPio - NFio) > 6
    valid_mask2 = DPio > -20
    valid_mask = valid_mask1*valid_mask2


    # Compute cumulative sum only for valid values
    cumsum_result = np.cumsum(np.where(valid_mask, nDPio, 0))

    # Store results
    DPio_selected_L.append(DPio)
    NFio_selected_L.append(NFio)
    nDPio_results_L.append(nDPio)
    cumsum_results_L.append(cumsum_result)

# Loop over all x in DPioNL
for x in range(len(DPioNL_R)):
    DPio = np.array(DPioNL_R[x])
    NFio = np.array(NOxNL_R[x])

    # Select elements where L2array is within the desired range
    mask = (L2array_R >= L2_min) & (L2array_R <= L2_max)
    DPio, NFio = DPio[mask], NFio[mask]

    # Normalize DPio values
    nDPio = (10**(DPio / 20)) / np.max(10**(DPio / 20))

    # Compute the valid mask for cumulative sum condition
    valid_mask1 = (DPio - NFio) > 6
    valid_mask2 = DPio > -20
    valid_mask = valid_mask1*valid_mask2

    # Compute cumulative sum only for valid values
    cumsum_result = np.cumsum(np.where(valid_mask, nDPio, 0))

    # Store results
    DPio_selected_R.append(DPio)
    NFio_selected_R.append(NFio)
    nDPio_results_R.append(nDPio)
    cumsum_results_R.append(cumsum_result)



fig, ax = plt.subplots(figsize=(7, 5))

colors = []  # Store colors for consistency

mask_L = (L2array_L >= L2_min) & (L2array_L <= L2_max)

mask_R = (L2array_L >= L2_min) & (L2array_L <= L2_max)


# Plot left ear data (solid lines)
for i in range(len(cumsum_results_L)):
    line, = ax.plot(L2list_L[1:len(cumsum_results_L[i])+1], cumsum_results_L[i] * 5, label=f'${{\it f}}_2$ = {CF[i] / 1000} kHz')
    colors.append(line.get_color())  # Store color for right ear

# Plot right ear data (dashed lines, same colors)
for i in range(len(cumsum_results_R)):
    ax.plot(L2list_R[1:len(cumsum_results_R[i])+1], cumsum_results_R[i] * 5, '--', color=colors[i], label='_nolegend_')

# Reverse X-axis (high to low L2 values)
ax.invert_xaxis()

# Labels
ax.set_xlabel(r'$L_2$ (dB FPL)', fontsize=14)
ax.set_ylabel('$\sum_{L_2=60}^{30} DPOAE(L_2) \Delta L$', fontsize=14)

# Custom legend handles
left_ear_line = mlines.Line2D([], [], color='black', linestyle='-', label='Left ear')
right_ear_line = mlines.Line2D([], [], color='black', linestyle='--', label='Right ear')

# Create separate legends
legend1 = ax.legend(handles=[left_ear_line, right_ear_line], loc='upper center', title="", fontsize=12)
l2_handles = [
    mlines.Line2D([], [], color=colors[i], linestyle='-', label=f'${{\it f}}_2$ = {int(CF[i] / 1000)}') for i in range(len(CF))
]
legend2 = ax.legend(handles=l2_handles, title=r"$\mathit{f}_2$ (kHz)", title_fontsize=12, fontsize=12, loc='upper left')
ax.add_artist(legend1)  # Add ear legend separately
ax.set_ylim([5,30])
# Save & show

# Increase tick label font sizes and set both x and y tick marks inward
ax.tick_params(axis='both', which='major', labelsize=label_fontsize, direction='in')


plt.tight_layout()
plt.savefig(f'Figures/CumsumBothEars{subjN_L}.png', format='png', dpi=300)
plt.show()
