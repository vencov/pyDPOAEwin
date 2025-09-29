
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
    
plt.close('all')


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

subjD['s096L'] = ['Results/s096/', '24_12_09_13_24_12_F2b_8000Hz', '24_12_09_13_25_28_F2b_8000Hz', '24_12_09_13_26_58_F2b_8000Hz', '24_12_09_13_28_28_F2b_8000Hz', '24_12_09_13_29_28_F2b_8000Hz', '24_12_09_13_30_43_F2b_8000Hz', '24_12_09_13_31_58_F2b_8000Hz']
subjD['s096R'] = ['Results/s096/', '24_12_09_13_46_18_F2b_8000Hz', '24_12_09_13_47_49_F2b_8000Hz']

subjD['s097L'] = ['Results/s097/', '24_12_10_09_33_01_F2b_8000Hz', '24_12_10_09_34_15_F2b_8000Hz', '24_12_10_09_35_29_F2b_8000Hz', '24_12_10_09_36_44_F2b_8000Hz', '24_12_10_09_37_58_F2b_8000Hz', '24_12_10_09_38_57_F2b_8000Hz', '24_12_10_09_40_12_F2b_8000Hz', '24_12_10_09_41_41_F2b_8000Hz']
subjD['s097R'] = ['Results/s097/', '24_12_10_09_55_12_F2b_8000Hz', '24_12_10_09_56_27_F2b_8000Hz', '24_12_10_09_57_41_F2b_8000Hz', '24_12_10_09_59_11_F2b_8000Hz', '24_12_10_10_00_41_F2b_8000Hz', '24_12_10_10_01_40_F2b_8000Hz', '24_12_10_10_02_55_F2b_8000Hz', '24_12_10_10_04_09_F2b_8000Hz', '24_12_10_10_05_24_F2b_8000Hz']

subjD['s098L'] = ['Results/s098/', '24_12_11_13_25_00_F2b_8000Hz', '24_12_11_13_25_43_F2b_8000Hz', '24_12_11_13_26_27_F2b_8000Hz', '24_12_11_13_27_28_F2b_8000Hz', '24_12_11_13_28_45_F2b_8000Hz', '24_12_11_13_30_31_F2b_8000Hz', '24_12_11_13_32_02_F2b_8000Hz']
subjD['s098R'] = ['Results/s098/', '24_12_11_13_44_03_F2b_8000Hz', '24_12_11_13_44_48_F2b_8000Hz', '24_12_11_13_45_34_F2b_8000Hz', '24_12_11_13_46_50_F2b_8000Hz', '24_12_11_13_48_06_F2b_8000Hz', '24_12_11_13_49_37_F2b_8000Hz', '24_12_11_13_51_08_F2b_8000Hz']

subjD['s099L'] = ['Results/s099/', '24_12_12_09_09_37_F2b_8000Hz', '24_12_12_09_10_37_F2b_8000Hz', '24_12_12_09_11_36_F2b_8000Hz', '24_12_12_09_12_36_F2b_8000Hz', '24_12_12_09_14_12_F2b_8000Hz', '24_12_12_09_15_27_F2b_8000Hz', '24_12_12_09_16_42_F2b_8000Hz', '24_12_12_09_17_42_F2b_8000Hz', '24_12_12_09_18_41_F2b_8000Hz']
subjD['s099R'] = ['Results/s099/', '24_12_12_09_31_39_F2b_8000Hz', '24_12_12_09_32_39_F2b_8000Hz', '24_12_12_09_33_54_F2b_8000Hz', '24_12_12_09_35_09_F2b_8000Hz', '24_12_12_09_36_08_F2b_8000Hz', '24_12_12_09_36_53_F2b_8000Hz', '24_12_12_09_38_08_F2b_8000Hz', '24_12_12_09_39_24_F2b_8000Hz']

subjD['s001L'] = ['Results/s001/GACR/', '24_12_18_10_11_56_F2b_8000Hz', '24_12_18_10_12_55_F2b_8000Hz', '24_12_18_10_13_54_F2b_8000Hz', '24_12_18_10_14_53_F2b_8000Hz', '24_12_18_10_16_08_F2b_8000Hz', '24_12_18_10_17_23_F2b_8000Hz', '24_12_18_10_18_56_F2b_8000Hz', '24_12_18_10_20_45_F2b_8000Hz']
subjD['s001R'] = ['Results/s001/GACR/', '24_12_18_10_36_45_F2b_8000Hz', '24_12_18_10_37_45_F2b_8000Hz', '24_12_18_10_38_45_F2b_8000Hz', '24_12_18_10_39_28_F2b_8000Hz', '24_12_18_10_40_43_F2b_8000Hz', '24_12_18_10_41_58_F2b_8000Hz', '24_12_18_10_42_42_F2b_8000Hz', '24_12_18_10_43_41_F2b_8000Hz', '24_12_18_10_44_56_F2b_8000Hz']

subjD['s101L'] = ['Results/s101/', '24_12_18_13_38_13_F2b_8000Hz', '24_12_18_13_39_17_F2b_8000Hz', '24_12_18_13_40_16_F2b_8000Hz', '24_12_18_13_41_33_F2b_8000Hz', '24_12_18_13_43_03_F2b_8000Hz', '24_12_18_13_44_49_F2b_8000Hz', '24_12_18_13_46_03_F2b_8000Hz', '24_12_18_13_47_49_F2b_8000Hz']
subjD['s101R'] = ['Results/s101/', '24_12_18_13_15_39_F2b_8000Hz', '24_12_18_13_17_11_F2b_8000Hz', '24_12_18_13_18_32_F2b_8000Hz', '24_12_18_13_19_31_F2b_8000Hz', '24_12_18_13_20_46_F2b_8000Hz', '24_12_18_13_21_44_F2b_8000Hz', '24_12_18_13_22_44_F2b_8000Hz', '24_12_18_13_23_43_F2b_8000Hz', '24_12_18_13_24_57_F2b_8000Hz']

subjD['s119L'] = ['Results/s119/', '25_01_22_11_32_21_F2b_8000Hz', '25_01_22_11_33_35_F2b_8000Hz', '25_01_22_11_34_50_F2b_8000Hz', '25_01_22_11_36_04_F2b_8000Hz', '25_01_22_11_37_18_F2b_8000Hz', '25_01_22_11_38_48_F2b_8000Hz', '25_01_22_11_40_18_F2b_8000Hz']
subjD['s119R'] = ['Results/s119/', '25_01_22_11_54_46_F2b_8000Hz', '25_01_22_11_55_45_F2b_8000Hz', '25_01_22_11_56_59_F2b_8000Hz', '25_01_22_11_58_13_F2b_8000Hz', '25_01_22_11_59_28_F2b_8000Hz', '25_01_22_12_00_42_F2b_8000Hz', '25_01_22_12_01_56_F2b_8000Hz']

subjD['s124L'] = ['Results/s124/', '25_01_22_14_11_05_F2b_8000Hz', '25_01_22_14_11_49_F2b_8000Hz', '25_01_22_14_12_49_F2b_8000Hz', '25_01_22_14_14_04_F2b_8000Hz', '25_01_22_14_15_18_F2b_8000Hz', '25_01_22_14_16_33_F2b_8000Hz', '25_01_22_14_17_47_F2b_8000Hz', '25_01_22_14_18_46_F2b_8000Hz']
subjD['s124R'] = ['Results/s124/', '25_01_22_13_49_12_F2b_8000Hz', '25_01_22_13_50_11_F2b_8000Hz', '25_01_22_13_51_10_F2b_8000Hz', '25_01_22_13_52_09_F2b_8000Hz', '25_01_22_13_53_08_F2b_8000Hz', '25_01_22_13_54_07_F2b_8000Hz', '25_01_22_13_55_06_F2b_8000Hz', '25_01_22_13_56_21_F2b_8000Hz', '25_01_22_13_57_35_F2b_8000Hz']

subjD['s126L'] = ['Results/s126/', '25_02_05_10_00_44_F2b_8000Hz', '25_02_05_10_01_47_F2b_8000Hz', '25_02_05_10_02_55_F2b_8000Hz', '25_02_05_10_04_11_F2b_8000Hz', '25_02_05_10_05_32_F2b_8000Hz', '25_02_05_10_06_48_F2b_8000Hz', '25_02_05_10_07_47_F2b_8000Hz', '25_02_05_10_09_03_F2b_8000Hz']
subjD['s126R'] = ['Results/s126/', '25_02_05_10_24_02_F2b_8000Hz', '25_02_05_10_25_17_F2b_8000Hz', '25_02_05_10_26_17_F2b_8000Hz', '25_02_05_10_27_32_F2b_8000Hz', '25_02_05_10_28_32_F2b_8000Hz', '25_02_05_10_29_32_F2b_8000Hz', '25_02_05_10_30_31_F2b_8000Hz', '25_02_05_10_31_31_F2b_8000Hz']

subjD['s127L'] = ['Results/s127/', '25_02_05_13_51_54_F2b_8000Hz', '25_02_05_13_53_25_F2b_8000Hz', '25_02_05_13_54_40_F2b_8000Hz', '25_02_05_13_55_55_F2b_8000Hz', '25_02_05_13_57_10_F2b_8000Hz', '25_02_05_13_58_40_F2b_8000Hz', '25_02_05_13_59_55_F2b_8000Hz']
subjD['s127R'] = ['Results/s127/', '25_02_05_14_18_12_F2b_8000Hz', '25_02_05_14_19_27_F2b_8000Hz', '25_02_05_14_20_42_F2b_8000Hz', '25_02_05_14_21_58_F2b_8000Hz', '25_02_05_14_23_29_F2b_8000Hz', '25_02_05_14_25_14_F2b_8000Hz', '25_02_05_14_25_58_F2b_8000Hz', '25_02_05_14_26_42_F2b_8000Hz', '25_02_05_14_27_44_F2b_8000Hz', '25_02_05_14_28_43_F2b_8000Hz']

subjD['s995L2s'] =  ['Results/s995/', '25_02_07_13_29_24_F2b_8000Hz', '25_02_07_13_31_10_F2b_8000Hz', '25_02_07_13_32_26_F2b_8000Hz', '25_02_07_13_37_50_F2b_8000Hz', '25_02_07_13_39_05_F2b_8000Hz', '25_02_07_13_40_21_F2b_8000Hz', '25_02_07_13_41_54_F2b_8000Hz']
subjD['s995R2s'] = ['Results/s995/', '25_02_07_14_00_07_F2b_8000Hz', '25_02_07_14_01_38_F2b_8000Hz', '25_02_07_14_02_46_F2b_8000Hz', '25_02_07_14_03_45_F2b_8000Hz', '25_02_07_14_05_01_F2b_8000Hz', '25_02_07_14_06_01_F2b_8000Hz', '25_02_07_14_07_17_F2b_8000Hz', '25_02_07_14_08_33_F2b_8000Hz']

subjD['s122L'] = ['Results/s122/', '25_02_10_16_23_36_F2b_8000Hz', '25_02_10_16_24_36_F2b_8000Hz', '25_02_10_16_25_20_F2b_8000Hz', '25_02_10_16_26_19_F2b_8000Hz', '25_02_10_16_27_35_F2b_8000Hz', '25_02_10_16_28_49_F2b_8000Hz', '25_02_10_16_30_05_F2b_8000Hz']
subjD['s122R'] = ['Results/s122/', '25_02_10_16_43_42_F2b_8000Hz', '25_02_10_16_44_46_F2b_8000Hz', '25_02_10_16_46_16_F2b_8000Hz', '25_02_10_16_47_31_F2b_8000Hz', '25_02_10_16_48_47_F2b_8000Hz', '25_02_10_16_50_01_F2b_8000Hz', '25_02_10_16_51_33_F2b_8000Hz', '25_02_10_16_52_48_F2b_8000Hz']
    

subjD['s116L'] = ['Results/s116/', '25_02_10_13_54_40_F2b_8000Hz', '25_02_10_13_55_58_F2b_8000Hz', '25_02_10_13_56_57_F2b_8000Hz', '25_02_10_13_57_58_F2b_8000Hz', '25_02_10_13_59_19_F2b_8000Hz', '25_02_10_14_00_35_F2b_8000Hz', '25_02_10_14_01_49_F2b_8000Hz']
subjD['s116R'] = ['Results/s116/', '25_02_10_14_18_29_F2b_8000Hz', '25_02_10_14_19_29_F2b_8000Hz', '25_02_10_14_20_45_F2b_8000Hz', '25_02_10_14_21_44_F2b_8000Hz', '25_02_10_14_22_44_F2b_8000Hz', '25_02_10_14_24_01_F2b_8000Hz', '25_02_10_14_25_32_F2b_8000Hz', '25_02_10_14_26_53_F2b_8000Hz']

subjD['s112L'] = ['Results/s112/', '25_02_10_11_30_13_F2b_8000Hz', '25_02_10_11_31_30_F2b_8000Hz', '25_02_10_11_32_30_F2b_8000Hz', '25_02_10_11_33_29_F2b_8000Hz', '25_02_10_11_34_28_F2b_8000Hz', '25_02_10_11_35_27_F2b_8000Hz', '25_02_10_11_36_26_F2b_8000Hz']
subjD['s112R'] = ['Results/s112/', '25_02_10_11_48_31_F2b_8000Hz', '25_02_10_11_49_30_F2b_8000Hz', '25_02_10_11_50_48_F2b_8000Hz', '25_02_10_11_52_18_F2b_8000Hz', '25_02_10_11_53_20_F2b_8000Hz', '25_02_10_11_54_39_F2b_8000Hz', '25_02_10_11_55_54_F2b_8000Hz']

subjD['s140L'] = ['Results/s140/', '25_02_13_16_53_28_F2b_8000Hz', '25_02_13_16_54_44_F2b_8000Hz', '25_02_13_16_55_51_F2b_8000Hz', '25_02_13_16_57_07_F2b_8000Hz', '25_02_13_16_58_39_F2b_8000Hz', '25_02_13_16_59_54_F2b_8000Hz', '25_02_13_17_01_26_F2b_8000Hz', '25_02_13_17_02_42_F2b_8000Hz']
subjD['s140R'] = ['Results/s140/', '25_02_13_16_24_56_F2b_8000Hz', '25_02_13_16_26_10_F2b_8000Hz', '25_02_13_16_27_27_F2b_8000Hz', '25_02_13_16_28_27_F2b_8000Hz', '25_02_13_16_29_42_F2b_8000Hz', '25_02_13_16_30_42_F2b_8000Hz', '25_02_13_16_32_14_F2b_8000Hz', '25_02_13_16_33_46_F2b_8000Hz']

subjD['s133R'] = ['Results/s133/', '25_02_10_18_06_48_F2b_8000Hz', '25_02_10_18_07_36_F2b_8000Hz', '25_02_10_18_08_37_F2b_8000Hz', '25_02_10_18_09_22_F2b_8000Hz', '25_02_10_18_10_24_F2b_8000Hz', '25_02_10_18_11_25_F2b_8000Hz', '25_02_10_18_12_27_F2b_8000Hz', '25_02_10_18_13_28_F2b_8000Hz', '25_02_10_18_14_43_F2b_8000Hz', '25_02_10_18_16_30_F2b_8000Hz', '25_02_10_18_18_01_F2b_8000Hz']

subjD['s135L'] = ['Results/s135/', '25_02_19_09_50_25_F2b_8000Hz', '25_02_19_09_51_55_F2b_8000Hz', '25_02_19_09_53_40_F2b_8000Hz', '25_02_19_09_55_26_F2b_8000Hz', '25_02_19_09_57_11_F2b_8000Hz', '25_02_19_09_58_50_F2b_8000Hz']
subjD['s135R'] = ['Results/s135/', '25_02_19_10_14_03_F2b_8000Hz', '25_02_19_10_15_52_F2b_8000Hz', '25_02_19_10_17_07_F2b_8000Hz', '25_02_19_10_18_38_F2b_8000Hz', '25_02_19_10_19_53_F2b_8000Hz', '25_02_19_10_21_15_F2b_8000Hz']

subjD['s102L'] = ['Results/s102/', '24_12_18_14_30_09_F2b_8000Hz', '24_12_18_14_31_39_F2b_8000Hz', '24_12_18_14_33_30_F2b_8000Hz', '24_12_18_14_35_09_F2b_8000Hz', '24_12_18_14_36_23_F2b_8000Hz', '24_12_18_14_38_09_F2b_8000Hz', '24_12_18_14_39_39_F2b_8000Hz', '24_12_18_14_41_10_F2b_8000Hz']
subjD['s102R'] = ['Results/s102/', '24_12_18_14_56_47_F2b_8000Hz', '24_12_18_14_57_31_F2b_8000Hz', '24_12_18_14_58_35_F2b_8000Hz', '24_12_18_14_59_34_F2b_8000Hz', '24_12_18_15_01_04_F2b_8000Hz', '24_12_18_15_02_18_F2b_8000Hz', '24_12_18_15_03_33_F2b_8000Hz', '24_12_18_15_04_48_F2b_8000Hz']


subjD['s120L'] = ['Results/s120/', '25_02_19_11_22_51_F2b_8000Hz', '25_02_19_11_23_50_F2b_8000Hz', '25_02_19_11_24_49_F2b_8000Hz', '25_02_19_11_25_48_F2b_8000Hz', '25_02_19_11_26_46_F2b_8000Hz', '25_02_19_11_27_45_F2b_8000Hz', '25_02_19_11_28_44_F2b_8000Hz', '25_02_19_11_29_59_F2b_8000Hz', '25_02_19_11_31_16_F2b_8000Hz']
subjD['s120R'] = ['Results/s120/', '25_02_19_11_41_44_F2b_8000Hz', '25_02_19_11_42_28_F2b_8000Hz', '25_02_19_11_43_43_F2b_8000Hz', '25_02_19_11_44_27_F2b_8000Hz', '25_02_19_11_45_25_F2b_8000Hz', '25_02_19_11_46_58_F2b_8000Hz', '25_02_19_11_48_12_F2b_8000Hz', '25_02_19_11_49_42_F2b_8000Hz', '25_02_19_11_50_57_F2b_8000Hz']

subjD['s136L'] = ['Results/s136/', '25_02_12_11_36_53_F2b_8000Hz', '25_02_12_11_38_24_F2b_8000Hz', '25_02_12_11_40_11_F2b_8000Hz', '25_02_12_11_41_43_F2b_8000Hz', '25_02_12_11_43_31_F2b_8000Hz', '25_02_12_11_45_19_F2b_8000Hz', '25_02_12_11_46_56_F2b_8000Hz', '25_02_12_11_47_13_F2b_8000Hz', '25_02_12_11_48_47_F2b_8000Hz']
subjD['s136R'] = ['Results/s136/', '25_02_12_12_05_55_F2b_8000Hz', '25_02_12_12_08_48_F2b_8000Hz', '25_02_12_12_10_04_F2b_8000Hz', '25_02_12_12_11_35_F2b_8000Hz', '25_02_12_12_13_23_F2b_8000Hz', '25_02_12_12_15_26_F2b_8000Hz', '25_02_12_12_17_21_F2b_8000Hz', '25_02_12_12_17_37_F2b_8000Hz', '25_02_12_12_19_08_F2b_8000Hz']


subjD['s146R'] = ['Results/s146/', '25_06_03_10_45_34_F2b_8000Hz', '25_06_03_10_46_16_F2b_8000Hz', '25_06_03_10_47_07_F2b_8000Hz', '25_06_03_10_47_50_F2b_8000Hz', '25_06_03_10_48_32_F2b_8000Hz', '25_06_03_10_49_15_F2b_8000Hz', '25_06_03_10_50_20_F2b_8000Hz', '25_06_03_10_51_15_F2b_8000Hz', '25_06_03_10_52_09_F2b_8000Hz']

subjD['s147L'] = ['Results/s147/', '25_06_04_13_20_20_F2b_8000Hz', '25_06_04_13_21_25_F2b_8000Hz', '25_06_04_13_22_30_F2b_8000Hz', '25_06_04_13_23_24_F2b_8000Hz', '25_06_04_13_24_18_F2b_8000Hz', '25_06_04_13_25_23_F2b_8000Hz', '25_06_04_13_26_28_F2b_8000Hz', '25_06_04_13_27_45_F2b_8000Hz']
subjD['s147R'] = ['Results/s147/', '25_06_04_13_49_37_F2b_8000Hz', '25_06_04_13_50_42_F2b_8000Hz', '25_06_04_13_51_35_F2b_8000Hz', '25_06_04_13_52_40_F2b_8000Hz', '25_06_04_13_53_56_F2b_8000Hz', '25_06_04_13_55_13_F2b_8000Hz', '25_06_04_13_56_19_F2b_8000Hz', '25_06_04_13_57_34_F2b_8000Hz']


subjN = 's147L'



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
    
    
    
    for k in range(10,400):
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



DPgr = {}
L2list = []
f2f1list = []
for i in range(1,len(subjD[subjN])):

    DPgrD, rateOct, L2, f2f1, Nchosen = getDPgram(subjD[subjN][0], subjD[subjN][i])
    DPgr[str(L2)] = DPgrD
    DPgr[str(L2)+'ch'] = Nchosen
    L2list.append(L2)  # list of L2 values
    f2f1list.append(f2f1)
    #DPgrD10 = getDPgram(path, deL_r10)


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

InfoOnData(DPgr)



#%%


fig,(ax1,ax2) = plt.subplots(2,1)
pREF = np.sqrt(2)*2e-5

Nopak = 4  # nuber of presentation

fxx = DPgr['60']['fxx']
f2xx = f2f1list[-1]*fxx[:int(len(fxx)//2)+1]/(2-f2f1)
cList = ['C1','C3','C2','C4','C5','C6','C7','C8','C9','C10','C11']
#fxx8 = DPgr['12']['fxx']
for i in range(len(L2list)):
#ax.plot(fxx[:int(len(fxx)//2)+1],20*np.log10(np.abs(DPgr['30']['NLgr'])/pREF),color='C1')
    ax1.plot(f2xx,20*np.log10(np.abs(DPgr[str(L2list[i])]['NLgr'])/pREF),color=cList[i],label=str(L2list[i]))
    ax1.plot(f2xx,20*np.log10(np.abs(DPgr[str(L2list[i])]['NLgrN'])/pREF),':',color=cList[i],label='_nolegend_')

ax1.set_xlim([500,8000])
ax1.set_ylim([-40,20])
ax1.legend()
ax1.set_ylabel('Amplitude (dB SPL)')

cycle = 2*np.pi
F2start = 700
idx1 = np.where(f2xx>=F2start)[0][0]  # freq index for unwraping
for i in range(len(L2list)):
#ax.plot(fxx[:int(len(fxx)//2)+1],20*np.log10(np.abs(DPgr['30']['NLgr'])/pREF),color='C1')
    ax2.plot(f2xx[idx1:],np.unwrap(np.angle(DPgr[str(L2list[i])]['NLgr'][idx1:]))/cycle,color=cList[i],label=str(L2list[i]))
    
ax2.set_ylabel('Phase (cycles)')


ax2.set_xlim([500,8000])
ax2.set_ylim([-5,1])
ax2.set_xlabel('Frequency $f_{2}$ (kHz)')

# Convert x-ticks to kHz
ax1.set_xticks([1000, 2000, 3000, 4000,5000,6000,7000,8000])
ax1.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])  # Update x-tick labels to kHz
ax2.set_xticks([1000, 2000, 3000, 4000,5000,6000,7000,8000])
ax2.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])  # Update x-tick labels to kHz

#ax2.set_xticks([500, 1000, 2000, 4000, 8000])
#ax2.set_xticklabels([0.5, 1, 2, 4, 8])  # Update x-tick labels to kHz

# Add subject name and ear information to the bottom-right corner
if subjN[-1] == 'L':
    ear = 'left ear'
elif subjN[-1] == 'R':
    ear = 'right ear'
subject_name = subjN[:-1]  # Exclude the last character from subject name
text_to_display = f'{subject_name} ({ear})'

# Increase font sizes by approximately 50%
label_fontsize = 16  # Adjust as necessary
legend_fontsize = 12
text_fontsize = 13
# Add the text in the bottom-right corner of the plot
ax2.text(0.3, 0.2, text_to_display, transform=ax2.transAxes, 
        fontsize=text_fontsize, verticalalignment='top', horizontalalignment='right')


# Save the second plot as well (optional)
plt.savefig('Figures/DPgrams/dpgr' + subjN + '.png', format='png', dpi=300)  # Save the second graph

#%%


#%% visualization


import numpy as np

CF = [2000,3000,4000,7000]
CFidx = np.zeros_like(CF)

DPioNL = []
GAdpNL = []
NOxNL = []  # Background noise array
for i in range(len(CF)):
    
    CFidx[i] = np.where(f2xx>=CF[i])[0][0]

    IOx = []
    GAx = []
    NOx = []  # Background noise array
    for j in range(len(L2list)):
        #IOx.append(20*np.log10(np.abs(DPgr[str(L2list[j])]['NLgr'][CFidx[i]])/pREF))
        IOx.append((np.abs(DPgr[str(L2list[j])]['NLgr'][CFidx[i]])/pREF))
        GAx.append(IOx[j]-L2list[j])
        # Extract background noise and convert to dB
        NOx.append(20 * np.log10(np.abs(DPgr[str(L2list[j])]['NLgrN'][CFidx[i]])/pREF))

    DPioNL.append(IOx)
    GAdpNL.append(GAx)
    NOxNL.append(NOx)

data_line = []
noise_lines = []
fig, ax = plt.subplots()
data_line.append(ax.plot(L2list, DPioNL[0], label=r'${\it f}_2$ = '+ str(CF[0]) + ' kHz'))
data_line.append(ax.plot(L2list, DPioNL[1], label=r'${\it f}_2$ = '+ str(CF[1]) + ' kHz'))
data_line.append(ax.plot(L2list, DPioNL[2], label=r'${\it f}_2$ = '+ str(CF[2]) + ' kHz'))
data_line.append(ax.plot(L2list, DPioNL[3], label=r'${\it f}_2$ = '+ str(CF[3]) + ' kHz'))
noise_lines.append(ax.plot(L2list, NOxNL[0], color=data_line[0][0].get_color(), 
                               linestyle=':', linewidth=0.5, label="_nolegend_"))
noise_lines.append(ax.plot(L2list, NOxNL[1], color=data_line[1][0].get_color(), 
                               linestyle=':', linewidth=0.5, label="_nolegend_"))
noise_lines.append(ax.plot(L2list, NOxNL[2], color=data_line[2][0].get_color(), 
                               linestyle=':', linewidth=0.5, label="_nolegend_"))                   
noise_lines.append(ax.plot(L2list, NOxNL[3], color=data_line[3][0].get_color(), 
                               linestyle=':', linewidth=0.5, label="_nolegend_"))                   

ax.tick_params(axis='both', direction='in')

# Convert L2list to a NumPy array for element-wise operations
L2array = np.array(L2list)

# Plot a gray dotted line with slope 1, shifted 35 dB down
ax.plot(L2array, L2array - 35, color='gray', linestyle='--', linewidth=1)

# Set x and y limits to the specified values
ax.set_xlim([20, 70])  # X-axis limits from 20 dB to 70 dB
ax.set_ylim([-20, 20])  # Y-axis limits from -20 dB to 20 dB

# Increase font sizes by approximately 50%
label_fontsize = 16  # Adjust as necessary
legend_fontsize = 12
text_fontsize = 13


# Labels and legend with increased font size
ax.set_ylabel('Amplitude (dB SPL)', fontsize=label_fontsize)
ax.set_xlabel('$L_2$ (dB SPL)', fontsize=label_fontsize)
ax.legend(fontsize=legend_fontsize)

# Add subject name and ear information to the bottom-right corner
if subjN[-1] == 'L':
    ear = 'left ear'
elif subjN[-1] == 'R':
    ear = 'right ear'
subject_name = subjN[:-1]  # Exclude the last character from subject name
text_to_display = f'{subject_name} ({ear})'

# Add the text in the bottom-right corner of the plot
ax.text(0.3, 0.05, text_to_display, transform=ax.transAxes, 
        fontsize=text_fontsize, verticalalignment='top', horizontalalignment='right')

# Increase tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=label_fontsize)

# Adjust layout to fit everything
plt.tight_layout()

# Second plot (optional)
#fig, ax = plt.subplots()
#ax.plot(L2list, GAdpNL[0])
#ax.plot(L2list, GAdpNL[1])
#ax.plot(L2list, GAdpNL[2])
#ax.plot(L2list, GAdpNL[3])

# Optionally, you can also add the shifted line here if relevant
# ax.plot(L2array, L2array - 35, color='gray', linestyle='--', linewidth=1)

# Set x and y limits for the second plot (if needed)
# ax.set_xlim([20, 70])  # X-axis limits for the second plot
# ax.set_ylim([-20, 20])  # Y-axis limits for the second plot

# Adjust layout for the second plot
plt.tight_layout()

#%% fitting


from numpy.polynomial.polynomial import Polynomial
from scipy.io import savemat
import numpy as np
from numpy.polynomial import Polynomial

def fit_polynomial(L2, y_data, degree=4, max_slope_limit=50):
    """
    Fit a polynomial to the given data and calculate key estimates.

    Parameters:
    - L2: array-like, input x-axis values (L2 levels)
    - y_data: array-like, input y-axis values (measured amplitudes)
    - degree: int, the degree of the polynomial to fit (default is 4)
    - max_slope_limit: float, the maximum allowable slope (default is 50 dB)

    Returns:
    - p: Polynomial, the fitted polynomial object
    - max_slope: float, maximum slope of the fitted curve (capped at max_slope_limit)
    - L2_at_max_slope: float, L2 level at maximum slope
    - OAE_level_at_max_slope: float, OAE level at maximum slope
    - L2_half_slope: float, L2 level where slope equals 1/2
    - OAE_level_half_slope: float, OAE level at slope 1/2
    - L2_half_max_slope: float, L2 level where slope equals max_slope/2 (above max slope)
    - OAE_level_half_max_slope: float, OAE level at max_slope/2
    """

    # Fit the polynomial
    p = Polynomial.fit(L2, y_data, deg=degree)

    # Generate fitted values
    x_fit = np.linspace(np.min(L2), np.max(L2), 100)
    y_fit = p(x_fit)

    

    # Calculate slopes numerically for the fitted data
    dy = np.gradient(y_fit, x_fit)

    # Find the maximum slope and its corresponding L2 level
    max_slope_index = np.argmax(dy[:70])
    max_slope = dy[max_slope_index]

    # Cap the maximum slope at the specified limit
    if max_slope > max_slope_limit:
        max_slope = max_slope_limit

    L2_at_max_slope = x_fit[max_slope_index]
    OAE_level_at_max_slope = y_fit[max_slope_index]

    # Calculate the target slopes (1/2 and max_slope/2)
    slope_half = 1 / 2
    slope_half_max = max_slope / 2

    # Find the first point where the slope is below or equal to 1/2, after the max slope
    indices_above_max_slope = np.where(x_fit > L2_at_max_slope)[0]
    half_slope_index = np.where(dy[indices_above_max_slope] <= slope_half)[0]

    if len(half_slope_index) > 0:
        L2_half_slope = x_fit[indices_above_max_slope[half_slope_index[0]]]
        OAE_level_half_slope = y_fit[indices_above_max_slope[half_slope_index[0]]]
    else:
        L2_half_slope = None
        OAE_level_half_slope = None

    # Find the first point where the slope is below or equal to max_slope/2, after the max slope
    half_max_slope_index = np.where(dy[indices_above_max_slope] <= slope_half_max)[0]
    if len(half_max_slope_index) > 0:
        L2_half_max_slope = x_fit[indices_above_max_slope[half_max_slope_index[0]]]
        OAE_level_half_max_slope = y_fit[indices_above_max_slope[half_max_slope_index[0]]]
    else:
        L2_half_max_slope = None
        OAE_level_half_max_slope = None

    return (p, max_slope, L2_at_max_slope, OAE_level_at_max_slope,
            L2_half_slope, OAE_level_half_slope,
            L2_half_max_slope, OAE_level_half_max_slope)


# Example usage

L2 = np.array(L2list)

# Create a dictionary to hold all estimated results
estimated_results = {}
for i in range(4):  # Loop through each dataset index
   
    y_data = DPioNL[i]
    # Call the fitting function
    fit_results = fit_polynomial(L2, y_data, degree=4)
    
    x_fit = np.linspace(np.min(L2), np.max(L2), 100)  # Smooth curve for the fit
    y_fit = fit_results[0](x_fit)
    
    # Plot fitted curve using the same color as the data but exclude from legend
    ax.plot(x_fit, y_fit, color=data_line[i][0].get_color(), linestyle='--', linewidth=1, 
            label="_nolegend_")  # No label in the legend for the fit

    # Extract key points from the fit results
    L2_at_max_slope = fit_results[2]
    OAE_level_at_max_slope = fit_results[3]
    L2_half_slope = fit_results[4]  # Slope of 1/2
    OAE_level_half_slope = fit_results[5]
    L2_half_max_slope = fit_results[6]  # Slope of max_slope / 2
    OAE_level_half_max_slope = fit_results[7]

    # Plot the point where the slope is maximum but exclude from legend
    ax.plot(L2_at_max_slope, OAE_level_at_max_slope, 'o', color=data_line[i][0].get_color(), 
            markersize=8, label="_nolegend_")  # Circle marker, no legend

    # Plot the point where the slope is 1/2 but exclude from legend
    if L2_half_slope is not None:
        ax.plot(L2_half_slope, OAE_level_half_slope, 's', color=data_line[i][0].get_color(), 
                markersize=8, label="_nolegend_")  # Square marker, no legend

    # Plot the point where the slope is max_slope/2 but exclude from legend
    if L2_half_max_slope is not None:
        ax.plot(L2_half_max_slope, OAE_level_half_max_slope, '^', color=data_line[i][0].get_color(), 
                markersize=8, label="_nolegend_")  # Triangle marker, no legend

    # Store results in the dictionary
    estimated_results[f'fit_results_{i}'] = {
        'fitted_polynomial': fit_results[0],
        'max_slope': fit_results[1],
        'L2_at_max_slope': fit_results[2],
        'OAE_level_at_max_slope': fit_results[3],
        'L2_half_slope': fit_results[4],
        'OAE_level_half_slope': fit_results[5],
        'L2_half_max_slope': fit_results[6],
        'OAE_level_half_max_slope': fit_results[7],
    }

# Show legend only for data
ax.legend()


# Save the second plot as well (optional)
plt.savefig('Figures/DPgrams/io' + subjN + '.png', format='png', dpi=300)  # Save the second graph

# Save the results to a .mat file
filename = f'estData{subjN}.mat'
savemat(filename, estimated_results)

print(f'Saved estimated results to {filename}')


#%%



# Define your desired range
L2_min, L2_max = 30, 60  # Example range

# Convert DPioNL[0] to a NumPy array
DPio01 = np.array(DPioNL[0])
NFio01 = np.array(NOxNL[0])
# Select elements where L2list is within the desired range
mask = (L2array >= L2_min) & (L2array <= L2_max)
DPio01 = DPio01[mask]
NFio01 = NFio01[mask]

nDPio01 = (10**(DPio01/20))/np.max(10**(DPio01/20))

# Compute the difference
valid_mask = (DPio01 - NFio01) > 6  # Boolean mask

# Apply cumulative sum only to valid values, setting others to zero
cumsum_result = np.cumsum(np.where(valid_mask, DPio01, 0))


DPio01 = np.array(DPioNL[0])


nDPio01 = (10**(DPio01/20))/np.max(10**(DPio01/20))

cnDPio01 = np.cumsum(nDPio01, axis=0)  # Cumulative sum along rows


DPio02 = np.array(DPioNL[1])


nDPio02 = (10**(DPio02/20))/np.max(10**(DPio02/20))

cnDPio02 = np.cumsum(nDPio02, axis=0)  # Cumulative sum along rows


DPio03 = np.array(DPioNL[2])


nDPio03 = (10**(DPio03/20))/np.max(10**(DPio03/20))

cnDPio03 = np.cumsum(nDPio03, axis=0)  # Cumulative sum along rows


DPio04 = np.array(DPioNL[3])


nDPio04 = (10**(DPio04/20))/np.max(10**(DPio04/20))

cnDPio04 = np.cumsum(nDPio04, axis=0)  # Cumulative sum along rows



fig,ax = plt.subplots()

ax.plot(nDPio01)  # Cumulative sum along rows)

fig,ax = plt.subplots()

ax.plot(cnDPio01*5)  # Cumulative sum along rows)
ax.plot(cnDPio02*5)  # Cumulative sum along rows)
ax.plot(cnDPio03*5)  # Cumulative sum along rows)
ax.plot(cnDPio04*5)  # Cumulative sum along rows)


#%%


# Define your desired range
L2_min, L2_max = 30, 60  

# Initialize lists to store results for each x
DPio_selected, NFio_selected, cumsum_results, nDPio_results = [], [], [], []

# Loop over all x in DPioNL
for x in range(len(DPioNL)):  
    DPio = np.array(DPioNL[x])
    NFio = np.array(NOxNL[x])

    # Select elements where L2array is within the desired range
    mask = (L2array >= L2_min) & (L2array <= L2_max)
    DPio, NFio = DPio[mask], NFio[mask]

    # Normalize DPio values
    nDPio = (10**(DPio / 20)) / np.max(10**(DPio / 20))

    # Compute the valid mask for cumulative sum condition
    valid_mask = (DPio - NFio) > 6  

    # Compute cumulative sum only for valid values
    cumsum_result = np.cumsum(np.where(valid_mask, nDPio, 0))

    # Store results
    DPio_selected.append(DPio)
    NFio_selected.append(NFio)
    nDPio_results.append(nDPio)
    cumsum_results.append(cumsum_result)



fig,ax = plt.subplots()

for i in range(len(cumsum_results)):
    ax.plot(cumsum_results[i]*5)  # Cumulative sum along rows)
    
    

#%% save into matfile

data = {'CF':CF,'DPioNL':DPioNL,'NOxNL':NOxNL,'L2array':L2array,'cres':cumsum_results,'L2min':L2_min,'L2max':L2_max}

# File name for saving
file_name = 'Estimace/DPioCS_results_' + subjN + '.mat'

# Save the data into a .mat file
savemat(file_name, data)

print(f"Data successfully saved to {file_name}")



    
