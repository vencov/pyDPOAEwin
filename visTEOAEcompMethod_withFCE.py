#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:20:57 2024

@author: audiobunka
"""


import numpy as np
from UserModules.pyUtilities import butter_highpass_filter
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from UserModules.pyDPOAEmodule import giveTEOAE_MCmp, roexwin
plt.close('all')



def giveMatricesTEOAE(FolderName,FileName,Nopak,latency_SC):

    #def docalcTEOAEs(Fname,)
    for counter in range(1,Nopak+1):
    
    
    #Lcv = np.arange(20,56+6,6)
    
        if counter<10:  # to add 0 before the counter number
            counterSTR = '0' + str(counter)
        else:
            counterSTR = str(counter)    
    
    
        data = loadmat(FolderName+FileName+counterSTR+".mat")
        Lc = data['Lc'][0][0]
        recsig20 = data['recsig'][:,0]
    
        
        Npulse = data['Npulse'][0][0]
        Nclicks = data['Nclicks'][0][0]
        fsamp = data['fsamp'][0][0]
      
        tTD01, tTD02, tTD03, tTD04, wz, midx, tN01, tN02, tN03, tN04 = giveTEOAE_MCmp(data,recsig20,latency_SC,Npulse,Nclicks,Tap=1e-3,TWt=17e-3,ArtRej=1000)
        
        if counter == 1:
            recMat1 = tTD01
            recMat2 = tTD02
            recMat3 = tTD03
            recMat4 = tTD04
        else:
            recMat1 = np.c_[recMat1, tTD01]  # add to make a matrix with columns for every run
            recMat2 = np.c_[recMat2, tTD02]  # add to make a matrix with columns for every run
            recMat3 = np.c_[recMat3, tTD03]  # add to make a matrix with columns for every run
            recMat4 = np.c_[recMat4, tTD04]  # add to make a matrix with columns for every run
        
    return recMat1, recMat2, recMat3, recMat4, wz, midx, Lc, fsamp




subjD = {}  # dictionary with subjects
subjD['s002L'] = ['Results/s002/','CMclickOAE_s002_24_03_13_16_16_30_Lc_40dB_Ncl228Npulse_3072_L_','CMclickOAE_s002_24_03_13_16_18_21_Lc_46dB_Ncl228Npulse_3072_L_',
                  'CMclickOAE_s002_24_03_13_16_20_14_Lc_52dB_Ncl228Npulse_3072_L_','CMclickOAE_s002_24_03_13_16_22_00_Lc_58dB_Ncl228Npulse_3072_L_',
                  'CMclickOAE_s002_24_03_13_16_23_47_Lc_64dB_Ncl228Npulse_3072_L_']
subjD['s002R'] = ['Results/s002/','CMclickOAE_s002_24_03_13_16_45_17_Lc_40dB_Ncl228Npulse_3072_R_','CMclickOAE_s002_24_03_13_16_47_05_Lc_46dB_Ncl228Npulse_3072_R_',
                  'CMclickOAE_s002_24_03_13_16_48_58_Lc_52dB_Ncl228Npulse_3072_R_','CMclickOAE_s002_24_03_13_16_50_46_Lc_58dB_Ncl228Npulse_3072_R_',
                  'CMclickOAE_s002_24_03_13_16_52_33_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s078L'] = ['Results/s078/', 'CMclickOAE_s078_24_03_13_18_22_39_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s078_24_03_13_18_24_37_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s078_24_03_13_18_26_55_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s078_24_03_13_18_28_49_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s078_24_03_13_18_30_44_Lc_64dB_Ncl228Npulse_3072_L_']

subjD['s078R'] = ['Results/s078/', 'CMclickOAE_s078_24_03_13_17_40_19_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s078_24_03_13_17_42_08_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s078_24_03_13_17_44_13_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s078_24_03_13_17_46_06_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s078_24_03_13_17_48_01_Lc_64dB_Ncl228Npulse_3072_R_']


subjD['s069L'] = ['Results/s069/Cmp/','CMclickOAE_s069_24_03_19_16_17_07_Lc_40dB_Ncl228Npulse_3072_L_','CMclickOAE_s069_24_03_19_16_18_57_Lc_46dB_Ncl228Npulse_3072_L_',
                  'CMclickOAE_s069_24_03_19_16_20_44_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s069_24_03_19_16_22_44_Lc_58dB_Ncl228Npulse_3072_L_',
                  'CMclickOAE_s069_24_03_19_16_24_33_Lc_60dB_Ncl228Npulse_3072_L_']

subjD['s069R'] = ['Results/s069/Cmp/','CMclickOAE_s069_24_03_19_16_28_41_Lc_40dB_Ncl228Npulse_3072_R_','CMclickOAE_s069_24_03_19_16_30_37_Lc_46dB_Ncl228Npulse_3072_R_',
   'CMclickOAE_s069_24_03_19_16_32_52_Lc_52dB_Ncl228Npulse_3072_R_','CMclickOAE_s069_24_03_19_16_34_39_Lc_58dB_Ncl228Npulse_3072_R_',
   'CMclickOAE_s069_24_03_19_16_36_45_Lc_60dB_Ncl228Npulse_3072_R_']


subjD['s080L'] = ['Results/s080/', 'CMclickOAE_s080_24_03_19_15_44_33_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_03_19_15_46_28_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_03_19_15_48_24_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_03_19_15_50_14_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_03_19_15_52_12_Lc_64dB_Ncl228Npulse_3072_L_']


subjD['s080R'] = ['Results/s080/', 'CMclickOAE_s080_24_03_19_15_08_14_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_03_19_15_10_03_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_03_19_15_11_54_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_03_19_15_13_45_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_03_19_15_15_36_Lc_61dB_Ncl228Npulse_3072_R_']


subjD['s081L'] = ['Results/s081/','CMclickOAE_s081_24_03_20_13_51_56_Lc_40dB_Ncl228Npulse_3072_L_','CMclickOAE_s081_24_03_20_13_53_54_Lc_46dB_Ncl228Npulse_3072_L_',
 'CMclickOAE_s081_24_03_20_13_55_43_Lc_52dB_Ncl228Npulse_3072_L_','CMclickOAE_s081_24_03_20_13_57_36_Lc_58dB_Ncl228Npulse_3072_L_',
 'CMclickOAE_s081_24_03_20_13_59_26_Lc_61dB_Ncl228Npulse_3072_L_']

subjD['s081R'] = ['Results/s081/','CMclickOAE_s081_24_03_20_14_14_57_Lc_60dB_Ncl228Npulse_3072_R_','CMclickOAE_s081_24_03_20_14_16_53_Lc_58dB_Ncl228Npulse_3072_R_',
                  'CMclickOAE_s081_24_03_20_14_18_43_Lc_52dB_Ncl228Npulse_3072_R_','CMclickOAE_s081_24_03_20_14_20_34_Lc_46dB_Ncl228Npulse_3072_R_',
                  'CMclickOAE_s081_24_03_20_14_22_24_Lc_40dB_Ncl228Npulse_3072_R_']

subjD['s004L'] = ['Results/s004/GACR24/', 'CMclickOAE_s004_24_03_20_17_03_22_Lc_63dB_Ncl228Npulse_3072_L_','CMclickOAE_s004_24_03_20_17_05_22_Lc_58dB_Ncl228Npulse_3072_L_',
 'CMclickOAE_s004_24_03_20_17_07_13_Lc_52dB_Ncl228Npulse_3072_L_','CMclickOAE_s004_24_03_20_17_09_04_Lc_46dB_Ncl228Npulse_3072_L_',
 'CMclickOAE_s004_24_03_20_17_10_59_Lc_40dB_Ncl228Npulse_3072_L_']


subjD['s083L'] = ['Results/s083/','CMclickOAE_s083_24_03_26_14_59_35_Lc_60dB_Ncl228Npulse_3072_L_','CMclickOAE_s083_24_03_26_15_01_27_Lc_58dB_Ncl228Npulse_3072_L_','CMclickOAE_s083_24_03_26_15_03_15_Lc_52dB_Ncl228Npulse_3072_L_',
                  'CMclickOAE_s083_24_03_26_15_05_12_Lc_46dB_Ncl228Npulse_3072_L_']


subjD['s083R'] = ['Results/s083/','CMclickOAE_s083_24_03_26_15_29_42_Lc_60dB_Ncl228Npulse_3072_R_','CMclickOAE_s083_24_03_26_15_31_33_Lc_58dB_Ncl228Npulse_3072_R_','CMclickOAE_s083_24_03_26_15_33_24_Lc_52dB_Ncl228Npulse_3072_R_',
                  'CMclickOAE_s083_24_03_26_15_35_10_Lc_46dB_Ncl228Npulse_3072_R_']

subjD['s002L'] = ['Results/s002/', 'CMclickOAE_s002_24_03_13_16_16_30_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s002_24_03_13_16_18_21_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s002_24_03_13_16_20_14_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s002_24_03_13_16_22_00_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s002_24_03_13_16_23_47_Lc_64dB_Ncl228Npulse_3072_L_']
subjD['s002R'] = ['Results/s002/', 'CMclickOAE_s002_24_03_13_16_45_17_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s002_24_03_13_16_47_05_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s002_24_03_13_16_48_58_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s002_24_03_13_16_50_46_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s002_24_03_13_16_52_33_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s082L'] = ['Results/s082/', 'CMclickOAE_s082_24_03_26_09_57_37_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s082_24_03_26_09_55_25_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s082_24_03_26_09_53_16_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s082_24_03_26_09_51_08_Lc_62dB_Ncl228Npulse_3072_L_']
subjD['s082R'] =['Results/s082/', 'CMclickOAE_s082_24_03_26_10_24_13_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s082_24_03_26_10_22_06_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s082_24_03_26_10_20_02_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s082_24_03_26_10_18_02_Lc_62dB_Ncl228Npulse_3072_R_']

subjD['s081L'] = ['Results/s081/', 'CMclickOAE_s081_24_03_20_13_51_56_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s081_24_03_20_13_53_54_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s081_24_03_20_13_55_43_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s081_24_03_20_13_57_36_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s081_24_03_20_13_59_26_Lc_61dB_Ncl228Npulse_3072_L_']
subjD['s081R'] = ['Results/s081/', 'CMclickOAE_s081_24_03_20_14_22_24_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s081_24_03_20_14_20_34_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s081_24_03_20_14_18_43_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s081_24_03_20_14_16_53_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s081_24_03_20_14_14_57_Lc_60dB_Ncl228Npulse_3072_R_']

subjD['s003R'] = ['Results/s003/Click400_8000/', 'CMclickOAE_s003_24_04_05_17_30_21_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_04_05_17_16_25_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_04_05_17_14_40_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_04_05_17_12_54_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_04_05_17_11_05_Lc_64dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_04_05_17_09_13_Lc_70dB_Ncl228Npulse_3072_R_']

subjD['s040L'] = ['Results/s040/GACR/', 'CMclickOAE_s040_24_03_26_16_49_46_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s040_24_03_26_16_48_01_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s040_24_03_26_16_46_17_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s040_24_03_26_16_44_22_Lc_62dB_Ncl228Npulse_3072_L_']
subjD['s040R'] = ['Results/s040/GACR/', 'CMclickOAE_s040_24_03_26_16_59_46_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s040_24_03_26_16_57_56_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s040_24_03_26_16_55_49_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s040_24_03_26_16_53_58_Lc_62dB_Ncl228Npulse_3072_R_']

subjD['s004L1'] = ['Results/s004/Click400_8000/', 'CMclickOAE_s004_24_05_06_15_07_12_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_15_05_26_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_15_03_41_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_15_01_56_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_14_59_59_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s004R1'] = ['Results/s004/Click400_8000/', 'CMclickOAE_s004_24_05_06_17_28_34_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_17_26_49_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_17_25_01_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_17_23_17_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_17_21_10_Lc_64dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_17_19_21_Lc_68dB_Ncl228Npulse_3072_R_']

subjD['s004L2'] = ['Results/s004/Click400_4000/', 'CMclickOAE_s004_24_05_06_15_49_03_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s004_24_05_06_15_47_19_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s004_24_05_06_15_45_31_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s004_24_05_06_15_43_43_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s004_24_05_06_15_41_56_Lc_64dB_Ncl228Npulse_3072_L_']
    
subjD['s004R2'] = ['Results/s004/Click400_4000/', 'CMclickOAE_s004_24_05_06_15_59_16_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_15_57_32_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_15_55_46_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_15_53_59_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_15_52_11_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s003R2'] = ['Results/s003/Click400_4000/', 'CMclickOAE_s003_24_05_07_14_11_48_Lc_30dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_10_03_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_08_13_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_06_27_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_04_38_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_02_50_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_13_59_16_Lc_64dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_00_58_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s003L2'] = ['Results/s003/Click400_4000/', 'CMclickOAE_s003_24_05_07_14_32_55_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s003_24_05_07_14_31_09_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s003_24_05_07_14_29_21_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s003_24_05_07_14_27_25_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s003_24_05_07_14_25_40_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s003_24_05_07_14_23_51_Lc_62dB_Ncl228Npulse_3072_L_']

subjD['s080L2'] = ['Results/s080/click400_4000/', 'CMclickOAE_s080_24_05_10_15_40_26_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_05_10_15_38_42_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_05_10_15_36_56_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_05_10_15_35_11_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_05_10_15_33_27_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_05_10_15_31_42_Lc_64dB_Ncl228Npulse_3072_L_']
subjD['s080R2'] =['Results/s080/click400_4000/', 'CMclickOAE_s080_24_05_10_15_28_41_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_05_10_15_26_53_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_05_10_15_25_08_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_05_10_15_23_21_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_05_10_15_21_33_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_05_10_15_19_48_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s074L2'] = ['Results/s074/c400_4000/', 'CMclickOAE_s074_24_05_28_13_49_18_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s074_24_05_28_13_47_23_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s074_24_05_28_13_45_12_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s074_24_05_28_13_43_22_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s074_24_05_28_13_41_16_Lc_60dB_Ncl228Npulse_3072_L_']
subjD['s074R2'] = ['Results/s074/c400_4000/', 'CMclickOAE_s074_24_05_28_14_04_07_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s074_24_05_28_14_02_10_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s074_24_05_28_13_59_57_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s074_24_05_28_13_57_49_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s074_24_05_28_13_56_11_Lc_60dB_Ncl228Npulse_3072_R_']


subjD['s069L3'] = ['Results/s069/', 'CMclickOAE_s069_24_05_30_15_54_42_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s069_24_05_30_15_52_32_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s069_24_05_30_15_50_26_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s069_24_05_30_15_48_36_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s069_24_05_30_15_46_43_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s069_24_05_30_15_44_34_Lc_64dB_Ncl228Npulse_3072_L_']
subjD['s069R3'] = ['Results/s069/', 'CMclickOAE_s069_24_05_30_16_10_05_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s069_24_05_30_16_08_03_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s069_24_05_30_16_06_04_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s069_24_05_30_16_04_01_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s069_24_05_30_16_01_34_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s069_24_05_30_15_59_31_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s055L3'] = ['Results/s055/', 'CMclickOAE_s055_24_06_04_13_34_35_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s055_24_06_04_13_32_18_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s055_24_06_04_13_30_24_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s055_24_06_04_13_28_19_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s055_24_06_04_13_26_14_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s055_24_06_04_13_24_19_Lc_62dB_Ncl228Npulse_3072_L_']
subjD['s055R3'] = ['Results/s055/', 'CMclickOAE_s055_24_06_04_13_54_07_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s055_24_06_04_13_52_07_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s055_24_06_04_13_49_30_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s055_24_06_04_13_47_14_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s055_24_06_04_13_45_01_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s055_24_06_04_13_42_51_Lc_62dB_Ncl228Npulse_3072_R_']

subjD['s088L'] = ['Results/s088/', 'CMclickOAE_s088_24_06_20_16_10_25_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s088_24_06_20_16_08_34_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s088_24_06_20_16_06_44_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s088_24_06_20_16_04_54_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s088_24_06_20_16_02_47_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s088_24_06_20_16_00_34_Lc_62dB_Ncl228Npulse_3072_L_']
subjD['s088R'] = ['Results/s088/', 'CMclickOAE_s088_24_06_20_16_37_52_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s088_24_06_20_16_35_42_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s088_24_06_20_16_33_32_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s088_24_06_20_16_31_16_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s088_24_06_20_16_29_06_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s088_24_06_20_16_26_56_Lc_62dB_Ncl228Npulse_3072_R_']

subjD['s089L'] = ['Results/s089/', 'CMclickOAE_s089_24_07_01_11_22_31_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s089_24_07_01_11_20_26_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s089_24_07_01_11_18_20_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s089_24_07_01_11_16_14_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s089_24_07_01_11_14_01_Lc_64dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s089_24_07_01_11_11_51_Lc_70dB_Ncl228Npulse_3072_L_']
['Results/s089/']

subjN = 's055R3'




Nopak = 12
Nfft = 1920
TEOAE = {}
tClick = {}
tLINm = {}
tHm = {}
tNLm = {}
fxD = {}
LcList = []
latency_SC = 20532
#latency_SC = 20544
latency_SC = 8304
for i in range(1,len(subjD[subjN])):
    recMat1, recMat2, recMat3, recMat4, wz, midx, Lc,fsamp = giveMatricesTEOAE(subjD[subjN][0], subjD[subjN][i], Nopak,latency_SC)
    
    if i==1:
        midxT = midx
        
    #t1 = wz*((np.mean(recMat1,1)+np.mean(recMat2,1)+np.mean(recMat3,1))-np.mean(recMat4,1))  # perform averaging in time and calculating TEOAE using compression method
    #tLin = wz*((np.mean(recMat1,1)+np.mean(recMat2,1)+np.mean(recMat3,1)))/3 # perform averaging in time and calculating TEOAE using compression method
    t1 = ((np.mean(recMat1,1)+np.mean(recMat2,1)+np.mean(recMat3,1))-np.mean(recMat4,1))  # perform averaging in time and calculating TEOAE using compression method
    tLin = ((np.mean(recMat1,1)+np.mean(recMat2,1)+np.mean(recMat3,1)))/3 # perform averaging in time and calculating TEOAE using compression method
    tHigh = np.mean(recMat4,1)
    # extract click
    Nrw = 20 # order of the roex win
    Trwc = 0.001 # half of rw for click 
    rwC = roexwin(2*midx+1,Nrw,fsamp,Trwc,Trwc)
    
    tClick[str(Lc)] = tLin[:len(rwC)]*rwC
    
    # extract CEOAE
    
    Tceoae = 10e-3
    rwS = roexwin(2400,Nrw,fsamp,Tceoae,Tceoae)
    tNLm[str(Lc)] = t1[midxT:2400+midxT]*rwS
    tLINm[str(Lc)] = tLin[midxT:2400+midxT]*rwS
    tHm[str(Lc)] = tHigh[midxT:2400+midxT]*rwS
    
    #TEOAE[str(Lc)] = 2*np.fft.rfft(np.concatenate((t1[midxT:],np.zeros(int(2**15)))))/Nfft  # calculate spectrum
    
    Ns = len(np.concatenate((t1[midxT:],np.zeros(int(2**15)))))
    fxx = np.arange(Ns)*fsamp/Ns
    
    fxD[str(Lc)] = fxx[:Ns//2+1]  # take half of the frequency axis
    LcList.append(str(Lc))
#fig,ax = plt.subplots()
#ax.plot(tTD01[midx:])
#ax.plot(tTD02[midx:])
#ax.plot(tTD03[midx:])
#ax.plot(tTD04[midx:])


#fig,ax = plt.subplots()
#ax.plot(np.mean(recMat1,1))
#ax.plot(np.mean(recMat2,1))
#ax.plot(np.mean(recMat3,1))
#ax.plot(np.mean(recMat4,1))


#[52,70,58,40,64,46]


#fig,ax = plt.subplots()
#ax.plot((np.mean(recMat1,1)+np.mean(recMat2,1)+np.mean(recMat3,1))/3-np.mean(recMat4,1)/10**(9/20))
#ax.plot(0.0001*wz)


'''
Nfft = 1920



#t1 = np.mean(recMat4,1)-np.mean(recMat1,1)/10**(12/20)
#t2 = np.mean(recMat6,1)-np.mean(recMat3,1)/10**(12/20)
#t3 = np.mean(recMat1,1)-np.mean(recMat5,1)/10**(12/20)
#t4 = np.mean(recMat3,1)-np.mean(recMat2,1)/10**(12/20)
#t5 = np.mean(recMat5,1)-np.mean(recMat2,1)/2


teoaeFD20 = 2*np.fft.rfft(np.concatenate((t1[midx:],np.zeros(int(2**15)))))/Nfft
#teoaeFD56 = 2*np.fft.rfft(np.concatenate((teoaeTD56[midx:],np.zeros(int(2**15)))))/Nfft

fsamp = 96000
#Ns = len(np.concatenate((t1[midx:],np.zeros(int(2**15)))))
Ns = len((TEOAE['40']))

cycle = 2*np.pi

fig,(ax1, ax2) = plt.subplots(2,1)
ax1.plot(fxD[LcList[0]],20*np.log10(np.abs(TEOAE[LcList[0]])/(np.sqrt(2)*2e-5)),color='C00')
ax1.plot(fxD[LcList[1]],20*np.log10(np.abs(TEOAE[LcList[1]])/(np.sqrt(2)*2e-5)),color='C01')
ax1.plot(fxD[LcList[2]],20*np.log10(np.abs(TEOAE[LcList[2]])/(np.sqrt(2)*2e-5)),color='C02')
ax1.plot(fxD[LcList[3]],20*np.log10(np.abs(TEOAE[LcList[3]])/(np.sqrt(2)*2e-5)),color='C03')
ax1.plot(fxD[LcList[4]],20*np.log10(np.abs(TEOAE[LcList[4]])/(np.sqrt(2)*2e-5)),color='C04')

#ax1.plot(fx[:Ns//2+1],20*np.log10(np.abs(teoaeFD56)/(np.sqrt(2)*2e-5)),color='C06')

ax1.set_xlim(200,5000)
ax1.set_ylim(-40,10)

cycle = 2*np.pi
ax2.plot(fxD[LcList[0]],np.unwrap(np.angle(TEOAE[LcList[0]]))/cycle,color='C00')
ax2.plot(fxD[LcList[1]],np.unwrap(np.angle(TEOAE[LcList[1]]))/cycle,color='C01')
ax2.plot(fxD[LcList[2]],np.unwrap(np.angle(TEOAE[LcList[2]]))/cycle,color='C02')
ax2.plot(fxD[LcList[3]],np.unwrap(np.angle(TEOAE[LcList[3]]))/cycle,color='C03')
ax2.plot(fxD[LcList[4]],np.unwrap(np.angle(TEOAE[LcList[4]]))/cycle,color='C04')
ax2.set_xlim(200,5000)
ax2.set_ylim(-60,10)
ax2.set_xlabel('Frequnecy (Hz)')
ax2.set_ylabel('Phase (cycles)')
plt.show()
'''

#%% wavelet tr.

def mother_wavelet2(Nw,Nt,df,dt):
    vlnky = np.zeros((Nt,Nw))
    tx = (np.arange(Nt)-Nw)*dt
    for k in range(Nw):
        vlnky[:,k] = np.cos(2*np.pi*(k+1)*df*tx)*1/(1+(0.075*(k+1)*2*np.pi*df*tx)**4)
    return vlnky


def wavelet_filter(signal, wavelet_basis):
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
    filtered_signal = np.zeros(N)
    coefwti = np.zeros_like(wavelet_basis)
    # Perform wavelet filtering
    
    
    
    for k in range(20,400):
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
        awltSh = 0.011 # time constant for long latency components
        awltZL = 0.005 # time constant for zero-latency components
        bwlt = 0.8  # exponent
        TrwcSh = awltSh*(fx[k]/1000)**(-bwlt)  # rwc time const for positive time (frequency dependent vis Moleti 2012)
        TrwcZL = awltZL*(fx[k]/1000)**(-bwlt)  # rwc time const for positive time (frequency dependent vis Moleti 2012)
        NsampShift = int(fsamp*TrwcSh)  # number of samples  for shifting the roexwin
        
        #rwC = roexwin(Nall,Nrw,fsamp,Trwc01,Trwc02) #  not shifted window
        rwC = roexwin(Nall,Nrw,fsamp,TrwcSh-TrwcZL,TrwcSh-TrwcZL) # 
        rwC = np.roll(rwC,NsampShift)
        
        
        # Compute the inverse Fourier transform
        coew = np.fft.ifft(coewtf)*np.fft.fftshift(rwC)*len(signal)
        coefwti[:,k] = np.fft.fftshift(coew)
        
        # Add the filtered signal to the overall result
        filtered_signal += coew.real
    
    return filtered_signal, coefwti



tLINmwf={}
tNLmwf={}
tHmwf={}
SLINmwf={}
SHmwf={}
SNLmwf={}
SClick={}
cwLIN = {}
cwH = {}
cwNL = {}
for keys in tLINm:
    

    f2max = 5000
    
    hm_50lin = np.concatenate((np.zeros(len(tLINm[keys])),tLINm[keys]))
    hm_50nl = np.concatenate((np.zeros(len(tNLm[keys])),tNLm[keys]))
    hm_50H = np.concatenate((np.zeros(len(tHm[keys])),tHm[keys]))
    
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


    tLINmwf[keys],cwLIN[keys] = wavelet_filter(hm_50lin, vlnky)
    
    tNLmwf[keys],cwNL[keys] = wavelet_filter(hm_50nl, vlnky)
    
    tHmwf[keys],cwH[keys] = wavelet_filter(hm_50H, vlnky)
    
    
    SLINmwf[keys] = 2*np.fft.rfft(np.concatenate((tLINmwf[keys],np.zeros(int(2**15)))))/Nw  # calculate spectrum
    SNLmwf[keys] = 2*np.fft.rfft(np.concatenate((tNLmwf[keys],np.zeros(int(2**15)))))/Nw  # calculate spectrum
    SHmwf[keys] = 2*np.fft.rfft(np.concatenate((3*tLINmwf[keys]-tHmwf[keys],np.zeros(int(2**15)))))/Nw  # calculate spectrum
    fxx = np.arange(Nt+int(2**15))*fsamp/(Nt+int(2**15))
    SClick[keys] = 2*np.fft.rfft(np.concatenate((tClick[keys],np.zeros(Nt-len(tClick[keys])),np.zeros(int(2**15)))))/len(tClick[keys])
     
    
    import numpy as np
   
    # Save
    
    #np.save('s069t01.npy', tLINmwf) 
    #np.save('s069t02.npy', tClick) 

    # Load
    #read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()
    #print(read_dictionary['hello']) # displays "world"

    

#%% show me results in the time domain

fig,ax = plt.subplots()
LcList = []
for keys in tLINmwf:
    ax.plot(tLINmwf[keys]/np.sqrt(np.mean(tClick[keys]**2)))
    LcList.append(keys)
ax.legend(LcList)


fig,ax = plt.subplots()
for keys in tNLmwf:
    ax.plot(tNLmwf[keys]/np.sqrt(np.mean(tClick[keys]**2)))
ax.legend(LcList)

#fig,ax = plt.subplots()
#for keys in tNLmwf:
#    ax.plot((3*tLINmwf[keys]-tHmwf[keys])/np.sqrt(np.mean(tClick[keys]**2)))
#ax.legend(LcList)


#%% show me results in the frequency domain


fig,ax = plt.subplots()
for keys in tNLmwf:
    ax.plot(fxx[:int(len(fxx)/2+1)],20*np.log10(np.abs(SNLmwf[keys]/SClick[keys])))
    
ax.set_xlim([500,5000])
ax.legend(LcList)

fig,ax = plt.subplots()
for keys in tNLmwf:
    ax.plot(fxx[:int(len(fxx)/2+1)],20*np.log10(np.abs(SLINmwf[keys]/SClick[keys])))
ax.set_xlim([500,5000])
ax.legend(LcList)


pREF = np.sqrt(2)*2e-5

#%% TEOAE grams
# draw a graph with NL and LIN


#plt.ylabel('ddd')


fig,(ax1,ax2) = plt.subplots(2,1)
Fc1 = 0.63
Fc2 = 1.19
Fc3 = 1.46


plt.gcf().text(0.5, 0.9, 'Nonlinear extraction method', fontsize=12, fontweight='bold',ha='center')

plt.rcParams["xtick.direction"]="in"
plt.rcParams["ytick.direction"]="in"
plt.rcParams["xtick.top"]=True
plt.rcParams["ytick.right"]=True

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 12}

plt.matplotlib.rc('font', **font)

for keys in tNLmwf:
    if keys!='70':
        ax1.plot(fxx[:int(len(fxx)/2+1)]/1e3,20*np.log10(np.abs(SNLmwf[keys])/pREF)-20*np.log10(3))
    
ax1.set_xlim([0.500,3.000])
ax1.set_ylim([-40,10])
#ax1.legend(LcList[:-1])
ax1.set_ylabel('Amplitude (dB SPL)')
ax1.plot([Fc1,Fc1],[-100,100],'--',color='gray')
ax1.plot([Fc2,Fc2],[-100,100],'--',color='gray')
ax1.plot([Fc3,Fc3],[-100,100],'--',color='gray')
#ax1.plot([Fc4,Fc4],[-100,100],'--',color='gray')

cycle = 2*np.pi

for keys in tNLmwf:
    if keys!='70':
        ax2.plot(fxx[:int(len(fxx)/2+1)]/1e3,np.unwrap(np.angle(SNLmwf[keys]))/cycle)
ax2.set_xlim([0.500,3.000])
ax2.legend(LcList,loc='lower left',ncol=2)    
ax2.set_ylim([-40,1])

#fig.legend(handles, labels, loc='right')

ax2.set_ylabel('Phase (cycles)')
fig.supxlabel('Frequency (kHz)')

#plt.savefig('Figures/ceoaeNLs003L2.eps', format='eps')

fig,(ax1,ax2) = plt.subplots(2,1)

plt.gcf().text(0.5, 0.9, 'Linear extraction method', fontsize=12, fontweight='bold',ha='center')

plt.rcParams["xtick.direction"]="in"
plt.rcParams["ytick.direction"]="in"
plt.rcParams["xtick.top"]=True
plt.rcParams["ytick.right"]=True


for keys in tLINmwf:
    if keys!='70':
        ax1.plot(fxx[:int(len(fxx)/2+1)]/1e3,20*np.log10(np.abs(SLINmwf[keys])/pREF))

ax1.plot([Fc1,Fc1],[-100,100],'--',color='gray')
ax1.plot([Fc2,Fc2],[-100,100],'--',color='gray')
ax1.plot([Fc3,Fc3],[-100,100],'--',color='gray')
#ax1.plot([Fc4,Fc4],[-100,100],'--',color='gray')

ax1.set_ylabel('Amplitude (dB SPL)')
ax1.set_xlim([0.500,3.000])
ax1.set_ylim([-40,10])
#ax1.legend(LcList[:-1])

cycle = 2*np.pi

for keys in tNLmwf:
    if keys!='70':
        ax2.plot(fxx[:int(len(fxx)/2+1)]/1e3,np.unwrap(np.angle(SLINmwf[keys]))/cycle)
ax2.set_xlim([0.500,3.000])
ax2.legend(LcList,loc='lower left',ncol=2)    
ax2.set_ylim([-40,1])

ax2.set_ylabel('Phase (cycles)')
fig.supxlabel('Frequency (kHz)')

#plt.savefig('Figures/ceoaeLINs003L2.eps', format='eps')

#%%


# combine two freq

fig,ax = plt.subplots()
for keys in tNLmwf:
    if keys=='64':
        ax.plot(fxx[:int(len(fxx)/2+1)],20*np.log10(np.abs(SNLmwf[keys])/pREF)-20*np.log10(3),color='k')
        ax.plot(fxx[:int(len(fxx)/2+1)],20*np.log10(np.abs(SLINmwf[keys])/pREF),color='r')
    
ax.set_xlim([500,5000])
ax.legend(LcList)


#fig,ax = plt.subplots()
#for keys in tNLmwf:
#    ax.plot(fxx[:int(len(fxx)/2+1)],20*np.log10(np.abs(SHmwf[keys]/SClick[keys])))
#ax.set_xlim([500,5000])
#ax.legend(LcList)


#%%
Nt = 4800
Nw = 2400
dt = 1/fsamp
tx = (np.arange(Nt)-Nw)*dt
X, Y = np.meshgrid(fx[:len(fx)//2], tx)

'''
fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
from matplotlib import cm
from matplotlib.ticker import LinearLocator
surf = ax.plot_surface(Y, X, img, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.view_init(azim=0, elev=90)
'''

fig,ax = plt.subplots()
ax.contourf(X,Y,np.abs(cwLIN['46'])**0.2)
ax.set_xlim((0, 2500))

#%%


#Fc1 = 0.9
#Fc2 = 1.6
#Fc3 = 2.08

idx1 = np.where(fxx>=Fc1*1e3)[0][0]
idx2 = np.where(fxx>=Fc2*1e3)[0][0]
idx3 = np.where(fxx>=Fc3*1e3)[0][0]


k = 0
ioG1 = np.zeros(len(SNLmwf))
ioG2 = np.zeros(len(SNLmwf))
ioG3 = np.zeros(len(SNLmwf))
io = np.zeros(len(SNLmwf))

Lvect = np.zeros(len(SNLmwf))
for keys in tNLmwf:
    ioG1[k] = 20*np.log10(np.abs(SNLmwf[keys][idx1])/SClick[keys][idx1])-20*np.log10(3)
    ioG2[k] = 20*np.log10(np.abs(SNLmwf[keys][idx2])/SClick[keys][idx2])-20*np.log10(3)
    ioG3[k] = 20*np.log10(np.abs(SNLmwf[keys][idx3])/SClick[keys][idx3])-20*np.log10(3)
    io[k] = 20*np.log10(np.abs(SNLmwf[keys][idx1])/pREF)-20*np.log10(3)
    Lvect[k] = int(keys)
    k += 1

k = 0
ioLin = np.zeros(len(SNLmwf))
ioLinG1 = np.zeros(len(SNLmwf))
ioLinG2 = np.zeros(len(SNLmwf))
ioLinG3 = np.zeros(len(SNLmwf))

for keys in tNLmwf:
    ioLinG1[k] = 20*np.log10(np.abs(SLINmwf[keys][idx1])/SClick[keys][idx1])
    ioLinG2[k] = 20*np.log10(np.abs(SLINmwf[keys][idx2])/SClick[keys][idx2])
    ioLinG3[k] = 20*np.log10(np.abs(SLINmwf[keys][idx3])/SClick[keys][idx3])
    ioLin[k] = 20*np.log10(np.abs(SLINmwf[keys][idx1])/pREF)
    k += 1


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


#dataO = loadmat('DPondra.mat')
#ioF1 = dataO['ioF1'].flatten()
#ioF2 = dataO['ioF2'].flatten()
#ioF3 = dataO['ioF3'].flatten()
#L2v = dataO['L2v'].flatten()

pREF= np.sqrt(2)*2e-5


#ax.plot(L2v,20*np.log10(ioF1/(pREF*10**(L2v/20))))


fig,ax = plt.subplots()
ax.plot(Lvect[:],ioG1[:],color=colors[0])
ax.plot(Lvect[:],ioLinG1[:],'-',color=colors[1])
#ax.plot(L2v,20*np.log10(ioF1/(pREF*10**(L2v/20))),color=colors[2])
ax.plot(Lvect[:],15-Lvect[:],':',color='gray')
ax.set_ylim([-60,-30])
ax.set_xlim([30,65])
ax.legend(('NL','Lin','DPOAE'))
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 17}

plt.matplotlib.rc('font', **font)
ax.set_title('0.8 kHz')
ax.set_xlabel('Click level (dB peSPL) / Tone level (dB FPL)')
ax.set_ylabel('Magnitude (dB re 1)')
fig.tight_layout()
#plt.savefig('Figures/IOg1s003.eps', format='eps')


fig,ax = plt.subplots()
ax.plot(Lvect[:],ioG2[:],color=colors[0])
ax.plot(Lvect[:],ioLinG2[:],color=colors[1])
#ax.plot(L2v,20*np.log10(ioF2/(pREF*10**(L2v/20))),color=colors[2])
ax.plot(Lvect[:],15-Lvect[:],':',color='gray')
ax.legend(('NL','Lin','DPOAE'))
plt.matplotlib.rc('font', **font)
ax.set_title('1.6 kHz')
ax.set_xlabel('Level (dB peSPL)')
ax.set_ylabel('Magnitude (dB re 1)')


ax.set_ylim([-60,-30])
ax.set_xlim([30,65])
fig.tight_layout()
#plt.savefig('Figures/IOg2s003.eps', format='eps')

fig,ax = plt.subplots()
ax.plot(Lvect[:],ioG3[:],color=colors[0])
ax.plot(Lvect[:],ioLinG3[:],color=colors[1])
#ax.plot(L2v,20*np.log10(ioF3/(pREF*10**(L2v/20))),color=colors[2])
ax.plot(Lvect[:],15-Lvect[:],':',color='gray')
ax.set_ylim([-60,-30])
ax.set_xlim([30,65])
ax.legend(('NL','Lin','DPOAE'))
plt.matplotlib.rc('font', **font)
ax.set_title('2.08 kHz')
ax.set_xlabel('Level (dB peSPL)')
ax.set_ylabel('Magnitude (dB re 1)')
ax.set_xlim([30,65])
fig.tight_layout()
plt.savefig('Figures/IOg3s003.eps', format='eps')





fig,ax = plt.subplots()
ax.plot(Lvect[:],ioG1[:],linestyle='-',color=colors[0])
ax.plot(Lvect[:],ioLinG1[:],linestyle='--',color=colors[0])
ax.plot(Lvect[:],15-Lvect[:],':',color='gray')



k = 0
ioG = np.zeros(len(SNLmwf))
io = np.zeros(len(SNLmwf))

Lvect = np.zeros(len(SNLmwf))
for keys in tNLmwf:
    ioG[k] = 20*np.log10(np.abs(SNLmwf[keys][idx])/SClick[keys][idx])-20*np.log10(3)
    io[k] = 20*np.log10(np.abs(SNLmwf[keys][idx])/pREF)-20*np.log10(3)
    Lvect[k] = int(keys)
    k += 1


idx = np.where(fxx>=f_xx)[0][0]
k = 0
ioLin = np.zeros(len(SNLmwf))
ioLinG = np.zeros(len(SNLmwf))

for keys in tNLmwf:
    ioLinG[k] = 20*np.log10(np.abs(SLINmwf[keys][idx])/SClick[keys][idx])
    ioLin[k] = 20*np.log10(np.abs(SLINmwf[keys][idx])/pREF)
    k += 1



ax.plot(Lvect[:],ioG[:])
ax.plot(Lvect[:],ioLinG[:])




f_xx = Fc4*1e3

idx = np.where(fxx>=f_xx)[0][0]
k = 0
ioG = np.zeros(len(SNLmwf))
io = np.zeros(len(SNLmwf))

Lvect = np.zeros(len(SNLmwf))
for keys in tNLmwf:
    ioG[k] = 20*np.log10(np.abs(SNLmwf[keys][idx])/SClick[keys][idx])-20*np.log10(3)
    io[k] = 20*np.log10(np.abs(SNLmwf[keys][idx])/pREF)-20*np.log10(3)
    Lvect[k] = int(keys)
    k += 1


idx = np.where(fxx>=f_xx)[0][0]
k = 0
ioLin = np.zeros(len(SNLmwf))
ioLinG = np.zeros(len(SNLmwf))

for keys in tNLmwf:
    ioLinG[k] = 20*np.log10(np.abs(SLINmwf[keys][idx])/SClick[keys][idx])
    ioLin[k] = 20*np.log10(np.abs(SLINmwf[keys][idx])/pREF)
    k += 1



ax.plot(Lvect[:],ioG[:])
ax.plot(Lvect[:],ioLinG[:])


ax.set_ylim([-55,-25])


#%% fitovani Gain fce

#L = np.arange(35,75,5)
x = 10**(Lvect[:]/20)*np.sqrt(2)*2e-5

y = 10**(ioLin[:]/20)


def p_to_dB(p):
    return 20*np.log10(np.abs(p))


def dB_to_p(I):
    return (10**(I/20))

def func(A, G0, Act, alpha):
    return (G0/((1 + (A/Act))**alpha))


from scipy.optimize import curve_fit
popt, pcov = curve_fit(func, x, y, bounds=((dB_to_p(-50), 0.001, 0),(dB_to_p(-10), 0.089, 3)))


Peak_str = popt[0] # Peak strength in dB
Comp_thresh = popt[1] # Compression threshold in dB
Comp_slope = popt[2]

print("Peak strength, G0 = "+str(p_to_dB(Peak_str))+" dB")
print("Compression slope, alpha = "+str(popt[2]))
print("Compression Threshold, Act = "+str(20*np.log10(Comp_thresh/(np.sqrt(2)*2e-5)))+" dB")



fig,ax = plt.subplots()
ax.plot(x,np.abs(y))
ax.plot(x,Peak_str/(1+x/Comp_thresh)**Comp_slope)


fig,ax = plt.subplots()
ax.plot(Lvect[:],20*np.log10(np.abs(y)))

x2 = 10**(np.arange(10,80,5)/20)*np.sqrt(2)*2e-5

ax.plot(np.arange(10,80,5),20*np.log10((Peak_str/(1+x2/Comp_thresh)**Comp_slope)))