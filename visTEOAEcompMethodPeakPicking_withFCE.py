# -*- coding: utf-8 -*-
"""
Code which analyzes measured TEOAE data. It finds peaks in the amplitude finestructure

Created on Wed Oct  9 14:08:48 2024

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
    
    
        try:
            data = loadmat(FolderName + FileName + counterSTR + ".mat")
           # Process your data here
        except FileNotFoundError:
            break
        
        Lc = data['Lc'][0][0]
        recsig20 = data['recsig'][:,0]
    
        
        Npulse = data['Npulse'][0][0]
        Nclicks = data['Nclicks'][0][0]
        fsamp = data['fsamp'][0][0]
      
        tTD01, tTD02, tTD03, tTD04, wz, midx, tN01, tN02, tN03, tN04, nTDest = giveTEOAE_MCmp(data,recsig20,latency_SC,Npulse,Nclicks,Tap=1e-3,TWt=17e-3,ArtRej=1000)
        
        if counter == 1:
            recMat1 = tTD01
            recMat2 = tTD02
            recMat3 = tTD03
            recMat4 = tTD04
            nMat1 = nTDest
        else:
            recMat1 = np.c_[recMat1, tTD01]  # add to make a matrix with columns for every run
            recMat2 = np.c_[recMat2, tTD02]  # add to make a matrix with columns for every run
            recMat3 = np.c_[recMat3, tTD03]  # add to make a matrix with columns for every run
            recMat4 = np.c_[recMat4, tTD04]  # add to make a matrix with columns for every run
            nMat1 = np.c_[nMat1, nTDest]
    return recMat1, recMat2, recMat3, recMat4, nMat1, wz, midx, Lc, fsamp




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

subjD['s004L'] = ['Results/s004/Click400_4000/', 'CMclickOAE_s004_24_05_06_15_49_03_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s004_24_05_06_15_47_19_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s004_24_05_06_15_45_31_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s004_24_05_06_15_43_43_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s004_24_05_06_15_41_56_Lc_64dB_Ncl228Npulse_3072_L_']
    
subjD['s004R'] = ['Results/s004/Click400_4000/', 'CMclickOAE_s004_24_05_06_15_59_16_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_15_57_32_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_15_55_46_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_15_53_59_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s004_24_05_06_15_52_11_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s003R'] = ['Results/s003/Click400_4000/', 'CMclickOAE_s003_24_05_07_14_11_48_Lc_30dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_10_03_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_08_13_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_06_27_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_04_38_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_02_50_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_13_59_16_Lc_64dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s003_24_05_07_14_00_58_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s003L'] = ['Results/s003/Click400_4000/', 'CMclickOAE_s003_24_05_07_14_32_55_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s003_24_05_07_14_31_09_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s003_24_05_07_14_29_21_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s003_24_05_07_14_27_25_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s003_24_05_07_14_25_40_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s003_24_05_07_14_23_51_Lc_62dB_Ncl228Npulse_3072_L_']

subjD['s080L'] = ['Results/s080/click400_4000/', 'CMclickOAE_s080_24_05_10_15_40_26_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_05_10_15_38_42_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_05_10_15_36_56_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_05_10_15_35_11_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_05_10_15_33_27_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s080_24_05_10_15_31_42_Lc_64dB_Ncl228Npulse_3072_L_']
subjD['s080R'] =['Results/s080/click400_4000/', 'CMclickOAE_s080_24_05_10_15_28_41_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_05_10_15_26_53_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_05_10_15_25_08_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_05_10_15_23_21_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_05_10_15_21_33_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s080_24_05_10_15_19_48_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s074L'] = ['Results/s074/c400_4000/', 'CMclickOAE_s074_24_05_28_13_49_18_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s074_24_05_28_13_47_23_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s074_24_05_28_13_45_12_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s074_24_05_28_13_43_22_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s074_24_05_28_13_41_16_Lc_60dB_Ncl228Npulse_3072_L_']
subjD['s074R'] = ['Results/s074/c400_4000/', 'CMclickOAE_s074_24_05_28_14_04_07_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s074_24_05_28_14_02_10_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s074_24_05_28_13_59_57_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s074_24_05_28_13_57_49_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s074_24_05_28_13_56_11_Lc_60dB_Ncl228Npulse_3072_R_']


subjD['s069L'] = ['Results/s069/', 'CMclickOAE_s069_24_05_30_15_54_42_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s069_24_05_30_15_52_32_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s069_24_05_30_15_50_26_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s069_24_05_30_15_48_36_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s069_24_05_30_15_46_43_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s069_24_05_30_15_44_34_Lc_64dB_Ncl228Npulse_3072_L_']
subjD['s069R'] = ['Results/s069/', 'CMclickOAE_s069_24_05_30_16_10_05_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s069_24_05_30_16_08_03_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s069_24_05_30_16_06_04_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s069_24_05_30_16_04_01_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s069_24_05_30_16_01_34_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s069_24_05_30_15_59_31_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s055L'] = ['Results/s055/', 'CMclickOAE_s055_24_06_04_13_34_35_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s055_24_06_04_13_32_18_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s055_24_06_04_13_30_24_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s055_24_06_04_13_28_19_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s055_24_06_04_13_26_14_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s055_24_06_04_13_24_19_Lc_62dB_Ncl228Npulse_3072_L_']
subjD['s055R'] = ['Results/s055/', 'CMclickOAE_s055_24_06_04_13_54_07_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s055_24_06_04_13_52_07_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s055_24_06_04_13_49_30_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s055_24_06_04_13_47_14_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s055_24_06_04_13_45_01_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s055_24_06_04_13_42_51_Lc_62dB_Ncl228Npulse_3072_R_']

subjD['s088L'] = ['Results/s088/', 'CMclickOAE_s088_24_06_20_16_10_25_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s088_24_06_20_16_08_34_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s088_24_06_20_16_06_44_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s088_24_06_20_16_04_54_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s088_24_06_20_16_02_47_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s088_24_06_20_16_00_34_Lc_62dB_Ncl228Npulse_3072_L_']
subjD['s088R'] = ['Results/s088/', 'CMclickOAE_s088_24_06_20_16_37_52_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s088_24_06_20_16_35_42_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s088_24_06_20_16_33_32_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s088_24_06_20_16_31_16_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s088_24_06_20_16_29_06_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s088_24_06_20_16_26_56_Lc_62dB_Ncl228Npulse_3072_R_']


subjD['s089L'] = ['Results/s089/', 'CMclickOAE_s089_24_07_01_11_22_31_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s089_24_07_01_11_20_26_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s089_24_07_01_11_18_20_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s089_24_07_01_11_16_14_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s089_24_07_01_11_14_01_Lc_64dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s089_24_07_01_11_11_51_Lc_70dB_Ncl228Npulse_3072_L_']
subjD['s089R'] = ['Results/s089/', 'CMclickOAE_s089_24_07_11_12_05_12_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s089_24_07_11_12_03_19_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s089_24_07_11_12_01_28_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s089_24_07_11_11_59_37_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s089_24_07_11_11_57_33_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s089_24_07_11_11_55_27_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s091L'] = ['Results/s091/', 'CMclickOAE_s091_24_07_11_13_56_45_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s091_24_07_11_13_54_37_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s091_24_07_11_13_52_29_Lc_64dB_Ncl228Npulse_3072_L_']
subjD['s091R'] = ['Results/s091/', 'CMclickOAE_s091_24_07_11_14_24_54_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s091_24_07_11_14_22_45_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s091_24_07_11_14_20_32_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s007L'] = ['Results/s007/', 'CMclickOAE_s007_24_05_30_15_18_24_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s007_24_05_30_15_16_11_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s007_24_05_30_15_13_56_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s007_24_05_30_15_11_38_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s007_24_05_30_15_09_23_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s007_24_05_30_15_07_05_Lc_61dB_Ncl228Npulse_3072_L_']
subjD['s007R'] = ['Results/s007/', 'CMclickOAE_s007_24_05_30_15_00_24_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s007_24_05_30_14_58_08_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s007_24_05_30_14_56_11_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s007_24_05_30_14_54_08_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s007_24_05_30_14_51_53_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s007_24_05_30_14_50_02_Lc_64dB_Ncl228Npulse_3072_R_']

subjD['s087L'] = ['Results/s087/', 'CMclickOAE_s087_24_06_20_13_56_24_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s087_24_06_20_13_54_33_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s087_24_06_20_13_52_42_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s087_24_06_20_13_50_48_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s087_24_06_20_13_48_37_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s087_24_06_20_13_46_17_Lc_62dB_Ncl228Npulse_3072_L_']
subjD['s087R'] = ['Results/s087/', 'CMclickOAE_s087_24_06_20_14_11_50_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s087_24_06_20_14_09_42_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s087_24_06_20_14_07_37_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s087_24_06_20_14_05_05_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s087_24_06_20_14_02_58_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s087_24_06_20_14_01_04_Lc_62dB_Ncl228Npulse_3072_R_']

subjD['s086L'] = ['Results/s086/', 'CMclickOAE_s086_24_06_20_12_27_33_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s086_24_06_20_12_25_44_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s086_24_06_20_12_23_55_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s086_24_06_20_12_21_43_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s086_24_06_20_12_19_31_Lc_58dB_Ncl228Npulse_3072_L_']
subjD['s086R'] = ['Results/s086/', 'CMclickOAE_s086_24_06_20_12_41_47_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s086_24_06_20_12_39_54_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s086_24_06_20_12_38_05_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s086_24_06_20_12_36_09_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s086_24_06_20_12_34_22_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s086_24_06_20_12_32_21_Lc_62dB_Ncl228Npulse_3072_R_']

subjD['s084L'] = ['Results/s084/', 'CMclickOAE_s084_24_06_04_15_21_27_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s084_24_06_04_15_16_54_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s084_24_06_04_15_14_35_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s084_24_06_04_15_12_23_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s084_24_06_04_15_09_54_Lc_60dB_Ncl228Npulse_3072_L_']
subjD['s084R'] = ['Results/s084/', 'CMclickOAE_s084_24_06_04_15_35_54_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s084_24_06_04_15_34_02_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s084_24_06_04_15_32_05_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s084_24_06_04_15_30_05_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s084_24_06_04_15_27_54_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s084_24_06_04_15_25_39_Lc_60dB_Ncl228Npulse_3072_R_']


subjD['s072L'] = ['Results/s072/', 'CMclickOAE_s072_24_06_05_12_06_59_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s072_24_06_05_12_04_50_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s072_24_06_05_12_02_41_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s072_24_06_05_12_00_29_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s072_24_06_05_11_58_16_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s072_24_06_05_11_56_05_Lc_62dB_Ncl228Npulse_3072_L_']
subjD['s072R'] = ['Results/s072/', 'CMclickOAE_s072_24_06_05_12_21_26_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s072_24_06_05_12_19_15_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s072_24_06_05_12_17_07_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s072_24_06_05_12_14_56_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s072_24_06_05_12_12_47_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s072_24_06_05_12_10_36_Lc_62dB_Ncl228Npulse_3072_R_']

subjD['s063L'] = ['Results/s063/', 'CMclickOAE_s063_24_06_05_13_31_28_Lc_34dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s063_24_06_05_13_29_37_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s063_24_06_05_13_27_46_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s063_24_06_05_13_25_56_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s063_24_06_05_13_24_04_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s063_24_06_05_13_21_43_Lc_62dB_Ncl228Npulse_3072_L_']
subjD['s063R'] = ['Results/s063/', 'CMclickOAE_s063_24_06_05_14_02_23_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s063_24_06_05_14_00_27_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s063_24_06_05_13_58_36_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s063_24_06_05_13_56_24_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s063_24_06_05_13_54_14_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s063_24_06_05_13_52_20_Lc_62dB_Ncl228Npulse_3072_R_']

subjD['s001L'] = ['Results/s001/GACR/','CMclickOAE_s001_24_12_18_10_26_34_Lc_40dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s001_24_12_18_10_26_34_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s001_24_12_18_10_26_34_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s001_24_12_18_10_26_34_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s001_24_12_18_10_26_34_Lc_64dB_Ncl228Npulse_3072_L_']

subjD['s001R'] = ['Results/s001/GACR/','CMclickOAE_s001_24_12_18_10_48_54_Lc_34dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s001_24_12_18_10_48_54_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s001_24_12_18_10_48_54_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s001_24_12_18_10_48_54_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s001_24_12_18_10_48_54_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s001_24_12_18_10_48_54_Lc_64dB_Ncl228Npulse_3072_R_']


subjD['s122L'] = ['Results/s122/', 'CMclickOAE_s122_25_02_10_16_35_14_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s122_25_02_10_16_35_14_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s122_25_02_10_16_35_14_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s122_25_02_10_16_35_14_Lc_64dB_Ncl228Npulse_3072_L_']

subjD['s122R'] = ['Results/s122/', 'CMclickOAE_s122_25_02_10_16_55_54_Lc_40dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s122_25_02_10_16_55_54_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s122_25_02_10_16_55_54_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s122_25_02_10_16_55_54_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s122_25_02_10_16_55_54_Lc_64dB_Ncl228Npulse_3072_R_']


subjD['s127L'] = ['Results/s127/', 'CMclickOAE_s127_25_02_05_14_07_01_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s127_25_02_05_14_07_01_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s127_25_02_05_14_07_01_Lc_64dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s127_25_02_05_14_12_31_Lc_68dB_Ncl228Npulse_3072_L_']
subjD['s127R'] = ['Results/s127/','CMclickOAE_s127_25_02_05_14_31_15_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s127_25_02_05_14_31_15_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s127_25_02_05_14_31_15_Lc_64dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s127_25_02_05_14_31_15_Lc_68dB_Ncl228Npulse_3072_R_']


subjD['pok'] = ['Results/pokus/', 'CMclickOAE_pok_25_03_12_15_47_52_Lc_50dB_Ncl228Npulse_3072_R_', 'CMclickOAE_pok_25_03_12_15_47_52_Lc_60dB_Ncl228Npulse_3072_R_', 'CMclickOAE_pok_25_03_12_15_47_52_Lc_70dB_Ncl228Npulse_3072_R_']

subjD['s121L'] = ['Results/s121/', 'CMclickOAE_s121_25_03_13_09_38_23_Lc_46dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s121_25_03_13_09_38_23_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s121_25_03_13_09_38_23_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s121_25_03_13_09_38_23_Lc_64dB_Ncl228Npulse_3072_L_']
subjD['s121R'] = ['Results/s121/', 'CMclickOAE_s121_25_03_13_09_16_38_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s121_25_03_13_09_16_38_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s121_25_03_13_09_16_38_Lc_64dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s121_25_03_13_09_16_08_Lc_70dB_Ncl228Npulse_3072_R_']

subjD['s138L'] = ['Results/s138/', 'CMclickOAE_s138_25_03_03_16_26_41_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s138_25_03_03_16_26_41_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s138_25_03_03_16_26_41_Lc_64dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s138_25_03_03_16_26_41_Lc_70dB_Ncl228Npulse_3072_L_']
subjD['s138R'] = ['Results/s138/', 'CMclickOAE_s138_25_03_03_16_47_30_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s138_25_03_03_16_47_30_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s138_25_03_03_16_47_30_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s138_25_03_03_16_47_30_Lc_64dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s138_25_03_03_16_47_30_Lc_70dB_Ncl228Npulse_3072_R_']

subjD['s995L'] = ['Results/s995/', 'CMclickOAE_s995_25_02_07_13_47_16_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s995_25_02_07_13_47_16_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s995_25_02_07_13_47_16_Lc_64dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s995_25_02_07_13_53_16_Lc_64dB_Ncl228Npulse_3072_L_','CMclickOAE_s995_25_02_07_14_15_42_Lc_64dB_Ncl228Npulse_3072_L_']
subjD['s995R'] = ['Results/s995/', 'CMclickOAE_s995_25_02_07_14_16_59_Lc_46dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s995_25_02_07_14_16_59_Lc_52dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s995_25_02_07_14_16_59_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s995_25_02_07_14_16_59_Lc_62dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s995_25_02_07_14_16_59_Lc_68dB_Ncl228Npulse_3072_R_']

subjD['s139L'] = [ 'Results/s139/', 'CMclickOAE_s139_25_03_03_13_58_41_Lc_52dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s139_25_03_03_13_58_41_Lc_58dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s139_25_03_03_13_58_41_Lc_64dB_Ncl228Npulse_3072_L_', 'CMclickOAE_s139_25_03_03_13_58_41_Lc_70dB_Ncl228Npulse_3072_L_']
subjD['s139R'] = ['Results/s139/', 'CMclickOAE_s139_25_03_03_14_23_01_Lc_58dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s139_25_03_03_14_23_01_Lc_64dB_Ncl228Npulse_3072_R_', 'CMclickOAE_s139_25_03_03_14_23_01_Lc_70dB_Ncl228Npulse_3072_R_']

subjN = 's139L'



#subjN = 'pok'




Nopak = 12
Nfft = 1920
TEOAE = {}
tClick = {}
tLINm = {}
tHm = {}
nEst = {}
tNLm = {}
fxD = {}
LcList = []
latency_SC = 20532
#latency_SC = 20544
latency_SC = 8304
for i in range(1,len(subjD[subjN])):
    recMat1, recMat2, recMat3, recMat4, recNMat, wz, midx, Lc,fsamp = giveMatricesTEOAE(subjD[subjN][0], subjD[subjN][i], Nopak,latency_SC)
    
    if i==1:
        midxT = midx
    print(i)    
    #t1 = wz*((np.mean(recMat1,1)+np.mean(recMat2,1)+np.mean(recMat3,1))-np.mean(recMat4,1))  # perform averaging in time and calculating TEOAE using compression method
    #tLin = wz*((np.mean(recMat1,1)+np.mean(recMat2,1)+np.mean(recMat3,1)))/3 # perform averaging in time and calculating TEOAE using compression method
    t1 = ((np.mean(recMat1,1)+np.mean(recMat2,1)+np.mean(recMat3,1))-np.mean(recMat4,1))  # perform averaging in time and calculating TEOAE using compression method
    tLin = ((np.mean(recMat1,1)+np.mean(recMat2,1)+np.mean(recMat3,1)))/3 # perform averaging in time and calculating TEOAE using compression method
    tHigh = np.mean(recMat4,1)
    nLin = np.mean(recNMat,1)
    
    # extract click
    Nrw = 20 # order of the roex win
    Trwc = 0.001 # half of rw for click 
    rwC = roexwin(2*midx+1,Nrw,fsamp,Trwc,Trwc)
    
    tClick[str(Lc)] = tLin[:len(rwC)]*rwC
    #nEst[str(Lc)] = nLin  # estimated noise signal
    # extract CEOAE
    
    Tceoae = 7e-3
    rwS = roexwin(2400,Nrw,fsamp,Tceoae,Tceoae)
    tNLm[str(Lc)] = t1[midxT:2400+midxT]*rwS
    tLINm[str(Lc)] = tLin[midxT:2400+midxT]*rwS
    tHm[str(Lc)] = tHigh[midxT:2400+midxT]*rwS
    nEst[str(Lc)] = nLin[midxT:2400+midxT]*rwS  # estimated noise signal
    TEOAE[str(Lc)] = 2*np.fft.rfft(np.concatenate((t1[midxT:],np.zeros(int(2**15)))))/Nfft  # calculate spectrum
    
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


nLINmwf={}
tLINmwf={}
tNLmwf={}
tHmwf={}
SLINmwf={}
SNLINmwf={}
SHmwf={}
SNLmwf={}
SClick={}
cwLIN = {}
ncwLIN = {}
cwH = {}
cwNL = {}
for keys in tLINm:
    

    f2max = 5000
    
    hm_50lin = np.concatenate((np.zeros(len(tLINm[keys])),tLINm[keys]))
    Nhm_50 = np.concatenate((np.zeros(len(nEst[keys])),nEst[keys]))
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
    
    nLINmwf[keys],ncwLIN[keys] = wavelet_filter(Nhm_50, vlnky)
    
    tNLmwf[keys],cwNL[keys] = wavelet_filter(hm_50nl, vlnky)
    
    tHmwf[keys],cwH[keys] = wavelet_filter(hm_50H, vlnky)
    
    
    SLINmwf[keys] = 2*np.fft.rfft(np.concatenate((tLINmwf[keys],np.zeros(int(2**15)))))/Nw  # calculate spectrum
    SNLINmwf[keys] = 2*np.fft.rfft(np.concatenate((nLINmwf[keys],np.zeros(int(2**15)))))/Nw  # calculate spectrum
    SNLmwf[keys] = 2*np.fft.rfft(np.concatenate((tNLmwf[keys],np.zeros(int(2**15)))))/Nw  # calculate spectrum
    SHmwf[keys] = 2*np.fft.rfft(np.concatenate((3*tLINmwf[keys]-tHmwf[keys],np.zeros(int(2**15)))))/Nw  # calculate spectrum
    fxx = np.arange(Nt+int(2**15))*fsamp/(Nt+int(2**15))
    SClick[keys] = 2*np.fft.rfft(np.concatenate((tClick[keys],np.zeros(Nt-len(tClick[keys])),np.zeros(int(2**15)))))/len(tClick[keys])
     
   
    # Save
    
    #np.save('s069t01.npy', tLINmwf) 
    #np.save('s069t02.npy', tClick) 

    # Load
    #read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()
    #print(read_dictionary['hello']) # displays "world"

    

#%% show me results in the time domain


plotTDres = 1
if plotTDres:
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





#%% PEAK PICKING OF TRANSFER FUNCTIONS TO FIND SEVERAL FREQUENCIES FOR ANALYSIS
# peak picking is performed for the largest level (last key?)
keyV = max(SNLmwf.keys())
FreqData = fxx[:int(len(fxx)/2+1)]
TrFuncNL = 20*np.log10(np.abs(SNLmwf[keyV]/SClick[keyV]))
TrFuncLin = 20*np.log10(np.abs(SLINmwf[keyV]/SClick[keyV]))

from scipy.signal import find_peaks
def find_curve_peaks(x, y, interval=None, height_threshold=None, distance=None, prominence=None):
    """
    Detects peaks in the curve y(x) within the given interval and allows peak control with a threshold.

    Parameters:
    x: 1D array-like
        The x-values of the curve.
    y: 1D array-like
        The y-values of the curve.
    interval: tuple (xmin, xmax), optional
        The x-range in which to search for peaks.
    height_threshold: float, optional
        Minimum height of peaks to detect.
    distance: float, optional
        Minimum horizontal distance between neighboring peaks.
    prominence: float, optional
        Minimum prominence of peaks (how much the peak stands out from the surroundings).
    
    Returns:
    peaks: 1D array
        Indices of the detected peaks in the original data.
    """
    # If an interval is provided, restrict the data to the specified interval
    if interval:
        # Find the mask of the data within the interval
        mask = (x >= interval[0]) & (x <= interval[1])
        x_interval = x[mask]
        y_interval = y[mask]
        # Store the starting index of the interval
        interval_start_idx = np.where(mask)[0][0]
    else:
        x_interval = x
        y_interval = y
        interval_start_idx = 0

    # Detect peaks in the restricted data
    peaks, properties = find_peaks(y_interval, height=height_threshold, distance=distance, prominence=prominence)

    # Adjust the peak indices to match the original data
    peaks_in_original = peaks + interval_start_idx

    # Plotting the results
    #plt.figure(figsize=(10, 6))
    #plt.plot(x, y, label="Data")
    #plt.plot(x[peaks_in_original], y[peaks_in_original], 'ro', label="Detected Peaks")
    #plt.title("Detected Peaks in the Curve")
    #plt.xlabel("X")
    #plt.ylabel("Y")
    #plt.legend()
    #plt.show()

    return peaks_in_original, properties


# Find peaks in the interval (2, 8) with a height threshold of 0.5
interval = (500, 4000)
height_threshold = -70
distance = 100  # Minimum distance between peaks
prominence = 2  # Minimum prominence of peaks


peaksNL, propertiesNL = find_curve_peaks(FreqData, TrFuncNL, interval=interval, height_threshold=height_threshold, distance=distance, prominence=prominence)
peaksLin, propertiesLin = find_curve_peaks(FreqData, TrFuncLin, interval=interval, height_threshold=height_threshold, distance=distance, prominence=prominence)


plotNLgain = 0
if plotNLgain:
    fig,ax = plt.subplots()
    for keys in tNLmwf:
        ax.plot(fxx[:int(len(fxx)/2+1)],20*np.log10(np.abs(SNLmwf[keys]/SClick[keys])))
        
    ax.plot(FreqData[peaksNL],TrFuncNL[peaksNL],'ro', label="Detected Peaks")
        
    ax.set_xlim([500,4000])
    ax.legend(LcList)

fig,ax = plt.subplots()
for keys in tNLmwf:
    ax.plot(fxx[:int(len(fxx)/2+1)],20*np.log10(np.abs(SLINmwf[keys]/SClick[keys])))
    
ax.plot(FreqData[peaksLin],TrFuncLin[peaksLin],'ro', label="Detected Peaks")
ax.set_xlim([500,4000])
ax.legend(LcList)


pREF = np.sqrt(2)*2e-5

FchosenLin = [FreqData[peaksLin[i]]/1000 for i in range(len(peaksLin))]
FchosenNL = [FreqData[peaksNL[i]]/1000 for i in range(len(peaksNL))]


#file_name = 'ceoaeFch' + subjN + '.mat'



#plt.ylabel('ddd')

#%% TEOAE Grams - Nonlinear Extraction Method

plotNLCEOAE = 0

if plotNLCEOAE:
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    plt.gcf().text(0.5, 0.9, 'Nonlinear extraction method', fontsize=12, fontweight='bold', ha='center')
    
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    
    font = {'family': 'Helvetica', 'weight': 'normal', 'size': 12}
    plt.matplotlib.rc('font', **font)
    
    # Plot for the nonlinear data
    for keys in tNLmwf:
        if keys != '70':
            ax1.plot(fxx[:int(len(fxx)/2+1)]/1e3, 20*np.log10(np.abs(SNLmwf[keys])/pREF) - 20*np.log10(3))
    
    # Set axis limits and labels for amplitude
    ax1.set_xlim([0.500, 4.000])
    ax1.set_ylim([-40, 10])
    ax1.set_ylabel('Amplitude (dB SPL)')
    
    # Plot vertical lines for each frequency in Fchosen
    for Fc in FchosenNL:
        ax1.plot([Fc, Fc], [-100, 100], '--', color='gray')
    
    # Plot for the phase
    cycle = 2 * np.pi
    for keys in tNLmwf:
        if keys != '70':
            ax2.plot(fxx[:int(len(fxx)/2+1)]/1e3, np.unwrap(np.angle(SNLmwf[keys])) / cycle)
    
    # Set axis limits and labels for phase
    ax2.set_xlim([0.500, 4.000])
    ax2.legend(LcList, loc='lower left', ncol=2)
    ax2.set_ylim([-40, 1])
    ax2.set_ylabel('Phase (cycles)')
    fig.supxlabel('Frequency (kHz)')

#%% TEOAE Grams - Linear Extraction Method
fig, (ax1, ax2) = plt.subplots(2, 1)

cycle = np.pi*2

plt.gcf().text(0.5, 0.9, 'Linear extraction method', fontsize=12, fontweight='bold', ha='center')

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True


# Store colors for consistency
colors = []
i = 0  
# Plot for the linear data
for keys in tLINmwf:
    if keys != '72':
        line, = ax1.plot(fxx[:int(len(fxx)/2+1)]/1e3, 20*np.log10(np.abs(SLINmwf[keys])/pREF))
        colors.append(line.get_color())
        ax1.plot(fxx[:int(len(fxx)/2+1)]/1e3, 20*np.log10(np.abs(SNLINmwf[keys])/pREF),':',color=colors[i])
        i += 1

# Set axis limits and labels for amplitude
ax1.set_xlim([0.500, 4.000])
ax1.set_ylim([-40, 10])
ax1.set_ylabel('Amplitude (dB SPL)')

# Plot vertical lines for each frequency in Fchosen
for Fc in FchosenLin:
    ax1.plot([Fc, Fc], [-100, 100], '--', color='gray')

# Plot for the phase
i = 0
for keys in tLINmwf:
    if keys != '70':
        ax2.plot(fxx[:int(len(fxx)/2+1)]/1e3, np.unwrap(np.angle(SLINmwf[keys])) / cycle,color=colors[i])
        i += 1

# Set axis limits and labels for phase
ax2.set_xlim([0.500, 4.000])
ax2.legend(LcList, loc='lower left', ncol=2)
ax2.set_ylim([-40, 1])
ax2.set_ylabel('Phase (cycles)')


if subjN[-1] == 'R':
    earS = ' right '
elif subjN[-1] == 'L':
    earS = ' left '
    
# Add "s088, right ear" text in the upper right corner of ax1
ax2.text(0.95, 0.9, subjN[:-1] + earS + 'ear', transform=ax2.transAxes, ha='right', va='top')


fig.supxlabel('Frequency (kHz)')

plt.savefig('Figures/s127/ceoaeLINs127Left.png', format='png')



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

plotCWT = 0 # plot wavelet filtering
if plotCWT:
    fig,ax = plt.subplots()
    ax.contourf(X,Y,np.abs(cwLIN['46'])**0.2)
    ax.set_xlim((0, 2500))

#%%
import numpy as np
import matplotlib.pyplot as plt

# Assuming peaksNL and peaksLin are numpy arrays with the indices for peak values
# SNLmwf and SLINmwf are the Nonlinear and Linear datasets
# SClick is used for normalization, pREF is your reference pressure
pREF = np.sqrt(2) * 2e-5  # Example reference pressure

# Number of points in peaksNL and peaksLin
num_peaksNL = len(peaksNL)
num_peaksLin = len(peaksLin)

# Initialize the io arrays based on the number of peaks
ioG = np.zeros((len(SNLmwf), num_peaksNL))     # For nonlinear peaks
ioLinG = np.zeros((len(SNLmwf), num_peaksLin)) # For linear peaks
io = np.zeros(len(SNLmwf))
ioLin = np.zeros(len(SNLmwf))

Lvect = np.zeros(len(SNLmwf))

k = 0
for keys in tNLmwf:
    # Calculate ioG for each peak in peaksNL
    for i, idxNL in enumerate(peaksNL):
        ioG[k, i] = 20 * np.log10(np.abs(SNLmwf[keys][idxNL]) / SClick[keys][idxNL]) - 20 * np.log10(3)

    # Calculate ioLinG for each peak in peaksLin
    for i, idxLin in enumerate(peaksLin):
        ioLinG[k, i] = 20 * np.log10(np.abs(SLINmwf[keys][idxLin]) / SClick[keys][idxLin])

    # Additional io values for the first index of peaksNL
    io[k] = 20 * np.log10(np.abs(SNLmwf[keys][peaksNL[0]]) / pREF) - 20 * np.log10(3)
    ioLin[k] = 20 * np.log10(np.abs(SLINmwf[keys][peaksLin[0]]) / pREF)

    # Store the level vector (Lvect)
    Lvect[k] = int(keys)
    k += 1

# Plotting the results for all peaks in one graph

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plotIOall = 0
if plotIOall:
    
    fig, ax = plt.subplots()
    
    # Plot all nonlinear peak data
    #for i in range(num_peaksNL):
    #    ax.plot(Lvect[:], ioG[:, i], color=colors[i % len(colors)], linestyle='-', label=f'NL {FreqData[peaksNL[i]] / 1e3:.2f} kHz')
    
    # Plot all linear peak data
    for i in range(num_peaksLin):
        ax.plot(Lvect[:], ioLinG[:, i], color=colors[(i + num_peaksNL) % len(colors)], linestyle='--', label=f'Lin {FreqData[peaksLin[i]] / 1e3:.2f} kHz')
    
    # Add a reference line for DPOAE (example: 15 - Lvect)
    ax.plot(Lvect[:], 15 - Lvect[:], ':', color='gray', label='DPOAE')
    
    # Set plot properties
    ax.set_ylim([-60, -30])
    ax.set_xlim([30, 65])
    ax.legend(loc='upper right', ncol=2)
    
    font = {'family': 'Helvetica', 'weight': 'normal', 'size': 17}
    plt.matplotlib.rc('font', **font)
    
    # Title and labels
    ax.set_title('TEOAE gain functions')
    ax.set_xlabel('Click level (dB peSPL) / Tone level (dB FPL)')
    ax.set_ylabel('Magnitude (dB re 1)')
    fig.tight_layout()
    
    # Save the figure
    #plt.savefig('Figures/FreqResponse_AllPeaks.eps', format='eps')
    
    plt.show()





#%% fitovani Gain fce

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.io import savemat

# Example functions (already provided)
def p_to_dB(p):
    return 20 * np.log10(np.abs(p))

def dB_to_p(I):
    return 10**(I / 20)

def func(A, G0, Act, alpha):
    return G0 / ((1 + (A / Act))**alpha)

# List of starting rows (one for each column of ioLinG)
start_rows = np.zeros(ioLinG.shape[1])  # Example values, adjust as needed

fig, ax = plt.subplots()
# Dictionary to store extracted fitting information for each frequency
fit_results = {}

# Loop over each column of ioLinG (assuming ioLinG is a numpy array)
for col_idx in range(ioLinG.shape[1]):
    done = False
    while not done:
        # Get the starting row for the current column
        print(f"\nProcessing Column {col_idx} - FchosenLin = {FchosenLin[col_idx]}")
        user_input = input("Enter a number (0 to max) to set starting row or 'e' to move to next point: ")
        
        if user_input.lower() == 'e':
            # Move to the next point if 'e' is pressed
            done = True
            #plt.close('all')
            start_rows[col_idx] = start_row
            continue
        else:
            try:
                # Validate and update the starting row if a number is provided
                start_row = start_rows[col_idx] = int(user_input)
                if start_row < 0 or start_row >= len(Lvect):
                    print("Invalid starting row. Please try again.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a number or 'e'.")
                continue

        # Prepare data for fitting
        x = 10**(Lvect[start_row:] / 20) * np.sqrt(2) * 2e-5
        y = 10**(ioLinG[start_row:, col_idx] / 20)
        
        # Fit parameters with bounds
        bounds = (
            (dB_to_p(-50), 0.001, 0),     # lower bounds
            (dB_to_p(-10), 0.089, 1)      # upper bounds (adjust alpha max slope)
        )
        
        # Fit the data
        popt, pcov = curve_fit(func, x, y, bounds=bounds)
        
        # Extract fitting results
        Peak_str = popt[0]  # Peak strength in pressure
        Comp_thresh = popt[1]  # Compression threshold in pressure
        Comp_slope = popt[2]  # Compression slope
        
        # Store results in a dictionary with the column index as the key
        fit_results[f'{round(1000*FchosenLin[col_idx])}'] = {
            'Peak_strength_dB': p_to_dB(Peak_str),
            'Compression_slope': Comp_slope,
            'Compression_threshold_dB': 20 * np.log10(Comp_thresh / (np.sqrt(2) * 2e-5)),
            'FchosenLin': FchosenLin[col_idx],  # Corresponding frequency element
            'Start_row': start_row  # Starting row used for this column
        }
        
        
        
        # Plot the results
        x_fit = np.linspace(min(x), max(x) + 0.3, 1000)
        y_fit = func(x_fit, *popt)
        plt.clf()
        plt.scatter(Lvect[start_row:], 20 * np.log10(y), label='Data', color='red')
        plt.plot(20 * np.log10(x_fit / (np.sqrt(2) * 2e-5)), 20 * np.log10(y_fit), label='Fitted Curve', color='blue')
        plt.xlabel('Level (dB peSPL)')
        plt.ylabel('Magnitude (dB re 1)')
        plt.plot(Lvect[start_row:] + 20, 0 - Lvect[start_row:], linestyle='--', label='Max slope -1 dB')
        
        # Set the title with ith value from FchosenLin and the starting row
        plt.title(f'Column {col_idx}: FchosenLin = {FchosenLin[col_idx]}, Start Row = {start_row}')
        
        # Write results onto the graph
        results_text = (
            f"Peak strength, G0 = {p_to_dB(Peak_str):.2f} dB\n"
            f"Compression slope, alpha = {Comp_slope:.2f}\n"
            f"Compression Threshold, Act = {20 * np.log10(Comp_thresh / (np.sqrt(2) * 2e-5)):.2f} dB"
        )
        plt.text(0.05, 0.95, results_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
        plt.legend()
        plt.show()
        plt.pause(0.1)  # Ensure the plot renders before continuing

print(start_rows)

# Prepare data to be saved, including chosen start_rows
fit_results['FchosenLin'] = FchosenLin
fit_results['FchosenNL'] = FchosenNL

# File name for saving
file_name = 'Estimace/CEOAEfit_results_' + subjN + '.mat'

# Save the data into a .mat file
savemat(file_name, fit_results)

print(f"Data successfully saved to {file_name}")
