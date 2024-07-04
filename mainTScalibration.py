

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 23:05:38 2023

@author: vencov
"""

import numpy as np
from scipy.io import loadmat

import datetime
import matplotlib.pyplot as plt
plt.close('all')
#caldata = {'Hinear1':Hinear1,'Hinear2':Hinear2,'fxinear':fxinear,'y1':y1,'y2':y2,'AmpChirp':AmpChirp,'Nsamp':Nsamp,'TL':TL}

#%% load tube responses
plt.close('all')
def loadTubeResp(Filename):

    #Tube01 = loadmat('Calibration_files/Files/TScalTube01.mat')
    #TubeLLS = loadmat('Calibration_files/Files/TScalLLS.mat')
    Tube = loadmat(Filename)
    fx = Tube['fxinear'].flatten()
    H1T01 = Tube['Hinear1'].flatten()
    H2T01 = Tube['Hinear2'].flatten()
    AmpChirp = Tube['AmpChirp'][0][0]
    Pe01 = H1T01*AmpChirp
    Pe02 = H2T01*AmpChirp
    return H1T01, H2T01, fx, Pe01, Pe02, Tube['TL'][0][0]   # return freq responses for both speakers


H1T01, H2T01, fxT01, Pe1T01, Pe2T01, TL1 = loadTubeResp('Calibration_files/Files/TScalTube01_040724vPH.mat')
H1T02, H2T02, fxT02, Pe1T02, Pe2T02, TL2 = loadTubeResp('Calibration_files/Files/TScalTube02_040724vPH.mat')
H1T03, H2T03, fxT03, Pe1T03, Pe2T03, TL3 = loadTubeResp('Calibration_files/Files/TScalTube03_040724vPH.mat')
H1T04, H2T04, fxT04, Pe1T04, Pe2T04, TL4 = loadTubeResp('Calibration_files/Files/TScalTube04_040724vPH.mat')
H1T05, H2T05, fxT05, Pe1T05, Pe2T05, TL5 = loadTubeResp('Calibration_files/Files/TScalTube05_040724vPH.mat')
H1LLS, H2LLS, fxLLS, Pe1LLT, Pe2LLT, TLLL = loadTubeResp('Calibration_files/Files/TScalLLT_040724vPH.mat')
    
fig,ax = plt.subplots()
ax.plot(fxLLS,20*np.log10(np.abs(H1LLS)))
ax.plot(fxLLS,20*np.log10(np.abs(H2LLS)))
ax.set_title('Long Lossy Tube responses')
ax.legend(('Speaker 1','Speaker 2'))


fig,ax = plt.subplots()
ax.plot(fxLLS,20*np.log10(np.abs(Pe1T01)))
ax.plot(fxLLS,20*np.log10(np.abs(Pe2T01)))
ax.set_title('Long Lossy Tube responses')
ax.legend(('Speaker 1','Speaker 2'))

fig,ax = plt.subplots()
ax.plot(fxLLS,20*np.log10(np.abs(Pe1T02)))
ax.plot(fxLLS,20*np.log10(np.abs(Pe2T02)))
ax.set_title('Long Lossy Tube responses')
ax.legend(('Speaker 1','Speaker 2'))


fig,ax = plt.subplots()
ax.plot(fxLLS,20*np.log10(np.abs(Pe1T03)))
ax.plot(fxLLS,20*np.log10(np.abs(Pe2T03)))
ax.set_title('Long Lossy Tube responses')
ax.legend(('Speaker 1','Speaker 2'))


fig,ax = plt.subplots()
ax.plot(fxLLS,20*np.log10(np.abs(Pe1T04)))
ax.plot(fxLLS,20*np.log10(np.abs(Pe2T04)))
ax.set_title('Long Lossy Tube responses')
ax.legend(('Speaker 1','Speaker 2'))


fig,ax = plt.subplots()
ax.plot(fxLLS,20*np.log10(np.abs(Pe1T05)))
ax.plot(fxLLS,20*np.log10(np.abs(Pe2T05)))
ax.set_title('Long Lossy Tube responses')
ax.legend(('Speaker 1','Speaker 2'))



#%%


def ZinTubeLambda(fx,L):

    rho =  1.1769e-3 # g/cm3 air density
    eta = 1.846e-4 # g/scm shear viscosity coefficient
    a = 0.8/2 # cm radius of tubes
    c = 3.4723e4 # cm/s speed of sound 
    
    Ro = rho*c/(np.pi*a**2)  # characteristic impedance of acoustic transmission line
    
    #%fx = 0:1:44.1e3
    
    rv = a*np.sqrt(rho*fx*2*np.pi/eta) #ratio of tube radius to viscous boundary layer
    
    Zci = Ro*(1+0.369*rv**(-1)) - 1j*Ro*(0.369*rv**(-1) + 1.149*rv**(-2) + 0.303*rv**(-3))
    
    alpha = (2*np.pi*fx/c)*(1.045*rv**(-1) + 1.080*rv**(-2) + 0.750*rv**(-3))
    vp = 1/(c**(-1)*(1 + 1.045*rv**(-1)))
    Lambda = alpha + 1j*2*np.pi*fx/vp
    
    Zi = Zci*1/np.tanh(L*Lambda); 
    
    return Lambda, Zi


Lambda, Zci = ZinTubeLambda(fxT01,TL1)  # calculate Lambda parameter for tube length estimation

from scipy.signal import savgol_filter

fig,ax = plt.subplots()
ax.plot(20*np.log10(np.abs(H1T01/H1LLS)))
ax.plot(20*np.log10(savgol_filter(np.abs(H1T01/H1LLS),40,3)))

def find_local_maxima(vector, threshold=3):
    local_maxima = []

    for i in range(1, len(vector) - 1):
        if vector[i] > vector[i - 1] and vector[i] > vector[i + 1] and vector[i] > threshold and i>200:
            local_maxima.append(i)

    return local_maxima

#IdxMax = find_local_maxima(savgol_filter(np.abs(H1T01/H1LLS),20,3))
# find freq indexes for lambda/2 maxima

nH1T01 = np.abs(H1T01/H1LLS); nH2T01 = np.abs(H2T01/H2LLS)
nH1T02 = np.abs(H2T02/H1LLS); nH2T02 = np.abs(H2T02/H2LLS)
nH1T03 = np.abs(H2T03/H1LLS); nH2T03 = np.abs(H2T03/H2LLS)
nH1T04 = np.abs(H2T04/H1LLS); nH2T04 = np.abs(H2T04/H2LLS)
nH1T05 = np.abs(H2T05/H1LLS); nH2T05 = np.abs(H2T05/H2LLS)

idxLH_H1T01= find_local_maxima(nH1T01); idxLH_H2T01= find_local_maxima(nH2T01);
nT1 = 5  # number of lambda/2 peaks for tube length estimation

TL_H1T01 = [np.pi/np.imag(Lambda[idxLH_H1T01[i]])*(i+1) for i in range(nT1)]
TL_H2T01 = [np.pi/np.imag(Lambda[idxLH_H2T01[i]])*(i+1) for i in range(nT1)]

nT2 = 5  # number of lambda/2 peaks for tube length estimation
idxLH_H1T02= find_local_maxima(nH1T02); idxLH_H2T02= find_local_maxima(nH2T02);
TL_H1T02 = [np.pi/np.imag(Lambda[idxLH_H1T02[i]])*(i+1) for i in range(nT2)]
TL_H2T02 = [np.pi/np.imag(Lambda[idxLH_H2T02[i]])*(i+1) for i in range(nT2)]

nT3 = 3  # number of lambda/2 peaks for tube length estimation
idxLH_H1T03= find_local_maxima(nH1T03,3); idxLH_H2T03= find_local_maxima(nH2T03,3);
TL_H1T03 = [np.pi/np.imag(Lambda[idxLH_H1T03[i]])*(i+1) for i in range(nT3)]
TL_H2T03 = [np.pi/np.imag(Lambda[idxLH_H2T03[i]])*(i+1) for i in range(nT3)]

nT4 = 2  # number of lambda/2 peaks for tube length estimation
idxLH_H1T04= find_local_maxima(nH1T04); idxLH_H2T04= find_local_maxima(nH2T04);
TL_H1T04 = [np.pi/np.imag(Lambda[idxLH_H1T04[i]])*(i+1) for i in range(nT4)]
TL_H2T04 = [np.pi/np.imag(Lambda[idxLH_H2T04[i]])*(i+1) for i in range(nT4)]

nT5 = 1  # number of lambda/2 peaks for tube length estimation
idxLH_H1T05= find_local_maxima(nH1T05); idxLH_H2T05= find_local_maxima(nH2T05);
TL_H1T05 = [np.pi/np.imag(Lambda[idxLH_H1T05[i]])*(i+1) for i in range(nT5)]
TL_H2T05 = [np.pi/np.imag(Lambda[idxLH_H2T05[i]])*(i+1) for i in range(nT5)]

fig,(ax1,ax2,ax3) = plt.subplots(3,2)

ax1[0].plot(fxT01,20*np.log10(nH1T01))
ax1[0].plot(fxT01,20*np.log10(nH2T01))
ax1[0].plot(fxT01[idxLH_H1T01],20*np.log10(nH1T01[idxLH_H1T01]),'x')
ax1[0].plot(fxT01[idxLH_H2T01],20*np.log10(nH2T01[idxLH_H2T01]),'x')
ax1[0].set_title('T01')
ax1[1].plot(fxT02,20*np.log10(nH1T02))
ax1[1].plot(fxT02,20*np.log10(nH2T02))
ax1[1].plot(fxT02[idxLH_H1T02],20*np.log10(nH1T02[idxLH_H1T02]),'x')
ax1[1].plot(fxT02[idxLH_H2T02],20*np.log10(nH2T02[idxLH_H2T02]),'x')
ax1[1].set_title('T01')

ax2[0].plot(fxT03,20*np.log10(nH1T03))
ax2[0].plot(fxT03,20*np.log10(nH2T03))
ax2[0].plot(fxT03[idxLH_H1T03],20*np.log10(nH1T03[idxLH_H1T03]),'x')
ax2[0].plot(fxT03[idxLH_H2T03],20*np.log10(nH2T03[idxLH_H2T03]),'x')

ax2[1].plot(fxT04,20*np.log10(nH1T04))
ax2[1].plot(fxT04,20*np.log10(nH2T04))
ax2[1].plot(fxT04[idxLH_H1T04],20*np.log10(nH1T04[idxLH_H1T04]),'x')
ax2[1].plot(fxT04[idxLH_H2T04],20*np.log10(nH2T04[idxLH_H2T04]),'x')


ax3[0].plot(fxT05,20*np.log10(nH1T05))
ax3[0].plot(fxT05,20*np.log10(nH2T05))
ax3[0].plot(fxT05[idxLH_H1T05],20*np.log10(nH1T05[idxLH_H1T05]),'x')
ax3[0].plot(fxT05[idxLH_H2T05],20*np.log10(nH2T05[idxLH_H2T05]),'x')

TLeH1T01 = np.mean(TL_H1T01); TLeH2T01 = np.mean(TL_H2T01)
TLeH1T02 = np.mean(TL_H1T02); TLeH2T02 = np.mean(TL_H2T02)
TLeH1T03 = np.mean(TL_H1T03); TLeH2T03 = np.mean(TL_H2T03);
TLeH1T04 = np.mean(TL_H1T04); TLeH2T04 = np.mean(TL_H2T04);
TLeH1T05 = np.mean(TL_H1T05); TLeH2T05 = np.mean(TL_H2T05);

TL1cm = TL1/10; TL2cm=TL2/10; TL3cm=TL3/10; TL4cm=TL4/10; TL5cm=TL5/10

print(f"Tube01: measured TL (cm): {TL1cm:.2f}, estimated TL (cm): {TLeH1T01:.2f} and {TLeH2T01:.2f}")
print(f"Tube02: measured TL (cm): {TL2cm:.2f}, estimated TL (cm): {TLeH1T02:.2f} and {TLeH2T02:.2f}")
print(f"Tube03: measured TL (cm): {TL3cm:.2f}, estimated TL (cm): {TLeH1T03:.2f} and {TLeH2T03:.2f}")
print(f"Tube04: measured TL (cm): {TL4cm:.2f}, estimated TL (cm): {TLeH1T04:.2f} and {TLeH2T04:.2f}")
print(f"Tube05: measured TL (cm): {TL5cm:.2f}, estimated TL (cm): {TLeH1T05:.2f} and {TLeH2T05:.2f}")


#%% optimization

def find_nearest_index(array, value):
    """
    Find the index of the element in the array that is closest to the given value.

    Parameters:
    - array: NumPy array
    - value: The target value

    Returns:
    - index: The index of the element in the array closest to the given value
    """
    array = np.asarray(array)
    index = np.abs(array - value).argmin()
    return index


import scipy.optimize

def ThevenParFce2(Lta):
    # Define lengths of tubes
    Lzas = 0
    Lt = np.array([L - Lzas for L in Lta])

    M = len(Lta) # Number of tubes

    # Frequency range for estimation
    Fmin = 800
    Fmax = 5000
    
    # Extract relevant frequency data
    #fx = np.arange(len(YT1)) * fsamp / len(YT1)
    idxFmin = find_nearest_index(fx, Fmin)
    idxFmax = find_nearest_index(fx, Fmax)
    #print(idxFmin)
    #print(idxFmax)
    # Initialize Zi matrix
    Zi = np.zeros((M, idxFmax-idxFmin),dtype=complex)

    # Build impedance matrix Zi for each tube
    for k in range(M):
        Lambda, Z1 = ZinTubeLambda(fx[idxFmin:idxFmax], Lt[k])
        Zi[k, :] = Z1

    # Load measured pressure data (Pi) for each tube
    Pi = np.vstack([YT1[idxFmin:idxFmax], YT2[idxFmin:idxFmax], YT3[idxFmin:idxFmax], YT4[idxFmin:idxFmax], YT5[idxFmin:idxFmax]])

    Delta = np.sum(np.abs(Zi) ** 2,0) * np.sum(np.abs(Pi) ** 2,0) - np.sum(np.conj(Pi) * Zi,0) * np.sum(np.conj(Zi) * Pi,0)

    # Jacobian matrix
    Jac = np.vstack((np.hstack((np.reshape(np.sum(np.abs(Pi) ** 2,0),(idxFmax-idxFmin,1)), np.reshape(-np.sum(np.conj(Zi) * Pi,0),(idxFmax-idxFmin,1)))),
        (np.hstack((np.reshape(np.sum(Zi * np.conj(Pi),0),(idxFmax-idxFmin,1)), np.reshape(-np.sum(np.abs(Zi) ** 2,0),(idxFmax-idxFmin,1)))))))

    # Jacobian effect matrix
    Jec = np.vstack((np.sum(Pi * np.abs(Zi) ** 2,0), np.sum(Zi * np.abs(Pi) ** 2,0)))
    
    # Calculate Thevenin parameters P0 and Z0
    #P0Z0 = np.linalg.solve(Jac, Jec)
    P0Z0 = 1/Delta*(Jac@Jec);
    P0 = np.diag(P0Z0[:idxFmax-idxFmin,:])
    Z0 = np.diag(P0Z0[idxFmax-idxFmin:,:])

    # Calculate PhiA
    #PhiA = np.zeros_like(fx[idxFmin:idxFmax])
    #for k in range(M):
    #    Phi = np.abs(Zi[k, :] * P0 - Pi[k, :] * Z0 - Pi[k, :] * Zi[k, :]) ** 2
    #    PhiA += Phi

    # Calculate EtaT (normalized error)
    Pip = P0 * Zi / (Z0 + Zi)
    EtaT = np.sum(np.sum(np.abs(Pi - Pip) ** 2,0)) / np.sum(np.sum(np.abs(Pi) ** 2,0))

    return EtaT

fx = fxT01  # frequency axis for which the pressure in the tubes was measured
# speaker 01
YT1 = Pe1T01; YT2 = Pe1T02; YT3 = Pe1T03; YT4 = Pe1T04; YT5 = Pe1T05

fsamp = 44100
LT0 = [TLeH1T01,TLeH1T02,TLeH1T03,TLeH1T04,TLeH1T05]  # initial condition

xopt1 = scipy.optimize.fmin(func=ThevenParFce2, x0=LT0, xtol=0.01)

# speaker 02
YT1 = Pe2T01; YT2 = Pe2T02; YT3 = Pe2T03; YT4 = Pe2T04; YT5 = Pe2T05
xopt2 = scipy.optimize.fmin(func=ThevenParFce2, x0=LT0, xtol=0.01)


#%% calculate Psrc and Zsrc for optimal values of tube lengths


def ThevenParFceGetForLta(Lta,Fmin,Fmax):
    # Define lengths of tubes
    #Lzas = 0
    #Lt = np.array([L - Lzas for L in Lta])

    M = len(Lta) # Number of tubes

    # Frequency range for estimation
    
    
    # Extract relevant frequency data
    #fx = np.arange(len(YT1)) * fsamp / len(YT1)
    #idxFmin = find_nearest_index(fx, Fmin)
    #idxFmax = find_nearest_index(fx, Fmax)
    #print(idxFmin)
    #print(idxFmax)
    idxFmin = 0
    idxFmax = len(fx)
    # Initialize Zi matrix
    Zi = np.zeros((M, idxFmax-idxFmin),dtype=complex)

    # Build impedance matrix Zi for each tube
    for k in range(M):
        Lambda, Z1 = ZinTubeLambda(fx[idxFmin:idxFmax], Lta[k])
        Zi[k, :] = Z1

    # Load measured pressure data (Pi) for each tube
    Pi = np.vstack([YT1[idxFmin:idxFmax], YT2[idxFmin:idxFmax], YT3[idxFmin:idxFmax], YT4[idxFmin:idxFmax], YT5[idxFmin:idxFmax]])

    Delta = np.sum(np.abs(Zi) ** 2,0) * np.sum(np.abs(Pi) ** 2,0) - np.sum(np.conj(Pi) * Zi,0) * np.sum(np.conj(Zi) * Pi,0)

    # Jacobian matrix
    Jac = np.vstack((np.hstack((np.reshape(np.sum(np.abs(Pi) ** 2,0),(idxFmax-idxFmin,1)), np.reshape(-np.sum(np.conj(Zi) * Pi,0),(idxFmax-idxFmin,1)))),
        (np.hstack((np.reshape(np.sum(Zi * np.conj(Pi),0),(idxFmax-idxFmin,1)), np.reshape(-np.sum(np.abs(Zi) ** 2,0),(idxFmax-idxFmin,1)))))))

    # Jacobian effect matrix
    Jec = np.vstack((np.sum(Pi * np.abs(Zi) ** 2,0), np.sum(Zi * np.abs(Pi) ** 2,0)))
    
    # Calculate Thevenin parameters P0 and Z0
    #P0Z0 = np.linalg.solve(Jac, Jec)
    P0Z0 = 1/Delta*(Jac@Jec);
    P0 = np.diag(P0Z0[:idxFmax-idxFmin,:])
    Z0 = np.diag(P0Z0[idxFmax-idxFmin:,:])

    # Calculate PhiA
    #PhiA = np.zeros_like(fx[idxFmin:idxFmax])
    #for k in range(M):
    #    Phi = np.abs(Zi[k, :] * P0 - Pi[k, :] * Z0 - Pi[k, :] * Zi[k, :]) ** 2
    #    PhiA += Phi

    # Calculate EtaT (normalized error)
    #Pip = P0 * Zi / (Z0 + Zi)
    #EtaT = np.sum(np.sum(np.abs(Pi - Pip) ** 2,0)) / np.sum(np.sum(np.abs(Pi) ** 2,0))
    fxS = fx[idxFmin:idxFmax]
    return P0,Z0,fxS


Psrc1,Zsrc1,fxS = ThevenParFceGetForLta(xopt1, 100, 12000)
Psrc2,Zsrc2,fxS = ThevenParFceGetForLta(xopt2, 100, 12000)
THpar = {}
THpar['Psrc1'] = Psrc1; THpar['Psrc2'] = Psrc2; THpar['Zsrc1'] = Zsrc1; THpar['Zsrc2'] = Zsrc2; THpar['fxTS'] = fxS

from scipy.io import savemat
savemat('Calibration_files/Files/THpar_040724.mat',THpar)

fig,ax = plt.subplots()
ax.plot(fxS,np.abs(Psrc1))
ax.plot(fxS,np.abs(Psrc2))
fig,ax = plt.subplots()
ax.plot(fxS,np.abs(Zsrc1))
ax.plot(fxS,np.abs(Zsrc2))

#%% calculation of 

def loadPecs(filename,Fmin,Fmax):
    data1 = loadmat(filename)  # first speaker
    Pecs1 = data1['Hinear1'][0]*data1['AmpChirp'][0][0]
    
    Pecs2 = data1['Hinear1'][0]*data1['AmpChirp'][0][0]
    
    fxPecs = data1['fxinear'][0]
        
    # Extract relevant frequency data
    #fx = np.arange(len(YT1)) * fsamp / len(YT1)
    idxFmin = find_nearest_index(fxPecs, Fmin)
    idxFmax = find_nearest_index(fxPecs, Fmax)
    idxFmin = 0
    
    fxPecs = fxPecs[idxFmin:idxFmax+1]
    Pecs1 = Pecs1[idxFmin:idxFmax+1]
    Pecs2 = Pecs2[idxFmin:idxFmax+1]
    
    return Pecs1, Pecs2, fxPecs
    
    
Pecs1, Pecs2, fxPecs = loadPecs('Calibration_files/Files/Syringe01.mat',100,20000)
Zec = Zsrc1*Pecs1/(Psrc - Pecs1) # impedance of ear canal

fig,ax = plt.subplots()
ax.plot(fxPecs,np.abs(Zec))
#smW = 30;
#Zec = (smooth(real(Zec),smW,'loess') + sqrt(-1)*smooth(imag(Zec),smW,'loess')).';


from scipy.interpolate import interp1d
from scipy.signal import blackman

# Constants
rho = 1.1769e-3  # g/cm^3 air density
eta = 1.846e-4  # g/scm shear viscosity coefficient
a = 0.8 / 2  # cm radius of tubes
c = 3.4723e4  # cm/s speed of sound
k = 1
kC = 1

# Parameters for estimation
num_iterations = 30
Fmin = 16e3
smW = 5

fsUp = 200000
fsOrig = 40000

# Estimation of Zsurge
for i in range(num_iterations):
    
    # Initialize Zsurge
    Zsurge = k * rho * c / (np.pi * a ** 2)

    R = (Zec - Zsurge) / (Zec + Zsurge)
    # Reflection coefficient
    R = (Zec - Zsurge) / (Zec + Zsurge)

    # reflection coefficient has samples from 0 to 20kHz, now upsample to 200 kHz
    fxUp = np.linspace(0, fsUp / 2, len(fxPecs) * (fsUp // fsOrig))

    # Zero-pad the frequency spectrum for upsampling
    upsampled_length = len(fxUp)
    Rup = np.pad(R, (0, upsampled_length - len(R)))

    # Make Blackman window
    UpIdx = np.argmax(fxUp >= 17e3)
    if not UpIdx % 2:
        UpIdx += 1

    BMWwidth = UpIdx + UpIdx - 1
    BMwin = blackman(BMWwidth)

    BMwin = BMwin[len(BMwin)//2:]
    BMwinA = np.concatenate((BMwin,np.zeros(len(Rup)-len(BMwin))))
    
    Rup = BMwinA*Rup
    
    # Smoothing
    #R = (np.real(R) + 1j * np.imag(R))
    #R = np.convolve(R, np.ones(smW) / smW, mode='same')

 #   Rzeros = np.zeros(4 * len(fxI) // 2)
 #   Rzeros[:len(BMwin) // 2] = np.flipud(BMwin[:len(BMwin) // 2])

    #Rappz = np.concatenate([np.zeros(FminId - 1), R, np.zeros(len(Rzeros) - len(R) - FminId + 1)])
    #fxIex = np.arange(1, len(R) + FminId)
    #fxPex = np.arange(FminId, len(R) + FminId)

    #Rir = interp1d(fxPex, np.real(R), kind='spline', fill_value='extrapolate')(fxIex)
    #Rim = interp1d(fxPex, np.imag(R), kind='spline', fill_value='extrapolate')(fxIex)
    #Rint = Rir + 1j * Rim
    #RappzI = np.concatenate([Rint, np.zeros(len(Rzeros) - len(Rint))])

    #RappzI[0] = np.abs(RappzI[0])
    #Rall = np.concatenate([Rzeros * Rappz, np.flipud(Rzeros[1:]) * np.flipud(np.conj(Rappz[1:]))])
    #RallI = np.concatenate([Rzeros * RappzI, np.flipud(Rzeros[1:]) * np.flipud(np.conj(RappzI[1:]))])

    Rtime = np.fft.irfft(Rup)/2
    RtimeNeg = np.flipud(Rtime)
    RtimeNeg = RtimeNeg[:len(Rtime)//2+1]
    
    Rtime = Rtime[:len(Rtime)//2+1] + RtimeNeg

    k = k + 40 * Rtime[0]
    print(k)

fig,ax = plt.subplots()
ax.plot(Rtime)

fig,ax = plt.subplots()
ax.plot(fxPecs,(np.abs(R)))
# Update Pecs
#Pecs_withoutR = Pecs
Pecs = Pecs1 / (1 + R)

fig,ax =plt.subplots()
ax.plot(fxPecs,np.abs(Pecs/Pe1LLT))
#ax.plot(fxPecs,np.abs(Pecs1))

