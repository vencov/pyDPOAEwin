# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:26:13 2024

@author: audiobunka
"""

import numpy as np
import scipy.signal as signal
from scipy.signal.windows import blackman, tukey
from UserModules.pyUtilities import butter_highpass_filter, rfft
from UserModules.pyRMEsd import choose_soundcard, RMEplayrec
from scipy.signal import savgol_filter  # import savitzky golay filter
from scipy.io import loadmat
import time

import matplotlib.pylab as plt




def generateClickTrainMEMR(cfname,Nclicks,Npulse=2048):

    from scipy.io import loadmat

    cf = loadmat(cfname)
    clickIn = cf['clickIn'][0][:400]
    clickIn = clickIn/max(clickIn)
    
    zero_pad = np.zeros(Npulse-len(clickIn))
    clickIn = np.concatenate((clickIn,zero_pad))
    clickT = np.tile(clickIn,Nclicks)


    return clickT


def generateClickTrain(cfname,Nclicks,Npulse=2048):

    from scipy.io import loadmat

    cf = loadmat(cfname)
    clickIn = cf['clickIn'][0]
    clickIn = clickIn/max(clickIn)
    zero_pad = np.zeros(Npulse-len(clickIn))
    clickIn = np.concatenate((clickIn,zero_pad))
    clickT = np.tile(clickIn,Nclicks)


    return clickT

def generateClickTrainShuffledInt(cfname,Nclicks,LevS,Npulse=2048):

    
    from scipy.io import loadmat

    cf = loadmat(cfname)
    clickIn = abs(cf['clickIn'][0])
    clickIn = clickIn/max(clickIn)
    zero_pad = np.zeros(Npulse-len(clickIn))
    clickIn = np.concatenate((clickIn,zero_pad))
    Env = np.ones_like(clickIn)
    for i in range(1,len(LevS)):
        Env = np.concatenate((Env,np.ones_like(clickIn)*10**((LevS[i])/20)))
    clickT = np.tile(clickIn,Nclicks)
    clickEnv = np.tile(Env,int(Nclicks/len(LevS)))
    

    return clickT, clickEnv


def generateClickTrainCmp(cfname,Nclicks,LevS,Npulse=2048):

    
    from scipy.io import loadmat

    cf = loadmat(cfname)
    clickIn = cf['clickIn'][0]
    clickIn = clickIn/max(clickIn)
    zero_pad = np.zeros(Npulse-len(clickIn))
    clickIn = np.concatenate((clickIn,zero_pad))
    #Env = np.ones_like(clickIn)*10**((LevS[0]-max(LevS))/20)  # incorrect until 25 March 24
    Env = np.ones_like(clickIn)  # correct after 25 March 24
    for i in range(1,len(LevS)):
        Env = np.concatenate((Env,np.ones_like(clickIn)*10**((LevS[i]-min(LevS))/20)))
    clickT = np.tile(clickIn,Nclicks)
    clickEnv = np.tile(Env,int(Nclicks/len(LevS)))
    

    return clickT, clickEnv
        


def makeChirp(f1,f2,Nsamp,fsamp):
    '''
    generate a short chirp (linearly swept sine)
    '''
    T = Nsamp/fsamp  # duration of the chirp
    
    tx = np.arange(0,Nsamp/fsamp,1/fsamp)  # time axis

    A = 1
    T = Nsamp/fsamp
    
    y = np.zeros_like(tx)
    for k in range(len(tx)):
        y[k] = A*np.sin(2*np.pi*(f1+(f2-f1)/(2*(T))*tx[k])*tx[k])

    return y


def makeChirpTrain(f1,f2,Nsamp,fsamp,Nchirps):
    '''
    generate a train of chirps (linearly swept sines)
    '''

    chirp = makeChirp(f1,f2,Nsamp,fsamp) # make one chirp 
    chirptrain = np.tile(chirp,(Nchirps,))  # repeate the chirp Nchirp times
    
    return chirptrain, chirp



def generateSSSPhase(fs,fstart,fstop,phase,octpersec,Level,channel):
    '''
    generate synchronized swept sine with chosen phase
    '''
    if fstop>fstart:
        numofoct = np.log2(fstop/fstart)
    else:
        numofoct = np.log2(fstart/fstop)

    T = numofoct/octpersec         # time duration of the sweep
    #print(T)
    
    #fade = [441, 441]   # samlpes to fade-in and fade-out the input signal
    fade = [int(1024/(octpersec/2)), int(2400/(octpersec/2))]   # samlpes to fade-in and fade-out the input signal (related to swept rate 4.4.25)
    L = T/np.log(fstop/fstart)
    t = np.arange(0,np.round(fs*T-1)/fs,1/fs)  # time axis
    s = np.sin(2*np.pi*(fstart)*L*np.exp(t/L)+phase)       # generated swept-sine signal

    #p0 = 2e-5
    #s = p0*10**(Level/20)*s/np.sqrt(np.mean(s**2))
    # fade-in the input signal
    if fade[0]>0:
        s[0:fade[0]] = s[0:fade[0]] * ((-np.cos(np.arange(fade[0])/fade[0]*np.pi)+1) / 2)

    # fade-out the input signal
    if fade[1]>0:
        s[-fade[1]:] = s[-fade[1]:] *  ((np.cos(np.arange(fade[1])/fade[1]*np.pi)+1) / 2)

    #s = np.pad(s, (0, 8192), 'constant')  # append zeros
    s = np.pad(s, (0, 16384), 'constant')  # append zeros

    fft_len = len(s) # number of samples for fft


    probecal = loadmat('Calibration_files/Files/InEarCalData.mat')
    if channel==1:
        Hprobe = probecal['Hinear1']
        fxprobe = probecal['fxinear']
    elif channel==2:
        Hprobe = probecal['Hinear2']
        fxprobe = probecal['fxinear']

    Sin = np.fft.rfft(s,fft_len)
    axe_w = np.linspace(0, np.pi, num=int(fft_len/2+1))
    fxSin = axe_w/(np.pi)*fs/2
    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Sin))
    #draw_DPOAE(fxSin, Sin)

    Hrint = np.interp(fxSin,fxprobe.flatten(),Hprobe.flatten())

    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Hrint))
    #draw_DPOAE(fxSin, Hrint) 

    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Sin/Hrint))
    #draw_DPOAE(fxSin, Sin/Hrint)

    SinC = Sin/Hrint
 
    sig = np.fft.irfft(SinC)

    s = 10**(Level/20)*np.sqrt(2)*2e-5*sig
    print('max value of signal: ',max(abs(s)))
    '''
    chan_in = 1 # channel of the sound card for recording
    chan_out = 3  # channel of the sound card for playing
    y = sd.playrec(s, samplerate=fs, channels=1, input_mapping=chan_in, output_mapping=chan_out, blocking=True)

    y = y.flatten()'''
    return s


def generateDPOAEstimulusPhase(f2f1,fs,f2start,f2stop,phase1,phase2,octpersec,L1,L2):
    '''
    generate stimulus to evoke DPOAEs (two synchronized swept sines)
    you can define initial phases of the tones
    '''
    f1start = f2start/f2f1
    f1stop = f2stop/f2f1

    tone1 = generateSSSPhase(fs,f1start,f1stop,phase1,octpersec,L1,1)

    tone2 = generateSSSPhase(fs,f2start,f2stop,phase2,octpersec,L2,2)

    
    return tone1, tone2  # returning back two tones separately for presentation by each speaker





def synchronized_swept_sine_spectra_shifted(f1s,L1sw,fsamp,Nsamp,tshift):
    '''
    returns SSS in freq domain
    '''
    # frequency axis
    #f = np.linspace(0,fsamp,Nsamp,endpoint=False)
    f = np.linspace(0,fsamp/2,num=round(Nsamp/2)+1)  # half of the spectrum
    # spectra of the synchronized swept-sine signal [1, Eq.(42)]
    X = 1/2*np.emath.sqrt(L1sw/f)*np.exp(1j*2*np.pi*f*L1sw*(1 - np.log(f/f1s)) - 1j*np.pi/4)*np.exp(-1j*2*np.pi*f*tshift)
    X[0] = np.inf # protect from devision by zero
    return (X,f)



def processChirpResp(recordedchirp1,recordedchirp2,latency_SC,fsamp,Nsamp,Nchirps):
    
    
    y_stripSClat1 = recordedchirp1[latency_SC:] # remove the SC latency
    y_reshaped1 = np.reshape(y_stripSClat1[:Nchirps*Nsamp],(Nchirps,Nsamp))
    
    y_stripSClat2 = recordedchirp2[latency_SC:] # remove the SC latency
    y_reshaped2 = np.reshape(y_stripSClat2[:Nchirps*Nsamp],(Nchirps,Nsamp))

    # take the mean across some responses:
    Nchskip = 10 # skip first ten chirps

    y_mean1 = np.mean(y_reshaped1.T[:,Nchskip:],axis=1)
    y_mean2 = np.mean(y_reshaped2.T[:,Nchskip:],axis=1)
    
    # calculate spectrum

    Nmean1 = len(y_mean1)  # length of the data
    NmeanUp1 = int(2**np.ceil(2+np.log2(Nmean1)))  # interpolated length of the spectrum
    ChResp1 =  np.fft.rfft(y_mean1,NmeanUp1)/NmeanUp1
    fxCh1 = np.arange(NmeanUp1)*fsamp/NmeanUp1
    fx = fxCh1[:NmeanUp1//2+1]
    
    ChResp2 =  np.fft.rfft(y_mean2,NmeanUp1)/NmeanUp1
    
    return ChResp1, ChResp2, fx

def sendChirpToEar(*,AmpChirp=0.01,fsamp=44100,MicGain=40,Nchirps=300,buffersize=2048,latency_SC=8236,SC=10):
    '''
    creates a chirptrain and sends it into the sound card
    recorded response is used for calibration of OAE probe
    Input parameters:
    AmpChirp .... Signal amplitude
    fsamp .... sampling frequency (constructed for 44100 Hz)
    MicGain .... gain on the OAE probe
    Nchirps .... number of chirps in the chirptrain
    buffersize .... number of samples in the chirp (buffer)
    latency_SC .... sound card latency
    '''
    #import numpy as np
    #import sys
    #import os
    #import matplotlib.pyplot as plt

    #current = os.path.dirname(os.path.realpath(__file__)) # find path to the current file
    #UMdir = os.path.dirname(current)+'/UserModules'  # set path to the UserModules folder
    #sys.path.append(UMdir) # add the path to the module into sys.path
     # import needed functions from pyRMEsd module
    #from pyDPOAEmodule import makeChirpTrain  # import needed functions from pyRMEsd module

    #fsamp = 44100
    plotflag = 0  # plot responses? for debuging purposes set to 1
    f1 = 0  # start frequency
    f2 = fsamp/2 # stop frequency
    Nsamp = buffersize  # number of samples in the chirp
    chirptrain, chirpIn = makeChirpTrain(f1,f2,Nsamp,fsamp,Nchirps)
    # make matrix with 3 columns, each column has signal for each channel: 1, 2, 3
    chirpmatrix1 = np.vstack((chirptrain,np.zeros_like(chirptrain),chirptrain)).T 
    
    # send to soundcard and record
    recordedchirp1 = RMEplayrec(AmpChirp*chirpmatrix1,fsamp,SC=SC,buffersize=buffersize)

    chirpmatrix2 = np.vstack((np.zeros_like(chirptrain),chirptrain,chirptrain)).T
    
    time.sleep(0.5)
    recordedchirp2 = RMEplayrec(AmpChirp*chirpmatrix2,fsamp,SC=SC,buffersize=buffersize)
    #recordedchirp2 = recordedchirp1

    # process the recorded chirps

    # high pass filter
    print('msize1:',np.shape(chirpmatrix1))
    print('msize2:',np.shape(chirpmatrix2))

    print('size1:',np.shape(recordedchirp1))
    print('size2:',np.shape(recordedchirp2))
    cutoff = 300 # cutoff frequency for high pass filter to filter out low frequency noise
    recordedchirp1 = butter_highpass_filter(recordedchirp1[:,0], cutoff, fsamp, 1)
    recordedchirp2 = butter_highpass_filter(recordedchirp2[:,0], cutoff, fsamp, 1)
    if plotflag:
        fig,ax = plt.subplots()
        ax.plot(recordedchirp1)
        ax.plot(recordedchirp2)
        ax.title("recorded chirptrains")
        ax.set_xlabel("samples")
        ax.set_ylabel("amplitude")
        plt.show()
    

    #latency_SC = 5170 # sound card latency (can be removed from the begining of recorded signal)
    #latency_SC = 8681 # sound card latency (can be removed from the begining of recorded signal)

    y_stripSClat1 = recordedchirp1[latency_SC:] # remove the SC latency
    y_reshaped1 = np.reshape(y_stripSClat1[:Nchirps*Nsamp],(Nchirps,Nsamp))
    
    y_stripSClat2 = recordedchirp2[latency_SC:] # remove the SC latency
    y_reshaped2 = np.reshape(y_stripSClat2[:Nchirps*Nsamp],(Nchirps,Nsamp))

    # take the mean across some responses:
    Nchskip = 10 # skip first ten chirps

    y_mean1 = np.mean(y_reshaped1.T[:,Nchskip:],axis=1)
    y_mean2 = np.mean(y_reshaped2.T[:,Nchskip:],axis=1)
    
    # artifact rejection (reject frams which are affected by some artifact)
    
    y_dev1 = y_reshaped1.T[:,Nchskip:] - y_mean1[:,np.newaxis]
    y_dev2 = y_reshaped2.T[:,Nchskip:] - y_mean2[:,np.newaxis]
    # Example threshold
    threshold = 0.01  # Adjust this based on your criteria

    # Compute the maximum absolute deviation for each frame
    max_dev_per_frame1 = np.max(np.abs(y_dev1), axis=0)
    max_dev_per_frame2 = np.max(np.abs(y_dev2), axis=0)

    # Identify frames where the deviation exceeds the threshold
    frames_to_skip1 = np.where(max_dev_per_frame1 > threshold)[0]
    frames_to_skip2 = np.where(max_dev_per_frame2 > threshold)[0]

    print("Frames to skip:", frames_to_skip1)
    print("Frames to skip:", frames_to_skip2)
    
    # Select only the columns that are NOT in frames_to_skip
    valid_columns1 = np.setdiff1d(np.arange(y_reshaped1.T[:, Nchskip:].shape[1]), frames_to_skip1)
    valid_columns2 = np.setdiff1d(np.arange(y_reshaped2.T[:, Nchskip:].shape[1]), frames_to_skip2)
    # Compute the mean only for the selected columns
    y_mean1 = np.mean(y_reshaped1.T[:, Nchskip:][:, valid_columns1], axis=1)
    y_mean2 = np.mean(y_reshaped2.T[:, Nchskip:][:, valid_columns2], axis=1)
    
    if plotflag:
        fig,ax = plt.subplots()
        ax.plot(y_mean1)
        ax.plot(y_mean2)
        ax.title("mean value of the recorded chirps")
        ax.set_xlabel("samples")
        ax.set_ylabel("amplitude")
        plt.show()
    #fig,(ax1,ax2) = plt.subplots(2,1)
    #ax1.plot(y_reshaped.T)
    #ax2.plot(y_mean)
    
    # calculate spectrum

    Nmean1 = len(y_mean1)  # length of the data
    NmeanUp1 = int(2**np.ceil(2+np.log2(Nmean1)))  # interpolated length of the spectrum
    ChResp1 =  np.fft.rfft(y_mean1,NmeanUp1)/np.fft.rfft(AmpChirp*chirpIn,NmeanUp1)
    fxCh1 = np.arange(NmeanUp1)*fsamp/NmeanUp1
    fxCh1 = fxCh1[:NmeanUp1//2+1]

    Nmean2 = len(y_mean2)  # length of the data
    NmeanUp2 = int(2**np.ceil(2+np.log2(Nmean2)))  # interpolated length of the spectrum
    ChResp2 =  np.fft.rfft(y_mean2,NmeanUp2)/np.fft.rfft(AmpChirp*chirpIn,NmeanUp2)
    fxCh2 = np.arange(NmeanUp2)*fsamp/NmeanUp2
    fxCh2 = fxCh2[:NmeanUp2//2+1]   

    # to smooth out noise, Savitzky-Golay filter is used
    ChRespAbs1 = savgol_filter(np.abs(ChResp1), 20, 2)
    ChRespImag1 = savgol_filter(np.unwrap(np.angle(ChResp1)), 20, 2)

    ChRespAbs2 = savgol_filter(np.abs(ChResp2), 20, 2)
    ChRespImag2 = savgol_filter(np.unwrap(np.angle(ChResp2)), 20, 2)
      
    # change on 8.3.2024 to remove miccal
    #MicC = loadmat('MicCalCurve.mat')  # load calibration curve for the microphone
    #fx = MicC['fx'][0]
    #Hoaemic = MicC['Hoaemic'][0]
    fx = np.arange(100,18e3,5)

    ChRespI1 = np.interp(fx,fxCh1,ChRespAbs1)*np.exp(1j*np.interp(fx,fxCh1,ChRespImag1))
    ChRespI2 = np.interp(fx,fxCh2,ChRespAbs2)*np.exp(1j*np.interp(fx,fxCh2,ChRespImag2))

    #Hinear1 = ChRespI1/(Hoaemic*(10**(MicGain/20))) # convert recorded response to Pascals
    #Hinear2 = ChRespI2/(Hoaemic*(10**(MicGain/20)))
    Hinear1 = ChRespI1/(0.003*10**(MicGain/20))
    Hinear2 = ChRespI2/(0.003*10**(MicGain/20))


    fxinear = fx
    return Hinear1, Hinear2, fxinear, y_mean1, y_mean2, recordedchirp1, recordedchirp2

def giveRforTScalibration(Zec,fxPecs,num_iterations=20):
    '''
    function which interatively calculates reflectio coeficients from the point of view of the probe
    in the ear canal for calibration purposes
    it is an implementation of the method described in Rasetshwane and Neely (2011) JASA 130:3873
    the method iteratively changes Zsurge in order to decrase reflection coeficients in the time domain
    for t=0
    '''    

    # Constants
    rho = 1.1769e-3  # g/cm^3 air density
    eta = 1.846e-4  # g/scm shear viscosity coefficient
    a = 0.8 / 2  # cm radius of tubes
    c = 3.4723e4  # cm/s speed of sound
    k = 1
    kC = 1

    # Parameters for estimation
    #num_iterations = 30
    Fmin = 16e3
    smW = 5

    fsUp = 200000
    fsOrig = 40000

    # Estimation of Zsurge
    for i in range(num_iterations):
        
        # Initialize Zsurge
        Zsurge = k * rho * c / (np.pi * a ** 2)

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

    return R, Zsurge


def getSClat(fsamp,buffersize,SC = 10):

    # make short chirp
    f1 = 1000
    f2 = 1200
    Nsamp = 2048*20  # cca 500ms
    #fsamp = 44100 # samplinMatN[:,i]g rate must be the same as in jack setup

    #buffersize = 2048  # buffersize for sounddevice stream


    def makeChirp(f1,f2,Nsamp,fsamp):
        T = Nsamp/fsamp  # duration of the chirp
        
        tx = np.arange(0,Nsamp/fsamp,1/fsamp)  # time axis

        A = 1
        T = Nsamp/fsamp
        
        y = np.zeros_like(tx)
        print(len(tx))
        for k in range(len(tx)):
            y[k] = A*np.sin(2*np.pi*(f1+(f2-f1)/(2*(T))*tx[k])*tx[k])

        return y

    y = makeChirp(f1,f2,Nsamp,fsamp)

    # fade in fade out of the chirp
    rampdur = 10e-3 # duration of onset offset ramp

    rampts = np.round(rampdur * fsamp)

    step = np.pi/(rampts-1)
    x= np.arange(0,np.pi+step,step)
    offramp = (1+np.cos(x))/2
    onramp = (1+np.cos(np.flip(x)))/2
    o=np.ones(len(y)-2*len(x))
    wholeramp = np.concatenate((onramp, o, offramp)) # envelope of the entire ramp

    y = y*wholeramp

    # add some zeros
    generated_signal = np.concatenate((np.zeros(4800), 0.1*y, np.zeros(4800)))  # add some zeros to begining

    '''
    fig,ax = plt.subplots()
    ax.plot(generated_signal)
    plt.show()
    '''
    # now send the input to the sound card and record response



    import sys
    import os

    current = os.path.dirname(os.path.realpath(__file__)) # find path to the current file
    UMdir = os.path.dirname(current)+'/UserModules'  # set path to the UserModules folder
    sys.path.append(UMdir) # add the path to the module into sys.path
  

    #SC = choose_soundcard() # choose sound card (which device to use)
    
    print(f'Chosen device is {SC}')
    chan_in = 3
    sig_in = np.tile(generated_signal,(chan_in,1)).T

    recorded_data = RMEplayrec(sig_in,fsamp,SC=SC,buffersize=buffersize)
    #from pyDPOAEmodule import sendDPOAEstimulustoRMEsoundcard

    '''
    #recorded_data = sendDPOAEstimulustoRMEsoundcard(generated_signal,3,3,fsamp)
    #print(np.shape(output))
    fig,ax = plt.subplots()
    ax.plot(recorded_data)
    ax.plot(generated_signal)
    #ax.set_ylim((-0.001,0.001))
    plt.show()
    '''

    # estimate delay


    def lag_finder(y1, y2, sr):
        n = len(y1)

        corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

        delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
        delay = delay_arr[np.argmax(corr)]
        print('y2 is ' + str(delay) + ' seconds behind y1, which is ' + str(np.argmax(corr)-round(n/2)) + ' samples')
        return np.argmax(corr)-round(n/2)
        '''
        plt.figure()
        plt.plot(delay_arr, corr)
        plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
        plt.xlabel('Lag')
        plt.ylabel('Correlation coeff')
        plt.show()
        '''



    latSC = lag_finder(generated_signal,recorded_data[:len(generated_signal),2],fsamp)


    return latSC



def changeSampleRate(fsamp,buffersize,SC = 10):

    # make short chirp
    f1 = 1000
    f2 = 1200
    Nsamp = 2048*10  # cca 500ms
    #fsamp = 44100 # samplinMatN[:,i]g rate must be the same as in jack setup

    #buffersize = 2048  # buffersize for sounddevice stream


    def makeChirp(f1,f2,Nsamp,fsamp):
        T = Nsamp/fsamp  # duration of the chirp
        
        tx = np.arange(0,Nsamp/fsamp,1/fsamp)  # time axis

        A = 1
        T = Nsamp/fsamp
        
        y = np.zeros_like(tx)
        print(len(tx))
        for k in range(len(tx)):
            y[k] = A*np.sin(2*np.pi*(f1+(f2-f1)/(2*(T))*tx[k])*tx[k])

        return y

    y = makeChirp(f1,f2,Nsamp,fsamp)

    # fade in fade out of the chirp
    rampdur = 10e-3 # duration of onset offset ramp

    rampts = np.round(rampdur * fsamp)

    step = np.pi/(rampts-1)
    x= np.arange(0,np.pi+step,step)
    offramp = (1+np.cos(x))/2
    onramp = (1+np.cos(np.flip(x)))/2
    o=np.ones(len(y)-2*len(x))
    wholeramp = np.concatenate((onramp, o, offramp)) # envelope of the entire ramp

    y = y*wholeramp

    # add some zeros
    generated_signal = np.concatenate((np.zeros(4800), 0*y, np.zeros(4800)))  # add some zeros to begining

    '''
    fig,ax = plt.subplots()
    ax.plot(generated_signal)
    plt.show()
    '''
    # now send the input to the sound card and record response




    #SC = choose_soundcard() # choose sound card (which device to use)
    
    print(f'Chosen device is {SC}')
    chan_in = 3
    sig_in = np.tile(generated_signal,(chan_in,1)).T

    recorded_data = RMEplayrec(sig_in,fsamp,SC=SC,buffersize=buffersize)
    #from pyDPOAEmodule import sendDPOAEstimulustoRMEsoundcard

    '''
    #recorded_data = sendDPOAEstimulustoRMEsoundcard(generated_signal,3,3,fsamp)
    #print(np.shape(output))
    fig,ax = plt.subplots()
    ax.plot(recorded_data)
    ax.plot(generated_signal)
    #ax.set_ylim((-0.001,0.001))
    plt.show()
    '''



def makeTwoPureTones(ftone1,ftone2,Ltone1,Ltone2,phitone1,phitone2,T,fs,ch1,ch2,fade=(4410,4410)):

    # tone 1    
    t1 =  makePureTone(ftone1,Ltone1,phitone1,T,fs,ch1)
    # tone 2
    t2 =  makePureTone(ftone2,Ltone2,phitone2,T,fs,ch2)
    
    if ch1==1 and ch2 == 2:
        twtones = np.column_stack((t1, t2, t1+t2))
    elif ch1==2 and ch2==1:
        twtones = np.column_stack((t2, t1, t1+t2))
    else:      
        raise ValueError("wrong cahnnels!!!")

    return twtones





def makeTwoArbTones(t1,t2,ch1,ch2):
    # return two tones for DPOAE measurement, tone1 and tone2 are sent into respective speakers
    # tone 1    
    
    if ch1==1 and ch2 == 2:
        twtones = np.column_stack((t1, t2, t1+t2))
    elif ch1==2 and ch2==1:
        twtones = np.column_stack((t2, t1, t1+t2))
    else:      
        raise ValueError("wrong cahnnels!!!")

    return twtones



def makeLFPureTone(ftone,Ltone,phitone,T,fs,fade=(4410,4410)):
    '''
    generate a LF pure tone for biasing experiments
    '''

    
    tx = np.arange(0,T,1/fs)

    s = np.sin(2*np.pi*ftone*tx + phitone)  # pure tone

   
    # fade-in the input signal
    if fade[0]>0:
        s[0:fade[0]] = s[0:fade[0]] * ((-np.cos(np.arange(fade[0])/fade[0]*np.pi)+1) / 2)

    # fade-out the input signal
    if fade[1]>0:
        s[-fade[1]:] = s[-fade[1]:] *  ((np.cos(np.arange(fade[1])/fade[1]*np.pi)+1) / 2)


 
    s = 10**(Ltone/20)*np.sqrt(2)*2e-5*s

       
    s = np.pad(s, (0, 4096), 'constant')  # add some zeros to the end
    
    return s


def makePureTone(ftone,Ltone,phitone,T,fs,channel,fade=(4410,4410)):
    '''
    generate a pure tone
    '''

    # load calibration data for in situ calibration
    probecal = loadmat('Calibration_files/Files/InEarCalData.mat')
    if channel==1:
        Hprobe = probecal['Hinear1']
        fxprobe = probecal['fxinear']
    elif channel==2:
        Hprobe = probecal['Hinear2']
        fxprobe = probecal['fxinear']

    tx = np.arange(0,T,1/fs)

    s = np.sin(2*np.pi*ftone*tx + phitone)  # pure tone

   
    # fade-in the input signal
    if fade[0]>0:
        s[0:fade[0]] = s[0:fade[0]] * ((-np.cos(np.arange(fade[0])/fade[0]*np.pi)+1) / 2)

    # fade-out the input signal
    if fade[1]>0:
        s[-fade[1]:] = s[-fade[1]:] *  ((np.cos(np.arange(fade[1])/fade[1]*np.pi)+1) / 2)


    s = np.pad(s, (0, 4096), 'constant')  # add some zeros to the end


    fft_len = len(s) # number of samples for fft


    Sin = np.fft.rfft(s,fft_len)
    axe_w = np.linspace(0, np.pi, num=int(fft_len/2+1))
    fxSin = axe_w/(np.pi)*fs/2
    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Sin))
    #draw_DPOAE(fxSin, Sin)

    Hrint = np.interp(fxSin,fxprobe.flatten(),Hprobe.flatten())

    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Hrint))
    #draw_DPOAE(fxSin, Hrint) 

    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Sin/Hrint))
    #draw_DPOAE(fxSin, Sin/Hrint)

    SinC = Sin/Hrint

    sig = np.fft.irfft(SinC)

    s = 10**(Ltone/20)*np.sqrt(2)*2e-5*sig

       
    
    return s



def makeAmpSweptPureTone(ftone,Ltoneb,Ltones,phitone,T,fs,channel,fade=(4410,4410)):
    '''
    generate a pure tone linearly swept in amplitude
    '''

    # load calibration data for in situ calibration
    probecal = loadmat('Calibration_files/Files/InEarCalData.mat')
    if channel==1:
        Hprobe = probecal['Hinear1']
        fxprobe = probecal['fxinear']
    elif channel==2:
        Hprobe = probecal['Hinear2']
        fxprobe = probecal['fxinear']

    tx = np.arange(0,T,1/fs)

    s = np.sin(2*np.pi*ftone*tx + phitone)  # pure tone
    
    
   
    # fade-in the input signal
    if fade[0]>0:
        s[0:fade[0]] = s[0:fade[0]] * ((-np.cos(np.arange(fade[0])/fade[0]*np.pi)+1) / 2)

    # fade-out the input signal
    if fade[1]>0:
        s[-fade[1]:] = s[-fade[1]:] *  ((np.cos(np.arange(fade[1])/fade[1]*np.pi)+1) / 2)

    dBsw = np.linspace(0,Ltones-Ltoneb,len(tx))
    ssw = s*10**(dBsw/20)   # amplitude swept tone


    s = np.pad(s, (0, 4096), 'constant')  # add some zeros to the end


    ssw = np.pad(ssw, (0, 4096), 'constant')  # add some zeros to the end


    fft_len = len(s) # number of samples for fft


    Sin = np.fft.rfft(s,fft_len)
    axe_w = np.linspace(0, np.pi, num=int(fft_len/2+1))
    fxSin = axe_w/(np.pi)*fs/2
    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Sin))
    #draw_DPOAE(fxSin, Sin)

    Hrint = np.interp(fxSin,fxprobe.flatten(),Hprobe.flatten())

    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Hrint))
    #draw_DPOAE(fxSin, Hrint) 

    #fig,ax = plt.subplots()
    #ax.plot(fxSin,np.abs(Sin/Hrint))
    #draw_DPOAE(fxSin, Sin/Hrint)

    SinC = Sin/Hrint

    sig = np.fft.irfft(SinC)

    scaleF = np.sqrt(np.mean(sig**2))/np.sqrt(np.mean(s**2))    

    sswC = 10**(Ltoneb/20)*np.sqrt(2)*2e-5*scaleF*ssw

       
    
    return sswC


def calcDPstst(oaeDS,f2f1,f2,f1,fsamp,Twin,T,GainMic,Thresh):

    # reshape to Twin frames

    Nwin = int(Twin*fsamp)
    Nframes = int(int(T*fsamp)/Nwin)
    print(Nwin)
    print(Nframes)
    oaeRSh = np.reshape(oaeDS[:Nwin*Nframes],(Nwin,Nframes),'F')

    fx = np.arange(Nwin)*fsamp/Nwin
    
    # two bins below and above Fdp

    fdp = 2*f1-f2  # cubic difference tone

    idxFdp = np.where(fx==fdp)[0][0]

    selected = np.empty(shape=(0,), dtype=int)
    
       
    
    for i in range(2,Nframes-2):
        Spect = 2*np.fft.rfft(oaeRSh[:,i])/Nwin

        Spnoise = np.concatenate((Spect[idxFdp-4:idxFdp-1],Spect[idxFdp+2:idxFdp+5]))
        fxnoise = np.concatenate((fx[idxFdp-4:idxFdp-1],fx[idxFdp+2:idxFdp+5]))

        #(MicCalFN,Val,fxV,MicGain):
        SpnoisePa = ValToPaSimple(Spnoise,GainMic) # convert to Pa
        Spn = np.mean(abs(SpnoisePa))  # mean value across the noise bins 

        if Spn>Thresh: # skip the frame
            continue
        else:
            selected = np.concatenate((selected,np.array([i])))

    
    mean_oae = np.mean(oaeRSh[:,selected],1)  # perform averaging across time
    dpspect = 2*np.fft.rfft(mean_oae)/Nwin

    Spcalib,fxI = spectrumToPaSimple(dpspect,fx,GainMic)

    return mean_oae, selected, Spcalib, fxI



def calcDPststBias(oaeDS,f2f1,f2,f1,fsamp,Twin,T,GainMic,Thresh):

    # reshape to Twin frames

    Nwin = int(Twin*fsamp)
    Nframes = int(int(T*fsamp)/Nwin)
    print(Nwin)
    print(Nframes)
    oaeRSh = np.reshape(oaeDS[:Nwin*Nframes],(Nwin,Nframes),'F')

    fx = np.arange(Nwin)*fsamp/Nwin
    
    # two bins below and above Fdp

    fdp = 2*f1-f2  # cubic difference tone

    idxFdp = np.where(fx==fdp)[0][0]

    selected = np.empty(shape=(0,), dtype=int)
    
       

    for i in range(2,Nframes-2):
        Spect = 2*np.fft.rfft(oaeRSh[:,i])/Nwin

        Spnoise = np.concatenate((Spect[idxFdp-4:idxFdp-1],Spect[idxFdp+2:idxFdp+5]))
        fxnoise = np.concatenate((fx[idxFdp-4:idxFdp-1],fx[idxFdp+2:idxFdp+5]))

        #(MicCalFN,Val,fxV,MicGain):
        SpnoisePa = ValToPaSimple(Spnoise,GainMic) # convert to Pa
        Spn = np.mean(abs(SpnoisePa))  # mean value across the noise bins 

        if Spn>Thresh: # skip the frame
            continue
        else:
            selected = np.concatenate((selected,np.array([i])))

    mean_oae = np.mean(oaeRSh[:,selected],1)  # perform averaging across time
    dpspect = 2*np.fft.rfft(mean_oae)/Nwin

    Spcalib,fxI = spectrumToPaSimple(dpspect,fx,GainMic)

    return mean_oae, selected, Spcalib, fxI



def calcDPgramTD(oaeDS,nfloorDS,f2f1,f2b,f2e,fsamp,octpersec,nfilt,MicGain):
    # function which calculates DP-gram from swept sine response

    rF = f2f1 # frequency ratio f2/f1
    dt = 1/fsamp  # inverse to sampling frequency

    Nsamp = len(oaeDS)  # number of samples in the recorded response
    oaeDS = oaeDS/(0.003*(10**(MicGain/20)))  # conver to Pascals
    resOct = 3  # resolution 3 - 1/3 oct bands, 2 - 1/2 oct bands
    #twin = nfilt/fsamp   # win size in samfrom UserModules.draw import *ples
    if f2e>f2b: # upward sweep
        T = np.log2(f2e/f2b)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        fc = np.array([f2b*2**(1/(resOct*2))])
        f2eH = f2e
    else:
        T = np.log2(f2b/f2e)/octpersec     # duration of the sweep given by freq range and speed (oct/sec)
        fc = np.array([f2e*2**(1/(resOct*2))])
        f2eH = f2b

    f1s = f2b/rF # calculate starting frequency for f1 tone
    f1e = f2e/rF # calculate stop frequnecy for f2 tone
    #Nwin = round(twin*fsamp) # number of samples in a window
    #Nwinpul = round(Nwin/2) # half the number of samples in a window

    L1sw = T/np.log(f1e/f1s) # overall L1sw parameter for synchronized swept sine

    #finst = inst_freq_synch_swept_sine(f1s,f1e,T,fsamp)  # instantaneous frequency for swept sine
    # return frequency in Hz for the swept f1 tone
    #fc = finst[Nwinpul::Nwin] # find center frequencies for recursive exponential window sizes
   
   
    while True:
        newfcVal = fc[-1]*2**(1/resOct) # calculate new center frequency for 1/2 oct bands
        if newfcVal < f2eH:
            fc = np.append(fc,newfcVal)
        else:
            break
       
       
   
    fdpc = 2*fc/rF - fc; # convert center freq from f1 to fdp (assumed cdt (low-side))
    if f2e<f2b: # downward sweep, flip the frequencies
        fdpc = fdpc[::-1]
    aW = 0.01
    #tauNL = 0.5*aW*(fdpc/1000)**(-0.8)  # NL window width as a function of fdp
    #tauAll = 5/2*aW*(fdpc/1000)**(-0.8) # overall window width (NL + CR components) as a function of fdp
    #hmfftlen = 2**13 # hm window size (affect the highest possible speed and latencies in the impulse response)
    hmfftlen = nfilt
    hm, h, dt_1, dt_2 = fceDPOAEinwinSShm(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,hmfftlen)
    hmN, hN, dt_1N, dt_2N = fceDPOAEinwinSSNoisehm(oaeDS, Nsamp, f1s, L1sw, rF, fsamp, hmfftlen)
    return hm, h, hmN, hN



def fceDPOAEinwinSShm(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,hmfftlen):
    '''
    % function [Hm, fx, hm, tx, hmnoise] = fceDPOAEinwinSS(oaeDS,Npts,f1s,L1sw,rF,fsD,tau01,tau02,tshift)
    %
    % calculates DPOAE with SSS technique
    % 
    % oaeDS - time domain signal from which DPOAE is extracted (response to
    % swept sines in the ear canal)
    % Npts - number of samples in oaeDS
    % fs1 - starting frequency of the synch. swept sine (the initial frequency
    % regardless windowing)
    % L1sw - parameter for the swept sine: L1sw = T/log(f2/f1) (for the overall
    % swept sine)
    % rF - ratio of the f2/f1 frequencies used to evoke DPOAE
    % fsD - sampling frequency
    % tau01 - tau parametMicCalFN = 'CalibMicData.mat'er (cutoff) for first half of the recursive exponential function
    % tau01 - tau parameter (cutoff) for the second half of the recursive exponential function
    % region around the DPOAE impulse response
    % tshift - time shift (for the swept sine response in the windows)

    % calculate spectrum of synchronized swept sine (shifted version in case of
    % time windows which does not start at 0 time)
    '''
    tshift = 0
    fft_len = int(2**np.ceil(np.log2(len(oaeDS)))) # number of samples for fft 
    S,f = synchronized_swept_sine_spectra_shifted(f1s,L1sw,fsamp,fft_len,tshift)

    # spectra of the output signal
    
    Y = np.fft.rfft(oaeDS,fft_len)/fsamp  # convert the response to frequency domain
    
    # frequency-domain deconvolution
    H = Y/S
    h = np.fft.irfft(H)  # calculated "virtual" impulse response
    
    rF1 = 2 - rF
    dt = -fsamp*L1sw*np.log(rF1)     # positions of the selected (coef2) IMD component [samples]
    dt_ = round(dt)
    dt_rem = dt - dt_
    hmfftlen = 2**12
    len_IRpul = int((hmfftlen)/2) # length of the impulse response window (adequate for the used time windows for DPOAEs)
    hm = h[dt_-len_IRpul:dt_+len_IRpul]
    #fig,ax = plt.subplots()
    #ax.plot(hm)
    #print(len(S))
    #print(len(hm))
    #print(len(h))
    #print(dt_-len_IRpul+1)
    #print(dt_+len_IRpul+1)
    #from scipy.io import savemat
    #data = {'hm':hm}
    #savemat('hms015python.mat',data)
    
    axe_w = np.linspace(0,np.pi,len_IRpul+1,endpoint=False)

    # Non-integer sample delay correction
    Hx = np.fft.rfft(hm) * np.exp(-1j*dt_rem*axe_w)
    hm = np.fft.irfft(Hx)
    
    #from scipy.io import savemat
    #data = {'hm':hm}
    #savemat('hms015python.mat',data)
    
    # apply roex windows to suppress noise and perform component separation
    

    
    
    return hm, h, dt_-len_IRpul, dt_+len_IRpul



def fceDPOAEinwinSSNoisehm(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,hmfftlen):
    '''
    % function [Hm, fx, hm, tx, hmnoise] = fceDPOAEinwinSS(oaeDS,Npts,f1s,L1sw,rF,fsD,tau01,tau02,tshift)
    %
    % calculates DPOAE with SSS technique
    %
    % oaeDS - time domain signal from which DPOAE is extracted (response to
    % swept sines in the ear canal)
    % Npts - number of samples in oaeDS
    % fs1 - starting frequency of the synch. swept sine (the initial frequency
    % regardless windowing)
    % L1sw - parameter for the swept sine: L1sw = T/log(f2/f1) (for the overall
    % swept sine)
    % rF - ratio of the f2/f1 frequencies used to evoke DPOAE
    % fsD - sampling frequency
    % tau01 - tau parametMicCalFN = 'CalibMicData.mat'er (cutoff) for first half of the recursive exponential function
    % tau01 - tau parameter (cutoff) for the second half of the recursive exponential function
    % region around the DPOAE impulse response
    % tshift - time shift (for the swept sine response in the windows)

    % calculate spectrum of synchronized swept sine (shifted version in case of
    % time windows which does not start at 0 time)
    '''
    tshift = 0
    fft_len = int(2**np.ceil(np.log2(len(oaeDS)))) # number of samples for fft
    S,f = synchronized_swept_sine_spectra_shifted(f1s,L1sw,fsamp,fft_len,tshift)

    # spectra of the output signal
   
    Y = np.fft.rfft(oaeDS,fft_len)/fsamp  # convert the response to frequency domain
   
    # frequency-domain deconvolution
    H = Y/S
    h = np.fft.irfft(H)  # calculated "virtual" impulse response
   
    rF1a = 2 - rF  # 2*f1-f2 component
    rF1b = 3 - 2*rF # for 3*f1-2*f2 component
    rF1 = np.mean((rF1a,rF1b))  # calculate the mean to be between these to Fdp components for noise estimation
    dt = -fsamp*L1sw*np.log(rF1)     # positions of the selected (coef2) IMD component [samples]
    dt_ = round(dt)
    dt_rem = dt - dt_
    hmfftlen = 2**12
    len_IRpul = int((hmfftlen)/2) # length of the impulse response window (adequate for the used time windows for DPOAEs)
    if dt_>0:  # upward sweep
        hm = h[dt_-len_IRpul:dt_+len_IRpul]
    else:  # downward sweep
        hm = h[len(h)+dt_-len_IRpul:len(h)+dt_+len_IRpul]
    #fig,ax = plt.subplots()
    #ax.plot(hm)
    #print(len(S))
    #print(len(hm))
    #print(len(h))
    #print(dt_-len_IRpul+1)
    #print(dt_+len_IRpul+1)
    #from scipy.io import savemat
    #data = {'hm':hm}
    #savemat('hms015python.mat',data)
   
    axe_w = np.linspace(0,np.pi,len_IRpul+1,endpoint=False)

    # Non-integer sample delay correction
    Hx = np.fft.rfft(hm) * np.exp(-1j*dt_rem*axe_w)
    hm = np.fft.irfft(Hx)
   
   
    #from scipy.io import savemat
    #data = {'hm':hm}
    #savemat('hms015python.mat',data)
   
    # apply roex windows to suppress noise and perform component separation
   

   
   
    return hm, h, dt_-len_IRpul, dt_+len_IRpul



def fceDPOAEinwinSS(oaeDS,Nsamp,f1s,L1sw,rF,fsamp,tau01,tau02,tshift,hmfftlen):
    '''
    % function [Hm, fx, hm, tx, hmnoise] = fceDPOAEinwinSS(oaeDS,Npts,f1s,L1sw,rF,fsD,tau01,tau02,tshift)
    %
    % calculates DPOAE with SSS technique
    % 
    % oaeDS - time domain signal from which DPOAE is extracted (response to
    % swept sines in the ear canal)
    % Npts - number of samples in oaeDS
    % fs1 - starting frequency of the synch. swept sine (the initial frequency
    % regardless windowing)
    % L1sw - parameter for the swept sine: L1sw = T/log(f2/f1) (for the overall
    % swept sine)
    % rF - ratio of the f2/f1 frequencies used to evoke DPOAE
    % fsD - sampling frequency
    % tau01 - tau parametMicCalFN = 'CalibMicData.mat'er (cutoff) for first half of the recursive exponential function
    % tau01 - tau parameter (cutoff) for the second half of the recursive exponential function
    % region around the DPOAE impulse response
    % tshift - time shift (for the swept sine response in the windows)

    % calculate spectrum of synchronized swept sine (shifted version in case of
    % time windows which does not start at 0 time)
    '''

    fft_len = int(2**np.ceil(np.log2(len(oaeDS)))) # number of samples for fft 
    S,f = synchronized_swept_sine_spectra_shifted(f1s,L1sw,fsamp,fft_len,tshift)

    # spectra of the output signal
    
    Y = np.fft.rfft(oaeDS,fft_len)/fsamp  # convert the response to frequency domain
    
    # frequency-domain deconvolution
    H = Y/S
    h = np.fft.irfft(H)  # calculated "virtual" impulse response
    
    rF1 = 2 - rF
    dt = -fsamp*L1sw*np.log(rF1)     # positions of the selected (coef2) IMD component [samples]
    dt_ = round(dt)
    dt_rem = dt - dt_
    hmfftlen = 2**12
    len_IRpul = int((hmfftlen)/2) # length of the impulse response window (adequate for the used time windows for DPOAEs)
    if dt_>0:  # upward sweep
        hm = h[dt_-len_IRpul:dt_+len_IRpul]
    else:  # downward sweep
        hm = h[len(h)+dt_-len_IRpul:len(h)+dt_+len_IRpul]
    #fig,ax = plt.subplots()
    #ax.plot(hm)
    #print(len(S))
    #print(len(hm))
    #print(len(h))
    #print(dt_-len_IRpul+1)
    #print(dt_+len_IRpul+1)
    #from scipy.io import savemat
    #data = {'hm':hm}
    #savemat('hms015python.mat',data)
    
    axe_w = np.linspace(0,np.pi,len_IRpul+1,endpoint=False)

    # Non-integer sample delay correction
    Hx = np.fft.rfft(hm) * np.exp(-1j*dt_rem*axe_w)
    hm = np.fft.irfft(Hx)
    
    #from scipy.io import savemat
    #data = {'hm':hm}
    #savemat('hms015python.mat',data)
    
    # apply roex windows to suppress noise and perform component separation
    Nwindow = 10 # degree of roex windows
    w = roexwin(len(hm),Nwindow,fsamp,tau01,tau02)
    
    hm = hm*w  # multiply with roex window
    # add zeros to achieve larger number of points
    hm = np.concatenate((np.zeros(2**11),hm,np.zeros(2**11)))
    Hm = np.fft.rfft(np.fft.fftshift(hm))

    
    hmfftlen = len(hm)

    fxall = np.arange(hmfftlen)*fsamp/hmfftlen # overall freq axis
    fx = fxall[:round(hmfftlen/2)+1]
    return Hm, fx

def ValToPaSimple(Val,MicGain):
    # convert to Pascals at frequency fxV

    SpectPa = Val/(0.003*(10**(MicGain/20)))
        
    return SpectPa

def spectrumToPaSimple(Spect,fxSpect,MicGain):
    # convert spectrum to Pascals
    
    
    SpectPa = Spect/(0.003*(10**(MicGain/20)))
    
    
    return SpectPa, fxSpect




def giveTEOAE_MCmp(data,recsig40,latency_SC,Npulse,Nclicks,Tap=1e-3,TWt=25e-3,ArtRej=500):
    '''
    perform averaging in the time domain for click responses measured for 
    compression method: 3 clicks + click of 3 times larger amplitude
    
    Parameters
    ----------
    recsig40 : TYPE
        time domain responses to a click train
    latency_SC : TYPE
        sound card latency
    Npulse : TYPE
        Number of samples in one frame (pulse + silence)
    Nclicks : TYPE
        Number of clicks
    Tap : TYPE, optional
        Onset of the tukey window. The default is 1e-3.
    TWt : TYPE, optional
        Duration fo the tukey window. The default is 20e-3.

    Returns
    -------
    

    '''
    cutoff = 300
    fsamp = data['fsamp'][0]
    frecsig40 = butter_highpass_filter(recsig40, cutoff, fsamp)
        
    #fig,ax = plt.subplots()
    #plt.plot(frecsig40)
             
    # average in the time domain
    rct01f = frecsig40[latency_SC:]
    MG = 40
    rct01f = rct01f/(0.003*10**(MG/20))  # convert to Pascals
    
    # noise rejections
    
    
    NnoiseF = len(rct01f[Nclicks*Npulse+6000:])//Npulse
    nct01f = np.reshape(rct01f[Nclicks*Npulse+6000:Nclicks*Npulse+6000+NnoiseF*Npulse],(Npulse,NnoiseF),order='F')
    # noise estimation from last samples
    rct01f = np.reshape(rct01f[:Nclicks*Npulse],(Npulse,Nclicks),order='F')
    # here we have a matrix, each response is in separate column, 
    # we should calcaulte mean across corresponding levels
    
    step = 4 # length of levelS vector (number of click levels in scramble)
    mrct01l = np.mean(rct01f[:,::step],1)
    nrct01l = np.mean(nct01f,1)
    #mrct02l = np.mean(rct01f[:,1:step],1)
    #mrct03l = np.mean(rct01f[:,2:step],1)
    #mrct04l = np.mean(rct01f[:,3:step],1)
    #mrct05l = np.mean(rct01f[:,4:step],1)
    #mrct06l = np.mean(rct01f[:,5:step],1)
        
    max_index = np.argmax(np.abs(mrct01l[:500]))  # find index of max value, this is 0 time
    TNafter = int(0.5e-3*fsamp)  # time after peak of the click for TEAOE analysis
    
    alpha = 0.2  # This controls the shape of the Tukey window
    zero_before = np.zeros(max_index+TNafter)
    #TWt = 17e-3
    TWn = int(fsamp*TWt)
    zero_after = np.zeros(Npulse-TWn-max_index-TNafter)
    # Concatenate the Tukey window with the zero-padding
    # Generate the Tukey window
    tukey_window = tukey(TWn, alpha)
    
    window_with_zeros = np.concatenate((zero_before,tukey_window, zero_after), axis=0)
    
    
    # first start with artifact rejection
    Matrice = [True]*Nclicks
   
    for i in range(Nclicks):
        ramec = window_with_zeros*rct01f[:,i]
        if max(np.abs(ramec))>2e-5*ArtRej:
             Matrice[i]=False
             print(f'{i+1}ta odezva vypadava')
    
    
    Mat01 = Matrice[::step]
    Mat02 = Matrice[1::step]      
    Mat03 = Matrice[2::step]
    Mat04 = Matrice[3::step]       
    
    rct01l = rct01f[:,::step]
    rct02l = rct01f[:,1::step]
    rct03l = rct01f[:,2::step]
    rct04l = rct01f[:,3::step]
    
    rch01 = np.mean(rct01l[:,Mat01],1)       
    rch02 = np.mean(rct02l[:,Mat02],1)
    rch03 = np.mean(rct03l[:,Mat03],1)       
    rch04 = np.mean(rct04l[:,Mat04],1)       
    
    rch01Mean = np.reshape(rch01, (-1,1))
    rch02Mean = np.reshape(rch02, (-1,1))
    rch03Mean = np.reshape(rch03, (-1,1))
    rch04Mean = np.reshape(rch04, (-1,1))
    
    MatN01 = rct01l[:,Mat01] - rch01Mean
    MatN02 = rct02l[:,Mat02] - rch02Mean
    MatN03 = rct03l[:,Mat03] - rch03Mean
    MatN04 = rct04l[:,Mat04] - rch04Mean
    
    
    
        
    
    return rch01, rch02, rch03, rch04, window_with_zeros, max_index, MatN01, MatN02, MatN03, MatN04, nrct01l




def calcTFatF(y,fb,fe,T,fs):
    # Use SSS technique to extract transfer function at frequency f
    # extract the spectrum
    
    fft_len = int(2**np.ceil(np.log2(len(y)))) # number of samples for fft 
    Y = np.fft.rfft(y,fft_len)/fs  # convert the response to frequency domain
    #ax.semilogx(fx,20*np.log10(np.abs(DPgram))/(np.sqrt(2)*2e-5))
    
    # synchronized swept sine in the frequency domain
    L = T/np.log(fe/fb)   # parameter of exp.swept-sine
    f_axis = np.linspace(0, fs/2, num=round(fft_len/2)+1) # frequency axis
    f_axis[0] = 1e-12
    SI = 2*np.emath.sqrt(f_axis/L)*np.exp(-1j*2*np.pi*f_axis*L*(1-np.log(f_axis/fb)) + 1j*np.pi/4)
    SI[0] = 0j
    
    # first Nyquist zone 
    H = Y*SI; 
    
    # ifft
    h = np.fft.irfft(H)
    '''
    _,ax1 = plt.subplots()
    plot_shift = -50000
    t_h = -(np.arange(len(h),0,-1)+plot_shift)/fs
    ax1.plot(t_h,np.roll(h,plot_shift))
    ax1.set_title('Output of the Nonlinear Convolution')
    ax1.set_xlabel('time [s]')
    ax1.autoscale(enable=True, axis='x', tight=True)
    '''
    N = 1 # separate 3 higher harmonics
    dt = L*np.log(np.arange(1,N+1))*fs  # positions of higher orders up to N
    dt_rem = dt - np.around(dt) # The time lags may be non-integer in samples, the non integer delay must be applied later
    
    len_Hammer = 2**13
    shft = round(len_Hammer/2)          # number of samples to make an artificail delay
    h_pos = np.hstack((h, h[0:shft + len_Hammer - 1]))  # periodic impulse response
    # separation of higher orders 
    hs = np.zeros((N,len_Hammer))
    
    axe_w = np.linspace(0, np.pi, num=int(len_Hammer/2+1)); # frequency axis 
    for k in range(N):
        hs[k,:] = h_pos[len(h)-int(round(dt[k]))-shft-1:len(h)-int(round(dt[k]))-shft+len_Hammer-1]
        H_temp = np.fft.rfft(hs[k,:])
    
        # Non integer delay application
        H_temp = H_temp * np.exp(-1j*dt_rem[k]*axe_w)
        hs[k,:] = np.fft.fftshift(np.fft.irfft(H_temp))
    
    # Higher Harmonics
    Hs = np.fft.rfft(hs)
    Hs = Hs[0,:].flatten()
    MicGain = 40
    #MicCalFN = 'MicCalCurve.mat'
    fxHs = axe_w/(np.pi)*fs/2
    #HsC,fxI = spectrumToPa(MicCalFN,Hs,fxHs,MicGain)
    HsC, fxI = spectrumToPaSimple(Hs,fxHs,MicGain)
    
    return fxI, HsC


def mother_wavelet2(Nw,Nt,df,dt):
    vlnky = np.zeros((Nt,Nw))
    tx = (np.arange(Nt)-Nw)*dt
    for k in range(Nw):
        vlnky[:,k] = np.cos(2*np.pi*(k+1)*df*tx)*1/(1+(0.075*(k+1)*2*np.pi*df*tx)**4)
    return vlnky


def wavelet_filterDPOAE(signal, wavelet_basis,fx,fsamp):
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
        awltSh = 0.015 # time constant for long latency components
        awltZL = 0.005 # time constant for zero-latency components
        bwlt = 0.8  # exponent
        TrwcSh = awltSh*(fx[k]/1000)**(-bwlt)  # rwc time const for positive time (frequency dependent vis Moleti 2012)
        TrwcZL = awltZL*(fx[k]/1000)**(-bwlt)  # rwc time const for positive time (frequency dependent vis Moleti 2012)
        #NsampShift = int(fsamp*TrwcSh)  # number of samples  for shifting the roexwin

        rwC = roexwin(Nall,Nrw,fsamp,Trwc01,TrwcSh) #  not shifted window
        rwCNL = roexwin(Nall,Nrw,fsamp,Trwc01,TrwcZL) #  not shifted window
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




def roexwin(N,n,fs,tc1,tc2):
    # construct window comosed of two halfs of roex windows
    # see Kalluri and Shera J. Acoust. Soc. Am. 109 (2), February 2001 p.622
    #tx = 0:1/fs:(N/2-1)/fs;
    tx = np.arange(0,round(N/2)/fs,1/fs)

    def GetExpFilt(t,tc,N):
        g0=1
        for i in range(1,N):
            g0=np.log(g0+1)
        
        T=np.sqrt(g0)*t/tc
        G=np.exp((T**2))
        for i in range(1,N):
            G=np.exp(G-1)
        
        return 1/G    

    G1 = GetExpFilt(tx,tc1,n)

    G2 = GetExpFilt(tx,tc2,n)

    G = np.concatenate((np.flip(G1), G2))

    return G


def create_noise(mod_len, fs, Hleft, Hright, fxL, fxR, lb, lend):
    """
    Create noise with a sinusoidal envelope.

    Parameters:
    - mod_len: float, modulation duration in seconds.
    - fs: int, sampling frequency in Hz.

    Returns:
    - noise_samples: int, total number of noise samples.
    - n_reps: int, number of repetitions per loop.
    - H: numpy.ndarray, noise amplitude matrix with repetitions.
    """
    clicks_per_condition = 1000
    loop_duration = mod_len * 30  # n/sec
    n_loops = int(np.ceil(clicks_per_condition / loop_duration))

    noise_samples = int(round(mod_len * fs))
    n_reps = int(loop_duration / (mod_len * 2))  # Number of repetition

    h = np.linspace(lb, lend, noise_samples)
    h[0] = np.finfo(float).eps
    h = np.concatenate((h, np.flipud(h)))
    #h = 10 ** (h / 20) # Conversion to linear values
    h = 10**(h/20)*np.sqrt(2)*2e-5
    # h /= np.max(np.abs(h)) # Normalization

    H = np.tile(h, (n_reps, 1)).T  # Matrix of noise amplitude
    
    noise = np.random.normal(0, 1, noise_samples * 2)

    noiserfft = np.fft.rfft(noise)

    Nnrfft = len(noiserfft)
    fx =np.arange(2*Nnrfft-2)/(2*Nnrfft-1)
    fx = 44100*fx

    hifreq = 14000
    lofreq = 200

    # Calculate bandwidth and RMS noise
    bw = hifreq - lofreq


    nptslofreq = np.where(fx >= lofreq)[0][0]
    nptshifreq = np.where(fx >= hifreq)[0][0]

    # Zero out undesired frequency bands
    noiserfft[:nptslofreq] = 0.0
    noiserfft[nptshifreq:] = 0.0

    Hrint = np.interp(fx[nptslofreq:nptshifreq],fxL,Hleft)
    noiseSc = np.zeros_like(noiserfft)
    noiseSc[nptslofreq:nptshifreq] = noiserfft[nptslofreq:nptshifreq]/np.abs(Hrint)

     # 10**(Ltone/20)*numpy.sqrt(2)*2e-5
    stim1 = h*np.real(np.fft.irfft(noiseSc))
    

    return stim1