# -*- coding: utf-8 -*-
"""
code for TEOAE measurement using, which will run automatically and will be controlled by user
Created on Thu Oct  3 14:10:48 2024

@author: audiobunka
"""


import matplotlib.pyplot as plt
import numpy as np
from UserModules.pyDPOAEmodule import generateClickTrain, RMEplayrec, generateClickTrainCmp, getSClat, giveTEOAE_MCmp
from UserModules.pyUtilities import butter_highpass_filter
import datetime
from scipy.signal.windows import tukey
from scipy.io import savemat
from UserModules.pyUtilities import butter_highpass_filter
import threading



fsamp = 96000; bufsize = 4096;

latency_SC = getSClat(fsamp,bufsize,SC=10)  # estimated latency

#latency_SC = 16448
micGain = 40  # gain of the probe microphone
ear_t = 'R' # which ear

# parameters of evoking stimuli

#Nclicks = 2000


Lc = 52 # required intensity of click in peSPL
Lref = 53 # recorded intensity for 0.1 maximum of the click

save_path = 'Results/s055/scrambled/'
save_path = 'Results/s004/'
#save_path = 'Results/s063/Cmp/'
#save_path = 'Results/s078/'
#save_path = 'Results/s080/'
#save_path = 'Results/s003/'
#save_path = 'Results/s004/NewClick'
#save_path = 'Results/s091/'
subj_name = 's004'

print(f"Playing for {Lc} dB peSPL")
def get_time() -> str:
    # to get current time
    now_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return now_time

t = get_time() # current date and time
# data intialization
    
# generate stimuli    
#s1,s2 = generateDPOAEstimulus(f2f1, fsamp, f2b, f2e, r, L1, L2)    

Npulse = 2048+1024
#Npulse = 2048

# measurement phase

    
#cfname = 'clickInFAV.mat'
#cfname = 'clickInBP_OK.mat'
cfname = 'clickInBP_OK400_4000.mat'
levelS = Lc+np.array([0,0,0,20*np.log10(3)])

NsamplesInPulse = 2048
ClickInRun = 228 # number of clicks (2048 sample buffers) in each run
# 114 yields 4.864 seconds
# the number was chosen to be multiple of 6 (6 levels)
Nclicks = ClickInRun
yupr,Env = generateClickTrainCmp(cfname,ClickInRun,levelS,Npulse)

#yupr = generateClickTrain(cfname, Nclicks,Npulse)
clicktrainmat01 = np.vstack((Env*yupr,np.zeros_like(yupr),yupr)).T 

LcStart = 64  # starting click level
Lstep = 6  # default step


counter = 0

#for counter in range(1,13):
    
    
    

import keyboard
import time

# Flag to control the loop


# Function to stop the loop
def stop_loop():
    global running
    running = False
    print("Loop stopped")
    
# Function to stop the loop
def stop_meas():
    global runningM
    runningM = False
    
    print("Measurement stopped")

# Register a hotkey (e.g., 's') to stop the loop
hotkey_s = keyboard.add_hotkey('s', stop_loop) # a key to stop one 
hotkey_e = keyboard.add_hotkey('e', stop_meas) # a key to stop measurement

runningM = True # measurement is running    
Lc = LcStart


# Global variables to share between threads
fxD = np.zeros(1)
TEOAE = np.zeros(1)
NM1 = NM2 = NM3 = NM4 = np.zeros(1)
plot_ready = threading.Event()  # To signal that data is ready for plotting
runningM = True
Lstep = 6
LcStart = 64

# Define the plotting function to run in a separate thread
def update_plot():
    global fxD, TEOAE, NM1, NM2, NM3, NM4, runningM

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    while runningM:
        plot_ready.wait()  # Wait until data is ready
        plot_ready.clear()  # Reset the event

        ax1.clear()
        ax2.clear()
        ax1.plot(fxD, 20 * np.log10(abs(TEOAE) / (np.sqrt(2) * 2e-5)))
        ax1.plot(fxD, 20 * np.log10(abs(NM1) / (np.sqrt(2) * 2e-5)), ':')
        ax1.plot(fxD, 20 * np.log10(abs(NM2) / (np.sqrt(2) * 2e-5)), ':')
        ax1.plot(fxD, 20 * np.log10(abs(NM3) / (np.sqrt(2) * 2e-5)), ':')
        ax1.plot(fxD, 20 * np.log10(abs(NM4) / (np.sqrt(2) * 2e-5)), ':')
        ax1.set_ylim((-40, 30))
        ax1.set_xlim((500, 6000))
        ax1.set_ylabel('Amplitude (dB SPL)')

        cycle = 2 * np.pi
        ax2.plot(fxD, np.unwrap(np.angle(TEOAE)) / cycle)
        ax2.set_xlim((500, 6000))
        ax2.set_xlabel('Frequency (Hz)$')
        ax2.set_ylabel('Phase (cycles)')

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to allow for event processing

    plt.close(fig)

# Function to run the measurement process
def measurement_loop():
    global fxD, TEOAE, NM1, NM2, NM3, NM4, plot_ready, runningM, running, Lstep, LcStart

    counter = 0
    Lc = LcStart

    while runningM:
        counter = 0
        running = True
        Amp = 0.1 * 10 ** ((Lc - Lref) / 20)

        print(f"Click level {Lc} dB")

        while running and runningM:
            counter += 1
            if counter == 12:
                running = False
            print('Rep: {}'.format(counter))

            # Simulate recording process (replace with actual logic)
            recordedclicktrain01 = RMEplayrec(Amp * clicktrainmat01, fsamp, SC=10, buffersize=bufsize)
            recsig = recordedclicktrain01

            # (Add your measurement logic here)

            # Simulate processing the signal
            cutoff = 300  # High-pass filter cutoff frequency
            recordedclicktrain01f = butter_highpass_filter(recsig[:, 0], cutoff, fsamp, 1)

            recsig = recordedclicktrain01

            data = {"recsig": recsig,"fsamp":fsamp,"Nclicks":Nclicks,"Lc":Lc,"Npulse":Npulse,"levelS":levelS,"latency_SC":latency_SC}  # dictionary
                    

            file_name = 'CMclickOAE_' + subj_name + '_' + t[2:] + '_' + 'Lc_' +str(Lc) + 'dB_Ncl' + str(Nclicks) + 'Npulse' + '_' + str(Npulse) + '_' + ear_t + '_' + counterSTR
                    
            savemat(save_path + '/' + file_name + '.mat', data)



            # visualization
            
            cutoff = 300 # cutoff frequency for high pass filter to filter out low frequency noise
            recordedclicktrain01f = butter_highpass_filter(recsig[:,0], cutoff, fsamp, 1)
            fsampa = np.zeros(1)
            fsampa[0] = fsamp
            data2 = {"fsamp":fsampa}
            tTD01, tTD02, tTD03, tTD04, wz, midx, nM01, nM02, nM03, nM04 = giveTEOAE_MCmp(data2,recordedclicktrain01f,latency_SC,Npulse,Nclicks,Tap=1e-3,TWt=20e-3,ArtRej=100000)
                
            if counter == 1:
                recMat1 = tTD01
                recMat2 = tTD02
                recMat3 = tTD03
                recMat4 = tTD04
                    
                nsMat1 = nM01
                nsMat2 = nM02
                nsMat3 = nM03
                nsMat4 = nM04
                
            else:
                recMat1 = np.c_[recMat1, tTD01]  # add to make a matrix with columns for every run
                recMat2 = np.c_[recMat2, tTD02]  # add to make a matrix with columns for every run
                recMat3 = np.c_[recMat3, tTD03]  # add to make a matrix with columns for every run
                recMat4 = np.c_[recMat4, tTD04]  # add to make a matrix with columns for every run
                
                nsMat1 += nM01
                nsMat2 += nM02
                nsMat3 += nM03
                nsMat4 += nM04

            if counter==1:
                midxT = midx
            else:    
                t1 = wz*((np.mean(recMat1,1)+np.mean(recMat2,1)+np.mean(recMat3,1))-np.mean(recMat4,1))  # perform averaging in time and calculating TEOAE using compression method
                Nfft = 1920
                TEOAE = 2*np.fft.rfft(np.concatenate((t1[midxT:],np.zeros(int(2**15)))))/Nfft  # calcualte spectrum
                
                NM1 = 0
                NM2 = 0
                NM3 = 0
                NM4 = 0
                for ni in range(57):
                    tn1 = wz*nsMat1[:,ni]
                    tn2 = wz*nsMat2[:,ni]
                    tn3 = wz*nsMat3[:,ni]
                    tn4 = wz*nsMat4[:,ni]
                    
                    NM1 += np.abs(2*np.fft.rfft(np.concatenate((tn1[midxT:],np.zeros(int(2**15)))))/Nfft)  # calcualte spectrum
                    NM2 += np.abs(2*np.fft.rfft(np.concatenate((tn2[midxT:],np.zeros(int(2**15)))))/Nfft)  # calcualte spectrum
                    NM3 += np.abs(2*np.fft.rfft(np.concatenate((tn3[midxT:],np.zeros(int(2**15)))))/Nfft)  # calcualte spectrum
                    NM4 += np.abs(2*np.fft.rfft(np.concatenate((tn4[midxT:],np.zeros(int(2**15)))))/Nfft)  # calcualte spectrum
                
                NM1/=57
                NM2/=57
                NM3/=57
                NM4/=57
                
                Ns = len(np.concatenate((t1[midxT:],np.zeros(int(2**15)))))
                fxx = np.arange(Ns)*fsamp/Ns

                fxD = fxx[:Ns//2+1]  # take half of the frequency axis

                # Update shared data for plotting
                #fxD = np.linspace(500, 6000, 500)  # Example frequency axis (replace with actual data)
                #TEOAE = np.random.rand(500)  # Simulated TEOAE data (replace with actual processed data)
                #NM1 = NM2 = NM3 = NM4 = np.random.rand(500)  # Simulated noise data

                plot_ready.set()  # Signal that data is ready for plotting

            # Simulation of level step change (replace with actual level step input)
            if runningM:
                time.sleep(0.5)  # Simulate waiting time between measurements
                Lstep = int(input('Write level step in dB: '))
                Lc = Lc - Lstep

# Hotkey setup to stop the loop
keyboard.add_hotkey('s', stop_loop)
keyboard.add_hotkey('e', stop_meas)

# Start the plotting in a separate thread
plot_thread = threading.Thread(target=update_plot)
plot_thread.start()

# Start the measurement loop (runs in main thread)
measurement_loop()

# Clean up after the threads finish
plot_thread.join()
keyboard.remove_hotkey('s')
keyboard.remove_hotkey('e')



