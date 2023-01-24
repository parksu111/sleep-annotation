#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 21:31:56 2017

Note:
Matplotlib constantly changes how to plot figures interactively.
In case, if there are no figures plotted, there might be three options to try out:
(1) First, start python interpreter using "ipython --matplotlib"
(2) Replace plt.plot(block=False) with plt.plot()
(3) Or, try to turn on interactive mode before any plotting (plt.show()) happens, 
    using plt.ion()

DATE 3/27/18:
added algorithm for NREM closed-loop detection (recursive_sleepstate_nrem)

DATE 3/28/18:
fixed bug in recursive_sleepstate_rem
fixed bug in sleep_spectrum

DATE 4/13/18
added function rem_online_analysis

DATE 5/9/18
nicer representation of data by sleep_timecourse_list

DATE 5/26/18
sleep_stats and sleep_timecourse_list: exclude states where K < 0

DATE 6/02/18
sleep_stats and sleep_timecourse_list: order mice the same way

DATE 3/04/19
sleep_spectrum; implemented that brain states, where K < 0 (the white line
in sleep annotation), are discarded

DATE 3/29/19
sleep_spectrum, sleep_timecourse_list: allowed for recording names including more than final directory:
E.g. my_files/M1_020219, instead of just M1_020219
ma_timecourse_list: added support for pandas DataFrames

DATE 5/1/19
Upgraded to py3

@author: Franz
"""
import scipy.signal
import numpy as np
import scipy.io as so
import scipy.stats as stats
import os.path
import re
import sys
# on a mac, include the following two lines:
# required to make Tkinter in data_processing work
if sys.platform == 'darwin2':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pylab as plt
import h5py
import matplotlib.patches as patches
import numpy.random as rand
import seaborn as sns
import pandas as pd
from functools import reduce
import pdb



class Mouse :    
    def __init__(self, idf, list=None, typ='') :
        self.recordings = []
        self.recordings.append(list)
        self.typ = typ
        self.idf = idf
    
    def add(self, rec) :
        self.recordings.append(rec)

    def __len__(self) :
        return len(self.recordings)

    def __repr__(self) :
        return ", ".join(self.recordings)



### FILE PROCESSING OF RECORDING DATA #################################################
def load_stateidx(ppath, name, ann_name=''):
    """ load the sleep state file of recording (folder) $ppath/$name
    @Return:
        M,K         sequence of sleep states, sequence of 
                    0'1 and 1's indicating non- and annotated states
    """
    ddir = os.path.join(ppath, name)
    ppath, name = os.path.split(ddir)

    if ann_name == '':
        ann_name = name

    sfile = os.path.join(ppath, name, 'remidx_' + ann_name + '.txt')
    
    f = open(sfile, 'r')
    lines = f.readlines()
    f.close()
    
    n = 0
    for l in lines:
        if re.match('\d', l):
            n += 1
            
    M = np.zeros(n)
    K = np.zeros(n)
    
    i = 0
    for l in lines :
        
        if re.search('^\s+$', l) :
            continue
        if re.search('\s*#', l) :
            continue
        
        if re.match('\d+\s+-?\d+', l) :
            a = re.split('\s+', l)
            M[i] = int(a[0])
            K[i] = int(a[1])
            i += 1
            
    return M,K

def load_stateprob(ppath, name):
    """ load the sleep state file of recording ppath/name
    @RETURN:
    M     -      sequency of sleep states
    """   
    file = os.path.join(ppath, name, 'remprob_' + name + '.txt')
    
    f = open(file, newline=None)    
    lines = f.readlines()
    f.close()
    
    n = 0
    for l in lines:
        if re.match('\d', l):
            n = n+1
            
    M = np.zeros(n)
    K = np.zeros(n)
    
    i = 0
    for l in lines :
        
        if re.search('^\s+$', l) :
            continue
        if re.search('\s*#', l) :
            continue
        
        if re.match('\d+\s+-?\d+', l) :
            a = re.split('\s+', l)
            M[i] = int(a[0])
            K[i] = int(a[1])
            i += 1
            
    return M,K


def load_recordings(ppath, rec_file) :
    """
    load_recordings(ppath, rec_file)
    
    load recording listing with syntax:
    [E|C] \s+ recording_name
    
    #COMMENT
    
    @RETURN:
        (list of controls, lis of experiments)
    """
    exp_list = []
    ctr_list = []    

    rfile = os.path.join(ppath, rec_file)
    f = open(rfile, newline=None)
    lines = f.readlines()
    f.close()

    for l in lines :
        if re.search('^\s+$', l) :
            continue
        if re.search('^\s*#', l) :
            continue
        
        a = re.split('\s+', l)
        
        if re.search('E', a[0]) :
            exp_list.append(a[1])
            
        if re.search('C', a[0]) :
            ctr_list.append(a[1])
            
    return ctr_list, exp_list



def load_dose_recordings(ppath, rec_file):
    """
    load recording list with following syntax:
    A line is either control or experiments; Control recordings look like:

    C \s recording_name

    Experimental recordings also come with an additional dose parameter 
    (allowing for comparison of multiple doses with controls)
    
    E \s recording_name \s dose_1
    E \s recording_name \s dose_2
    """
    
    rfile = os.path.join(ppath, rec_file)
    f = open(rfile, newline=None)
    lines = f.readlines()
    f.close()

    # first get all potential doses
    doses = {}
    ctr_list = []
    for l in lines :
        if re.search('^\s+$', l):
            continue
        if re.search('^\s*#', l):
            continue        
        a = re.split('\s+', l)
        
        if re.search('E', a[0]):
            if a[2] in doses:
                doses[a[2]].append(a[1])
            else:
                doses[a[2]] = [a[1]]

        if re.search('C', a[0]):
            ctr_list.append(a[1])

    return ctr_list, doses
    
def get_cycles(ppath, name):
    """
    extract the time points where dark/light periods start and end
    """
    act_dur = get_infoparam(os.path.join(ppath, name, 'info.txt'), 'actual_duration')

    time_param = get_infoparam(os.path.join(ppath, name, 'info.txt'), 'time')
    if len(time_param) == 0 or len(act_dur) == 0:
        return {'light': [(0,0)], 'dark': [(0,0)]}
    
    hour, mi, sec = [int(i) for i in re.split(':', time_param[0])]
    #a = sleepy.get_infoparam(os.path.join(ppath, name, 'info.txt'), 'actual_duration')[0]
    a,b,c = [int(i[0:-1]) for i in re.split(':', act_dur[0])]
    total_dur = a*3600 + b*60 + c
    
    # number of light/dark switches
    nswitch = int(np.floor(total_dur / (12*3600)))
    switch_points = [0]
    cycle = {'light': [], 'dark':[]}
        
    if hour >= 7 and hour < 19:
        # recording starts during light cycle
        a = 19*3600 - (hour*3600+mi*60+sec)
        for j in range(nswitch):
            switch_points.append(a+j*12*3600)
        for j in range(1, nswitch, 2):
            cycle['dark'].append(switch_points[j:j+2])
        for j in range(0, nswitch, 2):
            cycle['light'].append(switch_points[j:j+2])
        
    else:
        # recording starts during dark cycle
        a = 0
        if hour < 24:
            a = 24 - (hour*3600+mi*60+sec) + 7*3600
        else:
            a = 7*3600 - (hour*3600+mi*60+sec)
        for j in range(nswitch):
            switch_points.append(a+j*12*3600)
        for j in range(0, nswitch, 2):
            cycle['dark'].append(switch_points[j:j+2])
        for j in range(1, nswitch, 2):
            cycle['light'].append(switch_points[j:j+2])
        
    return cycle    


def get_snr(ppath, name):
    """
    read and return SR from file $ppath/$name/info.txt 
    """
    fid = open(os.path.join(ppath, name, 'info.txt'), newline=None)
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines :
        a = re.search("^" + 'SR' + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))            
    return float(values[0])



def get_infoparam(ifile, field):
    """
    NOTE: field is a single string
    and the function does not check for the type
    of the values for field.
    In fact, it just returns the string following field
    """
    fid = open(ifile, newline=None)
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines :
        a = re.search("^" + field + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))
            
    return values
    

def add_infoparam(ifile, field, vals):
    fid = open(ifile, 'a')
    vals = [str(s) for s in vals]
    param = " ".join(vals)
    fid.write('%s:\t%s' % (field, param))
    fid.write(os.linesep)
    fid.close()


def laser_start_end(laser, SR=1525.88, intval=5):
    """laser_start_end(ppath, name) ...
    print start and end index of laser stimulation trains: For example,
    if you was stimulated for 2min every 20 min with 20 Hz, return the
    start and end index of the each 2min stimulation period (train)

    returns the tuple (istart, iend), both indices are inclusive,
    i.e. part of the sequence
    @Param:
    laser    -    laser, vector of 0s and 1s
    intval   -    minimum time separation [s] between two laser trains
    @Return:
    (istart, iend) - tuple of two np.arrays with laser start and end indices
    """
    idx = np.where(laser > 0.5)[0]
    if len(idx) == 0 :
        #return (None, None)
        return ([], [])
    
    idx2 = np.nonzero(np.diff(idx)*(1./SR) > intval)[0]
    istart = np.hstack([idx[0], idx[idx2+1]])
    iend   = np.hstack([idx[idx2], idx[-1]])    

    return (istart, iend)



def load_laser(ppath, name):
    """
    load laser from recording ppath/name ...
    @RETURN: 
    @laser, vector of 0's and 1's 
    """ 
    # laser might be .mat or h5py file
    # perhaps we could find a better way of testing that
    file = os.path.join(ppath, name, 'laser_'+name+'.mat')
    try:
        laser = np.array(h5py.File(file,'r').get('laser'))
    except:
        laser = so.loadmat(file)['laser']
    return np.squeeze(laser)



def laser_protocol(ppath, name):
    """
    What was the stimulation frequency and the inter-stimulation interval for recording
    $ppath/$name?
    
    @Return:
        iinter-stimulation intervals, avg. inter-stimulation interval, frequency
    """
    laser = load_laser(ppath, name)
    SR = get_snr(ppath, name)
    
    # first get inter-stimulation interval
    (istart, iend) = laser_start_end(laser, SR)
    intv = np.diff(np.array(istart/float(SR)))
    d = intv/60.0
    print("The laser was turned on in average every %.2f min," % (np.mean(d)))
    print("with a min. interval of %.2f min and max. interval of %.2f min." % (np.min(d), np.max(d)))
    print("Laser stimulation lasted for %f s." % (np.mean(np.array(iend/float(SR)-istart/float(SR)).mean())))

    # print laser start times
    print("Start time of each laser trial:")
    j=1
    for t in istart:
        print("trial %d: %.2f" % (j, (t / float(SR)) / 60))
        j += 1

    # for each laser stimulation interval, check laser stimulation frequency
    dt = 1/float(SR)
    freq = []
    laser_up = []
    laser_down = []
    for (i,j) in zip(istart, iend):
        part = laser[i:j+1]
        (a,b) = laser_start_end(part, SR, 0.005)
        
        dur = (j-i+1)*dt
        freq.append(len(a) / dur)
        up_dur = (b-a+1)*dt*1000
        down_dur = (a[1:]-b[0:-1]-1)*dt*1000
        laser_up.append(np.mean(up_dur))
        laser_down.append(np.mean(down_dur))

    print(os.linesep + "Laser stimulation freq. was %.2f Hz," % np.mean(np.array(freq)))
    print("with laser up and down duration of %.2f and %.2f ms." % (np.mean(np.array(laser_up)), np.mean(np.array(laser_down))))
        
    return d, np.mean(d), np.mean(np.array(freq))

def load_trigger(ppath, name) :
    """load sleep-state detection signal
    recorded during closed loop stimulation
    from ppath, name ...
    @RETURN: @laser, sequency of 0's and 1's """
    file = os.path.join(ppath, name, 'rem_trig_'+name+'.mat')
    try:
        triggered = np.array( h5py.File(file,'r').get('rem_trig') )
    except:
        triggered = so.loadmat(file)['rem_trig']
    return np.squeeze(triggered)

def swap_eeg(ppath, rec, ch='EEG'):
    """
    swap EEG and EEG2 or EMG with EMG2 if $ch='EMG'
    """
    if ch == 'EEG':
        name = 'EEG'
    else:
        name = ch
    
    EEG = so.loadmat(os.path.join(ppath, rec, name+'.mat'))[name]
    EEG2 = so.loadmat(os.path.join(ppath, rec, name+'2.mat'))[name + '2']
        
    tmp = EEG
    EEG = EEG2
    EEG2 = tmp
    
    file_eeg1 = os.path.join(ppath, rec, '%s.mat' % name)
    file_eeg2 = os.path.join(ppath, rec, '%s2.mat' % name)
    so.savemat(file_eeg1, {name : EEG})        
    so.savemat(file_eeg2, {name+'2' : EEG2})



def eeg_conversion(ppath, rec, conv_factor=0.195):
    """
    multiply all EEG and EMG channels with the given
    conversion factor and write the conversion factor
    as parameter (conversion:) into the info file.
    Only if, there's no conversion factor in the info file
    specified, the conversion will be executed
    :param ppath: base filder
    :param rec: recording
    :param conv_factor: conversion factor
    :return: n/s
    """
    ifile = os.path.join(ppath, rec, 'info.txt')
    conv = get_infoparam(ifile, 'conversion')
    if len(conv) > 0:
        print("found conversion: parameter in info file")
        print("returning; no conversion!!!")
        return
    else:
        files = os.listdir(os.path.join(ppath, rec))
        files = [f for f in files if re.match('^EEG', f)]
        for f in files:
            name = re.split('\.', f)[0]

            EEG = so.loadmat(os.path.join(ppath, rec, name+'.mat'), squeeze_me=True)[name]
            EEG = EEG * conv_factor
            file_eeg = os.path.join(ppath, rec, '%s.mat' % name)
            print(file_eeg)
            so.savemat(file_eeg, {name: EEG})
            calculate_spectrum(ppath, rec)

        files = os.listdir(os.path.join(ppath, rec))
        files = [f for f in files if re.match('^EMG', f)]
        for f in files:
            name = re.split('\.', f)[0]

            EMG = so.loadmat(os.path.join(ppath, rec, name+'.mat'), squeeze_me=True)[name]
            EMG = EMG * conv_factor
            file_emg = os.path.join(ppath, rec, '%s.mat' % name)
            print(file_emg)
            so.savemat(file_emg, {name: EMG})
            calculate_spectrum(ppath, rec)

        add_infoparam(ifile, 'conversion', [conv_factor])



def video_pulse_detection(ppath, rec, SR=1000, iv = 0.01):
    """
    return index of each video frame onset
    ppath/rec  -  recording
    
    @Optional
    SR     -      sampling rate of EEG(!) recording
    iv     -      minimum time inverval (in seconds) between two frames
    
    @Return
    index of each video frame onset
    """
    
    V = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'videotime_' + rec + '.mat'))['video'])
    TS = np.arange(0, len(V))
    # indices where there's a jump in the signal
    t = TS[np.where(V<0.5)];
    if len(t) == 0:
        idx = []
        return idx
    
    # time points where the interval between jumps is longer than iv
    t2  = np.where(np.diff(t)*(1.0/SR)>=iv)[0]
    idx = np.concatenate(([t[0]],t[t2+1]))
    return idx



# SIGNAL PROCESSING ###########################################################
def my_lpfilter(x, w0, N=4):
    """
    create a lowpass Butterworth filter with a cutoff of w0 * the Nyquist rate. 
    The nice thing about this filter is that is has zero-phase distortion. 
    A conventional lowpass filter would introduce a phase lag.
    
    w0   -    filter cutoff; value between 0 and 1, where 1 corresponds to nyquist frequency.
              So if you want a filter with cutoff at x Hz, the corresponding w0 value is given by
                                w0 = 2 * x / sampling_rate
    N    -    order of filter 
    @Return:
        low-pass filtered signal
        
    See also my hp_filter, or my_bpfilter
    """    
    from scipy import signal
    
    b,a = signal.butter(N, w0)
    y = signal.filtfilt(b,a, x)
    
    return y



def my_hpfilter(x, w0, N=4):
    """
    create an N-th order highpass Butterworth filter with cutoff frequency w0 * sampling_rate/2
    """
    from scipy import signal
    # use scipy.signal.firwin to generate filter
    #taps = signal.firwin(numtaps, w0, pass_zero=False)
    #y = signal.lfilter(taps, 1.0, x)

    b,a = signal.butter(N, w0, 'high')
    y = signal.filtfilt(b,a, x)
        
    return y



def my_bpfilter(x, w0, w1, N=4):
    """
    create N-th order bandpass Butterworth filter with corner frequencies 
    w0*sampling_rate/2 and w1*sampling_rate/2
    """
    #from scipy import signal
    #taps = signal.firwin(numtaps, w0, pass_zero=False)
    #y = signal.lfilter(taps, 1.0, x)
    #return y
    from scipy import signal
    b,a = signal.butter(N, [w0, w1], 'bandpass')
    y = signal.filtfilt(b,a, x)
        
    return y



def downsample_vec(x, nbin):
    """
    y = downsample_vec(x, nbin)
    downsample the vector x by replacing nbin consecutive \
    bin by their mean \
    @RETURN: the downsampled vector 
    """
    n_down = int(np.floor(len(x) / nbin))
    x = x[0:n_down*nbin]
    x_down = np.zeros((n_down,))

    # 0 1 2 | 3 4 5 | 6 7 8 
    for i in range(nbin) :
        idx = list(range(i, int(n_down*nbin), int(nbin)))
        x_down += x[idx]

    return x_down / nbin



def smooth_data(x, sig):
    """
    y = smooth_data(x, sig)
    smooth data vector @x with gaussian kernel
    with standard deviation $sig
    """
    sig = float(sig)
    if sig == 0.0:
        return x
        
    # gaussian:
    gauss = lambda x, sig : (1/(sig*np.sqrt(2.*np.pi)))*np.exp(-(x*x)/(2.*sig*sig))

    p = 1000000000
    L = 10.
    while (p > p):
        L = L+10
        p = gauss((L, sig))

    #F = map(lambda x: gauss((x, sig)), np.arange(-L, L+1.))
    # py3:
    F = [gauss(x, sig) for x in np.arange(-L, L+1.)]
    F = F / np.sum(F)
    
    return scipy.signal.fftconvolve(x, F, 'same')



def power_spectrum(data, length, dt):
    """
    scipy's implementation of Welch's method using hanning window to estimate
    the power spectrum
    @Parameters
        data    -   time series; float vector!
        length  -   length of hanning window, even integer!
    
    @Return:
        power density, frequencies

        The function return power density in units V^2 / Hz
        Note that
            np.var(data) ~ np.sum(power density) * (frequencies[1]-frequencies[0])
    """
    f, pxx = scipy.signal.welch(data, fs=1/dt, window='hann', nperseg=int(length), noverlap=int(length/2))
    return pxx, f



def spectral_density(data, length, nfft, dt):
    """
    calculate the spectrogram for the time series given by data with time resolution dt
    The powerspectrum for each window of length $length is computed using 
    Welch's method.
    The windows for the powerspectrum calculation are half-overlapping. If length contains 5s of data,
    then the first windows goes from 0s to 5s, the second window from 2.5 to 7.5s, ...
    The last window ends at ceil(len(data)/length)*5s
    Another example, assume we have 13 s of data, with 5 s windows, the the powerdensity is calculated for the following
    time windows:
    0 -- 5, 2.5 -- 7.5, 5 -- 10, 7.5 -- 12.5, 10 -- 15
    In total there are thus 2*ceil(13/5)-1 = 5 windows
    The last window starts at 2*3-2 * (5/2) = 10 s

    Note: the returned time axis starts at time point goes from 0 to 10s in 2.5s steps

    @Parameters:
    data    -     time series
    length  -     window length of data used to calculate powerspectrum. 
                  Note that the time resolution of the spectrogram is length/2
    nfft    -     size of the window used to calculate the powerspectrum. 
                  determines the frequency resolution.
    @Return:
        Powspectrum, frequencies, time axis
    """
    n = len(data)
    k = int(np.ceil((1.0*n)/length))
    data = np.concatenate((data, np.zeros((length*k-n,))))
    fdt = length*dt/2 # time step for spectrogram
    t = np.arange(0, fdt*(2*k-2)+fdt/2.0, fdt)

    # frequency axis of spectrogram
    f = np.linspace(0, 1, int(np.ceil(nfft/2.0))+1) * (0.5/dt)
    # the power spectrum is calculated for 2*k-1 time points
    Pow = np.zeros((len(f), k*2-1))
    j = 0
    for i in range(0, k-2+1):
        w1=data[(length*i):(i+1)*length]
        w2=data[length*i+int(length/2):(i+1)*length+int(length/2)]
        Pow[:,j]   = power_spectrum(w1, nfft, dt)[0]
        Pow[:,j+1] = power_spectrum(w2, nfft, dt)[0]
        j += 2
    # last time points
    (Pow[:,j],f) = power_spectrum(data[length*(k-1):k*length], nfft, dt)
    
    return Pow, f, t



def calculate_spectrum(ppath, name, fres=0.5):
    """
    calculate EEG and EMG spectrogram used for sleep stage detection.
    Function assumes that data vectors EEG.mat and EMG.mat exist in recording
    folder ppath/name; these are used to calculate the powerspectrum
    fres   -   resolution of frequency axis
    
    all data saved in "true" mat files
    :return  EEG Spectrogram, EMG Spectrogram, frequency axis, time axis
    """
    
    SR = get_snr(ppath, name)
    swin = round(SR)*5
    fft_win = round(swin/5)
    if (fres == 1.0) or (fres == 1):
        fft_win = int(fft_win)
    elif fres == 0.5:
        fft_win = 2*int(fft_win)
    else:
        print("Resolution %f not allowed; please use either 1 or 0.5" % fres)
    
    (peeg2, pemg2) = (False, False)
    
    # Calculate EEG spectrogram
    EEG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EEG.mat'))['EEG'])
    Pxx, f, t = spectral_density(EEG, int(swin), int(fft_win), 1/SR)
    if os.path.isfile(os.path.join(ppath, name, 'EEG2.mat')):
        peeg2 = True
        EEG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EEG2.mat'))['EEG2'])
        Pxx2, f, t = spectral_density(EEG, int(swin), int(fft_win), 1/SR)        
    #save the stuff to a .mat file
    spfile = os.path.join(ppath, name, 'sp_' + name + '.mat')
    if peeg2 == True:
        so.savemat(spfile, {'SP':Pxx, 'SP2':Pxx2, 'freq':f, 'dt':t[1]-t[0],'t':t})
    else:
        so.savemat(spfile, {'SP':Pxx, 'freq':f, 'dt':t[1]-t[0],'t':t})


    # Calculate EMG spectrogram
    EMG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EMG.mat'))['EMG'])
    Qxx, f, t = spectral_density(EMG, int(swin), int(fft_win), 1/SR)
    if os.path.isfile(os.path.join(ppath, name, 'EMG2.mat')):
        pemg2 = True
        EMG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EMG2.mat'))['EMG2'])
        Qxx2, f, t = spectral_density(EMG, int(swin), int(fft_win), 1/SR)
    # save the stuff to .mat file
    spfile = os.path.join(ppath, name, 'msp_' + name + '.mat')
    if pemg2 == True:
        so.savemat(spfile, {'mSP':Qxx, 'mSP2':Qxx2, 'freq':f, 'dt':t[1]-t[0],'t':t})
    else:
        so.savemat(spfile, {'mSP':Qxx, 'freq':f, 'dt':t[1]-t[0],'t':t})
    
    return Pxx, Qxx, f, t



def whiten_spectrogram(ppath, name, fmax=50):
    P = so.loadmat(os.path.join(ppath, name,  'sp_' + name + '.mat'), squeeze_me=True)
    SPE = P['SP']
    freq = P['freq']
    ifreq = np.where(freq <= fmax)[0]

    SPE = SPE[0:300,:]
    nfilt = 5
    filt = np.ones((2, 2))
    filt = np.divide(filt, filt.sum())
    SPE = scipy.signal.convolve2d(SPE, filt, boundary='symm', mode='same')

    m = np.mean(SPE,axis=1)
    SPE -= np.tile(m, (SPE.shape[1], 1)).T
    SPE = SPE.T
    C = np.dot(SPE.T, SPE)

    [evals, L] = np.linalg.eigh(C)
    idx = np.argsort(evals)
    D = np.diag(np.sqrt(evals[idx]))
    L = L[:,idx]
    W = np.dot(L, np.dot(np.linalg.inv(D),np.dot(L.T,SPE.T)))

    nfilt = 5
    filt = np.ones((nfilt,nfilt))
    filt = np.divide(filt, filt.sum())
    W = scipy.signal.convolve2d(W, filt, boundary='symm', mode='same')

    return W, D, L



def laser_overlap_duration(ppath, recordings):
    pass


def recursive_spectrogram(ppath, name, sf=0.3, alpha=0.3, pplot=True):
    """
    calculate EEG/EMG spectrogram in a way that can be implemented by a closed-loop system.
    The spectrogram is temporally filtered using a recursive implementation of a lowpass filter
    @Parameters:
        ppath/name   -    mouse EEG recording
        sf           -    smoothing factor along frequency axis
        alpha        -    temporal lowpass filter time constant
        pplot        -    if pplot==True, plot figure 
    @Return:
        SE, SM       -    EEG, EMG spectrogram

    """
    
    EEG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EEG.mat'))['EEG'])
    EMG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EMG.mat'))['EMG'])
    len_eeg = len(EEG)
    fdt = 2.5
    SR = get_snr(ppath, name)
    # we calculate the powerspectrum for 5s windows
    swin = int(np.round(SR) * 5.0)
    # but we sample new data each 2.5 s
    swinh = int(swin/2.0)
    fft_win = int(swin / 5.0)
    # number of 2.5s long samples
    spoints = int(np.floor(len_eeg / swinh))

    SE = np.zeros((int(fft_win/2+1), spoints))
    SM = np.zeros((int(fft_win/2+1), spoints))
    print("Starting calculating spectrogram for %s..." % name)
    for i in range(2, spoints):
        # we take the last two swinh windows (the new 2.5 s long sample and the one from
        # the last iteration)
        x = EEG[(i-2)*swinh:i*swinh]
        
        [p, f] = power_spectrum(x.astype('float'), fft_win, 1.0/SR)
        p = smooth_data(p, sf)
        # recursive low pass filtering of spectrogram:
        # the current state is an estimate of the current sample and the previous state
        SE[:,i] = alpha*p + (1-alpha) * SE[:,i-1]

        # and the same of EMG        
        x = EMG[(i-2)*swinh:i*swinh]
        [p, f] = power_spectrum(x.astype('float'), fft_win, 1.0/SR)
        p = smooth_data(p, sf)
        SM[:,i] = alpha*p + (1-alpha) * SM[:,i-1]

    if pplot:
        # plot EEG spectrogram
        t = np.arange(0, SM.shape[1])*fdt
        plt.figure()
        ax1 = plt.subplot(211)
        im = np.where((f>=0) & (f<=30))[0]
        med = np.median(SE.max(axis=0))
        ax1.imshow(np.flipud(SE[im,:]), vmin=0, vmax=med*2)
        plt.xticks(())
        ix = list(range(0, 30, 10))
        fi = f[im][::-1]
        plt.yticks(ix, list(map(int, fi[ix])))
        box_off(ax1)
        plt.axis('tight')
        plt.ylabel('Freq (Hz)')
        
        # plot EMG amplitude
        ax2 = plt.subplot(212)
        im = np.where((f>=10) & (f<100))[0]
        df = np.mean(np.diff(f))
        # amplitude is the square root of the integral
        ax2.plot(t, np.sqrt(SM[im,:].sum(axis=0)*df)/1000.0)
        plt.xlim((0, t[-1]))
        plt.ylabel('EMG Ampl (mV)')
        plt.xlabel('Time (s)')
        box_off(ax2)
        plt.show(block=False)

    return SE, SM, f



def recursive_sleepstate_rem(ppath, recordings, sf=0.3, alpha=0.3, past_mu=0.2, std_thdelta = 1.5, past_len=120, sdt=2.5, psave=False, xemg=False):
    """
    predict a REM period only based on EEG/EMG history; the same algorithm is also used for 
    closed-loop REM sleep manipulation.
    The algorithm uses for REM sleep detection a threshold on delta power, EMG power, and theta/delta power.
    For theta/delta I use two thresholds: A hard (larger) threshold and a soft (lower) threshold. Initially,
    theta/delta has to cross the hard threshold to initiate a REM period. Then, as long as,
    theta/delta is above the soft threshold (and EMG power stays low) REM sleep continues.
    
    @Parameters:
        ppath        base folder with recordings
        recordings   list of recordings
        sf           smoothing factor for each powerspectrum
        past_mu      percentage (0 .. 1) of brain states that are allowed to have EMG power larger than threshold
                     during the last $past_len seconds
        past_len     window to calculate $past_mu
        std_thdelta  the hard theta/delta threshold is given by, mean(theta/delta) + $std_thdelta * std(theta/delta)  
        sdt          time bin for brain sttate, typically 2.5s
        psave        if True, save threshold parameters to file.
    
    """        
    idf = re.split('_', recordings[0])[0]
    past_len = int(np.round(past_len/sdt))
    
    # calculate spectrogram
    (SE, SM) = ([],[])
    for rec in recordings:
        A,B, freq = recursive_spectrogram(ppath, rec, sf=sf, alpha=alpha)       
        SE.append(A)
        SM.append(B)
        
    # fuse lists SE and SM
    SE = np.squeeze(reduce(lambda x,y: np.concatenate((x,y)), SE))
    if not xemg:
        SM = np.squeeze(reduce(lambda x,y: np.concatenate((x,y)), SM))
    else:
        SM = SE

    # EEG, EMG bands
    ntbins = SE.shape[1]
    r_delta = [0.5, 4]
    r_theta = [5,12]
    # EMG band
    r_mu = [300, 500]

    i_delta = np.where((freq >= r_delta[0]) & (freq <= r_delta[1]))[0]
    i_theta = np.where((freq >= r_theta[0]) & (freq <= r_theta[1]))[0]
    i_mu    = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    
    pow_delta = np.sum(SE[i_delta,:], axis=0)
    pow_theta = np.sum(SE[i_theta,:], axis=0)
    pow_mu = np.sum(SM[i_mu,:], axis=0)
    # theta/delta
    th_delta = np.divide(pow_theta, pow_delta)
    thr_th_delta1 = np.nanmean(th_delta) + std_thdelta*np.nanstd(th_delta)
    thr_th_delta2 = np.nanmean(th_delta) +  0.0*np.nanstd(th_delta)
    thr_delta = pow_delta.mean()
    thr_mu = pow_mu.mean() + 0.5*np.nanstd(pow_mu)


    ### The actual algorithm for REM detection
    rem_idx = np.zeros((ntbins,))
    prem = 0 # whether or not we are in REM
    for i in range(ntbins):
        
        if prem == 0 and pow_delta[i] < thr_delta and pow_mu[i] < thr_mu:
            ### could be REM
            
            if th_delta[i] > thr_th_delta1:
                ### we are potentially entering REM
                if (i - past_len) >= 0:
                    sstart = i-past_len
                else:
                    sstart = 0
                # count the percentage of brainstate bins with elevated EMG power
                c_mu = np.sum( np.where(pow_mu[sstart:i]>thr_mu)[0] ) / past_len
               
                if c_mu < past_mu:
                    ### we are in REM
                    prem = 1  # turn laser on
                    rem_idx[i] = 1
        
        # We are currently in REM; do we stay there?
        if prem == 1:
            ### REM continues, if theta/delta is larger than soft threshold and if there's
            ### no EMG activation
            if (th_delta[i] > thr_th_delta2) and (pow_mu[i] < thr_mu):
                rem_idx[i] = 1
            else:
                prem = 0 #turn laser off

    # Determine which channel is EEG, EMG
    ch_alloc = get_infoparam(os.path.join(ppath, recordings[0], 'info.txt'),  'ch_alloc')[0]
      
    # plot the whole stuff:
    # (1) spectrogram
    # (2) EMG Power
    # (3) Delta
    # (4) TH_Delta
    plt.figure()
    t = np.arange(0, sdt*(ntbins-1)+sdt/2.0, sdt)
    ax1 = plt.subplot(411)
    im = np.where((freq>=0) & (freq<=30))[0]
    med = np.median(SE.max(axis=0))
    ax1.imshow(np.flipud(SE[im,:]), vmin=0, vmax=med*2)
    plt.yticks(list(range(0, 31, 10)), list(range(30, -1, -10)))
    plt.ylabel('Freq. (Hz)')
    plt.axis('tight')

    ax2 = plt.subplot(412)
    ax2.plot(t, pow_mu, color='black')
    ax2.plot(t, np.ones((len(t),))*thr_mu, color='red')
    plt.ylabel('EMG Pow.')
    plt.xlim((t[0], t[-1]))

    ax3 = plt.subplot(413, sharex=ax2)
    ax3.plot(t, pow_delta, color='black')
    ax3.plot(t, np.ones((len(t),))*thr_delta, color='red')
    plt.ylabel('Delta Pow.')
    plt.xlim((t[0], t[-1]))

    ax4 = plt.subplot(414, sharex=ax3)
    ax4.plot(t, th_delta, color='black')
    ax4.plot(t, np.ones((len(t),))*thr_th_delta1, color='red')
    ax4.plot(t, np.ones((len(t),))*thr_th_delta2, color='pink')
    ax4.plot(t, rem_idx*thr_th_delta1, color='blue')
    plt.ylabel('Theta/Delta')
    plt.xlabel('Time (s)')
    plt.xlim((t[0], t[-1]))
    plt.show(block=False)
    
    # write config file
    if psave:
        cfile = os.path.join(ppath, idf + '_rem.txt')
        fid = open(cfile, 'w')
        fid.write(('IDF: %s'+os.linesep) % idf)
        fid.write(('ch_alloc: %s'+os.linesep) % ch_alloc)
        fid.write(('THR_DELTA: %.2f'+os.linesep) % thr_delta)
        fid.write(('THR_MU: %.2f'+os.linesep) % thr_mu)
        fid.write(('THR_TH_DELTA: %.2f %.2f'+os.linesep) % (thr_th_delta1, thr_th_delta2))
        fid.write(('STD_THDELTA: %.2f'+os.linesep) % std_thdelta)
        fid.write(('PAST_MU: %.2f'+os.linesep) % past_mu)
        fid.write(('SF: %.2f'+os.linesep) % sf)
        fid.write(('ALPHA: %.2f'+os.linesep) % alpha)
        if xemg:
            fid.write(('XEMG: %d'+os.linesep) % 1)
        else:
            fid.write(('XEMG: %d' + os.linesep) % 0)
        fid.close()
        print('wrote file %s' % cfile)



def recursive_sleepstate_rem_control(ppath, recordings, past_len=120, sdt=2.5, delay=120):
    """
    algorithm running laser control for REM sleep dependent activation/inhibtion.
    $delay s after a detected REM sleep period, the laser is turned on for the same duration. If a new REM period starts,
    the laser stops, but we keep track of the missing time. The next time is laser turns on again,
    it stays on for the duration of the most recent REM period + the remaining time.

    The algorithm for REM detection is the same as used forclosed-loop REM sleep manipulation.
    The function reads in the required parameters from the configuration file (MOUSEID_rem.txt)

    The algorithm uses for REM sleep detection a threshold on delta power, EMG power, and theta/delta power.
    For theta/delta I use two thresholds: A hard (larger) threshold and a soft (lower) threshold. Initially,
    theta/delta has to cross the hard threshold to initiate a REM period. Then, as long as,
    theta/delta is above the soft threshold (and EMG power stays low) REM sleep continues.

    @Parameters:
        ppath        base folder with recordings
        recordings   list of recordings
        past_len     window to calculate $past_mu
        sdt          time bin for brain sttate, typically 2.5s
        delay        delay to wait after a REM sleep periods ends, till the laser is turned on.
    """
    idf = re.split('_', recordings[0])[0]
    past_len = int(np.round(past_len/sdt))

    # load parameters
    cfile = os.path.join(ppath, idf + '_rem.txt')
    params = load_sleep_params(ppath, cfile)
    thr_th_delta1 = params['THR_TH_DELTA'][0]
    thr_th_delta2 = params['THR_TH_DELTA'][1]
    thr_delta = params['THR_DELTA'][0]
    thr_mu = params['THR_MU'][0]
    alpha = params['ALPHA'][0]
    sf = params['SF'][0]
    past_mu = params['PAST_MU'][0]
    xemg = params['XEMG'][0]

    # calculate spectrogram
    (SE, SM) = ([], [])
    for rec in recordings:
        A, B, freq = recursive_spectrogram(ppath, rec, sf=sf, alpha=alpha)
        SE.append(A)
        SM.append(B)

    # fuse lists SE and SM
    SE = np.squeeze(reduce(lambda x, y: np.concatenate((x, y)), SE))
    if not xemg:
        SM = np.squeeze(reduce(lambda x, y: np.concatenate((x, y)), SM))
    else:
        SM = SE

    # EEG, EMG bands
    ntbins = SE.shape[1]
    r_delta = [0.5, 4]
    r_theta = [5, 12]
    # EMG band
    r_mu = [300, 500]

    i_delta = np.where((freq >= r_delta[0]) & (freq <= r_delta[1]))[0]
    i_theta = np.where((freq >= r_theta[0]) & (freq <= r_theta[1]))[0]
    i_mu = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]

    pow_delta = np.sum(SE[i_delta, :], axis=0)
    pow_theta = np.sum(SE[i_theta, :], axis=0)
    pow_mu = np.sum(SM[i_mu, :], axis=0)
    th_delta = np.divide(pow_theta, pow_delta)


    ### The actual algorithm for REM detection
    rem_idx = np.zeros((ntbins,))
    prem = 0  # whether or not we are in REM

    # NEW variables:
    laser_idx = np.zeros((ntbins,))
    delay = int(np.round(delay/sdt))
    delay_count = 0
    curr_rem_dur = 0
    dur_count = 0
    on_delay = False
    laser_on = False
    for i in range(ntbins):

        if prem == 0 and pow_delta[i] < thr_delta and pow_mu[i] < thr_mu:
            ### could be REM

            if th_delta[i] > thr_th_delta1:
                ### we are potentially entering REM
                if (i - past_len) >= 0:
                    sstart = i - past_len
                else:
                    sstart = 0
                # count the percentage of brainstate bins with elevated EMG power
                c_mu = np.sum(np.where(pow_mu[sstart:i] > thr_mu)[0]) / past_len

                if c_mu < past_mu:
                    ### we are in REM
                    prem = 1  # turn laser on
                    rem_idx[i] = 1
                    curr_rem_dur += 1 #NEW

        # We are currently in REM; do we stay there?
        if prem == 1:
            ### REM continues, if theta/delta is larger than soft threshold and if there's
            ### no EMG activation
            if (th_delta[i] > thr_th_delta2) and (pow_mu[i] < thr_mu):
                rem_idx[i] = 1
                curr_rem_dur += 1
            else:
                prem = 0  # turn laser off
                dur_count += curr_rem_dur #NEW
                delay_count = delay #NEW
                curr_rem_dur = 0 #NEW
                on_delay = True #NEW

        # NEW:
        if on_delay:
            if prem == 0:
                delay_count -=1

                if delay_count == 0:
                    laser_on = True
                    on_delay = False

        if laser_on:
            if prem == 0:
                if dur_count >= 0:
                    dur_count -= 1
                    laser_idx[i] = 1
                else:
                    laser_on = False
            else:
                laser_on = False

    # plot the whole stuff:
    # (1) spectrogram
    # (2) EMG Power
    # (3) Delta
    # (4) TH_Delta
    plt.figure()
    t = np.arange(0, sdt*(ntbins-1)+sdt/2.0, sdt)
    ax1 = plt.subplot(411)
    im = np.where((freq>=0) & (freq<=30))[0]
    med = np.median(SE.max(axis=0))
    ax1.imshow(np.flipud(SE[im,:]), vmin=0, vmax=med*2)
    plt.yticks(list(range(0, 31, 10)), list(range(30, -1, -10)))
    plt.ylabel('Freq. (Hz)')
    plt.axis('tight')

    ax2 = plt.subplot(412)
    ax2.plot(t, pow_mu, color='black')
    ax2.plot(t, np.ones((len(t),))*thr_mu, color='red')
    plt.ylabel('EMG Pow.')
    plt.xlim((t[0], t[-1]))

    ax3 = plt.subplot(413, sharex=ax2)
    ax3.plot(t, pow_delta, color='black')
    ax3.plot(t, np.ones((len(t),))*thr_delta, color='red')
    plt.ylabel('Delta Pow.')
    plt.xlim((t[0], t[-1]))

    ax4 = plt.subplot(414, sharex=ax3)
    ax4.plot(t, th_delta, color='black')
    ax4.plot(t, np.ones((len(t),))*thr_th_delta1, color='red')
    ax4.plot(t, np.ones((len(t),))*thr_th_delta2, color='pink')
    ax4.plot(t, rem_idx*thr_th_delta1, color='green', label='REM')
    ax4.plot(t, laser_idx * thr_th_delta1, color='blue', label='Laser')
    plt.ylabel('Theta/Delta')
    plt.xlabel('Time (s)')
    plt.xlim((t[0], t[-1]))
    plt.legend()
    plt.show(block=False)



def load_sleep_params(path, param_file):
    """
    load parameter file generated by &recursive_sleepstate_rem || &recursive_sleepstate_nrem
    @Return:
        Dictionary: Parameter --> Value
    """    
    fid = open(os.path.join(path, param_file), 'r')
    lines = fid.readlines()
    params = {}
    for line in lines:
        if re.match('^[\S_]+:', line):
            a = re.split('\s+', line)
            key = a[0][:-1]
            params[key] = a[1:-1]
            
    # transform number strings to floats
    for k in params:
        vals = params[k] 
        new_vals = []
        for v in vals:
            if re.match('^[\d\.]+$', v):
                new_vals.append(float(v))
            else:
                new_vals.append(v)
        params[k] = new_vals
                    
    return params
            
        

def recursive_sleepstate_nrem(ppath, recordings, sf=0.3, alpha=0.3, std_thdelta = 1.5, sdt=2.5, psave=False, xemg=False):
    """
    predict NREMs period only based on EEG/EMG history; the same algorithm is also used for
    closed-loop NREM sleep manipulation.
    The algorithm uses for NREM sleep detection thresholds for delta power, EMG power, and theta/delta power.
    For delta I use two thresholds: A hard (larger) threshold and a soft (lower) threshold. Initially,
    theta/delta has to cross the hard threshold to initiate a NREM period. Then, as long as,
    theta/delta is above the soft threshold (and EMG power stays low) REM sleep continues.
    The values for hard and soft threshold are fitted using a Gaussian mixture model

    :param ppath: base folder
    :param recordings: list of recordings
    :param sf: smoothing factor for each powerspectrum
    :param alpha: spatial smoothing factor
    :param std_thdelta: factor to set threshold for theta/delta
    :param sdt: time step of brain state classification, typically 2.5 s
    :param psave: save parameters to text file?
    :param xemg: use EEG instead of EMG?
    """

    # to fit Gaussian mixture model to delta power distributino
    from sklearn import mixture

    idf = re.split('_', recordings[0])[0]

    # calculate spectrogram
    (SE, SM) = ([],[])
    for rec in recordings:
        A,B, freq = recursive_spectrogram(ppath, rec, sf=sf, alpha=alpha)       
        SE.append(A)
        SM.append(B)
        
    # fuse lists SE and SM
    SE = np.squeeze(reduce(lambda x,y: np.concatenate((x,y)), SE))
    if not xemg:
        SM = np.squeeze(reduce(lambda x,y: np.concatenate((x,y)), SM))
    else:
        SM = SE

    # EEG, EMG bands        
    ntbins = SE.shape[1]
    r_delta = [0.5, 4]
    r_theta = [5,12]
    # EMG band
    r_mu = [300, 500]

    i_delta = np.where((freq >= r_delta[0]) & (freq <= r_delta[1]))[0]
    i_theta = np.where((freq >= r_theta[0]) & (freq <= r_theta[1]))[0]
    i_mu    = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    
    pow_delta = np.sum(SE[i_delta,:], axis=0)
    pow_theta = np.sum(SE[i_theta,:], axis=0)
    pow_mu = np.sum(SM[i_mu,:], axis=0)
    # theta/delta
    th_delta = np.divide(pow_theta, pow_delta)
    thr_th_delta1 = np.nanmean(th_delta) + std_thdelta*np.nanstd(th_delta)
    thr_th_delta2 = np.nanmean(th_delta) +  0.0*np.nanstd(th_delta)
    thr_mu = pow_mu.mean() + 0.5*np.nanstd(pow_mu)

    med_delta = np.median(pow_delta)
    pow_delta_fit = pow_delta[np.where(pow_delta<=3*med_delta)]
    
    # fit Gaussian mixture model to delta power
    # see http://www.astroml.org/book_figures/chapter4/fig_GMM_1D.html
    gm = mixture.GaussianMixture(n_components=2)
    fit = gm.fit(pow_delta_fit.reshape(-1, 1))
    means = np.squeeze(fit.means_)

    x = np.arange(0, med_delta*3, 100)
    plt.figure()
    plt.hist(pow_delta_fit, 100, normed=True, histtype='stepfilled', alpha=0.4)
    
    pdb.set_trace()
    logprob = fit.score_samples(x.reshape(-1,1))
    responsibilities = fit.predict_proba(x.reshape((-1,1)))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    plt.plot(x, pdf, '-k')
    plt.plot(x, pdf_individual, '--k')
    plt.xlim((0, med_delta*3))

    plt.ylabel('p(x)')
    plt.xlabel('x = Delta power')
    
    # get point where curves cut each other
    if means[0] < means[1]:
        idx = np.where((x>= means[0]) & (x<= means[1]))[0]
    else:
        idx = np.where((x >= means[1]) & (x <= means[0]))[0]
    imin = np.argmin(pdf[idx])
    xcut = x[idx[0]+imin]
    
    plt.plot(xcut, pdf[idx[0]+imin], 'ro')
    
    ilow = np.argmin(np.abs(x-means[0]))
    plt.plot(x[ilow], pdf[ilow], 'bo')

    ihigh = np.argmin(np.abs(x-means[1]))
    plt.plot(x[ihigh], pdf[ihigh], 'go')
    plt.show(block=False)

    # set parameters for hard and soft delta thresholds
    tmp = np.array([x[ihigh], xcut, x[ilow]])
    tmp.sort()
    thr_delta1 = tmp[-1] # x[ihigh]; right peak of distribution
    thr_delta2 = tmp[1]  # trough of distribution

    # NREM yes or no according to thresholds
    # However, this variable does not directly control whether laser should
    # be on or off; whether NREM sleep is really on or off is determined
    # by nrem_idx; if pnrem_hidden == 1, then all threshold critera, but not
    # sleep history criteria are fulfilled
    pnrem_hidden = 0
    # if nrem_idx[i] == 1, time point i is NREM
    nrem_idx = np.zeros((ntbins,), dtype='int8')

    # NREM stays on after thresholds are NOT fulfilled to avoid interruptions by microarousals
    grace_period = int(20 / sdt)

    # nrem_delay: NREM only starts with some delay
    nrem_delay = int(10 / sdt)

    grace_count = grace_period
    delay_count = nrem_delay
    for i in range(ntbins):
        if pnrem_hidden == 0:
            ### Entering NREM:
            # Delta power laser than high threshold
            if pow_delta[i] > thr_delta1 and pow_mu[i] < thr_mu  and th_delta[i] < thr_th_delta1:
                ### NOT-NREM -> NREM
                pnrem_hidden = 1
                nrem_idx[i] = 0
                delay_count -= 1

                # we are fully in NREM, that's why grace_count is reset:
                grace_count = grace_period
            else:
                ### NOT-NREM -> NOT-NREM
                if grace_count > 0:
                    grace_count -= 1
                    nrem_idx[i] = 1
                else:
                    nrem_idx[i] = 0
        else:
            ### pnrem_hidden == 1
            if pow_delta[i] > thr_delta2 and pow_mu[i] < thr_mu and th_delta[i] < thr_th_delta1:
                if delay_count > 0:
                    delay_count -= 1
                    nrem_idx[i] = 0
                else :
                    nrem_idx[i] = 1
            else:
                ### Exit NREM -> NOT-NREM
                # were are fully out of NREM, so delay_count can be reset:
                delay_count = nrem_delay
                pnrem_hidden = 0
                
                if grace_count > 0:
                    grace_count -= 1
                    nrem_idx[i] = 1

    #### figure ##############################################
    plt.figure()
    t = np.arange(0, sdt * (ntbins - 1) + sdt / 2.0, sdt)
    ax1 = plt.subplot(411)
    im = np.where((freq >= 0) & (freq <= 30))[0]
    med = np.median(SE.max(axis=0))
    ax1.imshow(np.flipud(SE[im, :]), vmin=0, vmax=med * 2, cmap='jet')
    ax1.pcolorfast(t, freq[im], np.flipud(SE[im, :]), vmin=0, vmax=med * 2, cmap='jet')
    plt.yticks(list(range(0, 31, 10)), list(range(30, -1, -10)))
    plt.ylabel('Freq. (Hz)')
    plt.axis('tight')

    ax2 = plt.subplot(412, sharex=ax1)
    ax2.plot(t, pow_mu, color='black')
    ax2.plot(t, np.ones((len(t),)) * thr_mu, color='red')
    plt.ylabel('EMG Pow.')
    plt.xlim((t[0], t[-1]))

    ax3 = plt.subplot(413, sharex=ax2)
    ax3.plot(t, pow_delta, color='black')
    ax3.plot(t, np.ones((len(t),)) * thr_delta1, color='red')
    ax3.plot(t, np.ones((len(t),)) * thr_delta2, color=[1, 0.6, 0.6])
    ax3.plot(t, nrem_idx * thr_delta1, color=[0.6, 0.6, 0.6])

    plt.ylabel('Delta Pow.')
    plt.xlim((t[0], t[-1]))

    ax4 = plt.subplot(414, sharex=ax3)
    ax4.plot(t, th_delta, color='black')
    ax4.plot(t, np.ones((len(t),)) * thr_th_delta1, color='red')

    plt.ylabel('Theta/Delta')
    plt.xlabel('Time (s)')
    plt.xlim((t[0], t[-1]))
    plt.show(block=False)

    # Determine which channel is EEG, EMG
    ch_alloc = get_infoparam(os.path.join(ppath, recordings[0], 'info.txt'),  'ch_alloc')[0]

    # write config file
    if psave:
        cfile = os.path.join(ppath, idf + '_nrem.txt')
        fid = open(cfile, 'w')
        fid.write(('IDF: %s' + os.linesep) % idf)
        fid.write(('ch_alloc: %s' + os.linesep) % ch_alloc)
        fid.write(('THR_DELTA: %.2f %.2f' + os.linesep) % (thr_delta1, thr_delta2))
        fid.write(('THR_MU: %.2f' + os.linesep) % thr_mu)
        fid.write(('THR_TH_DELTA: %.2f %.2f' + os.linesep) % (thr_th_delta1, thr_th_delta2))
        fid.write(('STD_THDELTA: %.2f' + os.linesep) % std_thdelta)
        fid.write(('SF: %.2f' + os.linesep) % sf)
        fid.write(('ALPHA: %.2f' + os.linesep) % alpha)
        if xemg:
            fid.write(('XEMG: %d' + os.linesep) % 1)
        else:
            fid.write(('XEMG: %d' + os.linesep) % 0)
        fid.close()
        print('wrote file %s' % cfile)



def rem_online_analysis(ppath, recordings, backup='', single_mode=False, fig_file=''):
    """
    analyze results from closed-loop experiments
    :param ppath: base folder
    :param recordings: list of strings, recordinds
    :param backup: string, potential second backup folder with recordings
    :param single_mode: boolean, if True, average across all REM periods (irrespective of mouse)
           and plot each single REM period as dot
    :return: df, pd.DataFrame, with control and experimental REM durations as data columns
    """
    import pandas as pd

    if type(recordings) != list:
        recordings = [recordings]

    paths = dict()
    for rec in recordings:
        if os.path.isdir(os.path.join(ppath, rec)):
            paths[rec] = ppath
        else:
            paths[rec] = backup

    mice = dict()
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    if len(mice) == 1:
        single_mode=True

    dur_exp = {m:[] for m in mice}
    dur_ctr = {m:[] for m in mice}

    for rec in recordings:
        idf = re.split('_', rec)[0]
        M,S = load_stateidx(paths[rec], rec)
        sr = get_snr(paths[rec], rec)
        nbin = int(np.round(sr)*2.5)
        dt = (1.0/sr)*nbin

        laser = load_laser(paths[rec], rec)
        rem_trig = so.loadmat(os.path.join(paths[rec], rec, 'rem_trig_%s.mat'%rec), squeeze_me=True)['rem_trig']

        laser = downsample_vec(laser, nbin)
        laser[np.where(laser>0)] = 1
        rem_trig = downsample_vec(rem_trig, nbin)
        rem_trig[np.where(rem_trig>0)] = 1

        laser_idx = np.where(laser==1)[0]
        rem_idx = np.where(rem_trig==1)[0]

        # REM sequences from offline analysis (assumed to be the
        # "ground truth"
        seq = get_sequences(np.where(M==1)[0])
        for s in seq:
            # check true REM sequences overlapping with online detected sequences
            if len(np.intersect1d(s, rem_idx)) > 0:
                drn = (s[-1]-s[0]+1)*dt
                # does the sequence overlap with laser?
                if len(np.intersect1d(s, laser_idx))>0:
                    dur_exp[idf].append(drn)
                else:
                    dur_ctr[idf].append(drn)

    data = {'exp':[], 'ctr':[]}

    # if single_mode put all REM periods together,
    # otherwise average across REM periods for each mouse
    if len(mice) == 1 or single_mode==True:
        for m in mice:
            data['exp'] += dur_exp[m]
            data['ctr'] += dur_ctr[m]
    else:
        for idf in dur_ctr:
            dur_ctr[idf] = np.array(dur_ctr[idf]).mean()
            dur_exp[idf] = np.array(dur_exp[idf]).mean()
        data['exp'] = np.array(list(dur_exp.values()))
        data['ctr'] = np.array(list(dur_ctr.values()))

    df = pd.DataFrame({'ctr':pd.Series(data['ctr']), 'exp' : pd.Series(data['exp'])})

    # plot everything
    if not single_mode:
        plt.ion()
        plt.figure()
        ax = plt.axes([0.2, 0.15, 0.3, 0.7])
        df_mean = df.mean()
        plt.bar([1], [df_mean['ctr']], color='grey', label='W/o Laser')
        plt.bar([2], [df_mean['exp']], color='blue', label='With laser')
        plt.xticks([1,2])
        box_off(ax)
        #ax.set_xticklabels(['ctr', 'exp'], rotation=30)
        plt.ylabel('REM duration (s)')
        for (a,b) in zip(df['ctr'], df['exp']):
            plt.plot([1,2], [a,b], color='black')
        plt.legend(bbox_to_anchor=(0., 1.0, 1., .102), loc=3, mode='expand', ncol=1, frameon=False)
    else:
        plt.figure()
        ax = plt.axes([0.2, 0.15, 0.3, 0.7])
        df_mean = df.mean()
        plt.bar([1], [df_mean['ctr']], color='grey')
        plt.bar([2], [df_mean['exp']], color='blue')
        plt.xticks([1,2])
        box_off(ax)
        #ax.set_xticklabels(['ctr', 'exp'], rotation=30)
        plt.ylabel('REM duration (s)')
        a = df['ctr']
        b = df['exp']
        plt.plot(np.ones((len(a),)), a, '.', color='black', label='W/o Laser')
        plt.plot(2*np.ones((len(b),)), b, '.', color='black', label='With laser')
        plt.legend(bbox_to_anchor=(0., 1.0, 1., .102), loc=3, mode='expand', ncol=1, frameon=False)
    plt.show()

    if len(fig_file) > 0:
        save_figure(fig_file)

    return df



def online_homeostasis(ppath, recordings, backup='', mode=0, single_mode=False, pplot=True):
    """
    Further analysis of data obtained from closed loop stimulation
    Assume the sleep structure looks like this
    R R R R W W N N N N N W W N N N N R R R R R
    REM_pre   -- inter REM ----      REM_post

    REM_pre is the duration of the first REM period, inter-REM is everything between REM_pre and the
    next REM period REM_post.

    The function calculates the inter REM duration after REM periods with laser and after REM periods w/o laser

    :param ppath: base folder
    :param recordings: list of recording, or file listing
    :param backup: backup folder for $ppath
    :param mode: mode == 0, calculate complete inter REM duration
                 mode == 2, only calculate duration of wake in inter REM periods
                 mode == 3, only calculate duration of NREM in inter REM periods
    :param single_mode: consider each single recording, instead of mice
    :param pplot: if True, plot figure; errorbars show 95% confidence intervals,
           calculated using bootstrapping

    :return: df, if single_mode == True $df is a pandas DataFrame:
             REM    iREM   laser
             mouse - mouse ID
             REM   - REM duration
             iREM  - inter REM duration after REM periods with laser
             laser - 'y' or 'n'; depending on whether laser was on during REM sleep period (for "REM") or during the
                     preceding REM sleep period (for "iREM")

             if single_mode == False, mouse is the data frame index
    """
    if type(recordings) != list:
        recordings = [recordings]

    paths = dict()
    for rec in recordings:
        if os.path.isdir(os.path.join(ppath, rec)):
            paths[rec] = ppath
        else:
            paths[rec] = backup

    mice = dict()
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())
    if len(mice) == 1:
        single_mode=True

    remdur_exp = {m:[] for m in mice}
    remdur_ctr = {m:[] for m in mice}
    itdur_exp = {m:[] for m in mice}
    itdur_ctr = {m:[] for m in mice}

    for rec in recordings:
        idf = re.split('_', rec)[0]
        M = load_stateidx(paths[rec], rec)[0]
        sr = get_snr(paths[rec], rec)
        nbin = int(np.round(sr)*2.5)
        dt = (1.0/sr)*nbin

        laser = load_laser(paths[rec], rec)
        rem_trig = so.loadmat(os.path.join(paths[rec], rec, 'rem_trig_%s.mat' % rec), squeeze_me=True)['rem_trig']

        laser = downsample_vec(laser, nbin)
        laser[np.where(laser>0)] = 1
        rem_trig = downsample_vec(rem_trig, nbin)
        rem_trig[np.where(rem_trig>0)] = 1

        laser_idx = np.where(laser==1)[0]
        rem_idx = np.where(rem_trig==1)[0]

        # REM sequences from offline analysis (assumed to be the
        # "ground truth"
        seq = get_sequences(np.where(M==1)[0])
        for (p,q) in zip(seq[0:-1], seq[1:]):
            # check true REM sequences overlapping with online detected sequences
            if len(np.intersect1d(p, rem_idx)) > 0:
                drn = (p[-1]-p[0]+1)*dt
                #it_drn = (q[0]-p[-1])*dt

                it_M = M[p[-1]+1:q[0]]
                if mode == 0:
                    it_drn = len(it_M)*dt
                elif mode == 2:
                    it_drn = len(np.where(it_M==2)[0]) * dt
                else:
                    it_drn = len(np.where(it_M == 3)[0]) * dt

                #it_idx = range(p[-1]+1, q[0])
                #it_drn = len(np.where(M[it_idx] == 3)[0])*dt
                # does the sequence overlap with laser?
                if len(np.intersect1d(p, laser_idx))>0:
                    remdur_exp[idf].append(drn)
                    itdur_exp[idf].append(it_drn)
                else:
                    remdur_ctr[idf].append(drn)
                    itdur_ctr[idf].append(it_drn)


    # if single_mode put all REM periods together,
    # otherwise average across REM periods for each mouse
    if len(mice) == 1 or single_mode==True:
        data = {'itexp':[], 'itctr':[], 'remexp':[], 'remctr':[]}

        for m in mice:

            data['itexp'] += itdur_exp[m]
            data['itctr'] += itdur_ctr[m]
            data['remexp'] += remdur_exp[m]
            data['remctr'] += remdur_ctr[m]

        df = pd.DataFrame({'REM': data['remexp']+data['remctr'], 'iREM':data['itexp']+data['itctr'], 'laser': ['y']*len(data['remexp']) + ['n']*len(data['remctr'])})

    else:
        for idf in mice:
            itdur_ctr[idf] = np.array(itdur_ctr[idf]).mean()
            itdur_exp[idf] = np.array(itdur_exp[idf]).mean()
            remdur_ctr[idf] = np.array(remdur_ctr[idf]).mean()
            remdur_exp[idf] = np.array(remdur_exp[idf]).mean()

        data = {}
        for s in ['itexp', 'itctr', 'remexp', 'remctr']:
            data[s] = np.zeros((len(mice),))
        i = 0
        for m in mice:
            data['itexp'][i] = itdur_exp[m]
            data['itctr'][i] = itdur_ctr[m]
            data['remexp'][i] = remdur_exp[m]
            data['remctr'][i] = remdur_ctr[m]
            i += 1

        df = pd.DataFrame({'REM': np.concatenate((data['remexp'], data['remctr'])),
                           'iREM': np.concatenate((data['itexp'], data['itctr'])),
                           'laser': ['y']*len(mice) + ['n']*len(mice), 
                           'mouse': mice+mice})

    if pplot and not single_mode:
        dfm = pd.melt(df, id_vars=['laser', 'mouse'], var_name='state')
        sns.set_style('whitegrid')


        plt.ion()

        sns.barplot(data=dfm, hue='laser', x='state', y='value', palette=['blue', 'gray'])
        sns.swarmplot(data=dfm, hue='laser', x='state', y='value', dodge=True, color='black')

        sns.despine()
        plt.ylabel('Duration (s)')


    if pplot and single_mode:
        dfm = pd.melt(df, id_vars=['laser'], var_name='state')
        pdb.set_trace()
        plt.ion()
        plt.figure()
        sns.set(style="whitegrid")
        
        #sns.swarmplot(data=df[['itctr', 'itexp']], color='black')
        #sns.barplot(data=df[['itctr', 'itexp']], palette=['gray', 'blue'], errcolor='black')

        sns.barplot(data=dfm, hue='laser', x='state', y='value', palette=['blue', 'gray'])
        sns.swarmplot(data=dfm, hue='laser', x='state', y='value', dodge=True, color='black')
        sns.despine()
        plt.ylabel('Duration (s)')

    return df



### FUNCTIONS USED BY SLEEP_STATE #####################################################
def get_sequences(idx, ibreak=1) :  
    """
    get_sequences(idx, ibreak=1)
    idx     -    np.vector of indices
    @RETURN:
    seq     -    list of np.vectors
    """
    diff = idx[1:] - idx[0:-1]
    breaks = np.nonzero(diff>ibreak)[0]
    breaks = np.append(breaks, len(idx)-1)
    
    seq = []    
    iold = 0
    for i in breaks:
        r = list(range(iold, i+1))
        seq.append(idx[r])
        iold = i+1
        
    return seq


def threshold_crossing(data, th, ilen, ibreak, m):
    """
    seq = threshold_crossing(data, th, ilen, ibreak, m)
    """
    if m>=0:
        idx = np.where(data>=th)[0]
    else:
        idx = np.where(data<=th)[0]

    # gather sequences
    j = 0
    seq = []
    while (j <= len(idx)-1):
        s = [idx[j]]
        
        for k in range(j+1,len(idx)):
            if (idx[k] - idx[k-1]-1) <= ibreak:
                # add j to sequence
                s.append(idx[k])
            else:
                break

        if (s[-1] - s[0]+1) >= ilen and not(s[0] in [i[1] for i in seq]):
            seq.append((s[0], s[-1]))
        
        if j == len(idx)-1:
            break           
        j=k
        
    return seq



def closest_precessor(seq, i):
    """
    find the preceding element in seq which is closest to i
    helper function for sleep_state
    """
    tmp = seq-i;
    d = np.where(tmp<0)[0]
    
    if len(d)>0:
        id = seq[d[-1]];
    else:
        id = 0;
    
    return id


def write_remidx(M, K, ppath, name, mode=1) :
    """
    rewrite_remidx(idx, states, ppath, name)
    replace the indices idx in the remidx file of recording name
    with the assignment given in states
    """
   
    if mode == 0 :
        outfile = os.path.join(ppath, name, 'remidx_' + name + '.txt')
    else :
        outfile = os.path.join(ppath, name, 'remidx_' + name + '_corr.txt')

    f = open(outfile, 'w')
    s = ["%d\t%d\n" % (i,j) for (i,j) in zip(M[0,:],K)]
    f.writelines(s)
    f.close()

def rewrite_remidx(M, K, ppath, name):
    """
    rewrite_remidx(idx, states, ppath, name)
    replace the indices idx in the remidx file of recording name
    with the assignment given in states
    """
    outfile = os.path.join(ppath, name, 'remidx_' + name + '.txt')

    f = open(outfile, 'w')
    s = ["%d\t%d\n" % (i,j) for (i,j) in zip(M[0,:],K)]
    f.writelines(s)
    f.close()

def rewrite_stateprob(M,K,ppath,name):
    outfile = os.path.join(ppath, name, 'stateprob_'+name+'.txt')
    f = open(outfile, 'w')
    s = ["%d\t%d\n" % (i,j) for (i,j) in zip(M[0,:],K)]
    f.writelines(s)
    f.close()


#######################################################################################


### MANIPULATING FIGURES ##############################################################
def set_fontsize(fs):
    import matplotlib
    matplotlib.rcParams.update({'font.size': fs})


def set_fontarial():
    """
    set Arial as default font
    """
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = "Arial"


def save_figure(fig_file):
    # alternative way of setting nice fonts:
    
    #matplotlib.rcParams['pdf.fonttype'] = 42
    #matplotlib.rcParams['ps.fonttype'] = 42
    #matplotlib.pylab.savefig(fig_file, dpi=300)

    #matplotlib.rcParams['text.usetex'] = False
    #matplotlib.rcParams['text.usetex'] = True
    plt.savefig(fig_file, bbox_inches="tight", dpi=200)
    #matplotlib.rcParams['text.usetex'] = False


def box_off(ax):
    """
    similar to Matlab's box off
    """
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  

#######################################################################################



def sleep_state(ppath, name, th_delta_std=1, mu_std=0, sf=1, sf_delta=3, pwrite=0, pplot=True, pemg=True, vmax=2.5, pspec_norm=False, use_idx=[]):
    """
    automatic sleep state detection based on
    delta, theta, sigma, gamma and EMG power.

    New: use also sigma band: that's very helpful to classify pre-REM periods
    as NREM; otherwise they tend to be classified as wake.
    Gamma peaks nicely pick up during microarousals.
    My strategy is the following:
    I smooth delta band a lot to avoid strong fragmentation of sleep; but to
    still pick up microarousals I use the gamma power.

    spectrogram data has to be calculated before using calculate_spectrum

    :param ppath: base folder
    :param name: recording name
    :param th_delta_std: threshold for theta/delta band is calculated as mean(theta/delta) + th_delta_std*std(theta/delta)
    :param mu_std: threshold for EMG power is calculate as "mean(EMG) + mu_std * mean(EMG)
    :param sf: smoothing factor for gamma and sigma power
    :param sf_delta: smoothing factor for delta power
    :param pwrite: if True, save sleep classification to file remidx_$name.txt
    :param pplot: if True, plot figures
    :param pemg: if True, use EMG as EMG, otherwise use EEG gamma power instead
    :param vmax: float, set maximum of color range of EEG heatmap.
    :param pspec_norm: boolean, if True, normalized EEG spectrogram by deviding each frequency band by its mean; only affects
           plotting, no effect on sleep state calculation
    :param use_idx: list, if not empty, use only given indices to calculate sleep state
    :return:
    """
    PRE_WAKE_REM = 30.0
    
    # Minimum Duration and Break in 
    # high theta/delta, high emg, and high delta sequences
    # Synatax: duration(i,0) is the minimum duration of sequency i
    # duration(i,2) is maximal break duration allowed in a sequence
    # of state i
    duration = np.zeros((5,2))
    # high theta/delta
    duration[0,:] = [5,15]
    # high emg
    duration[1,:] = [0, 5]
    # high delta
    duration[2,:] = [10, 10]
    # high sigma
    duration[3,:] = [10, 10]
    # gamma
    duration[4,:] = [0, 5]
    
    # Frequency Bands/Ranges for delta, theta, and, gamma
    r_delta = [0.5, 4]
    r_sigma = [12, 20]
    r_theta = [5,12]
    # EMG band
    r_mu = [50, 500]
    if not pemg:
        r_mu = [250, 500]
    # high gamma power
    r_gamma = [100, 150]

    #load EEG and EMG spectrum, calculated by calculate_spectrum
    P = so.loadmat(os.path.join(ppath, name,  'sp_' + name + '.mat'))
    if pemg:
        Q = so.loadmat(os.path.join(ppath, name, 'msp_' + name + '.mat'))
    else:
        #Q = so.loadmat(os.path.join(ppath, name, 'sp_' + name + '.mat'))
        pass
    
    SPEEG = np.squeeze(P['SP'])
    if pemg == 1:
        SPEMG = np.squeeze(Q['mSP'])
    else:
        SPEMG = np.squeeze(P['SP'])

    if use_idx == []:
        use_idx = range(0, SPEEG.shape[1])
    freq  = np.squeeze(P['freq'])
    t     = np.squeeze(P['t'])
    dt    = float(np.squeeze(P['dt']))
    N     = len(t)
    duration = np.divide(duration,dt)
    
    # get indices for frequency bands
    i_delta = np.where((freq >= r_delta[0]) & (freq <= r_delta[1]))[0]
    i_theta = np.where((freq >= r_theta[0]) & (freq <= r_theta[1]))[0]
    i_mu    = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    i_sigma = np.where((freq >= r_sigma[0]) & (freq <= r_sigma[1]))[0]
    i_gamma = np.where((freq >= r_gamma[0]) & (freq <= r_gamma[1]))[0]

    p_delta = smooth_data( SPEEG[i_delta,:].mean(axis=0), sf_delta )
    p_theta = smooth_data( SPEEG[i_theta,:].mean(axis=0), 0 )
    # now filtering for EMG to pick up microarousals
    p_mu    = smooth_data( SPEMG[i_mu,:].mean(axis=0), sf )
    p_sigma = smooth_data( SPEEG[i_sigma,:].mean(axis=0), sf )
    p_gamma = smooth_data( SPEEG[i_gamma,:].mean(axis=0), 0 )

    th_delta = np.divide(p_theta, p_delta)
    #th_delta = smooth_data(th_delta, 2);

    seq = {}
    seq['high_theta'] = threshold_crossing(th_delta, np.nanmean(th_delta[use_idx])+th_delta_std*np.nanstd(th_delta[use_idx]), \
       duration[0,1], duration[0,1], 1)
    seq['high_emg'] = threshold_crossing(p_mu, np.nanmean(p_mu[use_idx])+mu_std*np.nanstd(p_mu[use_idx]), \
       duration[1,0], duration[1,1], 1)
    seq['high_delta'] = threshold_crossing(p_delta, np.nanmean(p_delta[use_idx]), duration[2,0], duration[2,1], 1)
    seq['high_sigma'] = threshold_crossing(p_sigma, np.nanmean(p_sigma[use_idx]), duration[3,0], duration[3,1], 1)
    seq['high_gamma'] = threshold_crossing(p_gamma, np.nanmean(p_gamma[use_idx]), duration[4,0], duration[4,1], 1)

    # Sleep-State Rules
    idx = {}
    for k in seq:
        tmp = [list(range(i,j+1)) for (i,j) in seq[k]]
        # now idea why this works to flatten a list
        # idx[k] = sum(tmp, [])
        # alternative that I understand:
        if len(tmp) == 0:
            idx[k] = np.array([])
        else:
            idx[k] = np.array(reduce(lambda x,y: x+y, tmp))

    idx['low_emg']    = np.setdiff1d(np.arange(0,N), np.array(idx['high_emg']))
    idx['low_delta'] = np.setdiff1d(np.arange(0,N), np.array(idx['high_delta']))
    idx['low_theta'] = np.setdiff1d(np.arange(0,N), np.array(idx['high_theta']))
        
        
    #REM Sleep: thdel up, emg down, delta down    
    a = np.intersect1d(idx['high_theta'], idx['low_delta'])
    # non high_emg phases
    b = np.setdiff1d(a, idx['high_emg'])
    rem = get_sequences(b, duration[0,1])
    rem_idx = reduce(lambda x,y: np.concatenate((x,y)), rem)


    # SWS Sleep
    # delta high, no theta, no emg
    a = np.setdiff1d(idx['high_delta'], idx['high_emg']) # no emg activation
    b = np.setdiff1d(a, idx['high_theta'])               # no theta;
    sws = get_sequences(b)
    sws_idx = reduce(lambda x,y: np.concatenate((x,y)), sws)
    #print a

    # Wake
    # low delta + high emg and not rem
    a = np.unique(np.union1d(idx['low_delta'], idx['high_emg']))
    b = np.setdiff1d(a, rem_idx)
    wake = get_sequences(b)
    wake_idx = reduce(lambda x,y: np.concatenate((x,y)), wake)

    # sequences with low delta, high sigma and low emg are NREM
    a = np.intersect1d(np.intersect1d(idx['high_sigma'], idx['low_delta']), idx['low_emg'])
    a = np.setdiff1d(a, rem_idx)
    sws_idx = np.unique(np.union1d(a, sws_idx))
    wake_idx = np.setdiff1d(wake_idx, a)

    #NREM sequences with high gamma are wake
    a = np.intersect1d(sws_idx, idx['high_gamma'])    
    sws_idx = np.setdiff1d(sws_idx, a)
    wake_idx = np.unique(np.union1d(wake_idx,a))

    # Wake and Theta
    wake_motion_idx = np.intersect1d(wake_idx, idx['high_theta'])

    # Wake w/o Theta
    wake_nomotion_idx = np.setdiff1d(wake_idx, idx['low_theta'])

    # Are there overlapping sequences?
    a = np.intersect1d(np.intersect1d(rem_idx, wake_idx), sws_idx);

    # Are there undefined sequences?
    undef_idx = np.setdiff1d(np.setdiff1d(np.setdiff1d(np.arange(0,N), rem_idx), wake_idx), sws_idx);
    
    # Wake wins over SWS
    sws_idx = np.setdiff1d(sws_idx, wake_idx);

    # Special rules
    # if there's a REM sequence directly following a short wake sequence (PRE_WAKE_REM),
    # this wake sequence goes to SWS
    # NREM to REM transitions are sometimes mistaken as quite wake periods
    for rem_seq in rem:
        if len(rem_seq) > 0:
            irem_start = rem_seq[0]
            # is there wake in the preceding bin?
            if irem_start-1 in wake_idx:
                # get the closest sws bin in the preceding history
                isws_end = closest_precessor(sws_idx, irem_start)
                if (irem_start - isws_end)*dt < PRE_WAKE_REM:
                    new_rem = np.arange(isws_end+1,irem_start)
                    rem_idx = np.union1d(rem_idx, new_rem)
                    wake_idx = np.setdiff1d(wake_idx, new_rem)
                else:
                    new_wake = rem_seq
                    wake_idx = np.union1d(wake_idx, new_wake)
                    rem_idx = np.setdiff1d(rem_idx, new_wake)


    # two different representations for the results:
    S = {}
    S['rem']    = rem_idx
    S['nrem']   = sws_idx
    S['wake']   = wake_idx
    S['awake']  = wake_motion_idx
    S['qwake']  = wake_nomotion_idx
    
    M = np.zeros((N,))
    if len(rem_idx) > 0:
        M[rem_idx]           = 1
    if len(wake_idx) > 0:
        M[wake_idx]          = 2
    if len(sws_idx) > 0:
        M[sws_idx]           = 3
    if len(undef_idx) > 0:
        M[undef_idx]         = 0
    
    # write sleep annotation to file
    if pwrite:
        outfile = os.path.join(ppath, name, 'remidx_' + name + '.txt')
        print("writing annotation to %s" % outfile)
        f = open(outfile, 'w')
        s = ["%d\t%d\n" % (i,j) for (i,j) in zip(M,np.zeros((N,)))]
        f.writelines(s)
        f.close()
        
    # nice plotting
    plt.ion()
    if pplot:
        plt.figure(figsize=(18,9))
        axes1=plt.axes([0.1, 0.9, 0.8, 0.05])
        A = np.zeros((1,len(M)))
        A[0,:] = M
        cmap = plt.cm.jet
        my_map = cmap.from_list('ha', [[0,0,0], [0,1,1],[0.5,0,1], [0.8, 0.8, 0.8]], 4)
        #tmp = axes1.imshow(A, vmin=0, vmax=3)
        tmp = axes1.pcolorfast(t, [0,1], A, vmin=0, vmax=3)
        tmp.set_cmap(my_map)
        axes1.axis('tight')
        tmp.axes.get_xaxis().set_visible(False)
        tmp.axes.get_yaxis().set_visible(False)
        box_off(axes1)
        
        # show spectrogram
        axes2=plt.axes([0.1, 0.75, 0.8, 0.1], sharex=axes1)
        ifreq = np.where(freq <= 30)[0]
        med = np.median(SPEEG.max(axis=0))
        if pspec_norm:
            ifreq = np.where(freq <= 80)[0]

            filt = np.ones((6, 1))
            filt = filt / np.sum(filt)
            SPEEG = scipy.signal.convolve2d(SPEEG, filt, mode='same')
            spec_mean = SPEEG.mean(axis=1)
            SPEEG = np.divide(SPEEG, np.repeat([spec_mean], SPEEG.shape[1], axis=0).T)
            med = np.median(SPEEG.max(axis=0))
            axes2.pcolorfast(t, freq[ifreq], SPEEG[ifreq, :], vmax = med*vmax, cmap='jet')
        else:
            axes2.pcolorfast(t, freq[ifreq], SPEEG[ifreq, :], vmax=med * vmax, cmap='jet')
        axes2.axis('tight')        
        plt.ylabel('Freq (Hz)')
        box_off(axes2)

        # show delta power
        axes3=plt.axes([0.1, 0.6, 0.8, 0.1], sharex=axes2)
        axes3.plot(t,p_delta, color='gray')
        plt.ylabel('Delta (a.u.)')
        plt.xlim((t[0], t[-1]))
        seq = get_sequences(S['nrem'])
        #for s in seq:
        #    plt.plot(t[s],p_delta[s], color='red')
        s = idx['high_delta']
        seq = get_sequences(s)
        for s in seq:
            plt.plot(t[s],p_delta[s], color='red')
        box_off(axes3)

        axes4=plt.axes([0.1, 0.45, 0.8, 0.1], sharex=axes3)
        axes4.plot(t,p_sigma, color='gray')
        plt.ylabel('Sigma (a.u.)')
        plt.xlim((t[0], t[-1]))
        s = idx['high_sigma']
        seq = get_sequences(s)
        for s in seq:
            plt.plot(t[s],p_sigma[s], color='red')
        box_off(axes4)

        axes5=plt.axes([0.1, 0.31, 0.8, 0.1], sharex=axes4)
        axes5.plot(t,th_delta, color='gray')
        plt.ylabel('Th/Delta (a.u.)')
        plt.xlim((t[0], t[-1]))
        s = idx['high_theta']
        seq = get_sequences(s)
        for s in seq:            
            plt.plot(t[s],th_delta[s], color='red')
        box_off(axes5)

        axes6=plt.axes([0.1, 0.17, 0.8, 0.1], sharex=axes5)
        axes6.plot(t,p_gamma, color='gray')
        plt.ylabel('Gamma (a.u.)')
        plt.xlim((t[0], t[-1]))
        s = idx['high_gamma']
        seq = get_sequences(s)
        for s in seq:
            plt.plot(t[s],p_gamma[s], color='red')
        box_off(axes6)

        axes7=plt.axes([0.1, 0.03, 0.8, 0.1], sharex=axes6)
        axes7.plot(t,p_mu, color='gray')        
        plt.xlabel('Time (s)')
        plt.ylabel('EMG (a.u.)')
        plt.xlim((t[0], t[-1]))
        s = idx['high_emg']
        seq = get_sequences(s)
        for s in seq:
            plt.plot(t[s],p_mu[s], color='red')
        box_off(axes7)
        plt.show()

        
        # 2nd figure showing distribution of different bands
        plt.figure(figsize=(20,3))
        axes1 = plt.axes([0.05, 0.1, 0.13, 0.8])
        plt.hist(p_delta, bins=100)
        plt.plot(np.nanmean(p_delta), 10, 'ro')
        plt.title('delta')
        plt.ylabel('# Occurances')
        box_off(axes1)
        
        axes1 = plt.axes([0.25, 0.1, 0.13, 0.8])
        plt.hist(th_delta, bins=100)
        plt.plot(np.nanmean(th_delta)+th_delta_std*np.nanstd(th_delta), 10, 'ro')
        plt.title('theta/delta')
        box_off(axes1)

        axes1 = plt.axes([0.45, 0.1, 0.13, 0.8])
        plt.hist(p_sigma, bins=100)
        plt.plot(np.nanmean(p_sigma), 10, 'ro')
        plt.title('sigma')
        box_off(axes1)
                
        axes1 = plt.axes([0.65, 0.1, 0.13, 0.8])
        plt.hist(p_gamma, bins=100)
        plt.plot(np.nanmean(p_gamma), 10, 'ro')
        plt.title('gamma')
        box_off(axes1)
        
        axes1 = plt.axes([0.85, 0.1, 0.13, 0.8])
        plt.hist(p_mu, bins=100)
        plt.plot(np.nanmean(p_mu)+np.nanstd(p_mu), 10, 'ro')
        plt.title('EMG')
        plt.show(block=False)
        box_off(axes1)
        
        plt.show()
    
    return M,S
    

def plot_hypnograms(ppath, recordings, tbin=0, unit='h', ma_thr=20, title='', tstart=0, tend=-1):
    """
    plot all hypnograms specified in @recordings
    :param ppath: base folder
    :param recordings: list of recordings
    :param tbin: tbin for xticks
    :param unit: time unit; h - hour, min - minute, s - second
    :param ma_thr: float, wake periods shorter than $ma_thr are considered as microarousals and further converted to NREM
    :param tstart: float, start time point (in seconds!) of hypnograms
    :param tend: float, last shown time point (in seconds!)
    :param title: optional title for figure
    """
    recordings = recordings[::-1]

    sr = get_snr(ppath, recordings[0])
    nbin = int(np.round(sr) * 2.5)
    dt_sec = (1.0 / sr) * nbin
    istart = int(np.round(tstart/dt_sec))

    dt = dt_sec
    if unit == 'h':
        dt /= 3600
    elif unit == 'min':
        dt /= 60

    rec_len = dict()
    irec = 0
    ny = (1.0-0.2) / len(recordings)
    dy = ny * 0.75
    cmap = plt.cm.jet
    my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.5, 0, 1], [0.8, 0.8, 0.8]], 4)
    plt.ion()
    plt.figure(figsize=(9,4))
    axes = []
    for rec in recordings:
        M,K = load_stateidx(ppath, rec)
        kcut = np.where(K<0)[0]
        #M = M[kcut]
        #M[kcut] = 0

        if tend == -1:
            iend = len(M)
        else:
            iend = int(tend/dt_sec)
        M = M[istart:iend]

        if ma_thr>0:
            seq = get_sequences(np.where(M==2)[0])
            for s in seq:
                if len(s)*dt_sec <= ma_thr:
                    M[s] = 3

        rec_len[rec] = len(M)*dt
        t = np.arange(0, len(M))*dt

        ax = plt.axes([0.05, ny*irec+0.15, 0.75, dy])

        tmp = ax.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3, cmap=my_map)
        box_off(ax)
        ax.axis('tight')
        tmp.axes.get_yaxis().set_visible(False)
        if irec > 0:
            tmp.axes.get_xaxis().set_visible(False)
        if irec == 0:
            plt.xlabel('Time (%s)' % unit)

        irec += 1
        axes.append(ax)
    if len(title) > 0:
        plt.title(title)

    max_dur = max(rec_len.values())
    if tbin > 0:
        xtick = np.arange(0, max_dur, tbin)

    for (ax, rec) in zip(axes, recordings):
        ax.set_xlim([0, max_dur])
        if tbin > 0:
            ax.set_xticks(xtick)
        ax.text(max_dur+max_dur*0.01, 0.5, rec)

    plt.show()



def laser_triggered_eeg(ppath, name, pre, post, f_max, pnorm=2, pplot=False, psave=False, tstart=0, tend=-1,
                        peeg2=False, vm=2.5, prune_trials=True, mu=[10, 200]):
    """
    calculate laser triggered, averaged EEG and EMG spectrum
    ppath   -    base folder containing mouse recordings
    name    -    recording
    pre     -    time before laser
    post    -    time after laser
    f_max   -    calculate/plot frequencies up to frequency f_max
    p_norm  -    normalization: 
                 pnorm = 0, no normalization
                 pnorm = 1, normalize each frequency band by its average power
                 pnorm = 2, normalize each frequency band by the average power 
                            during the preceding baseline period
    vm      -    float to set saturation level of colormap
    pplot   -    plot figure yes=True, no=False
    psave   -    save the figure, yes=True, no = False
    """
    SR = get_snr(ppath, name)
    NBIN = np.round(2.5*SR)
    lsr = load_laser(ppath, name)
    idxs, idxe = laser_start_end(lsr)
    laser_dur = np.mean((idxe-idxs)/SR)
    print('Average laser duration: %f; Number of trials %d' % (laser_dur, len(idxs)))

    # downsample EEG time to spectrogram time    
    idxs = [int(i/NBIN) for i in idxs]
    idxe = [int(i/NBIN) for i in idxe]
    #load EEG and EMG
    if not peeg2:
        P = so.loadmat(os.path.join(ppath, name,  'sp_' + name + '.mat'))
    else:
        P = so.loadmat(os.path.join(ppath, name,  'sp_' + name + '.mat'))
    Q = so.loadmat(os.path.join(ppath, name, 'msp_' + name + '.mat'))

    if not peeg2:
        SPEEG = np.squeeze(P['SP'])
    else:
        SPEEG = np.squeeze(P['SP2'])
    SPEMG = np.squeeze(Q['mSP'])
    freq  = np.squeeze(P['freq'])
    t     = np.squeeze(P['t'])
    dt    = float(np.squeeze(P['dt']))

    ifreq = np.where(freq<=f_max)[0]
    ipre  = int(np.round(pre/dt))
    ipost = int(np.round(post/dt))

    speeg_mean = SPEEG.mean(axis=1)
    spemg_mean = SPEMG.mean(axis=1)

    #pdb.set_trace()
    if tend > -1:
        i = np.where((np.array(idxs)*dt >= tstart) & (np.array(idxs)*dt <= tend))[0]
    else:
        i = np.where(np.array(idxs)*dt >= tstart)[0]

    idxs = [idxs[j] for j in i]
    idxe = [idxe[j] for j in i]

    skips = []
    skipe = []
    if prune_trials:
        for (i,j) in zip(idxs, idxe):
            A = SPEEG[0,i-ipre:i+ipost+1] / speeg_mean[0]
            B = SPEMG[0,i-ipre:i+ipost+1] / spemg_mean[0]
            k = np.where(A >= np.median(A)*50)[0]
            l = np.where(B >= np.median(B)*500)[0]
            if len(k) > 0 or len(l) > 0:
                skips.append(i)
                skipe.append(j)
    print("kicking out %d trials" % len(skips))

    idxs_new = []
    idxe_new = []
    for (i,j) in zip(idxs, idxe):
        if not i in skips:
            idxs_new.append(i)
            idxe_new.append(j)
    idxs = idxs_new
    idxe = idxe_new

    # Spectrogram for EEG and EMG normalized by average power in each frequency band
    if pnorm == 1:
        SPEEG = np.divide(SPEEG, np.repeat(speeg_mean, len(t)).reshape(len(speeg_mean), len(t)))
        SPEMG = np.divide(SPEMG, np.repeat(spemg_mean, len(t)).reshape(len(spemg_mean), len(t)))

    speeg_parts = []
    spemg_parts = []
    for (i,j) in zip(idxs, idxe):
        if i>=ipre and j+ipost < len(t):
            eeg_part = SPEEG[ifreq,i-ipre:i+ipost+1]
            speeg_parts.append(eeg_part)
            spemg_parts.append(SPEMG[ifreq,i-ipre:i+ipost+1])
            
    
    EEGLsr = np.array(speeg_parts).mean(axis=0)
    EMGLsr = np.array(spemg_parts).mean(axis=0)
    
    # smooth spectrogram
    nfilt = 3
    filt = np.ones((nfilt,nfilt))
    filt = np.divide(filt, filt.sum())
    EEGLsr = scipy.signal.convolve2d(EEGLsr, filt, boundary='symm', mode='same')
    EMGLsr = scipy.signal.convolve2d(EMGLsr, filt, boundary='symm', mode='same')

    if pnorm == 2:    
        for i in range(EEGLsr.shape[0]):
            EEGLsr[i,:] = np.divide(EEGLsr[i,:], np.sum(np.abs(EEGLsr[i,0:ipre]))/(1.0*ipre))
            EMGLsr[i,:] = np.divide(EMGLsr[i,:], np.sum(np.abs(EMGLsr[i,0:ipre]))/(1.0*ipre))
    
    # get time axis    
    dt = (1.0/SR)*NBIN
    #t = np.arange(-ipre,ipost+1)*dt
    t = np.linspace(-ipre*dt, ipost*dt, ipre+ipost+1)
    f = freq[ifreq]

    if pplot:
        # get rid of boxes around matplotlib plots
        def box_off(ax):
            ax.spines["top"].set_visible(False)    
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()  
            ax.get_yaxis().tick_left() 
                
        plt.ion()
        plt.figure(figsize=(10,8))
        ax = plt.axes([0.1, 0.55, 0.4, 0.35])
        plt.pcolormesh(t,f,EEGLsr, vmin=0, vmax=np.median(EEGLsr)*vm, cmap='jet')
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        box_off(ax)
        plt.title('EEG', fontsize=12)
        cbar = plt.colorbar()
        if pnorm >0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        
        ax = plt.axes([0.62, 0.55, 0.35, 0.35])
        #ipre = np.where(t<0)[0]
        ilsr = np.where((t>=0) & (t<=120))[0]        
        plt.plot(f,EEGLsr[:,0:ipre].mean(axis=1), color='gray', label='baseline', lw=2)
        plt.plot(f,EEGLsr[:,ilsr].mean(axis=1), color='blue', label='laser', lw=2)
        box_off(ax)
        plt.xlabel('Freq. (Hz)')
        plt.ylabel('Power (uV^2)')
        #plt.legend(loc=0)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
        
        ax = plt.axes([0.1, 0.1, 0.4, 0.35])
        plt.pcolormesh(t,f,EMGLsr, cmap='jet')
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        box_off(ax)    
        plt.title('EMG', fontsize=12)
        cbar = plt.colorbar()
        if pnorm >0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')

        
        ax = plt.axes([0.62, 0.1, 0.35, 0.35])
        mf = np.where((f>=mu[0]) & (f <= mu[1]))[0]
        df = f[1]-f[0]
        # amplitude is square root of (integral over each frequency)
        avg_emg = np.sqrt(EMGLsr[mf,:].sum(axis=0)*df)    
        m = np.max(avg_emg)*1.5
        plt.plot([0,0], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.plot([laser_dur,laser_dur], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.xlim((t[0], t[-1]))
        plt.ylim((0,m))
        plt.plot(t,avg_emg, color='black', lw=2)
        box_off(ax)     
        plt.xlabel('Time (s)')
        plt.ylabel('EMG ampl. (uV)')
        
        plt.show()
        
        if psave:
            img_file = os.path.join(ppath, name, 'fig_'+name+'_spec.png')
            save_figure(img_file)
                
    return EEGLsr, EMGLsr, freq[ifreq], t
    


def laser_triggered_eeg_avg(ppath, recordings, pre, post, f_max, laser_dur, pnorm=1, pplot=True, tstart=0, tend=-1,
                            vm=[], cb_ticks=0, mu=[10, 200], fig_file=''):
    """
    calculate average spectrogram for all recordings listed in @recordings; for averaging take
    mouse identity into account
    :param ppath: base folder
    :param recordings: list of recordings
    :param pre: time before laser onset
    :param post: time after laser onset
    :param f_max: maximum frequency shown for EEG spectrogram
    :param laser_dur: duration of laser stimulation
    :param pnorm: type of normalization for spectrogram (see laser_triggered_eeg)
    :param pplot: pplot==0 - no figure;
                  pplot==1 - conventional figure;
                  pplot==2 - pretty figure showing EEG spectrogram
                             along with EMG amplitude
    :param tstart: only consider laser trials with laser onset after tstart seconds
    :param tend: only consider laser trials with laser onset before tend seconds
    :param vm: saturation of heatmap for EEG spectrogram
    :param cb_ticks: ticks for colorbar (only applies for pplot==2)
    :param mu: frequencies for EMG amplitude calculation
    :param fig_file: if specified, save figure to given file
    :return: n/a
    """
    EEGSpec = {}
    EMGSpec = {}
    mice = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not(idf in mice):
            mice.append(idf)
        EEGSpec[idf] = []
        EMGSpec[idf] = []
    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        EEG, EMG, f, t = laser_triggered_eeg(ppath, rec, pre, post, f_max, pnorm=pnorm, pplot=False, psave=False, tstart=tstart, tend=tend)
        EEGSpec[idf].append(EEG)
        EMGSpec[idf].append(EMG)
    
    for idf in mice:
        EEGSpec[idf] = np.array(EEGSpec[idf]).mean(axis=0)
        EMGSpec[idf] = np.array(EMGSpec[idf]).mean(axis=0)
        
    EEGLsr = np.array([EEGSpec[k] for k in mice]).mean(axis=0)
    EMGLsr = np.array([EMGSpec[k] for k in mice]).mean(axis=0)

    mf = np.where((f >= mu[0]) & (f <= mu[1]))[0]
    df = f[1] - f[0]

    EMGAmpl = np.zeros((len(mice), EEGLsr.shape[1]))
    i=0
    for idf in mice:
        # amplitude is square root of (integral over each frequency)
        EMGAmpl[i,:] = np.sqrt(EMGSpec[idf][mf,:].sum(axis=0)*df)
        i += 1
    avg_emg = EMGAmpl.mean(axis=0)
    sem_emg = EMGAmpl.std(axis=0) #/ np.sqrt(len(mice))

    if pplot==1:
        plt.ion()
        plt.figure(figsize=(12,10))
        ax = plt.axes([0.1, 0.55, 0.4, 0.4])
        plt.pcolormesh(t,f,EEGLsr, cmap='jet', vmin=vm[0], vmax=vm[1])
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        box_off(ax)
        plt.title('EEG')
        cbar = plt.colorbar()
        if pnorm >0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')
        
        ax = plt.axes([0.6, 0.55, 0.3, 0.4])
        ipre = np.where(t<0)[0]
        ilsr = np.where((t>=0) & (t<=120))[0]        
        plt.plot(f,EEGLsr[:,ipre].mean(axis=1), color='gray', label='baseline', lw=2)
        plt.plot(f,EEGLsr[:,ilsr].mean(axis=1), color='blue', label='laser', lw=2)
        box_off(ax)
        plt.xlabel('Freq. (Hz)')
        plt.ylabel('Power (uV^2)')
        #plt.legend(loc=0)
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0.)
        
        ax = plt.axes([0.1, 0.05, 0.4, 0.4])
        plt.pcolormesh(t,f,EMGLsr, cmap='jet')
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')    
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        box_off(ax)    
        plt.title('EMG')
        cbar = plt.colorbar()
        if pnorm >0:
            cbar.set_label('Rel. Power')
        else:
            cbar.set_label('Power uV^2s')

        ax = plt.axes([0.6, 0.05, 0.3, 0.4])

        m = np.max(avg_emg)*1.5
        plt.plot([0,0], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.plot([laser_dur,laser_dur], [0,np.max(avg_emg)*1.5], color=(0,0,0))
        plt.xlim((t[0], t[-1]))
        plt.ylim((0,m))
        plt.plot(t,avg_emg, color='black', lw=2)
        plt.fill_between(t, avg_emg-sem_emg, avg_emg+sem_emg)
        box_off(ax)     
        plt.xlabel('Time (s)')
        plt.ylabel('EMG ampl. (uV)')
        
        plt.show()

    elif pplot==2:
        plt.figure()
        if len(vm) > 0:
            cb_ticks = vm

        # plot EEG spectrogram
        axes_cbar = plt.axes([0.8, 0.75, 0.1, 0.2])
        ax = plt.axes([0.1, 0.55, 0.75, 0.4])
        im=ax.pcolorfast(t,f,EEGLsr, cmap='jet', vmin=vm[0], vmax=vm[1])
        plt.plot([0,0], [0,f[-1]], color=(1,1,1))
        plt.plot([laser_dur,laser_dur], [0,f[-1]], color=(1,1,1))
        plt.axis('tight')
        plt.xlabel('Time (s)')
        plt.ylabel('Freq (Hz)')
        box_off(ax)

        # colorbar for EEG spectrogram
        cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0, orientation='vertical')
        cb.set_label('Rel. Power')
        cb.ax.xaxis.set_ticks_position("bottom")
        cb.ax.xaxis.set_label_position('top')
        if len(cb_ticks) > 0:
            cb.set_ticks(cb_ticks)
        axes_cbar.set_alpha(0.0)
        axes_cbar.spines["top"].set_visible(False)
        axes_cbar.spines["right"].set_visible(False)
        axes_cbar.spines["bottom"].set_visible(False)
        axes_cbar.spines["left"].set_visible(False)
        axes_cbar.axes.get_xaxis().set_visible(False)
        axes_cbar.axes.get_yaxis().set_visible(False)

        # EMG amplitude
        ax = plt.axes([0.1, 0.1, 0.75, 0.3])
        m = np.max(avg_emg) * 1.5
        ax.add_patch(patches.Rectangle((0, 0), laser_dur, np.max(avg_emg)*1.5, facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1]))
        plt.xlim((t[0], t[-1]))
        plt.ylim((0, m))
        plt.fill_between(t, avg_emg-sem_emg, avg_emg+sem_emg, color='gray', zorder=2)
        plt.plot(t, avg_emg, color='black', lw=2)
        box_off(ax)
        plt.xlabel('Time (s)')
        plt.ylabel('EMG ampl. ($\mathrm{\mu V}$)')

        plt.show()

    if len(fig_file) > 0:
        save_figure(fig_file)



def laser_brainstate(ppath, recordings, pre, post, pplot=True, fig_file='', start_time=0, end_time=-1,
                     ma_thr=0, edge=0, sf=0, cond=0, single_mode=False, backup=''):
    """
    calculate laser triggered probability of REM, Wake, NREM
    ppath        -    base folder holding all recording
    recordings   -    list of recording
    pre          -    time before laser onset
    post         -    time after laser onset

    @Optional:
    pplot        -    pplot==True: plot figure
    fig_file     -    specify filename including ending, if you wish to save figure
    start_time   -    in [s], only consider laser onsets starting after $start_time
    end_time     -    in [s], only consider laset onsets starting before $end_time
    sf           -    smoothing factor for Gaussian kernel; if sf=0, no filtering
    edge         -    only use if $sf > 0: to avoid smoothing artifacts, set edge to a value > 0, e.g. 20
    ma_thr       -    if > 0, smooth out microarousals with duration < $ma_thr
    cond         -    cond==0: consider all trials; cond==[1,2,3] only plot trials,
                      where mouse was in REM, Wake, or NREM as laser turned on
    single_mode  -    if True, plot every single mouse
    backup       -    optional backup folder; if specified each single recording folder can be either on $ppath or $backup;
                      if it's on both folders, the version in ppath is used

    @Return:  BS, t, df
    BS, t, df    -   BS,t: np.array(mice x time x [REM|Wake|NREM]), np.array(time vector)
                     df: pd.DataFrame.
                     columns are:
                     mouse_id  REM, NREM, Wake, Lsr

                     Lsr has three values: 0 - before laser, 1 - during laser, 2 - after laser
                     if laser was on for laser_dur s, then
                     df[df['Lsr'] == 0]['REM'] is the average % of REM sleep during laser stimulation for each mouse
                     df[df['Lsr'] == 0]['REM'] is the average % of REM sleep
                     during the laser_dur s long time interval preceding laser onset.
                     df[df['Lsr'] == 2]['REM'] is the average during the time inverval of duration laser_dur that
                     directly follows laser stimulation
    """
    if type(recordings) != list:
        recordings = [recordings]

    rec_paths = dict()
    for rec in recordings:
        if os.path.isdir(os.path.join(ppath, rec)):
            rec_paths[rec] = ppath
        else:
            rec_paths[rec] = backup

    pre += edge
    post += edge

    BrainstateDict = {}
    mouse_order = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        BrainstateDict[idf] = []
        if not idf in mouse_order:
            mouse_order.append(idf)
    nmice = len(BrainstateDict)

    for rec in recordings:
        ppath = rec_paths[rec]
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        dt = NBIN * 1.0/SR
        istart_time = int(np.round(start_time / dt))
        M = load_stateidx(ppath, rec)[0]
        if end_time == -1:
            iend_time = len(M)
        else:
            iend_time = int(np.round(end_time / dt))

        seq = get_sequences(np.where(M==2)[0])
        for s in seq:
            if len(s)*dt <= ma_thr:
                M[s] = 3

        (idxs, idxe) = laser_start_end(load_laser(ppath, rec))
        idf = re.split('_', rec)[0]

        ipre  = int(np.round(pre/dt))
        ipost = int(np.round(post/dt))

        idxs = [int(i/NBIN) for i in idxs]
        idxe = [int(i/NBIN) for i in idxe]
        laser_dur = np.mean((np.array(idxe) - np.array(idxs))) * dt

        for (i,j) in zip(idxs, idxe):
            if i>=ipre and i+ipost<=len(M)-1 and i>istart_time and i < iend_time:
                bs = M[i-ipre:i+ipost+1]                
                BrainstateDict[idf].append(bs) 
    # I assume here that every recording has same dt
    t = np.linspace(-ipre*dt, ipost*dt, ipre+ipost+1)
    # first time point where the laser was fully on (during the complete bin).
    izero = np.where(t>0)[0][0]
    # the first time bin overlapping with laser is then
    izero -= 1
    # @BS: mouse x time x state
    BS = np.zeros((nmice, len(t), 3))
    Trials = []
    imouse = 0
    for mouse in mouse_order:
        if cond==0:
            M = np.array(BrainstateDict[mouse])
            Trials.append(M)
            for state in range(1,4):
                C = np.zeros(M.shape)
                C[np.where(M==state)] = 1
                BS[imouse,:,state-1] = C.mean(axis=0)
        if cond>0:
            M = BrainstateDict[mouse]
            Msel = []
            for trial in M:
                if trial[izero] == cond:
                    Msel.append(trial)
            M = np.array(Msel)
            Trials.append(M)
            for state in range(1,4):
                C = np.zeros(M.shape)
                C[np.where(M==state)] = 1
                BS[imouse,:,state-1] = C.mean(axis=0)
        imouse += 1

    # flatten Trials
    Trials = reduce(lambda x,y: np.concatenate((x,y), axis=0),  Trials)

    if sf > 0:
        for state in [2, 1, 0]:
            for i in range(nmice):
                BS[i, :, state] = smooth_data(BS[i, :, state], sf)

    nmice = imouse
    if pplot:
        state_label = {0:'REM', 1:'Wake', 2:'NREM'}
        it = np.where((t >= -pre + edge) & (t <= post - edge))[0]

        plt.ion()

        if not single_mode:
            plt.figure()
            ax = plt.axes([0.15, 0.15, 0.6, 0.7])
            colors = [[0, 1, 1 ],[0.5, 0, 1],[0.6, 0.6, 0.6]]
            for state in [2,1,0]:
                tmp = BS[:, :, state].mean(axis=0)
                plt.plot(t[it], tmp[it], color=colors[state], lw=3, label=state_label[state])
                if nmice > 1:
                    smp = BS[:,:,state].std(axis=0) / np.sqrt(nmice)
                    plt.fill_between(t[it], tmp[it]-smp[it], tmp[it]+smp[it], color=colors[state], alpha=0.4, zorder=3)

            plt.xlim([-pre+edge, post-edge])
            plt.ylim([0,1])
            ax.add_patch(patches.Rectangle((0,0), laser_dur, 1, facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1]))
            box_off(ax)
            plt.xlabel('Time (s)')
            plt.ylabel('Probability')
            #plt.legend(bbox_to_anchor=(0., 1.02, 0.5, .102), loc=3, ncol=3, borderaxespad=0.)
            plt.draw()

        else:
            plt.figure(figsize=(7,7))
            clrs = sns.color_palette("husl", nmice)
            for state in [2,1,0]:
                ax = plt.subplot('31' + str(3-state))
                for i in range(nmice):
                    plt.plot(t[it], BS[i,it,state]*100, color=clrs[i], label=mouse_order[i])
                ax.add_patch(patches.Rectangle((0, 0), laser_dur, 100, facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1], alpha=0.8))
                plt.xlim((t[it][0], t[it][-1]))
                plt.ylim((0,100))
                plt.ylabel('% ' + state_label[state])
                if state==0:
                    plt.xlabel('Time (s)')
                else:
                    ax.set_xticklabels([])
                if state==2:
                    #plt.legend(bbox_to_anchor=(1.0, 0.7, 1., .102), loc=1, mode='expand', ncol=nmice, frameon=False)
                    ax.legend(mouse_order, bbox_to_anchor=(0., 1.0, 1., .102), loc=3, mode='expand', ncol=len(mouse_order),
                          frameon=False)
                box_off(ax)

        # figure showing all trials
        plt.figure(figsize=(4,6))
        set_fontarial()
        plt.ion()
        ax = plt.axes([0.15, 0.1, 0.8, 0.8])
        cmap = plt.cm.jet
        my_map = cmap.from_list('ha', [[0,1,1],[0.5,0,1], [0.6, 0.6, 0.6]], 3)
        x = list(range(Trials.shape[0]))
        plt.pcolormesh(t,np.array(x), np.flipud(Trials), cmap=my_map, vmin=1, vmax=3)
        plt.plot([0,0], [0, len(x)-1], color='white')
        plt.plot([laser_dur,laser_dur], [0, len(x)-1], color='white')
        ax.axis('tight')
        plt.draw()
        plt.xlabel('Time (s)')
        plt.ylabel('Trial No.')
        box_off(ax)

        plt.show()
        if len(fig_file)>0:
            plt.savefig(fig_file)

        # compile dataframe with all baseline and laser values
        ilsr   = np.where((t>=0) & (t<=laser_dur))[0]
        ibase  = np.where((t>=-laser_dur) & (t<0))[0]
        iafter = np.where((t>=laser_dur) & (t<laser_dur*2))[0]
        df = pd.DataFrame(columns = ['Mouse', 'REM', 'NREM', 'Wake', 'Lsr'])
        mice = mouse_order + mouse_order + mouse_order
        lsr  = np.concatenate((np.ones((nmice,), dtype='int'), np.zeros((nmice,), dtype='int'), np.ones((nmice,), dtype='int')*2))
        df['Mouse'] = mice
        df['Lsr'] = lsr
        df['REM']  = np.concatenate((BS[:,ilsr,0].mean(axis=1), BS[:,ibase,0].mean(axis=1), BS[:,iafter,0].mean(axis=1)))
        df['NREM'] = np.concatenate((BS[:,ilsr,2].mean(axis=1), BS[:,ibase,2].mean(axis=1), BS[:,iafter,2].mean(axis=1)))
        df['Wake'] = np.concatenate((BS[:,ilsr,1].mean(axis=1), BS[:,ibase,1].mean(axis=1), BS[:,iafter,1].mean(axis=1)))

    return BS, t, df



def laser_brainstate_bootstrap(ppath, recordings, pre, post, edge=0, sf=0,
                               nboots=10000, alpha=0.05, backup='',
                               start_time=0, ma_thr=20, fig_file=''):
    """
    Align brain state with laser stimulation and calculate two-sided 1-$alpha confidence intervals using
    bootstrap
    :param ppath: base folder
    :param recordings: list of recordings
    :param pre: time before laser
    :param post: time after laser onset
    :param edge: add $edge seconds add beginning and end (that are not shown in the plot) to avoid filtering artifacts
    :param sf: smoothing factor for Gaussian filter; better do not use
    :param nboots: int, how many times the whole data set is resampled for boot-strapping
    :param alpha: plot shows 1-$alpha confidence interval
    :param backup: optional backup folder where recordings are stored
    :param start_time: start time of recordding used for analysis
    :param ma_thr: sleep periods < ma_thr are thrown away
    :param fig_file, if file name is specified, the figure will be saved
    :return: P   - p-values for NREM, REM, Wake
             Mod - by how much the percentage of NREM, REM, Wake is increased compared to baseline
    """

    pre += edge
    post += edge

    rec_paths = dict()
    for rec in recordings:
        if os.path.isdir(os.path.join(ppath, rec)):
            rec_paths[rec] = ppath
        else:
            rec_paths[rec] = backup

    # dict: mouse_id --> laser trials, R W N sequence
    BrainstateDict = {}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        BrainstateDict[idf] = []
    mice = list(BrainstateDict.keys())
    nmice = len(BrainstateDict)

    for rec in recordings:
        ppath = rec_paths[rec]
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5 * SR)
        dt = NBIN * 1 / SR
        istart_time = int(np.round(start_time / dt))

        M = load_stateidx(ppath, rec)[0]

        seq = get_sequences(np.where(M == 2)[0])
        for s in seq:
            if len(s) * dt <= ma_thr:
                M[s] = 3

        (idxs, idxe) = laser_start_end(load_laser(ppath, rec))
        idf = re.split('_', rec)[0]

        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5 * SR)
        ipre = int(np.round(pre / dt))
        ipost = int(np.round(post / dt))

        idxs = [int(i / NBIN) for i in idxs]
        idxe = [int(i / NBIN) for i in idxe]
        laser_dur = np.mean((np.array(idxe) - np.array(idxs))) * dt

        for (i, j) in zip(idxs, idxe):
            if i >= ipre and j + ipost <= len(M) - 1 and i > istart_time:
                bs = M[i - ipre:i + ipost + 1]
                BrainstateDict[idf].append(bs)

    for mouse in mice:
        BrainstateDict[mouse] = np.array(BrainstateDict[mouse])

    # I assume here that every recording has same dt
    t = np.linspace(-ipre*dt, ipost*dt, ipre+ipost+1)
    BS = np.zeros((nmice, len(t), 3))
    Trials = dict()
    for mouse in BrainstateDict:
        Trials[mouse] = np.zeros((BrainstateDict[mouse].shape[0], len(t), 3))

    for mouse in BrainstateDict:
        M = np.array(BrainstateDict[mouse])
        for state in range(1, 4):
            C = np.zeros(M.shape)
            C[np.where(M == state)] = 100.
            Trials[mouse][:,:,state-1] = C

    Prob = np.zeros((nboots, len(t), 3))
    #ntrials = sum([s.shape[0] for s in BrainstateDict.values()])
    #bBS = dict()
    #for s in [1,2,3]:
    #    bBS[s] = np.zeros((ntrials, len(t)))
    for b in range(nboots):
        # average brain state percentage for each mouse during iteration b
        mouse_mean_state = np.zeros((nmice, len(t), 3))
        offset = 0
        i = 0
        for mouse in mice:
            mmouse = Trials[mouse].shape[0]
            iselect = rand.randint(0, mmouse, (mmouse,))
            for s in [1,2,3]:
                #bBS[s][offset:offset+mmouse,:] = Trials[mouse][iselect,:,s-1]
                mouse_mean_state[i,:,s-1] = Trials[mouse][iselect,:,s-1].mean(axis=0)
            i += 1
            offset += mmouse

        for s in [1,2,3]:
            Prob[b,:,s-1] = mouse_mean_state[:,:,s-1].mean(axis=0)

    # simple average for each brainstate across mice (w/o) bootstrapping
    Prob_mean = np.zeros((nmice, len(t), 3))
    for s in [1,2,3]:
        i = 0
        for mouse in mice:
            Prob_mean[i,:,s-1] = Trials[mouse][:,:,s-1].mean(axis=0)
            i += 1

    usProb = Prob.copy()
    Prob = np.sort(Prob, axis=0)
    Bounds = np.zeros((2, len(t), 3))
    a = int((nboots * alpha) / 2.0)
    #pdb.set_trace()
    for s in [1,2,3]:
        Bounds[0,:,s-1] = Prob[a,:,s-1]
        Bounds[1,:,s-1] = Prob[-a,:, s-1]

    # smooth_data
    if sf > 0:
        for s in range(3):
            Bounds[0, :, s] = smooth_data(Bounds[0, :, s], sf)
            Bounds[1, :, s] = smooth_data(Bounds[1, :, s], sf)
        for i in range(nmice):
            for s in range(3):
                Prob_mean[i, :, s] = smooth_data(Prob_mean[i,:,s], sf)

    # plot figure
    colors = np.array([[0, 1, 1], [0.5, 0, 1], [0.6, 0.6, 0.6]])
    br_states = {1:'REM', 2:'Wake', 3:'NREM'}
    #colors = np.array([[55,255,255], [153,255,153],[153,153,153]])/255.
    it = np.where((t>=-pre+edge) & (t<=post-edge))[0]
    plt.ion()
    plt.figure()
    ax = plt.axes([0.15, 0.15, 0.6, 0.7])
    for s in [3,2,1]:
        ax.fill_between(t[it], Bounds[0,it,s-1], Bounds[1,it,s-1], color=colors[s-1,:], alpha=0.8, zorder=3, edgecolor=None)
        ax.plot(t[it], Prob_mean[:, it, s-1].mean(axis=0), color=colors[s-1,:], label=br_states[s])
    ax.add_patch(patches.Rectangle((0, 0), laser_dur, 100, facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1]))
    plt.xlim([-pre+edge, post-edge])
    plt.ylim([0,100])
    plt.xlabel('Time (s)')
    plt.ylabel('Brain state (%)')
    plt.legend(bbox_to_anchor = (1.0, 0.7, 1., .102), loc = 3, mode = 'expand', ncol = 1, frameon = False)
    box_off(ax)
    plt.draw()

    # statistics
    ibase = np.where((t>=-laser_dur) & (t<0))[0]
    ilsr  = np.where((t>=0) & (t<laser_dur))[0]
    P   = np.zeros((3,))
    Mod = np.zeros((3,))
    for istate in [1,2,3]:
        basel = usProb[:,ibase,istate-1].mean(axis=1)
        laser = usProb[:,ilsr, istate-1].mean(axis=1)
        d = laser - basel
        if np.mean(d) >= 0:
            # now we want all values be larger than 0
            p = len(np.where(d>0)[0]) / (1.0*nboots)
            sig = 1 - p
            if sig == 0:
                sig = 1.0/nboots
            Mod[istate-1] = (np.mean(laser) / np.mean(basel) - 1) * 100
        else:
            p = len(np.where(d<0)[0]) / (1.0*nboots)
            sig = 1 - p
            if sig == 0:
                sig = 1.0/nboots
            Mod[istate-1] = -(1 - np.mean(laser) / np.mean(basel)) * 100
        P[istate-1] = sig

    labels = {1:'REM', 2:'Wake', 3:'NREM'}
    for s in [1,2,3]:
        print('%s is changed by %f perc.; P = %f, bootstrap' % (labels[s], Mod[s-1], P[s-1]))
    print("n = %d mice" % len(mice))

    if len(fig_file) > 0:
        plt.savefig(fig_file, bbox_inches="tight")

    return P, Mod



def sleep_example(ppath, name, tlegend, tstart, tend, fmax=30, fig_file='', vm=[], ma_thr=10,
                  fontsize=12, cb_ticks=[], emg_ticks=[], r_mu = [10, 100], fw_color=True):
    """
    plot sleep example
    :param ppath: base folder
    :param name: recording name
    :param tstart: start (in seconds) of shown example interval
    :param tend: end of example interval
    :param tlegend: length of time legend
    :param fmax: maximum frequency shown for EEG spectrogram
    :param fig_file: file name where figure will be saved
    :param vm: saturation of EEG spectrogram
    :param fontsize: fontsize
    :param cb_ticks: ticks for colorbar
    :param emg_ticks: ticks for EMG amplitude axis (uV)
    :param r_mu: range of frequencies for EMG amplitude
    :param fw_color: if True, use standard color scheme for brainstate (gray - NREM, violet - Wake, cyan - REM);
            otherwise use Shinjae's color scheme
    """
    set_fontarial()
    set_fontsize(fontsize)

    # True, if laser exists, otherwise set to False
    plaser = True

    sr = get_snr(ppath, name)
    nbin = np.round(2.5 * sr)
    dt = nbin * 1 / sr

    istart = int(np.round(tstart/dt))
    iend   = int(np.round(tend/dt))
    dur = (iend-istart+1)*dt

    M,K = load_stateidx(ppath, name)
    #kcut = np.where(K>=0)[0]
    #M = M[kcut]
    if tend==-1:
        iend = len(M)
    M = M[istart:iend]

    seq = get_sequences(np.where(M==2)[0])
    for s in seq:
        if len(s)*dt <= ma_thr:
            M[s] = 3

    t = np.arange(0, len(M))*dt

    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name), squeeze_me=True)
    SPEEG = P['SP']#/1000000.0
    # calculate median for choosing right saturation for heatmap
    med = np.median(SPEEG.max(axis=0))
    if len(vm) == 0:
        vm = [0, med*2.5]
    #t = np.squeeze(P['t'])
    freq = P['freq']
    P = so.loadmat(os.path.join(ppath, name, 'msp_%s.mat' % name), squeeze_me=True)
    SPEMG = P['mSP']#/1000000.0

    # load laser
    if not os.path.isfile(os.path.join(ppath, name, 'laser_%s.mat' % name)):
        plaser = False
    if plaser:
        laser = load_laser(ppath, name)
        idxs, idxe = laser_start_end(laser, SR=sr)
        idxs = [int(i / nbin) for i in idxs]
        idxe = [int(i / nbin) for i in idxe]

    # laser
    if plaser:
        laser_start = []
        laser_end = []
        for (i,j) in zip(idxs, idxe):
            if i>=istart and j <= iend:
                laser_start.append(i-istart)
                laser_end.append(j-istart)

    # create figure
    plt.ion()
    plt.figure(figsize=(8,4))

    # axis in the background to draw laser patches
    axes_back = plt.axes([0.1, .4, 0.8, 0.52])
    axes_back.get_xaxis().set_visible(False)
    axes_back.get_yaxis().set_visible(False)
    axes_back.spines["top"].set_visible(False)
    axes_back.spines["right"].set_visible(False)
    axes_back.spines["bottom"].set_visible(False)
    axes_back.spines["left"].set_visible(False)

    if plaser:
        for (i,j) in zip(laser_start, laser_end):
            axes_back.add_patch(patches.Rectangle((i*dt, 0), (j-i+1)*dt, 1, facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1]))
        plt.text(laser_end[0] * dt + dur * 0.01, 0.94, 'Laser', color=[0.6, 0.6, 1])

    plt.ylim((0,1))
    plt.xlim([t[0], t[-1]])


    # show brainstate
    axes_brs = plt.axes([0.1, 0.4, 0.8, 0.05])
    cmap = plt.cm.jet
    if fw_color:
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    else:
        my_map = cmap.from_list('brs', [[0, 0, 0], [153 / 255.0, 76 / 255.0, 9 / 255.0],
                                        [120 / 255.0, 120 / 255.0, 120 / 255.0], [1, 0.75, 0]], 4)

    tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    axes_brs.axis('tight')
    axes_brs.axes.get_xaxis().set_visible(False)
    axes_brs.axes.get_yaxis().set_visible(False)
    axes_brs.spines["top"].set_visible(False)
    axes_brs.spines["right"].set_visible(False)
    axes_brs.spines["bottom"].set_visible(False)
    axes_brs.spines["left"].set_visible(False)

    axes_legend = plt.axes([0.1, 0.33, 0.8, 0.05])
    plt.ylim((0,1.1))
    plt.xlim([t[0], t[-1]])
    plt.plot([0, tlegend], [1, 1], color='black')
    plt.text(tlegend/4.0, 0.1, str(tlegend) + ' s')
    axes_legend.spines["top"].set_visible(False)
    axes_legend.spines["right"].set_visible(False)
    axes_legend.spines["bottom"].set_visible(False)
    axes_legend.spines["left"].set_visible(False)
    axes_legend.axes.get_xaxis().set_visible(False)
    axes_legend.axes.get_yaxis().set_visible(False)

    # show spectrogram
    ifreq = np.where(freq <= fmax)[0]
    # axes for colorbar
    axes_cbar = plt.axes([0.82, 0.68, 0.1, 0.2])
    # axes for EEG spectrogram
    axes_spec = plt.axes([0.1, 0.68, 0.8, 0.2], sharex=axes_brs)
    im = axes_spec.pcolorfast(t, freq[ifreq], SPEEG[ifreq, istart:iend], cmap='jet', vmin=vm[0], vmax=vm[1])
    axes_spec.axis('tight')
    axes_spec.set_xticklabels([])
    axes_spec.set_xticks([])
    axes_spec.spines["bottom"].set_visible(False)
    plt.ylabel('Freq (Hz)')
    box_off(axes_spec)
    plt.xlim([t[0], t[-1]])

    # colorbar for EEG spectrogram
    cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0)
    cb.set_label('Power ($\mathrm{\mu}$V$^2$s)')
    if len(cb_ticks) > 0:
        cb.set_ticks(cb_ticks)
    axes_cbar.set_alpha(0.0)
    axes_cbar.spines["top"].set_visible(False)
    axes_cbar.spines["right"].set_visible(False)
    axes_cbar.spines["bottom"].set_visible(False)
    axes_cbar.spines["left"].set_visible(False)
    axes_cbar.axes.get_xaxis().set_visible(False)
    axes_cbar.axes.get_yaxis().set_visible(False)

    # show EMG
    i_mu = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    # * 1000: to go from mV to uV
    p_mu = np.sqrt(SPEMG[i_mu, :].sum(axis=0) * (freq[1] - freq[0])) #* 1000.0 # back to muV
    axes_emg = plt.axes([0.1, 0.5, 0.8, 0.1], sharex=axes_spec)
    axes_emg.plot(t, p_mu[istart:iend], color='black')
    axes_emg.patch.set_alpha(0.0)
    axes_emg.spines["bottom"].set_visible(False)
    if len(emg_ticks) > 0:
        axes_emg.set_yticks(emg_ticks)
    plt.ylabel('Ampl. ' + '$\mathrm{(\mu V)}$')
    plt.xlim((t[0], t[-1] + 1))
    box_off(axes_emg)

    if len(fig_file) > 0:
        save_figure(fig_file)

    plt.show()



def sleep_stats(ppath, recordings, ma_thr=10.0, tstart=0, tend=-1, pplot=True, csv_file=''):
    """
    Calculate average percentage of each brain state,
    average duration and average frequency
    plot histograms for REM, NREM, and Wake durations
    @PARAMETERS:
    ppath      -   base folder
    recordings -   single string specifying recording or list of recordings

    @OPTIONAL:
    ma_thr     -   threshold for wake periods to be considered as microarousals
    tstart     -   only consider recorded data starting from time tstart, default 0s
    tend       -   only consider data recorded up to tend s, default -1, i.e. everything till the end
    pplot      -   generate plot in the end; True or False
    csv_file   -   file where data should be saved as csv file (e.g. csv_file = '/home/Users/Franz/Documents/my_data.csv')

    @RETURN:
        ndarray of percentages (# mice x [REM,Wake,NREM])
        ndarray of state durations
        ndarray of transition frequency / hour
    """
    if type(recordings) != list:
        recordings = [recordings]

    Percentage = {}
    Duration = {}
    Frequency = {}
    mice = []
    for rec in recordings:
        idf = re.split('_', os.path.split(rec)[-1])[0]
        if not idf in mice:
            mice.append(idf)
        Percentage[idf] = {1:[], 2:[], 3:[]}
        Duration[idf] = {1:[], 2:[], 3:[]}
        Frequency[idf] = {1:[], 2:[], 3:[]}
    nmice = len(Frequency)    

    for rec in recordings:
        idf = re.split('_', os.path.split(rec)[-1])[0]
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        dt = NBIN * 1/SR

        # load brain state
        M, K = load_stateidx(ppath, rec)
        kcut = np.where(K >= 0)[0]
        M = M[kcut]
        istart = int(np.round((1.0 * tstart) / dt))
        if tend==-1:
            iend = len(M)-1
        else:
            iend = int(np.round((1.0*tend) / dt))

        M[np.where(M==5)] = 2        
        # polish out microarousals
        seq = get_sequences(np.where(M==2)[0])
        for s in seq:
            if len(s)*dt <= ma_thr:
                M[s] = 3

        midx = np.arange(istart,iend+1)
        Mcut = M[midx]
        nm = len(Mcut)*1.0
        
        # get percentage of each state
        for s in [1,2,3]:
            Percentage[idf][s].append(len(np.where(Mcut==s)[0]) / nm)
            
        # get frequency of each state
        for s in [1,2,3]:
            Frequency[idf][s].append( len(get_sequences(np.where(Mcut==s)[0])) * (3600. / (nm*dt)) )
            
        # get average duration for each state
        for s in [1,2,3]:
            seq = get_sequences(np.where(Mcut==s)[0])
            Duration[idf][s] += [len(i)*dt for i in seq] 
        
    PercMx = np.zeros((nmice,3))
    i=0
    for k in mice:
        for s in [1,2,3]:
            PercMx[i,s-1] = np.array(Percentage[k][s]).mean()
        i += 1
    PercMx *= 100
        
    FreqMx = np.zeros((nmice,3))
    i = 0
    for k in mice:
        for s in [1,2,3]:
            FreqMx[i,s-1] = np.array(Frequency[k][s]).mean()
        i += 1
    
    DurMx = np.zeros((nmice,3))
    i = 0
    for k in mice:
        for s in [1,2,3]:
            DurMx[i,s-1] = np.array(Duration[k][s]).mean()
        i += 1
        
    DurHist = {1:[], 2:[], 3:[]}
    for s in [1,2,3]:
        DurHist[s] = np.squeeze(np.array(reduce(lambda x,y: x+y, [Duration[k][s] for k in Duration])))

    if pplot:
        clrs = sns.color_palette("husl", nmice)
        plt.ion()
        # plot bars summarizing results - Figure 1
        plt.figure(figsize=(10, 5))
        ax = plt.axes([0.1, 0.15, 0.2, 0.8])
        plt.bar([1,2,3], PercMx.mean(axis=0), align='center', color='gray', fill=False)
        plt.xticks([1,2,3], ['REM', 'Wake', 'NREM'], rotation=60)
        for i in range(nmice):
            plt.plot([1,2,3], PercMx[i,:], 'o', label=mice[i], color=clrs[i])
        plt.ylabel('Percentage (%)')
        plt.legend(fontsize=9)
        plt.xlim([0.2, 3.8])
        box_off(ax)
            
        ax = plt.axes([0.4, 0.15, 0.2, 0.8])
        plt.bar([1,2,3], DurMx.mean(axis=0), align='center', color='gray', fill=False)
        plt.xticks([1,2,3], ['REM', 'Wake', 'NREM'], rotation=60)
        for i in range(nmice):
            plt.plot([1, 2, 3], DurMx[i, :], 'o', label=mice[i], color=clrs[i])
        plt.ylabel('Duration (s)')
        plt.xlim([0.2, 3.8])
        box_off(ax)
            
        ax = plt.axes([0.7, 0.15, 0.2, 0.8])
        plt.bar([1,2,3], FreqMx.mean(axis=0), align='center', color='gray', fill=False)
        plt.xticks([1,2,3], ['REM', 'Wake', 'NREM'], rotation=60)
        for i in range(nmice):
            plt.plot([1, 2, 3], FreqMx[i, :], 'o', label=mice[i], color=clrs[i])
        plt.ylabel('Frequency (1/h)')
        plt.xlim([0.2, 3.8])
        box_off(ax)
        plt.show(block=False)    

        # plot histograms - Figure 2            
        plt.figure(figsize=(5, 10))
        ax = plt.axes([0.2,0.1, 0.7, 0.2])
        h, edges = np.histogram(DurHist[1], bins=40, range=(0, 300), normed=1)
        binWidth = edges[1] - edges[0]
        plt.bar(edges[0:-1], h*binWidth, width=5)
        plt.xlim((edges[0], edges[-1]))
        plt.xlabel('Duration (s)')
        plt.ylabel('Freq. REM')
        box_off(ax)
        
        ax = plt.axes([0.2,0.4, 0.7, 0.2])
        h, edges = np.histogram(DurHist[2], bins=40, range=(0, 1200), normed=1)
        binWidth = edges[1] - edges[0]
        plt.bar(edges[0:-1], h*binWidth, width=20)
        plt.xlim((edges[0], edges[-1]))
        plt.xlabel('Duration (s)')
        plt.ylabel('Freq. Wake')
        box_off(ax)
        
        ax = plt.axes([0.2,0.7, 0.7, 0.2])
        h, edges = np.histogram(DurHist[3], bins=40, range=(0, 1200), normed=1)
        binWidth = edges[1] - edges[0]
        plt.bar(edges[0:-1], h*binWidth, width=20)
        plt.xlim((edges[0], edges[-1]))
        plt.xlabel('Duration (s)')
        plt.ylabel('Freq. NREM')
        box_off(ax)
        plt.show()

    if len(csv_file) > 0:
        mouse_list = [[m]*3 for m in mice]
        mouse_list = sum(mouse_list, [])
        state_list = ['REM', 'Wake', 'NREM']*nmice
        df = pd.DataFrame({'mouse':mouse_list, 'state':state_list, 'Perc':PercMx.flatten(), 'Dur':DurMx.flatten(), 'Freq':FreqMx.flatten()})
        df.to_csv(csv_file)

    return PercMx, DurMx, FreqMx, df
  
    

def sleep_timecourse_list(ppath, recordings, tbin, n, tstart=0, tend=-1, ma_thr=-1, pplot=True, single_mode=False, csv_file=''):
    """
    simplified version of sleep_timecourse
    plot sleep timecourse for a list of recordings
    The function does not distinguish between control and experimental mice.
    It computes/plots the how the percentage, frequency (1/h) of brain states and duration 
    of brain state episodes evolves over time.
    
    See also sleep_timecourse
    
    @Parameters:
        ppath                       Base folder with recordings
        recordings                  list of recordings as e.g. generated by &load_recordings
        tbin                        duration of single time bin in seconds
        n                           number of time bins
    @Optional:
        tstart                      start time of first bin in seconds
        tend                        end time of last bin; end of recording if tend==-1
        ma_thr                      set microarousals (wake periods <= ma_thr seconds) to NREM
                                    if ma_thr==-1, don't do anything
        pplot                       plot figures summarizing results
        single_mode                 if True, plot each single mouse
        csv_file                    string, if non-empty, write data into file $ppath/$csv_file .csv
                                    to load csv file: data = pd.read_csv($csv_file.csv, header=[0,1,2])

    @Return:
        TimeMx, DurMx, FreqMx, df   Dict[state][time_bin x mouse_id]

        df is a pandas DataFrame of the format
               Perc                                         DUR
               REM            Wake           NREM           REM           Wake etc.
               bin1 ... binn  bin1 ... binn  bin1 ... bin2  bin1 ... bin2 etc.
        mouse1
        .
        .
        .
        mousen

    """
    if type(recordings) != list:
        recordings = [recordings]
    
    Mice = {}
    mouse_order = []
    for rec in recordings:
        #idf = re.split('_', rec)[0]
        idf = re.split('_', os.path.split(rec)[-1])[0]
        if not idf in mouse_order:
            mouse_order.append(idf)
        Mice[idf] = 1
    Mice = list(Mice.keys())
    
    TimeCourse = {}
    FreqCourse = {}
    DurCourse = {}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        # time bin in Fourier time
        dt = NBIN * 1/SR

        M,K = load_stateidx(ppath, rec)
        kcut = np.where(K>=0)[0]
        #kidx = np.setdiff1d(np.arange(0, M.shape[0]), kcut)
        M = M[kcut]
        M[np.where(M)==5] = 2
        # polish out microarousals
        if ma_thr>0:
            seq = get_sequences(np.where(M==2)[0])
            for s in seq:
                if len(s)*dt <= ma_thr:
                    M[s] = 3

        if tend==-1:
            iend = len(M)-1
        else:
            iend = int(np.round((1.0*tend) / dt))
        M = M[0:iend+1]
        istart = int(np.round((1.0*tstart) / dt))
        ibin = int(np.round(tbin / dt))
        # how brain state percentage changes over time
        perc_time = []
        for i in range(n):
            midx = np.arange(istart+i*ibin, istart+(i+1)*ibin)
            #midx = np.setdiff1d(midx, kcut)
            M_cut = M[midx]
            perc = []
            for s in [1,2,3]:
                perc.append( len(np.where(M_cut==s)[0]) / (1.0*len(M_cut)) )

            perc_time.append(perc)
        
        perc_vec = np.zeros((n,3))
        for i in range(3):
            perc_vec[:,i] = np.array([v[i] for v in perc_time])
        TimeCourse[rec] = perc_vec

        # how frequency of sleep stage changes over time
        freq_time = []
        for i in range(n):
            midx = np.arange(istart+i*ibin, istart+(i+1)*ibin)
            #midx = np.setdiff1d(midx, kcut)
            M_cut = M[midx]
            freq = []
            for s in [1,2,3]:
                tmp = len(get_sequences(np.where(M_cut==s)[0])) * (3600. / (len(M_cut)*dt))
                freq.append(tmp)
            freq_time.append(freq)
        
        freq_vec = np.zeros((n,3))
        
        for i in range(3):
            freq_vec[:,i] = np.array([v[i] for v in freq_time])
        FreqCourse[rec] = freq_vec
        
        # how duration of sleep stage changes over time
        dur_time = []
        for i in range(n):
            midx = np.arange(istart+i*ibin, istart+(i+1)*ibin)
            #midx = np.setdiff1d(midx, kcut)
            M_cut = M[midx]
            dur = []
            for s in [1,2,3]:
                tmp = get_sequences(np.where(M_cut==s)[0])
                tmp = np.array([len(j)*dt for j in tmp]).mean()                
                dur.append(tmp)
            dur_time.append(dur)
        
        dur_vec = np.zeros((n,3))
        
        for i in range(3):
            dur_vec[:,i] = np.array([v[i] for v in dur_time])
        DurCourse[rec] = dur_vec

    # collect all recordings belonging to a Control mouse        
    TimeCourseMouse = {}
    DurCourseMouse = {}
    FreqCourseMouse = {}
    # Dict[mouse_id][time_bin x br_state]
    for mouse in Mice:
        TimeCourseMouse[mouse] = []
        DurCourseMouse[mouse] = []
        FreqCourseMouse[mouse] = []

    for rec in recordings:
        idf = re.split('_', rec)[0]
        TimeCourseMouse[idf].append(TimeCourse[rec])
        DurCourseMouse[idf].append(DurCourse[rec])
        FreqCourseMouse[idf].append(FreqCourse[rec])
    
    mx = np.zeros((n, len(Mice)))
    TimeMx = {1:mx, 2:mx.copy(), 3:mx.copy()}
    mx = np.zeros((n, len(Mice)))
    DurMx = {1:mx, 2:mx.copy(), 3:mx.copy()}
    mx = np.zeros((n, len(Mice)))
    FreqMx = {1:mx, 2:mx.copy(), 3:mx.copy()}        
    # Dict[R|W|N][time_bin x mouse_id]
    i = 0
    for k in mouse_order:
        for s in range(1,4):
            tmp = np.array(TimeCourseMouse[k]).mean(axis=0)
            TimeMx[s][:,i] = tmp[:,s-1]
            
            tmp = np.array(DurCourseMouse[k]).mean(axis=0)
            DurMx[s][:,i] = tmp[:,s-1]
            
            tmp = np.array(FreqCourseMouse[k]).mean(axis=0)
            FreqMx[s][:,i] = tmp[:,s-1]
        i += 1

    if pplot:
        clrs = sns.color_palette("husl", len(mouse_order))
        label = {1:'REM', 2:'Wake', 3:'NREM'}
        tlabel = np.linspace(istart*dt, istart*dt+n*ibin*dt, n+1)
        t = np.linspace(istart*dt, istart*dt+n*ibin*dt, n+1)[0:-1] + (ibin*dt/2.0)
        t /= 3600.0
        tlabel /= 3600.0

        # plot percentage of brain state as function of time
        plt.ion()
        plt.figure()
        for s in range(1,4):
            ax = plt.axes([0.1, (s-1)*0.3+0.1, 0.8, 0.2])
            if not single_mode:
                plt.errorbar(t, np.nanmean(TimeMx[s],axis=1), yerr = np.nanstd(TimeMx[s],axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            else:
                for i in range(len(mouse_order)):
                    plt.plot(t, TimeMx[s][:,i], 'o-', linewidth=1.5, color=clrs[i])
                if s==3:
                    ax.legend(mouse_order, bbox_to_anchor = (0., 1.0, 1., .102), loc=3, mode='expand', ncol=len(mouse_order), frameon=False)
            box_off(ax)
            plt.xlim((-0.5, n-0.5))
            if s==1:
                #plt.ylim([0, 0.2])
                pass
            else:
                plt.ylim([0, 1.0])
            plt.ylabel('Perc ' + label[s] + '(%)')
            plt.xticks(tlabel)
            plt.xlim([tlabel[0], tlabel[-1]])
            if s == 1:
                plt.xlabel('Time (h)')
        plt.draw()

        # plot duration as function of time
        plt.figure()
        for s in range(1,4):
            ax = plt.axes([0.1, (s-1)*0.3+0.1, 0.8, 0.2])
            if not single_mode:
                plt.errorbar(t, np.nanmean(DurMx[s],axis=1), yerr = np.nanstd(DurMx[s],axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            else:
                for i in range(len(mouse_order)):
                    plt.plot(t, DurMx[s][:,i], 'o-', linewidth=1.5, color=clrs[i])
                if s==3:
                    ax.legend(mouse_order, bbox_to_anchor = (0., 1.0, 1., .102), loc=3, mode='expand', ncol=len(mouse_order), frameon=False)

            box_off(ax)
            plt.xlim((-0.5, n-0.5))
            plt.ylabel('Dur ' + label[s] + '(s)')
            plt.xticks(tlabel)
            plt.xlim([tlabel[0], tlabel[-1]])
            if s==1:
                plt.xlabel('Time (h)')
        plt.draw()

        # plot frequency as function of time
        plt.figure()
        for s in range(1,4):
            ax = plt.axes([0.1, (s-1)*0.3+0.1, 0.8, 0.2])
            if not single_mode:
                plt.errorbar(t, np.nanmean(FreqMx[s],axis=1), yerr = np.nanstd(FreqMx[s],axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            else:
                for i in range(len(mouse_order)):
                    plt.plot(t, FreqMx[s][:,i], 'o-', linewidth=1.5, color=clrs[i])
                if s==3:
                    ax.legend(mouse_order, bbox_to_anchor = (0., 1.0, 1., .102), loc=3, mode='expand', ncol=len(mouse_order), frameon=False)

            box_off(ax)
            plt.xlim((-0.5, n-0.5))
            plt.ylabel('Freq ' + label[s] + '(1/h)')
            plt.xticks(tlabel)
            plt.xlim([tlabel[0], tlabel[-1]])
            if s==1:
                plt.xlabel('Time (h)')
        plt.draw()


    # write data into dataframe and csv file
    bins = ['bin' + str(i+1) for i in range(n)]
    columns = pd.MultiIndex.from_product([['Perc', 'Dur', 'Freq'], ['REM', 'Wake', 'NREM'], bins],
                                         names=['stats', 'state', 'bin'])
    D = np.concatenate((TimeMx[1].T, TimeMx[2].T, TimeMx[3].T, DurMx[1].T, DurMx[2].T, DurMx[3].T, FreqMx[1].T, FreqMx[2].T, FreqMx[3].T), axis=1)
    df = pd.DataFrame(D, index=mouse_order, columns=columns)

    if len(csv_file) > 0:
        df.to_csv(csv_file)

    return TimeMx, DurMx, FreqMx, df



def ma_timecourse_list(ppath, recordings, tbin, n, tstart=0, tend=-1, ma_thr=20, pplot=True, single_mode=False, csv_file=''):
    """
    Calculate percentage, duration, and frequency of microarousals
    :param ppath: base folder
    :param recordings: single recording or list of recordings
    :param tbin: time bin in seconds
    :param n: number of time bins
    :param tstart: start time in recording(s) for analysi
    :param tend: end time for analysis
    :param ma_thr: microarousal threshold; any wake period shorter than $ma_thr will be considered as microarousal
    :param pplot: if True, plot figure
    :param single_mode: if True, plot each single mouse with different color
    :param csv_file: string, if non-empty, write data into file "csv_file";
                     file name should end with ".csv"

    :return: TimeMX, DurMX, FreqMx, df - np.array(# time bins x mice)
        TimeMX, DurMX, FreqMx: arrays with shape "time bins x mice"
        df: DataFrame with columns: mouse, perc, dur, freq, bin
            For example, to get the first time bins of perc, dur, freq of mouse M1 type
            df[(df.mouse == 'M1') & (df.bin == 't0')]
    """
    if type(recordings) != list:
        recordings = [recordings]

    mouse_order = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mouse_order:
            mouse_order.append(idf)

    TimeCourse = {}
    FreqCourse = {}
    DurCourse = {}
    for rec in recordings:
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5 * SR)
        # time bin in Fourier time
        dt = NBIN * 1 / SR

        M, K = load_stateidx(ppath, rec)
        kcut = np.where(K >= 0)[0]
        # kidx = np.setdiff1d(np.arange(0, M.shape[0]), kcut)
        M = M[kcut]
        Mnew = np.zeros(M.shape)
        Mnew[np.where(M) == 5] = 1
        # polish out microarousals
        if ma_thr > 0:
            seq = get_sequences(np.where(M == 2)[0])
            for s in seq:
                if len(s) * dt <= ma_thr:
                    Mnew[s] = 1
        M = Mnew
        if tend == -1:
            iend = len(M) - 1
        else:
            iend = int(np.round((1.0 * tend) / dt))
        M = M[0:iend + 1]
        istart = int(np.round((1.0 * tstart) / dt))
        ibin = int(np.round(tbin / dt))
        # how brain state percentage changes over time
        perc_time = []
        for i in range(n):
            midx = np.arange(istart + i * ibin, istart + (i + 1) * ibin)
            # midx = np.setdiff1d(midx, kcut)
            M_cut = M[midx]
            perc = len(np.where(M_cut == 1)[0]) / (1.0 * len(M_cut))
            perc_time.append(perc)

        TimeCourse[rec] = np.array(perc_time)

        # how frequency of sleep stage changes over time
        freq_time = []
        for i in range(n):
            midx = np.arange(istart + i * ibin, istart + (i + 1) * ibin)
            # midx = np.setdiff1d(midx, kcut)
            M_cut = M[midx]
            s = 1
            freq = len(get_sequences(np.where(M_cut == s)[0])) * (3600. / (len(M_cut) * dt))
            freq_time.append(freq)
        FreqCourse[rec] = np.array(freq_time)

        # how duration of microarousals changes over time
        dur_time = []
        for i in range(n):
            midx = np.arange(istart + i * ibin, istart + (i + 1) * ibin)
            # midx = np.setdiff1d(midx, kcut)
            M_cut = M[midx]
            s = 1
            tmp = get_sequences(np.where(M_cut == s)[0])
            dur = np.array([len(j) * dt for j in tmp]).mean()
            dur_time.append(dur)

        DurCourse[rec] = np.array(dur_time)

    # collect all recordings belonging to a Control mouse
    TimeCourseMouse = {}
    DurCourseMouse = {}
    FreqCourseMouse = {}
    # Dict[mouse_id][time_bin x br_state]
    for mouse in mouse_order:
        TimeCourseMouse[mouse] = []
        DurCourseMouse[mouse] = []
        FreqCourseMouse[mouse] = []

    for rec in recordings:
        idf = re.split('_', rec)[0]
        TimeCourseMouse[idf].append(TimeCourse[rec])
        DurCourseMouse[idf].append(DurCourse[rec])
        FreqCourseMouse[idf].append(FreqCourse[rec])

    # np.array(time x mouse_id)
    TimeMx = np.zeros((n, len(mouse_order)))
    DurMx = np.zeros((n, len(mouse_order)))
    FreqMx = np.zeros((n, len(mouse_order)))

    i = 0
    for k in mouse_order:
        tmp = np.array(TimeCourseMouse[k]).mean(axis=0)
        TimeMx[:,i] = tmp
        tmp = np.array(DurCourseMouse[k]).mean(axis=0)
        DurMx[:,i] = tmp
        tmp = np.array(FreqCourseMouse[k]).mean(axis=0)
        FreqMx[:,i] = tmp
        i += 1

    # plotting
    if pplot:
        plot_dict = {0:TimeMx, 1:DurMx, 2:FreqMx}

        clrs = sns.color_palette("husl", len(mouse_order))
        ylabel = {0:'Perc (%)', 1:'Dur (s)', 2:'Freq ($h^{-1}$)'}
        tlabel = np.linspace(istart*dt, istart*dt+n*ibin*dt, n+1)
        t = np.linspace(istart*dt, istart*dt+n*ibin*dt, n+1)[0:-1] + (ibin*dt/2.0)
        t /= 3600.0
        tlabel /= 3600.0

        # plot percentage of brain state as function of time
        plt.ion()
        plt.figure()
        for s in range(0, 3):
            ax = plt.axes([0.15, s*0.3+0.1, 0.8, 0.2])
            if not single_mode:
                plt.errorbar(t, np.nanmean(plot_dict[s],axis=1), yerr = np.nanstd(plot_dict[s],axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            else:
                for i in range(len(mouse_order)):
                    plt.plot(t, plot_dict[s][:,i], 'o-', linewidth=1.5, color=clrs[i])
                if s==2:
                    ax.legend(mouse_order, bbox_to_anchor = (0., 1.0, 1., .102), loc=3, mode='expand', ncol=len(mouse_order), frameon=False)
            box_off(ax)
            plt.xlim((-0.5, n-0.5))
            plt.ylabel(ylabel[s])
            plt.xticks(tlabel)
            plt.xlim([tlabel[0], tlabel[-1]])
            if s == 0:
                plt.xlabel('Time (h)')
        plt.draw()


    bins = [['t' + str(i)]*len(mouse_order) for i in range(n)]
    bins = sum(bins, [])
    cols = ['mouse', 'perc', 'dur', 'freq', 'bin']
    mice = mouse_order*n

    df = pd.DataFrame(columns=cols)
    df['mouse'] = mice
    df['bin'] = bins
    df['perc'] = TimeMx.T.flatten()
    df['dur'] = DurMx.T.flatten()
    df['freq'] = FreqMx.T.flatten()

    if len(csv_file) > 0:
        df.to_csv(csv_file)

    return TimeMx, DurMx, FreqMx, df



def sleep_through_days(ppath, recordings, tstart=0, tend=-1, stats=0, xticks=[], ma_thr=20, min_dur=[0,0,0], single_mode=True, csv_file = ''):
    """
    Follow sleep quantity (percentage, bout duration, or frequency / hour) over multiple days
    :param ppath: base folder
    :param recordings: list of lists of recordings, for example [[F1_010118n1, F2_010118n1], [F1_010218n1, F1_010218n1]]
    specifies the recordings of F1 and F2 for two days
    :param tstart: float, quantificiation of sleep starts at $start s
    :param tend: float, quantification of sleep ends at $tend s
    :param stats: Measured sleep variable (statistics):
            0 - percentage, 1 - episode duration, 2 - episode frequency, 3 - latency to first state occurance REM, Wake, and NREM
    :param xticks: list of string, specifying the xticks
    :param ma_thr: float, wake periods shorter than $ma_thr are considered as microarousals and further converted to NREM
    :param min_dur: list with 3 floats, specifying the minimum duration of the first REM, Wake, and NREM period,
            only relevant if $stats == 3
    :param single_mode: if True, plot each single mouse in different color
    :param csv_file: string, save pd.DataFrame to file; the actual file name will be "$csv_file + stats + $stats .csv" and
            will be save in the folder $ppath

    :return: np.array, mice x [REM,Wake,NREM] x days AND pd.DataFrame
             the pd.DataFrame has the following format:
             state  REM              Wake           NREM
             day    Day1   ... Dayn  Day1 ... Dayn  Day1 ... Dayn
             mouse1
             .
             .
             .
             mousen
    """

    states = {1:'REM', 2:'Wake', 3:'NREM'}
    stats_label = {0:'(%)', 1:'Dur (s)', 2: 'Freq. (1/h)', 3: 'Lat. (min)'}

    mice_per_day = {}
    iday = 0
    for day in recordings:
        mice = []
        for rec in day:
            idf = re.split('_', os.path.split(rec)[-1])[0]
            if not idf in mice:
                mice.append(idf)
        mice_per_day[iday] = mice
        iday += 1

    ndays = len(mice_per_day)
    nmice = len(mice_per_day[0])

    for i in range(ndays):
        for j in range(i+1, ndays):
            if mice_per_day[i] != mice_per_day[j]:
                print("ERROR: mice on day %d and %d not consistent" % (i+1, j+1))
                return

    #DayResults: mice x [R|W|N] x days
    DayResults = np.zeros((nmice, 3, ndays))
    for day in range(ndays):
        if stats<=2:
            res = sleep_stats(ppath, recordings[day], tstart=tstart, tend=tend, pplot=False, ma_thr=ma_thr)[stats]
        else:
            res = np.zeros((nmice, 3))
            for s in range(1,4):
                res[:,s-1] = state_onset(ppath, recordings[day], s, min_dur=min_dur[s-1], tstart=tstart, tend=tend, pplot=False)
        DayResults[:,:,day] = res

    plt.ion()
    clrs = sns.color_palette("husl", nmice)
    plt.figure(figsize=(10,6))
    for s in range(1, 4):
        ax = plt.axes([0.1, (s - 1) * 0.3 + 0.1, 0.8, 0.2])
        if single_mode:
            for i in range(nmice):
                plt.plot(list(range(1,ndays+1)), DayResults[i,s-1,:], 'o-', color=clrs[i], label=mice[i])
        else:
            plt.errorbar(list(range(1, ndays+1)), DayResults[:, s-1, :].mean(axis=0), yerr=DayResults[:, s-1, :].std(axis=0),
                         color='gray', label='avg', linewidth=2)

        if s == 1:
            if len(xticks) == 0:
                plt.xticks(list(range(1,ndays+1)))
            else:
                plt.xticks(list(range(1, ndays + 1), xticks))
        else:
            plt.xticks(list(range(1, ndays + 1)))
            ax.set_xticklabels([])
        if s == 3:
            ax.legend(bbox_to_anchor=(0., 1.0, 1., .102), loc=3, mode='expand', ncol=nmice,
                      frameon=False)
        box_off(ax)
        if s == 1:
            plt.xlabel('Day')
        plt.ylabel(states[s] + ' ' + stats_label[stats])

    if len(xticks) > 0:
        col = xticks
    else:
        col = ['Day' + str(i+1) for i in range(ndays)]

    # put data into pandas dataframe
    columns = pd.MultiIndex.from_product([['REM', 'Wake', 'NREM'], col], names=['state', 'day'])
    D = np.concatenate((DayResults[:,0,:], DayResults[:,1,:], DayResults[:,2,:]), axis=1)
    df = pd.DataFrame(D, index=mice, columns=columns)
    if len(csv_file) > 0:
        csv_file += '_stats' + str(stats) + '.csv'
        df.to_csv(os.path.join(ppath, csv_file))

    return DayResults, df



def sleep_timecourse(ppath, trace_file, tbin, n, tstart=0, tend=-1, pplot=True, csv_file=''):
    """
    plot how percentage of REM,Wake,NREM changes over time;
    compares control with experimental data; experimental recordings can have different "doses"
    a simpler version is sleep_timecourse_list
    
    @Parameters
    trace_file-  text file, specifies control and experimental recordings
    tbin    -    size of time bin in seconds
    n       -    number of time bins
    @Optional:
    tstart  -    beginning of recording (time <tstart is thrown away)
    tend    -    end of recording (time >tend is thrown away)
    pplot   -    plot figure if True
    
    @Return:
    TimeMxCtr - Dict[R|W|N][time_bin x mouse_id] 
    TimeMxExp - Dict[R|W|N][dose][time_bin x mouse_id]
    """
    (ctr_rec, exp_rec) = load_dose_recordings(ppath, trace_file)
    
    Recordings = []
    Recordings += ctr_rec
    for k in exp_rec.keys():
        Recordings += exp_rec[k]
        
    CMice = []
    for mouse in ctr_rec:
        idf = re.split('_', mouse)[0]
        if not idf in CMice:
            CMice.append(idf)

    EMice = {}
    for d in exp_rec:
        mice = exp_rec[d]
        EMice[d] = []
        for mouse in mice:
            idf = re.split('_', mouse)[0]
            if not idf in EMice[d]:
                EMice[d].append(idf)

    TimeCourse = {}
    for rec in Recordings:
        idf = re.split('_', rec)[0]
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        # time bin in Fourier time
        dt = NBIN * 1/SR

        M = load_stateidx(ppath, rec)[0]        
        M[np.where(M)==5] = 2
        if tend==-1:
            iend = len(M)-1
        else:
            iend = int(np.round((1.0*tend) / dt))
        M = M[0:iend+1]
        istart = int(np.round((1.0*tstart) / dt))
        ibin = int(np.round(tbin / dt))
        
        perc_time = []
        for i in range(n):
            # return something even if istart+i+1)*ibin >= len(M)
            M_cut = M[istart+i*ibin:istart+(i+1)*ibin]            
            perc = []
            for s in [1,2,3]:
                perc.append( len(np.where(M_cut==s)[0]) / (1.0*len(M_cut)) )
            perc_time.append(perc)
        perc_vec = np.zeros((n,3))
        
        for i in range(3):
            perc_vec[:,i] = np.array([v[i] for v in perc_time])
        TimeCourse[rec] = perc_vec


    # define data frame containing all data
    bins = ['t' + str(i) for i in range(n)]
    cols = ['mouse', 'dose', 'state', 'time', 'perc']
    df = pd.DataFrame(columns=cols)
    state_map = {1: 'REM', 2:'Wake', 3:'NREM'}

    # collect all recordings belonging to a Control mouse
    TimeCourseCtr = {}
    # Dict[mouse_id][time_bin x br_state]
    for mouse in CMice:
        TimeCourseCtr[mouse] = []

    for rec in Recordings:
        idf = re.split('_', rec)[0]
        if rec in ctr_rec:
            TimeCourseCtr[idf].append(TimeCourse[rec])
    
    mx = np.zeros((n, len(CMice)))
    TimeMxCtr = {1:mx, 2:mx.copy(), 3:mx.copy()}
    # Dict[R|W|N][time_bin x mouse_id]
    i = 0
    for k in TimeCourseCtr:
        for s in range(1,4):
            # [time_bin x br_state]
            tmp = np.array(TimeCourseCtr[k]).mean(axis=0)
            TimeMxCtr[s][:,i] = tmp[:,s-1]
            for j in range(n):
                df = df.append(pd.Series([k, '0', state_map[s], 't'+str(j), tmp[j,s-1]], index=cols), ignore_index=True)

        i += 1
                
    # collect all recording belonging to one Exp mouse with a specific dose
    TimeCourseExp = {}
    # Dict[dose][mouse_id][time_bin x br_state]
    for d in EMice:
        TimeCourseExp[d]={}
        for mouse in EMice[d]:
            TimeCourseExp[d][mouse] = []
    
    for rec in Recordings:
        idf = re.split('_', rec)[0]
        for d in exp_rec:
            if rec in exp_rec[d]:
                TimeCourseExp[d][idf].append(TimeCourse[rec])
    
    # dummy dictionally to initialize TimeMxExp
    # Dict[R|W|N][dose][time_bin x mouse_id]
    TimeMxExp = {1:{}, 2:{}, 3:{}}
    for s in [1,2,3]:
        TimeMxExp[s] = {}
        for d in EMice:
            TimeMxExp[s][d] = np.zeros((n, len(EMice[d])))
    
    for d in TimeCourseExp:
        i = 0    
        for k in TimeCourseExp[d]:
            print(k)
            tmp = np.array(TimeCourseExp[d][k]).mean(axis=0)
            for s in [1,2,3]:
                # [time_bin x br_state] for mouse k
                #tmp = sum(TimeCourseExp[d][k]) / (1.0*len(TimeCourseExp[d][k]))                
                TimeMxExp[s][d][:,i] = tmp[:,s-1]
                for j in range(n):
                    df = df.append(pd.Series([k, d, state_map[s], 't'+str(j), tmp[j, s-1]], index=cols), ignore_index=True)

            i += 1

    if pplot:
        # new
        tlabel = np.linspace(istart*dt, istart*dt+n*ibin*dt, n+1)
        t = np.linspace(istart*dt, istart*dt+n*ibin*dt, n+1)[0:-1] + (ibin*dt/2.0)
        t /= 3600.0
        tlabel /= 3600.0
        # new end

        plt.ion()
        plt.figure()
        ndose = len(EMice)
        ax = plt.axes([0.1, 0.7, 0.8, 0.2])
        plt.errorbar(t, TimeMxCtr[1].mean(axis=1), yerr = TimeMxCtr[1].std(axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
        box_off(ax)
        plt.xlim([t[0], t[-1]])
        plt.yticks([0, 0.1, 0.2])
        plt.xticks(tlabel)
        plt.ylabel('% REM')    
        #plt.ylim((0,0.2))
        
        i = 1
        for d in TimeMxExp[1]:            
            c = 1 - 1.0/ndose*i
            plt.errorbar(t, TimeMxExp[1][d].mean(axis=1), yerr = TimeMxExp[1][d].std(axis=1),  color=[c, c, 1], fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            i += 1
    
        ax = plt.axes([0.1, 0.4, 0.8, 0.2])
        plt.errorbar(t, TimeMxCtr[2].mean(axis=1), yerr = TimeMxCtr[2].std(axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
        box_off(ax)
        plt.xlim([t[0], t[-1]])
        plt.yticks([0, 0.1, 0.2])
        plt.xticks(tlabel)
        plt.ylabel('% Wake')    
        #plt.ylim((0,1))
        plt.yticks(np.arange(0, 1.1, 0.25))
    
        i = 1
        for d in TimeMxExp[2]:            
            c = 1 - 1.0/ndose*i
            plt.errorbar(t, TimeMxExp[2][d].mean(axis=1), yerr = TimeMxExp[2][d].std(axis=1),  color=[c, c, 1], fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            i += 1
                
        ax = plt.axes([0.1, 0.1, 0.8, 0.2])
        plt.errorbar(t, TimeMxCtr[3].mean(axis=1), yerr = TimeMxCtr[3].std(axis=1),  color='gray', fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
        box_off(ax)
        plt.xlim([t[0], t[-1]])
        plt.yticks([0, 0.1, 0.2])
        plt.xticks(tlabel)
        plt.ylabel('% NREM')    
        #plt.ylim((0,1))
        plt.yticks(np.arange(0, 1.1, 0.25))
        plt.xlabel('Time (h)')
        plt.show()
    
        i = 1
        for d in TimeMxExp[2]:            
            c = 1 - 1.0/ndose*i
            plt.errorbar(t, TimeMxExp[3][d].mean(axis=1), yerr = TimeMxExp[3][d].std(axis=1),  color=[c, c, 1], fmt = 'o', linestyle='-', linewidth=2, elinewidth=2)
            i += 1

    if len(csv_file) > 0:
        df.to_csv(os.path.join(csv_file))

    return TimeMxCtr, TimeMxExp, df



def state_onset(ppath, recordings, istate, min_dur, iseq=0, ma_thr=10, tstart=0, tend=-1, pplot=True):
    """
    calculate time point of first occurance of state $istate in @recordings
    :param ppath: base folder
    :param recordings: list of recordings
    :param istate: 1 = REM, 2 = Wake, 3 = NREM
    :param min_dur: minimum duration in [s] to be counted as first occurance
    :param iseq: calculate the $iseq-th occurance state $istate
    :param ma_thr: microarousal threshould
    :param tstart: float, quantificiation of sleep starts at $start s
    :param tend: float, quantification of sleep ends at $tend s
    :return: np.array, average latency (in minutes) for each mouse. If one mouse contributes several recordings
             the average latency is computed
    """
    if type(recordings) != list:
        recordings = [recordings]

    # get all mice in recordings
    mice = []
    dt = 2.5
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice.append(idf)

    latency = {m:[] for m in mice}
    for rec in recordings:
        SR = get_snr(ppath, rec)
        # Number of EEG bins / brainstate bin
        NBIN = np.round(2.5 * SR)
        # Precise time bin duration of each brain state:
        dt = NBIN * 1.0 / SR

        idf = re.split('_', rec)[0]
        M,K = load_stateidx(ppath, rec)
        # flatten out microarousals
        if istate == 3 and ma_thr > 0:
            seq = get_sequences(np.where(M==2)[0])
            for s in seq:
                if len(s)*dt <= ma_thr:
                    M[s] = 3

        kcut = np.where(K>=0)[0]
        M = M[kcut]
        istart = int(np.round(tstart/dt))
        iend = int(np.round(tend/dt))
        if tend == -1:
            iend = len(M)
        M = M[istart:iend]

        seq = get_sequences(np.where(M==istate)[0])
        seq = [s for s in seq if len(s) * dt >= min_dur]
        #ifirst = seq[seq[iseq][0]]
        ifirst = seq[iseq][0]

        latency[idf].append(ifirst*dt)

    for m in mice:
        latency[m] = np.nanmax(np.array(latency[m]))

    values = np.array([latency[m]/60. for m in mice])
    clrs = sns.color_palette("husl", len(mice))
    if pplot:
        # print latencies
        for m in mice:
            print("%s - %.2f min" % (m, latency[m] / 60.))

        plt.ion()
        plt.figure()
        ax = plt.subplot(111)
        set_fontsize(14)
        #plt.bar(range(0, len(values)), values, color='gray')
        for i in range(len(mice)):
            plt.plot(i, values[i], 'o', color=clrs[i], label=m)

        #plt.legend(bbox_to_anchor=(0., 1.0, 1., .102), loc=3, mode='expand', ncol=len(mice), frameon=False)
        plt.xticks(list(range(0, len(values))), mice)
        plt.ylabel('Onset Latency (min)')
        box_off(ax)
        plt.show()

    return values



def sleep_spectrum(ppath, recordings, istate=1, pmode=1, twin=3, ma_thr=20.0, f_max=-1, pplot=True, sig_type='EEG', mu=[10, 100],
                   tstart=0, tend=-1, sthres=np.inf, peeg2=False, pnorm=False, single_mode=False, conv=1.0, fig_file='', laser_color='blue'):
    """
    calculate power spectrum for brain state i state for the given recordings 
    @Param:
    ppath    -    folder containing all recordings
    recordings -  single recording (string) or list of recordings
    @Optional:
    istate   -    state for which to calculate power spectrum
    twin     -    time window (in seconds) for power spectrum calculation
                  the longer the higher frequency resolution, but the more noisy
    ma_thr   -    short wake periods <= $ma_thr are considered as sleep
    f_max    -    maximal frequency, if f_max==-1: f_max is maximally possible frequency
    pplot    -    if True, plot figure showing result
    pmode    -    mode: 
                  pmode == 1, compare state during laser with baseline outside laser interval
                  pmode == 0, just plot power spectrum for state istate and don't care about laser
    tstart   -    use EEG starting from time point tstart [seconds]
    sig_type -    string, if 'EMG' calculate EMG amplitude (from the EMG spectrum). E.g.,
                  sleepy.sleep_spectrum(ppath, E, istate=2, f_max=30, sig_type='EMG')
    mu       -    tuple, lower and upper range for EMG frequencies used for amplitude calculation
    tend     -    use data up to tend [seconds], if tend == -1, use data till end
    sthres   -    maximum length of bout duration of state $istate used for calculation. If bout duration > $sthres, only
                  use the bout up to $sthres seconds after bout onset.
    peeg2    -    if True, use EEG2 channel for spectrum analysis
    pnorm    -    if True, normalize powerspectrum by dividing each frequency through each average power
                  over the whole EEG recording
    single_mode - if True, plot each single mouse
    fig_file -    if specified save to given file

    errorbars: If it's multiple mice make errorbars over mice; if it's multiple
    recordings of ONE mouse, show errorbars across recordings; 
    if just one recording show now errorbars
                  
    @Return:
    Pow     -    Dict[No loaser = 0|Laser = 1][array], where array: mice x frequencies, if more than one mouse;
                 otherwise, array: recordings x frequencies
    F       -    Frequencies
    """
    if type(recordings) != list:
        recordings = [recordings]

    Mice = {}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in Mice:
            Mice[idf] = Mouse(idf, rec, 'E')
        else:
            Mice[idf].add(rec)

    mouse_order = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mouse_order:
            mouse_order.append(idf)

    # Spectra: Dict[mouse_id][laser_on|laser_off][list of powerspectrum_arrays]
    Spectra = {}
    Ids = list(Mice.keys())
    for i in Ids:
        Spectra[i] = {0:[], 1:[]}
        Spectra[i] = {0:[], 1:[]}

    for idf in mouse_order:
        for rec in Mice[idf].recordings:
            # load EEG
            if sig_type =='EEG':
                if not peeg2:
                    EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'EEG.mat'))['EEG']).astype('float')*conv
                else:
                    EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'EEG2.mat'))['EEG2']).astype('float')*conv
            elif sig_type == 'EMG':
                if not peeg2:
                    EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'EMG.mat'))['EMG']).astype('float')*conv
                else:
                    EEG = np.squeeze(so.loadmat(os.path.join(ppath, rec, 'EMG2.mat'))['EMG2']).astype('float')*conv
            else:
                pass

            # load brain state
            M,K = load_stateidx(ppath, rec)
            # set brain states where K<0 to zero;
            # this whay they are effectively discarded
            M[K<0] = 0
            sr = get_snr(ppath, rec)
            # number of time bins for each time bin in spectrogram
            nbin = int(np.round(sr) * 2.5)
            # duration of time bin in spectrogram / brainstate
            dt = nbin * 1/sr
            nwin = np.round(twin*sr)

            istart = int(np.round(tstart/dt))
            if tend==-1:
                iend = M.shape[0]
            else:
                iend = int(np.round(tend/dt))
            istart_eeg = istart*nbin
            iend_eeg   = (iend-1)*nbin+1

            M[np.where(M==5)]=2
            # flatten out microarousals
            seq = get_sequences(np.where(M==2)[0])
            for s in seq:
                if len(s)*dt <= ma_thr:
                    M[s] = 3

            # get all sequences of state $istate
            M = M[istart:iend]
            seq = get_sequences(np.where(M==istate)[0])

            EEG = EEG[istart_eeg:iend_eeg]

            if pnorm:
                pow_norm = power_spectrum(EEG, nwin, 1 / sr)[0]

            if pmode == 1:
                laser = load_laser(ppath, rec)[istart_eeg:iend_eeg]
                (idxs, idxe) = laser_start_end(laser, SR=sr)
                # downsample EEG time to spectrogram time    
                idxs = [int(i/nbin) for i in idxs]
                idxe = [int(i/nbin) for i in idxe]
                
                laser_idx = []
                for (i,j) in zip(idxs, idxe):
                    laser_idx += list(range(i,j+1))
                laser_idx = np.array(laser_idx)
                
            if pmode == 1:
                # first analyze frequencies not overlapping with laser
                seq_nolsr = []
                for s in seq:
                    s = np.setdiff1d(s, laser_idx)
                    if len(s) > 0:
                        q = get_sequences(s)
                        seq_nolsr += q

                for s in seq_nolsr:
                    #s = np.setdiff1d(s, laser_idx)
                    if len(s)*nbin >= nwin:
                        drn = (s[-1]-s[0])*dt
                        if drn > sthres:
                            # b is the end of segment used for power spectrum calculation;
                            # that is, the last index (in raw EEG) of the segment
                            b = (s[0] + int(np.round(sthres/dt)))*nbin
                        else:
                            b = int((s[-1]+1)*nbin)

                        sup = list(range(int(s[0]*nbin), b))

                        if sup[-1]>len(EEG):
                            sup = list(range(int(s[0]*nbin), len(EEG)))
                        if len(sup) >= nwin:
                            Pow, F = power_spectrum(EEG[sup], nwin, 1/sr)
                            if pnorm:
                                Pow = np.divide(Pow, pow_norm)
                            Spectra[idf][0].append(Pow)
                        
                # now analyze sequences overlapping with laser
                seq_lsr = []
                for s in seq:
                    s = np.intersect1d(s, laser_idx)
                    if len(s) > 0:
                        q = get_sequences(s)
                        seq_lsr += q

                for s in seq_lsr:
                    s = np.intersect1d(s, laser_idx)
                    
                    if len(s)*nbin >= nwin:
                        # calculate power spectrum
                        # upsample indices
                        # brain state time 0     1         2
                        # EEG time         0-999 1000-1999 2000-2999
                        drn = (s[-1]-s[0])*dt
                        if drn > sthres:
                            b = (s[0] + int(np.round(sthres/dt)))*nbin
                            #pdb.set_trace()
                        else:
                            b = int((s[-1]+1)*nbin)
                        sup = list(range(int(s[0]*nbin), b))
                        if sup[-1]>len(EEG):
                            sup = list(range(int(s[0]*nbin), len(EEG)))
                        # changed line on 02/08/2019
                        if len(sup) >= nwin:
                            Pow, F = power_spectrum(EEG[sup], nwin, 1/sr)
                            if pnorm:
                                Pow = np.divide(Pow, pow_norm)
                            Spectra[idf][1].append(Pow)
                        
            # don't care about laser
            if pmode == 0:
                for s in seq:
                    if len(s)*nbin >= nwin:
                        drn = (s[-1]-s[0])*dt
                        if drn > sthres:
                            b = (s[0] + int(np.round(sthres/dt)))*nbin
                        else:
                            b = int((s[-1]+1)*nbin)
                        sup = list(range(int(s[0]*nbin), b))
                        if sup[-1]>len(EEG):
                            sup = list(range(int(s[0]*nbin), len(EEG)))
                        # changed line on 02/08/2019
                        if len(sup) >= nwin:
                            Pow, F = power_spectrum(EEG[sup], nwin, 1/sr)
                            if pnorm:
                                Pow = np.divide(Pow, pow_norm)
                            Spectra[idf][0].append(Pow)
                
            Pow = {0:[], 1:[]}
            if len(Ids)==1:
                # only one mouse
                Pow[0] = np.array(Spectra[Ids[0]][0])
                Pow[1] = np.array(Spectra[Ids[0]][1])
            else:
                # several mice
                Pow[0] = np.zeros((len(Ids),len(F)))
                Pow[1] = np.zeros((len(Ids),len(F)))
                i = 0
                for m in Ids:
                    Pow[0][i,:] = np.array(Spectra[m][0]).mean(axis=0)
                    if pmode == 1:
                        Pow[1][i,:] = np.array(Spectra[m][1]).mean(axis=0)
                    i += 1

    if f_max > -1:
        ifreq = np.where(F<=f_max)[0]
        F = F[ifreq]
        Pow[0] = Pow[0][:,ifreq]
        if pmode==1:
            Pow[1] = Pow[1][:,ifreq]
    else:
        f_max = F[-1]
    
    if pplot:
        plt.ion()
        plt.figure()
        if sig_type == 'EEG':
            ax = plt.axes([0.2, 0.15, 0.6, 0.7])
            n = Pow[0].shape[0]
            clrs = sns.color_palette("husl", len(mouse_order))
            if pmode==1:
                if not single_mode:
                    a = Pow[1].mean(axis=0) - Pow[1].std(axis=0) / np.sqrt(n)
                    b = Pow[1].mean(axis=0) + Pow[1].std(axis=0) / np.sqrt(n)
                    plt.fill_between(F, a, b, alpha=0.5, color=laser_color)
                    plt.plot(F, Pow[1].mean(axis=0), color=laser_color, lw=2, label='With laser')
                else:
                    for i in range(len(mouse_order)):
                        plt.plot(F, Pow[1][i,:], '--', color=clrs[i])
                    #plt.legend(bbox_to_anchor=(0., 1.0, 1., .102), loc=3, mode='expand', ncol=len(mouse_order),
                    #           frameon=False)

            if not single_mode:
                a = Pow[0].mean(axis=0)-Pow[0].std(axis=0)/np.sqrt(n)
                b = Pow[0].mean(axis=0)+Pow[0].std(axis=0)/np.sqrt(n)
                plt.fill_between(F, a, b, alpha=0.5, color='gray')
                plt.plot(F, Pow[0].mean(axis=0), color='gray', lw=2, alpha=0.5, label='W/o laser')
            else:
                for i in range(len(mouse_order)):
                    plt.plot(F, Pow[0][i, :], label=mouse_order[i], color=clrs[i])
                plt.legend(bbox_to_anchor=(0., 1.0, 1., .102), loc=3, mode='expand', ncol=len(mouse_order), frameon=False)

            if pmode==1 and not single_mode:
                plt.legend(bbox_to_anchor=(0., 1.0, 1., .102), loc=3, mode='expand', frameon=False)

            box_off(ax)
            plt.xlim([0, f_max])
            plt.xlabel('Freq. (Hz)')
            plt.ylabel('Power ($\mathrm{\mu V^2}$)')
            plt.show()

        else:
            # plot EMG amplitude
            nmice = len(mouse_order)
            clrs = sns.color_palette("husl", nmice)
            # plot EMG
            Ampl = {}
            # range of frequencies
            mfreq = np.where((F >= mu[0]) & (F <= mu[1]))[0]
            df = F[1] - F[0]
            if pmode==1:
                for i in [0, 1]:
                    Ampl[i] = np.sqrt(Pow[i][:,mfreq].sum(axis=1)*df)
            else:
                Ampl[0] = np.sqrt(Pow[0][:,mfreq].sum(axis=1)*df)

            if pmode==1:
                ax = plt.axes([0.2, 0.15, 0.4, 0.7])
                ax.bar([0], Ampl[0].mean(), color='gray', label='w/o laser')
                ax.bar([1], Ampl[1].mean(), color='blue', label='laser')
                plt.legend(bbox_to_anchor=(0., 1.0, 1., .102), loc=3, mode='expand', frameon=False)

                for i in range(nmice):
                    plt.plot([0,1], [Ampl[0][i], Ampl[1][i]], color=clrs[i], label=mouse_order[i])

                box_off(ax)
                plt.ylabel('EMG Ampl. ($\mathrm{\mu V}$)')
                ax.set_xticks([0,1])
                ax.set_xticklabels(['', ''])

                # some basic stats
                [tstats, p] = stats.ttest_rel(Ampl[0], Ampl[1])
                print("Stats for EMG amplitude: t-statistics: %.3f, p-value: %.3f" % (tstats, p))

    if len(fig_file) > 0:
        save_figure(fig_file)

    return Pow, F




### TRANSITION ANALYSIS #########################################################
def transition_analysis(ppath, rec_file, pre, laser_tend, tdown, large_bin,
                        backup='', stats_mode=0, after_laser=0, tstart=0, tend=-1, fig_file='', fontsize=12):
    """
    transition analysis
    :param ppath: base recording folder
    :param rec_file: file, list of recordings
    :param pre: time before laser onset [s]
    :param laser_tend: duration of laser stimulation
    :param tdown: time bin for brainstate (>= 2.5 s)
    :param large_bin: >= tdown; average large_bin/tdown consecutive transition probabilities
    :param backup: possible backup base folder, e.g. on an external hard drive
    :param stats_mode: int; The transition probability during the baseline period is
            either by averaging over individual transition probabilities
            during laser and baseline period (stats_mode == 1);
            or by calculating a markov matrix on its own (at large_bin resolution)
            for laser and baseline period (state_mode == 0)
    :param after_laser: float, exclude $after_laser seconds after laser offset for
            baseline calculation
    :param tstart only consider laser trials starting after tstart seconds
    :param tend   only consider laser trials starting before tend seconds
    :return: dict, transitions id --> 3 x 3 x time bins np.array
    """
    nboot = 1000
    mouse_trials = 50

    if type(rec_file) == str:
        E = load_recordings(ppath, rec_file)[1]
    else:
        E = rec_file
    post = pre + laser_tend

    # set path for each recording: either ppath or backup
    rec_paths = {}
    mouse_ids = {}
    for m in E:
        idf = re.split('_', m)[0]
        if len(backup) == 0:
            rec_paths[m] = ppath
        else:
            if os.path.isdir(os.path.join(ppath, m)):
                rec_paths[m] = ppath
            else:
                rec_paths[m] = backup
        if not idf in mouse_ids:
            mouse_ids[idf] = 1
    mouse_ids = list(mouse_ids.keys())

    # Dict:  Mouse_id --> all trials of this mouse 
    MouseMx = {idf:[] for idf in mouse_ids}

    for m in E:
        trials = _whole_mx(rec_paths[m], m, pre, post, tdown, tstart=tstart, tend=tend)
        idf = re.split('_', m)[0]
        MouseMx[idf] += trials
    ntdown = len(trials[0])

    for idf in mouse_ids:
        MouseMx[idf] = np.array(MouseMx[idf])

    # dict mouse_id --> number of trials
    num_trials = {k:len(MouseMx[k]) for k in MouseMx}
    ntrials = sum(num_trials.values())

    # Markov Computation & Bootstrap 
    # time axis:
    t = np.linspace(-pre, post-large_bin, int((pre+post)/large_bin))

    # to make sure that the bar for block 0 to $large_bin is centered around $large_bin/2
    # such that the bars left end touches 0 s
    t += large_bin/2.0
    dt = t[1]-t[0]

    if tdown == large_bin:
        # the first bin that includes 0s (i.e. that touches the laser),
        # should be centered around 0s
        t += dt/2

    tfine = np.linspace(-pre, post-tdown, int((pre+post)/tdown))
    ilsr_fine = np.where((tfine>=0) & (tfine<laser_tend))[0]
    tmp = np.where((tfine >= 0) & (tfine < (laser_tend + after_laser)))[0]
    ibase_fine = np.setdiff1d(np.arange(0, len(tfine)), tmp)

    ### indices during and outside laser stimulation
    # indices of large bins during laser
    ilsr = np.where((t>=0) & (t<laser_tend))[0]
    tmp = np.where((t>=0) & (t<(laser_tend+after_laser)))[0]
    # indices of large bins outsize laser (and $after_laser)
    ibase = np.setdiff1d(np.arange(0, len(t)), tmp)
    
    # number of large time bins
    nseq = len(t)

    # states
    M = dict()
    Base = dict()
    Laser = dict()
    states = {1:'R', 2:'W', 3:'N'}
    for si in range(1,4):
        for sj in range(1,4):
            id = states[si] + states[sj]
            M[id] = np.zeros((nboot, nseq))
            Base[id] = np.zeros((nboot,))
            Laser[id] = np.zeros((nboot,))

    MXsel = np.zeros((mouse_trials*len(mouse_ids), ntdown))
    for b in range(nboot):
        i = 0
        for idf in mouse_ids:
            num = num_trials[idf]
            itrials = rand.randint(0, num, (mouse_trials,))
            sel = MouseMx[idf][itrials,:]
            MXsel[i*mouse_trials:(i+1)*mouse_trials,:] = sel
            i += 1

        # calculate average laser and baseline transition probability to have two values to compare
        # for statistics
        if stats_mode == 0:
            baseline = complete_transition_matrix(MXsel, ibase_fine)
            lsr      = complete_transition_matrix(MXsel, ilsr_fine)

        # caluclate actual transition probilities
        if not large_bin/tdown == 1:
            MXb = build_markov_matrix_blocks(MXsel, tdown, large_bin)
        else:
            MXb = build_markov_matrix_seq(MXsel)

        for si in states:
            for sj in states:
                id = states[si] + states[sj]
                M[id][b,:] = np.squeeze(MXb[si-1, sj-1,:])
                if stats_mode == 0:
                    Base[id][b]  = baseline[si-1, sj-1]
                    Laser[id][b] = lsr[si-1, sj-1]

    if stats_mode == 1:
        for si in states:
            for sj in states:
                id = states[si] + states[sj]
                Base[id]  = np.mean(M[id][:,ibase], axis=1)
                Laser[id] = np.mean(M[id][:, ilsr], axis=1)

    # plotting
    alpha = 0.05
    plt.ion()
    set_fontsize(fontsize)
    plt.figure(figsize=(8,6))
    for si in states.keys():
        for sj in states.keys():
            id = states[si] + states[sj]
            pi = (si - 1) * 3 + sj
            ax = plt.subplot(int('33'+str(pi)))
            ax.add_patch(patches.Rectangle((0, 0), laser_tend, 1, facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1]))
            plt.xlim([t[0]-dt, t[-1]])
            box_off(ax)

            P = M[id]
            for i in range(P.shape[1]):
                P[:,i].sort()
            Bounds = np.zeros((2, P.shape[1]))
            Bounds[0,:] = -1.0 * P[int(nboot*(alpha/2)),:] + np.nanmean(P, axis=0)
            Bounds[1,:] = P[-int(nboot * (alpha / 2)),:] - np.nanmean(P, axis=0)

            baseline = Base[id]
            baseline.sort()
            Bounds_baseline = np.zeros((2,))
            basel_mean = np.nanmean(baseline)
            Bounds_baseline[0] = -baseline[int(nboot*(alpha / 2))] + basel_mean
            Bounds_baseline[1] = baseline[-int(nboot*(alpha / 2))] - basel_mean
            aa = (basel_mean-Bounds_baseline[0]) * np.ones((nseq,))
            bb = (basel_mean+Bounds_baseline[1]) * np.ones((nseq,))

            ax.bar(t, np.nanmean(P, axis=0), yerr=Bounds, width=large_bin-large_bin*0.05, color='gray')
            plt.fill_between(t, aa, bb, alpha=0.5, lw=0, color='red', zorder=2)

            # set title
            if si == 1 and sj == 1:
                plt.ylabel('REM')
                plt.title('REM', fontsize=fontsize)
            if si == 1 and sj == 2:
                plt.title('Wake', fontsize=fontsize)
            if si == 1 and sj == 3:
                plt.title('NREM', fontsize=fontsize)
            if si == 2 and sj == 1:
                plt.ylabel('Wake')
            if si == 3 and sj == 1:
                plt.ylabel('NREM')
            if si == 3 and sj == 1:
                plt.ylim([0, 0.5])

            ax.set_xticks([-pre, 0, laser_tend, laser_tend+pre])

            if si != 3:
                ax.set_xticklabels([])

            if si==3:
                plt.xlabel('Time (s)')
    plt.draw()

    # Statistics summary
    Mod = np.zeros((3,3))
    Sig = np.zeros((3,3))
    for si in states:
        for sj in states:
            id = states[si] + states[sj]

            # probabilities during laser stimulation
            laser = Laser[id]

            # probabilities outside laser stimulation
            basel = Base[id]

            d = laser - basel
            p = len(np.where(d >= 0)[0]) / (1.0*len(d))
            Mod[si-1, sj-1] = np.nanmean(laser) / np.nanmean(basel)
            s = '='
            if Mod[si-1, sj-1] > 1:
                val = 1 - p
                if val == 1:
                    s = '>'
                    val = 1 - 1.0 / nboot
                if val == 0:
                    s = '<'
                    val = 1.0 / nboot
                Sig[si-1, sj-1] = val
                if val < alpha:
                    print('%s -> %s: Trans. prob. is INCREASED by a factor of %.3f; P %s %.4f'
                          % (states[si], states[sj], Mod[si-1, sj-1], s, val))
                else:
                    print('%s -> %s: Trans. prob. is increased by a factor of %.3f; P %s %.4f'
                        % (states[si], states[sj], Mod[si-1, sj-1], s, val))
            else:
                val = p
                if val == 1:
                    s = '>'
                    val = 1 - 1.0 / nboot
                if val == 0:
                    s = '<'
                    val = 1.0 / nboot
                Sig[si-1, sj-1] = val
                if val < alpha:
                    print('%s -> %s: Trans. prob. is DECREASED by a factor of %.3f; P %s %.4f'
                            % (states[si], states[sj], Mod[si - 1, sj - 1], s, val))
                else:
                    print('%s -> %s: Trans. prob. is decreased by a factor of %.3f; P %s %.4f'
                            % (states[si], states[sj], Mod[si - 1, sj - 1], s, val))

    if len(fig_file) > 0:
        save_figure(fig_file)

    return M



def build_markov_matrix_blocks(MX, tdown, large_bin):
    """
    pMX = build_markov_matrix_blocks(MX, down, large_bin)
    build sequence of Markov matrices; build one Matrix for each large bin (block) of
    duration $large_bin by using smaller bins of duration $down;
    i.e. $down <= $large_bin
    :param MX, np.array, with time bins on fine (tdown) time scale
    :param tdown: fine time scale
    :param large_bin: coarse time scale
    :return: pMX, 3x3xtime np.array, series of markov matrices; time is the third dimension
    """
    nbins = MX.shape[1]        # number of time bins on fine time scale
    ndown = large_bin/tdown     # number of fine bins in large bin
    nstep = nbins/ndown        # number of large time bins
    nrows = MX.shape[0]

    pMX = np.zeros((3, 3, nstep))
    for s in range(0, nstep):
        mx = MX[:, s*ndown:(s+1)*ndown]
        pmx = np.zeros((3,3))
        c = np.zeros((3,))

        for i in range(0, nrows):
            seq = mx[i,:]
            for j in range(0, ndown-1):
                pmx[int(seq[j])-1, int(seq[j+1])-1] += 1
                c[int(seq[j])-1] += 1

        for i in range(3):
            pmx[i,:] = pmx[i,:] / c[i]

        pMX[:,:, s] = pmx

    return pMX



def build_markov_matrix_seq(MX):
    nseq = MX.shape[1]
    nrows = MX.shape[0]
    MX[np.where(MX==4)] = 3

    pMX = np.zeros((3,3,nseq))
    C   = np.zeros((3,nseq))
    for i in range (0, nrows):
        seq = MX[i,:]
        for j in range (0, nseq-1):
            pMX[int(seq[j])-1,int(seq[j+1])-1,j] += 1
            C[int(seq[j]-1),j] +=1

    for t in range(0, nseq):
        for s in range(3):
            pMX[s,:,t] = pMX[s,:,t] / C[s,t]

    return pMX



def complete_transition_matrix(MX, idx):
    idx_list = get_sequences(idx)

    nrows = MX.shape[0]

    pmx = np.zeros((3, 3))
    c = np.zeros((3,))

    for idx in idx_list:
        for i in range(0, nrows):
            seq = MX[i, idx]
            for j in range(0, len(seq)-1):
                pmx[int(seq[j])-1, int(seq[j+1])-1] += 1
                c[int(seq[j])-1] += 1

    for i in range(3):
        pmx[i, :] = pmx[i, :] / c[i]

    return pmx



def _whole_mx(ppath, name, pre, post, tdown, ptie_break=1, tstart=0, tend=-1):
    """
    @Return:
        List of all laser stimulation trials of recording ppath/name
    """
    SR = get_snr(ppath, name)
    # Number of EEG bins / brainstate bin
    NBIN = np.round(2.5*SR)
    # Precise time bin duration of each brain state:
    dt = NBIN * 1.0/SR
    # ds - number how often dt fits into coarse brain state bin tdown:
    ds = int(np.round(tdown/dt))
    NBIN *= ds

    ipre  = int(np.round(pre/tdown))
    ipost = int(np.round(post/tdown))

    # load brain state
    M = load_stateidx(ppath, name)[0]
    if tend == -1:
        iend = M.shape[0] - 1
    istart = np.round(tstart/dt)

    # downsample brain states
    M = downsample_states(M, ds, ptie_break)
    len_rec = len(M)

    (idxs, idxe) = laser_start_end(load_laser(ppath, name))
    idxs = [s for s in idxs if s >= istart*(NBIN/ds) and s<= iend*(NBIN/ds)]
    idxs = [int(i/NBIN) for i in idxs]
    #idxe = [int(i/NBIN) for i in idxe]
    trials = []
    for s in idxs:
        # i.e. element ii+1 is the first overlapping with laser
        if s>=ipre-1 and s+ipost < len_rec:
            trials.append(M[s-ipre:s+ipost])

    return trials

    
def downsample_states(M, nbin, ptie_break=True):
    """
    downsample brain state sequency by replacing $nbin consecutive bins by the most frequent state
    ptie_break     -    tie break rule: 
                        if 1, wake wins over NREM which wins over REM (Wake>NREM>REM) in case of tie
    """
    n = int(np.floor(len(M))/(1.0*nbin))
    Mds = np.zeros((n,))

    for i in range(n):
        m = M[i*nbin:(i+1)*nbin]
            
        S = np.array([len(np.where(m==s)[0]) for s in [1,2,3]])
        
        if not ptie_break:
            Mds[i] = np.argmax(S)+1  
        else:
            tmp = S[[1,2,0]]
            ii = np.argmax(tmp)
            ii = [1,2,0][ii]
            Mds[i] = ii+1
    
    return Mds
### END Transition Analysis ####################################



def infraslow_rhythm(ppath, recordings, ma_thr=20.0, min_dur = 160,
                     band=[10,15], state=3, win=64, pplot=True, pflipx=True, pnorm=False, tstart=0, tend=-1, peeg2=False):
    """
    calculate powerspectrum of EEG spectrogram to identify oscillations in sleep activity within different frequency bands;
    only contineous NREM periods are considered for
    @PARAMETERS:
    ppath        -       base folder of recordings
    recordings   -       single recording name or list of recordings
    
    @OPTIONAL:
    ma_thr       -       microarousal threshold; wake periods <= $min_dur are transferred to NREM
    min_dur      -       minimal duration [s] of a NREM period
    band         -       frequency band used for calculation
    win          -       window (number of indices) for FFT calculation
    pplot        -       if True, plot window showing result
    pflipx       -       if True, plot wavelength instead of frequency on x-axis
    pnorm        -       if True, normalize spectrum (for each mouse) by its total power
    
    @RETURN:
    SpecMx, f    -       ndarray [mice x frequencies], vector [frequencies]
    """
    import scipy.linalg as LA

    #min_dur = win*2.5
    min_dur = np.max([win*2.5, min_dur])
    
    if type(recordings) != list:
        recordings = [recordings]

    Spec = {}    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        Spec[idf] = []
    mice = list(Spec.keys())
    
    for rec in recordings:
        idf = re.split('_', rec)[0]

        # sampling rate and time bin for spectrogram
        SR = get_snr(ppath, rec)
        NBIN = np.round(2.5*SR)
        dt = NBIN * 1/SR
        istart = int(np.round(tstart/dt))
        if tend > -1:
            iend   = int(np.round(tend/dt))


        # load sleep state
        M = load_stateidx(ppath, rec)[0]
        if tend == -1:
            iend = M.shape[0]
        M = M[istart:iend]
        seq = get_sequences(np.where(M==state)[0], np.round(ma_thr/dt))
        seq = [list(range(s[0], s[-1]+1)) for s in seq]
        
        # load frequency band
        P = so.loadmat(os.path.join(ppath, rec,  'sp_' + rec + '.mat'))
        if not peeg2:
            SP = np.squeeze(P['SP'])[:,istart:iend]
        else:
            SP = np.squeeze(P['SP2'])[:, istart:iend]
        freq = np.squeeze(P['freq'])
        ifreq = np.where((freq>=band[0]) & (freq<=band[1]))
        pow_band = SP[ifreq,:].mean(axis=0)
        
        seq = [s for s in seq if len(s)*dt >= min_dur]   
        for s in seq:
            y,f = power_spectrum(pow_band[:,s], win, dt)
            y = y.mean(axis=0)
            Spec[idf].append(y)
        
    # Transform %Spec to ndarray
    SpecMx = np.zeros((len(Spec), len(f)))
    i=0
    for idf in Spec:
        SpecMx[i,:] = np.array(Spec[idf]).mean(axis=0)
        if pnorm==True:
            SpecMx[i,:] = SpecMx[i,:]/LA.norm(SpecMx[i,:])
        i += 1

    if pplot == True:
        plt.figure()
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])
        
        x = f[1:]
        if pflipx == True: x = 1.0/f[1:]
        y = SpecMx[:,1:]
        if len(mice) <= 1:
            ax.plot(x, y.mean(axis=0), color='gray', lw=2)
            
        else:
            ax.errorbar(x, y.mean(axis=0), yerr=y.std(axis=0), color='gray', fmt='-o')

        box_off(ax)
        if pflipx == True:
            plt.xlabel('Wavelength (s)')
        else:
            plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (uV^2)')
        plt.show()

    return SpecMx, f



def ma_rhythm(ppath, recordings, ma_thr=20.0, min_dur = 160, band=[10,15],
              state=3, win=64, pplot=True, pflipx=True, pnorm=False):
    """
    calculate powerspectrum of EEG spectrogram to identify oscillations in sleep activity within different frequency bands;
    only contineous NREM periods are considered for
    @PARAMETERS:
    ppath        -       base folder of recordings
    recordings   -       single recording name or list of recordings
    
    @OPTIONAL:
    ma_thr       -       microarousal threshold; wake periods <= $min_dur are transferred to NREM
    min_dur      -       minimal duration [s] of a NREM period
    band         -       frequency band used for calculation
    win          -       window (number of indices) for FFT calculation
    pplot        -       if True, plot window showing result
    pflipx       -       if True, plot wavelength instead of frequency on x-axis
    pnorm        -       if True, normalize spectrum (for each mouse) by its total power
    
    @RETURN:
    SpecMx, f    -       ndarray [mice x frequencies], vector [frequencies]
    """
    import scipy.linalg as LA

    min_dur = np.max([win*2.5, min_dur])
    
    if type(recordings) != list:
        recordings = [recordings]

    Spec = {}    
    for rec in recordings:
        idf = re.split('_', rec)[0]
        Spec[idf] = []
    mice = list(Spec.keys())
    
    for rec in recordings:
        idf = re.split('_', rec)[0]

        # sampling rate and time bin for spectrogram
        #SR = get_snr(ppath, rec)
        #NBIN = np.round(2.5*SR)
        #dt = NBIN * 1/SR
        dt = 2.5
        
        # load sleep state
        M = load_stateidx(ppath, "", ann_name=rec)[0]
        Mseq = M.copy()
        Mseq[np.where(M != 2)] = 0
        Mseq[np.where(M == 2)] = 1
        seq = get_sequences(np.where(M==state)[0], ibreak=int(np.round(ma_thr/dt))+1)
        seq = [range(s[0], s[-1]+1) for s in seq]
        
        #pdb.set_trace()
        # load frequency band
        #P = so.loadmat(os.path.join(ppath, rec,  'sp_' + rec + '.mat'));
        #SP = np.squeeze(P['SP'])
        #freq = np.squeeze(P['freq'])
        #ifreq = np.where((freq>=band[0]) & (freq<=band[1]))
        #pow_band = SP[ifreq,:].mean(axis=0)
        
        #pdb.set_trace()
        seq = [s for s in seq if len(s)*dt >= min_dur]   
        #pdb.set_trace()
        for s in seq:
            y,f = power_spectrum(Mseq[s], win, dt)
            #y = y.mean(axis=0)
            Spec[idf].append(y)
        
    # Transform %Spec to ndarray
    SpecMx = np.zeros((len(Spec), len(f)))
    i=0
    for idf in Spec:
        SpecMx[i,:] = np.array(Spec[idf]).mean(axis=0)
        if pnorm==True:
            SpecMx[i,:] = SpecMx[i,:]/LA.norm(SpecMx[i,:])
        i += 1

    if pplot:
        plt.figure()
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])
        
        x = f[1:]
        if pflipx == True: x = 1.0/f[1:]
        y = SpecMx[:,1:]
        if len(mice) <= 1:
            ax.plot(x, y.mean(axis=0), color='gray', lw=2)
            
        else:
            ax.errorbar(x, y.mean(axis=0), yerr=y.std(axis=0), color='gray', fmt='-o')

        box_off(ax)
        if pflipx == True:
            plt.xlabel('Wavelength (s)')
        else:
            plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (uV^2)')
        plt.show()

    return SpecMx, f

