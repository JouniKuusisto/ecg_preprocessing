# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:48:51 2022

@author: Jouni
"""

import wfdb
import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#path of locally saved dataset
PATH="C:/Users/Jouni/Datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/"

#following database-file contains human interpretations of the ECG, including information on rhythm during the ECG recording
y_df = pd.read_csv(PATH+"ptbxl_database.csv", index_col = "ecg_id")
y_df["scp_codes"] = y_df["scp_codes"].apply(lambda x: ast.literal_eval(x))

n_afib = 0
n_sr = 0

#for this ML project relevant label information is gathered from "ptbxl_database.csv" files column "scp_codes"
y_df["normal_ecg"] = y_df.scp_codes.apply(lambda x: int("NORM" in x.keys()))
y_df["normal_rhythm"] = y_df.scp_codes.apply(lambda x: int("SR" in x.keys()))
y_df["atrial_fibrillation"] = y_df.scp_codes.apply(lambda x: int("AFIB" in x.keys()))

#following calculates FFT of all the summed ECG signal - all channels are added together and the FFT is performed
def karvalakki_ft(file, n_peaks = False, plot = True, raja = False, limit = False):
    record = wfdb.rdrecord(file)
    signaali = record.p_signal
    if limit:
        signaali = signaali[:limit,:]
    summasignaali = np.sum(signaali, axis = 1)
    ft = np.fft.rfft(summasignaali)
    abs_ft = np.abs(ft)
    power_spectrum = np.square(abs_ft)
    frequency = np.linspace(0, record.fs/2, len(power_spectrum))
    power_spectrum[0:5] = 0 # we dont need baseline
    peaks = find_peaks(power_spectrum, height=np.max(power_spectrum)/200, distance=5)
    if n_peaks:
        sorted_indices = peaks[1]["peak_heights"].argsort()
        indices = np.flip(sorted_indices[-n_peaks:])
        peak_magnitudes = peaks[1]["peak_heights"][indices]
        peak_frequencies = peaks[0][indices]
    else:
        sorted_indices = peaks[1]["peak_heights"].argsort()
        indices = np.flip(sorted_indices)
        peak_magnitudes = peaks[1]["peak_heights"][indices]
        peak_frequencies = peaks[0][indices]
    if plot:
        if raja:
            plt.plot(frequency[0:raja], power_spectrum[0:raja])
            plt.scatter(peak_frequencies/10, peak_magnitudes, c = "r") #jako 10 koska freq bin on 0.1 Hz levyinen tjs
            plt.ylabel("amplitude")
            plt.xlabel("frequency (Hz)")
            plt.xlim(0,50)
            #plt.yscale("log")
            plt.show()
        else:
            plt.plot(frequency, power_spectrum)
            plt.scatter(peak_frequencies/10, peak_magnitudes, c = "r") #jako 10 koska freq bin on 0.1 Hz levyinen tjs
            plt.ylabel("amplitude")
            plt.xlabel("frequency (Hz)")
            plt.xlim(0,50)
            #plt.yscale("log")
            plt.show()
    return (power_spectrum, frequency, peak_magnitudes, peak_frequencies/10)

#for ECG visualization
def ecg_plot(file, summed = False):
    record = wfdb.rdrecord(file)
    signaali = record.p_signal
    if summed:
        signaali = np.sum(signaali, axis = 1)
    plt.plot(signaali)
    
#this creates and returns the dataset: datapoints will be FFT data and labels are either normal "0" or atrial fibrillation "1"
def create_dataset(): # af == 1 and norm == 0
    af_df = y_df[y_df.atrial_fibrillation == 1]
    norm_df = y_df[y_df.normal_rhythm == 1]
    m = len(af_df)+len(norm_df)
    X = np.zeros((m, 20)) # rows will have 20 features: 10 highest amplitude peaks and their frequency
    y = np.zeros(m)
    for i in range(len(af_df)):
        _ , _ , peak_magnitudes, peak_frequencies = karvalakki_ft(PATH + af_df.iloc[i].filename_hr, n_peaks = False, plot = False)
        padded_peak = np.zeros(10)
        padded_freq = np.zeros(10)
        padded_peak[:peak_magnitudes.shape[0]] = peak_magnitudes[:10]
        padded_freq[:peak_frequencies.shape[0]] = peak_frequencies[:10]
        
        X[i,:10] = padded_peak
        X[i,10:] = padded_freq
        y[i] = 1
        print(i)
        print("af") #printing the iterator to get follow the progress
        
    for i in range(len(norm_df)):
        _ , _ , peak_magnitudes, peak_frequencies = karvalakki_ft(PATH + norm_df.iloc[i].filename_hr, n_peaks = False, plot = False)
        padded_peak = np.zeros(10)
        padded_freq = np.zeros(10)
        padded_peak[:peak_magnitudes.shape[0]] = peak_magnitudes[:10]
        padded_freq[:peak_frequencies.shape[0]] = peak_frequencies[:10]
        X[i+len(af_df),:10] = padded_peak
        X[i+len(af_df),10:] = padded_freq
        y[i+len(af_df)] = 0
        print(i)
        print("norm")
        
    
    return (X, y)
#by inspecting all 10 max frequencies, frequencies >= 50 could be discarded

#logarithmic transformation of features to be assessed

X, y = create_dataset()
tallennus = np.zeros((18296, 21))
tallennus[:,:-1] = X
tallennus[:,-1] = y
pd.DataFrame(tallennus).to_csv("dataset_all_freq.csv", header = False, index = False)