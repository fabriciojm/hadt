from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import pickle
from wfdb import rdrecord, rdann, processing
from sklearn import preprocessing
from scipy.signal import resample

import numpy as np
import pandas as pd

def preproc(X):
    # to be called in inference/api
    in_shape = X.shape
    if X.shape[1] != 180:
        print('File shape is not (n, 180) but ', in_shape)

    X = to_time_series_dataset(X)
    X = X.reshape(in_shape[0], -1)
    scaler = TimeSeriesScalerMeanVariance()
    X = scaler.fit_transform(X)
    return X.reshape(in_shape)

def apple_csv_to_data(filepath_csv):
    # extract sampling rate 
    with open(filepath_csv, 'r') as file:
        for il,line in enumerate(file):
            if line.startswith("Sample Rate"):
                # Extract the sample rate
                sample_rate = int(line.split(",")[1].split()[0])  # Split and get the numerical part
                print(f"Sample Rate: {sample_rate}")
                break
            if il > 30:
                print("Could not find sample rate in first 30 lines")
                return None, None  
    X = pd.read_csv(filepath_csv, skiprows=14, header=None)
    return X, sample_rate

def apple_trim_join(X, sample_rate=512, ns=2):
    # There should be a less horrible way of doing this
    # Ignore first two and last two seconds, that tend to be noisy --> 26 seconds ecg
    X[1] = X[1].fillna(0)
    X = X[0] + X[1] / (10 ** (X[1].astype(str).str.len() - 2)) # Ignoring the trailing ".0"
    print(f"Ignoring first and last {ns} seconds")
    X = X[ns*sample_rate:-ns*sample_rate].to_frame().T
    X = X.iloc[0].to_numpy()
    return X

def apple_extract_beats(X, sample_rate=512):
    X = apple_trim_join(X, sample_rate=sample_rate, ns=3)
    # Scale and remove nans (should not happen anymore)
    X = preprocessing.scale(X[~np.isnan(X)])

    # I tried to hack the detection to make it learn peaks and
    # not go with default, but it doesn't work
    # I have tried:
    # - Hardwiring n_calib_beats (not possible from user side)
    #   to a lower number (5, 3).
    # - Setting qrs_width to lower and higher values
    # - Relax the correlation requirement to Rikers wavelet
    # Maybe explore correlation with more robust wavelets
    # wavelet = pywt.Wavelet('db4')
    # (lib/python3.10/site-packages/wfdb/processing/qrs.py)
 
    # Conf = processing.XQRS.Conf(qrs_width=0.1)
    # qrs = processing.XQRS(sig = X,fs = sample_rate, conf=Conf)
    # wfdb library doesn't allow to set n_calib_beats

    qrs = processing.XQRS(sig = X,fs = sample_rate)
    qrs.detect()
    peaks = qrs.qrs_inds
    print("Number of beats detected: ", len(peaks))
    target_length = 180
    beats = np.zeros((len(peaks), target_length))
    
    for i, peak in enumerate(peaks[1:-1]):
        rr_interval = peaks[i + 1] - peaks[i]  # Distance to the next peak
        window_size = int(rr_interval * 1.2)  # Extend by 20% to capture full P-QRS-T cycle
        # Define the dynamic window around the R-peak
        start = max(0, peak - window_size // 2)
        end = min(len(X), peak + window_size // 2)
        beat = resample(X[start:end], target_length)
        beats[i] = beat
    return beats

def save_beats_csv(beats, filepath_csv):
    pd.DataFrame(beats).to_csv(filepath_csv, index=False)

def label_decoding(values, path):
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    inverse_mapping = {v: k for k, v in mapping.items()}
    # return inverse_mapping[values]
    return [inverse_mapping[value] for value in values]