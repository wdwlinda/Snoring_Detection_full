
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
from torch.utils.data import dataset
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from pprint import pprint
from analysis.resample_test import DEFAULT_RESAMPLING_METHOD
from dataset.dataset_utils import load_content_from_txt
from scipy import signal
from pydub import AudioSegment
import librosa


# TODO: in progress
def balancing_indexing(data_path):
    dir_list = get_dir_list(data_path)
    dir_samples_pair = {}
    for d in dir_list:
        if os.path.isdir(os.path.join(data_path, d, '1')):
            p = len(os.listdir(os.path.join(data_path, d, '1')))
        else:
            p = 0
        if os.path.isdir(os.path.join(data_path, d, '0')):
            n = len(os.listdir(os.path.join(data_path, d, '0')))
        else:
            n = 0
        dir_samples_pair[d] = (p, n)

    # minimum sampling
    # class balance sampling


def get_dir_list(data_path):
    dir_list = np.array([], dtype=object)
    for f in os.listdir(data_path):
        folder_path = os.path.join(data_path, f)
        if os.path.isdir(folder_path):
            dir_list = np.append(dir_list, folder_path)
    return list(dir_list)


# def get_audio_waveform(filename):
#     if filename.endswith('wav'):
#         y, sr = librosa.load(filename)
#     elif filename.endswith('m4a'):
#         y = AudioSegment.from_file(filename, format='m4a')
#         sr = y.frame_rate
#     return y, sr


def get_audio_clip(signal, time_interval, sample_rate):
    signal_interval = [int(time_interval[0]*sample_rate), int(time_interval[1]*sample_rate)]
    return signal[signal_interval[0]:signal_interval[1]]


# TODO: more functional
def f_high(y, sr):
    b,a = signal.butter(10, 3000/(sr/2), btype='highpass')
    yf = signal.lfilter(b,a,y)
    return yf

def load_audio_waveform(filename, audio_format, sr=None, channels=None):
    y = AudioSegment.from_file(filename, audio_format)
    if sr: y = y.set_frame_rate(sr)
    if channels: y = y.set_channels(channels)
    return y


def main():
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\raw_mono_16k_h'
    balancing_indexing(data_path)


if __name__ == '__main__':
    main()
    pass
