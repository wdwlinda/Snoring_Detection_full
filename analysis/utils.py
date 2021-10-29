
import os
import matplotlib
import matplotlib.pyplot as plt
from numpy.ma.core import concatenate
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
from dataset.dataset_utils import load_content_from_txt, save_content_in_txt
from scipy import signal
from pydub import AudioSegment
import librosa


# # TODO: in progress
# def balancing_indexing(data_path):
#     dir_list = get_dir_list(data_path)
#     dir_samples_pair = {}
#     for d in dir_list:
#         if os.path.isdir(os.path.join(data_path, d, '1')):
#             p = len(os.listdir(os.path.join(data_path, d, '1')))
#         else:
#             p = 0
#         if os.path.isdir(os.path.join(data_path, d, '0')):
#             n = len(os.listdir(os.path.join(data_path, d, '0')))
#         else:
#             n = 0
#         dir_samples_pair[d] = (p, n)

#     # minimum sampling
#     # class balance sampling


def get_dir_list(data_path):
    dir_list = np.array([], dtype=object)
    for f in os.listdir(data_path):
        folder_path = os.path.join(data_path, f)
        if os.path.isdir(folder_path):
            dir_list = np.append(dir_list, folder_path)
    return list(dir_list)


def balancing_indexing(path):
    dir_list = get_dir_list(path)
    total_p, total_n = np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    for d in dir_list:
        path_p = os.path.join(path, d, '1')
        path_n = os.path.join(path, d, '0')
        if os.path.isdir(path_p):
            p = len(os.listdir(path_p))
            total_p = np.append(total_p, p)
        else:
            p = 0
        if os.path.isdir(path_n):
            n = len(os.listdir(path_n))
            total_n = np.append(total_n, n)
        else:
            n = 0

    # minimum
    # print(np.min(total_p), np.min(total_n))
    total_positive_samples, total_negative_samples = np.array([], dtype=object), np.array([], dtype=object)
    min_sample = min(np.min(total_p), np.min(total_n))
    for d in dir_list:
        path_p = os.path.join(path, d, '1')
        if os.path.isdir(path_p):
            p_list = [os.path.join(d, '1', f) for f in os.listdir(path_p)]
            positive_samples = np.random.choice(p_list, min_sample)
            total_positive_samples = np.append(total_positive_samples, positive_samples)
        
        path_n = os.path.join(path, d, '0')
        if os.path.isdir(path_n):
            n_list = [os.path.join(d, '0', f) for f in os.listdir(path_n)]
            negative_samples = np.random.choice(n_list, min_sample)
            total_negative_samples = np.append(total_negative_samples, negative_samples)

    # balancing_samples = list(np.concatenate([total_positive_samples, total_negative_samples]))
    
    # print(3)


    # total balancing
    return total_positive_samples, total_negative_samples

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
    
    p, n = balancing_indexing(data_path)
    index_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\ASUS_h_min_balance'
    content = np.concatenate([p, n])

    with open(os.path.join(index_path, 'train.txt'), 'w+') as fw:
        for c in content:
            fw.write(f'{c}\n')

if __name__ == '__main__':
    main()
    pass
