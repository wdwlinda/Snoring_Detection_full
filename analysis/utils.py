
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from pprint import pprint
from dataset.dataset_utils import load_content_from_txt
from scipy import signal
from pydub import AudioSegment
import librosa


def get_audio_waveform(filename):
    if filename.endswith('wav'):
        y, sr = librosa.load(filename)
    elif filename.endswith('m4a'):
        y = AudioSegment.from_file(filename, format='m4a')
        sr = y.frame_rate
    return y, sr


def get_audio_clip(signal, time_interval, sample_rate):
    signal_interval = [int(time_interval[0]*sample_rate), int(time_interval[1]*sample_rate)]
    return signal[signal_interval[0]:signal_interval[1]]


# TODO: more functional
def f_high(y, sr):
    b,a = signal.butter(10, 3000/(sr/2), btype='highpass')
    yf = signal.lfilter(b,a,y)
    return yf


if __name__ == '__main__':
    pass
