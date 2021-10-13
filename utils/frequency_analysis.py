# Frequency analysis
from __future__ import print_function
import os
import librosa
import librosa.display
import numpy as np
import datetime
import csv
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp


def get_audio_waveform(filename):
    if filename.endswith('wav'):
        y, sr = librosa.load(filename)
    elif filename.endswith('m4a'):
        y = AudioSegment.from_file(filename, format='m4a')
        sr = y.frame_rate
    return y, sr


def get_frquency(signal, sr, n_fft):
    # short time fourier transform
    # (n_fft and hop length determine frequency/time resolution)
    S = librosa.stft(signal, n_fft=n_fft, hop_length=n_fft//2)
    # print(S.shape)
    # convert to db
    # (for your CNN you might want to skip this and rather ensure zero mean and unit variance)
    cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
    # return librosa.amplitude_to_db(np.abs(S), ref=np.max)

    return cent[0]
    # return S

def get_audio_clip(signal, time_interval, sample_rate):
    signal_interval = [int(time_interval[0]*sample_rate), int(time_interval[1]*sample_rate)]
    return signal[signal_interval[0]:signal_interval[1]]


def convert_and_plot_frequency(signal, sr, time_interval, n_fft, save_path):
    # Get audio clip
    clip = get_audio_clip(signal, time_interval, sr)

    # Process
    # print(clip.shape)
    # clip = clip[list(range(0, len(clip), 20))]
    # print(clip.shape)
    # clip[np.abs(clip)<50] = 0

    # Convert to frquency domain
    f = get_frquency(clip, sr, n_fft)
    # print(f)
    
    # # plot
    # plt.figure(figsize=(6,6))
    print(signal.shape, clip.shape)
    fig, ax = plt.subplots(2,1)
    ax[0].plot(f)
    ax[0].set_xlabel('sample')
    ax[0].set_ylabel('frequency')
    ax[1].plot(clip)
    # fig.show()
    plt.savefig(save_path)


def freq_histogram(path, n_fft, color):
    snoring_files = os.listdir(path)
    total_freq = []
    os.chdir(path)
    for idx, f in enumerate(snoring_files):
        print(f'{idx+1}/{len(snoring_files)} processing {f}')
        signal, sr = get_audio_waveform(f)
        # y = np.float32(np.array(signal.get_array_of_samples()))
        y = signal

        # Convert to frquency
        f = np.mean(get_frquency(y, sr, n_fft))
        total_freq.append(f)

    n, bins, patches = plt.hist(np.array(total_freq), 50, density=True, facecolor=color, alpha=0.75)
    plt.xlabel('Frequency')
    plt.ylabel('Probability')
    plt.title('Histogram of Frequency')
    plt.grid(True)
    


def plot_freq_histogram():
    snoring_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\subset2\1'
    non_snoring_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\subset2\0'
    n_fft = 2048
    freq_histogram(snoring_path, n_fft, color='b')
    freq_histogram(non_snoring_path, n_fft, color='g')
    plt.show()

def frequency_analysis():
    n_fft = 2048
    time_range = 1
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1620055140118_ASUS_I002D\1620055140118_ASUS_I002D_12.m4a'
    filename = rf'C:\Users\test\Downloads\1007\env_sounds\alarm\1630345236867_100.m4a'
    save_path = rf'C:\Users\test\Downloads\1007\env_sounds\alarm'
    # time_interval = [56, 57]
    # time_interval = [62, 63]
    # time_interval = [56, 63]

    # Load audio
    signal, sr = get_audio_waveform(filename)
    y = np.float32(np.array(signal.get_array_of_samples()))
    for i in range(0, 180, time_range):
        print(i+1)
        name = os.path.basename(filename).split('.')[0]
        convert_and_plot_frequency(y, sr, [i, i+time_range], n_fft, os.path.join(save_path, f'{name}_{i+1:03d}.png'))
    
    name = os.path.basename(filename).split('.')[0]
    convert_and_plot_frequency(filename, [0, 180], n_fft, os.path.join(save_path, f'{name}_full.png'))


def main():
    # frequency_analysis()
    plot_freq_histogram()


if __name__ == '__main__':
    main()