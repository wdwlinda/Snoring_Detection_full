from __future__ import print_function
import librosa
from librosa.core import audio
import librosa.display
from numpy.core.numeric import _outer_dispatcher
from numpy.lib.npyio import save
import soundfile as sf
import numpy as np
import datetime
import time
import csv
import pydub
from pydub import AudioSegment
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy import ndimage
import random
from scipy.io.wavfile import read
from analysis import data_splitting
import pandas as pd
import test
from pprint import pprint
from analysis import utils
from dataset import dataset_utils
import array


def audio_loading_exp():
    # They are all same
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1620055140118_ASUS_I002D\1620055140118_ASUS_I002D_12.m4a'
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1598482996718_NA\1598482996718_8.m4a'
    save_path = rf'C:\Users\test\Downloads\1007'
    y1 = AudioSegment.from_file(filename, format='m4a')
    y2 = AudioSegment.from_file(filename, format='m4a', frame_rate=44100, channels=2, sample_width=2)
    y5 = AudioSegment.from_file(filename, format='raw', frame_rate=44100, channels=2, sample_width=2)

    w1 = np.array(y1.get_array_of_samples())
    w2 = np.array(y2.get_array_of_samples())
    w5 = np.array(y5.get_array_of_samples())

    c1 = y1[1000:5000]
    c2 = y2[1000:5000]

    y3 = pydub.AudioSegment(
        w1.tobytes(), 
        frame_rate=44100,
        sample_width=w1.dtype.itemsize, 
        channels=1
    )
    c3 = y3[1000:5000]

    c4 = w1[int(1*44100):int(5*44100)]
    c4 = pydub.AudioSegment(
        c4.tobytes(), 
        frame_rate=44100,
        sample_width=c4.dtype.itemsize, 
        channels=1
    )
    
    c5 = w2[int(1*y2.frame_rate):int(5*y2.frame_rate)]
    c5 = pydub.AudioSegment(
        c5.tobytes(), 
        frame_rate=y2.frame_rate,
        sample_width=c5.dtype.itemsize, 
        channels=1
    )

    c1.export(os.path.join(save_path, 'c1.wav'), format='wav')
    c1.export(os.path.join(save_path, 'c1.m4a'), format='mp4')
    c2.export(os.path.join(save_path, 'c2.wav'), format='wav')
    c2.export(os.path.join(save_path, 'c2.m4a'), format='mp4')
    c3.export(os.path.join(save_path, 'c3.m4a'), format='mp4')
    c3.export(os.path.join(save_path, 'c3.wav'), format='wav')
    c4.export(os.path.join(save_path, 'c4.m4a'), format='mp4')
    c4.export(os.path.join(save_path, 'c4.wav'), format='wav')
    c5.export(os.path.join(save_path, 'c5.m4a'), format='mp4')
    c5.export(os.path.join(save_path, 'c5.wav'), format='wav')


def enframe(x, win, inc):
    """
    Splits the vector up into (overlapping) frames beginning at increments
    of inc. Each frame is multiplied by the window win().
    The length of the frames is given by the length of the window win().
    The centre of frame I is x((I-1)*inc+(length(win)+1)/2) for I=1,2,...
    :param x: signal to split in frames
    :param win: window multiplied to each frame, length determines frame length
    :param inc: increment to shift frames, in samples
    :return f: output matrix, each frame occupies one row
    :return length, no_win: length of each frame in samples, number of frames
    """
    nx = len(x)
    nwin = len(win)
    if (nwin == 1):
        length = win
    else:
        # length = next_pow_2(nwin)
        length = nwin
    nf = int(np.fix((nx - length + inc) // inc))
    # f = np.zeros((nf, length))
    indf = inc * np.arange(nf)
    inds = np.arange(length) + 1
    f = x[(np.transpose(np.vstack([indf] * length)) +
           np.vstack([inds] * nf)) - 1]
    if (nwin > 1):
        w = np.transpose(win)
        f = f * np.vstack([w] * nf)
    f = signal.detrend(f, type='constant')
    no_win, _ = f.shape
    return f, length, no_win


def check_audio_sample_rate():
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring'
    data_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\_sample_data\resample_test\resample'
    # data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw'
    filelist = []
    check = []
    audio_foemat = 'wav'
    save_name = 'resample_sr'
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if audio_foemat in f:
                filelist.append(os.path.join(data_path, root, f))
                # break

    with open(f'{save_name}.txt', 'w+') as fw:
        for idx, filename in enumerate(filelist):
            print(f'{idx+1}/{len(filelist)} {filename}')
            audio = AudioSegment.from_file(filename, format=audio_foemat)
            sr = audio.frame_rate
            channels = audio.channels
            fw.write(f'{filename} sr: {sr} channels: {channels}')
            fw.write('\n')




def show_volume():
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1606921286802_sargo\1606921286802_sargo_15.m4a'
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1620055140118_ASUS_I002D\1620055140118_ASUS_I002D_5.m4a'
    waveform, sr = utils.load_audio_waveform(filename, 'm4a')
    waveform = waveform.get_array_of_samples()
    waveform = np.array(waveform)
    waveform = utils.get_audio_clip(waveform, [82, 85], sr)
    get_audio_volume(waveform, frame_size=512)


def get_audio_volume(signal, frame_size):
    volume = []
    for i in range(0, len(signal), frame_size):
        if i%100 == 0:
            print(f'{i}/{len(signal)}')
        if i+frame_size < len(signal):
            clip = signal[i:i+frame_size]
        else:
            clip = signal[i:]
        volume.append(np.abs(np.sum(clip)))
    fig, ax = plt.subplots(2,1)
    ax[0].plot(signal)
    ax[0].set_title('Singal')
    ax[1].plot(volume)
    ax[1].set_title('Volume')
    plt.show()


def show_median_filter():
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1606921286802_sargo\1606921286802_sargo_15.m4a'
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1620055140118_ASUS_I002D\1620055140118_ASUS_I002D_5.m4a'
    waveform, sr = utils.load_audio_waveform(filename, 'm4a')
    waveform = waveform.get_array_of_samples()
    waveform = np.array(waveform)
    waveform = utils.get_audio_clip(waveform, [82, 85], sr)
    _1d_median_filtering(waveform, frame_size=4096)


def _1d_median_filtering(signal, frame_size):
    proceesed_signal = []
    for i in range(len(signal)):
        if i%1000 == 0:
            print(f'{i}/{len(signal)}')
        if i+frame_size < len(signal):
            med_val = np.median(signal[i:i+frame_size])
        else:
            med_val = np.median(signal[i:])
        proceesed_signal.append(med_val)
    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(signal)
    # ax[0].set_title('Singal')
    # ax[1].plot(proceesed_signal)
    # ax[1].set_title(f'Median filter (frame_size={frame_size})')
    # plt.show()
    return proceesed_signal


def _1d_mean_filtering(signal, frame_size):
    proceesed_signal = []
    for i in range(len(signal)):
        if i%1000 == 0:
            print(f'{i}/{len(signal)}')
        if i+frame_size < len(signal):
            med_val = np.mean(signal[i:i+frame_size])
        else:
            med_val = np.mean(signal[i:])
        proceesed_signal.append(med_val)
    return proceesed_signal


def check_two_channels():
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw\1631033725248_AA1600174\1631033725248_28.m4a'
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw\1631385387541_NA\1631385387541_17.m4a'
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw\1631456400568_AA2001038\1631456400568_8.m4a' # mono
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw\1631468777871_NA\1631468777871_8.m4a'
    save_path = rf'C:\Users\test\Downloads\1018\test\result'
    time_range = [2, 11]
    y, sr = utils.load_audio_waveform(filename, 'm4a')
    print(f'Sample rate: {y.frame_rate} Channel: {y.channels}')
    left, right = y.split_to_mono()
    left = np.float32(np.array(left.get_array_of_samples()))
    right = np.float32(np.array(right.get_array_of_samples()))

    # librosa.display.waveplot(left, sr, x_axis='s')
    # plt.show()
    fig, ax = plt.subplots(2,1, figsize=(16, 12))
    # ax[0].plot(left)
    ax[0].set_title('Left')
    ax[0].xaxis.grid()
    librosa.display.waveplot(left, sr, x_axis='s', ax=ax[0], color='b')
    # ax[1].plot(right, 'g')
    ax[1].set_title('Right')
    ax[1].xaxis.grid()
    librosa.display.waveplot(right, sr, x_axis='s', ax=ax[1], color='g')
    fig_name = os.path.basename(filename).split('.')[0] + f'_left_and_right_{time_range[0]}_{time_range[1]}' 
    plt.savefig(os.path.join(save_path, fig_name))
    # plt.show()


def show_frequency():
    filenames = [
                 rf'C:\Users\test\Downloads\1018\test\1620055140118_ASUS_I002D_5.m4a',
                 rf'C:\Users\test\Downloads\1018\test\1620144382079_ASUS_I002D_117.m4a',
                 rf'C:\Users\test\Downloads\1018\test\1620231545598_ASUS_I002D_30.m4a',
                 rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1620231545598_ASUS_I002D\1620231545598_ASUS_I002D_25.m4a',
                 rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1620231545598_ASUS_I002D\1620231545598_ASUS_I002D_25.m4a',
                 rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1620231545598_ASUS_I002D\1620231545598_ASUS_I002D_1.m4a',
                 rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw\1630600693454_Johnason\1630600693454_15.m4a',
                 rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw\1631456400568_AA2001038\1631456400568_154.m4a',
                 rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw\1631456400568_AA2001038\1631456400568_154.m4a',
                ]
    
    time_ranges = [
                   [82, 91],
                   [38, 47],
                   [115, 120],
                   [101, 110],
                   [14, 18],
                   [2, 6],
                   [81, 94],
                   [72, 100],
                   [1, 180],
                  ]

    time_ranges = [
                   [1, 180],
                   [1, 180],
                   [1, 180],
                   [1, 180],
                   [1, 180],
                   [1, 180],
                   [1, 180],
                   [1, 180],
                   [1, 180],
                  ]
    filenames = data_splitting.get_files(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw\1630345236867', 'm4a')
    filenames.extend(data_splitting.get_files(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw\1630866536302', 'm4a'))
    filenames.extend(data_splitting.get_files(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw\1630600693454', 'm4a'))
    time_ranges = [list([1, 180]) for _ in range(len(filenames))]
    # save_path = rf'C:\Users\test\Downloads\1018\test\result\meanspec'
    
    frame_size = 8
    sr = 16000
    channels = 1
    # amplitude_factor = 2
    # first_erosion = 17
    amplitude_factor_list = [2, 5, 8]
    first_erosion_list = [17, 21, 25]
    amplitude_factor_list = [4,6]
    first_erosion_list = [13,19,25]

    for amplitude_factor in amplitude_factor_list:
        for first_erosion in first_erosion_list:
            save_path = rf'C:\Users\test\Downloads\1018\test\result2\{amplitude_factor}_{first_erosion}_test'
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            print(save_path)
            for filename, time_range in zip(filenames, time_ranges):
                y = utils.load_audio_waveform(filename, 'm4a', sr, channels)
                sr = y.frame_rate
                channels = y.channels
                # y = utils.get_audio_clip(y, time_range, 1000)
                waveform = y.get_array_of_samples()
                waveform = np.array(waveform)
                # waveform = utils.get_audio_clip(waveform, time_range, sr)
                # waveform = _1d_median_filtering(waveform, frame_size)
                duration = time_range[1] - time_range[0]
                # get_audio_frequency(waveform, sr, duration, frame_size, filename, save_path, time_range)
                get_audio_frequency_thrshold(y, waveform, sr, duration, frame_size, filename, save_path, time_range, amplitude_factor, first_erosion)


# def get_audio_frequency(signal, sr, duration, frame_size, filename, save_path, time_range):
#     signal = np.float32(signal)
#     spec = np.abs(librosa.stft(signal))
#     mean_spec = np.mean(spec, axis=0)
#     melspec = librosa.feature.melspectrogram(signal, sr)
#     # melspec -= melspec.mean()
#     mean_melspec = np.mean(melspec, axis=0)

#     print(f'Shape of Spectrogram: {np.shape(spec)} Shape of Mel-Spectrogram: {np.shape(melspec)}', 
#           f'Duration: {time_range[1]-time_range[0]+1} sec',
#           f'Sample rate: {sr}')

    
#     hop_length = 512
#     n_fft = 2048
    
#     # D = np.abs(librosa.stft(signal, n_fft=n_fft,  hop_length=hop_length))
#     # librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
#     # plt.colorbar()

#     fig = plt.figure(figsize=(16, 12))
#     S = librosa.feature.melspectrogram(np.float32(signal), sr=sr, n_fft=n_fft, hop_length=hop_length)
#     S_DB = librosa.power_to_db(S, ref=np.max)
#     S_mean = S_DB - np.mean(S_DB, 0, keepdims=True)
#     librosa.display.specshow(S_mean, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
#     plt.title('Mel-Spectrogram')
#     plt.colorbar(format='%+2.0f dB')
#     fig_name = os.path.basename(filename).split('.')[0] + f'_melspec.png' 
#     if not os.path.isdir(save_path):
#         os.mkdir(save_path)
#     fig.savefig(os.path.join(save_path, fig_name))
#     plt.close(fig)
#     # plt.show()

#     # figure(figsize=(8, 6), dpi=80)
#     fig, ax = plt.subplots(3,1, figsize=(18, 16))
#     ax[0].plot(signal)
#     ax[0].set_title('Singal')
#     ax[0].set_xlabel('Time')

#     # ax[1].plot(mean_spec)
#     ax[1].plot(mean_melspec)
#     ax[1].hlines(np.mean(mean_melspec), 0, len(mean_spec), 'r')
#     ax[1].set_title('Average of Spectrogram')
#     ax[1].set_xlabel('Time')
#     ax[1].set_ylabel('Hz (linear)')
    

#     # mean_melspec = np.diff(mean_melspec, n=5)
#     # mean_melspec[] = 0
#     mean_melspec = np.where(np.abs(mean_melspec)<np.mean(mean_melspec), 0, 1)
#     mean_melspec = ndimage.morphology.binary_erosion(mean_melspec, structure=np.ones((9))).astype(mean_melspec.dtype)
#     mean_melspec = ndimage.morphology.binary_dilation(mean_melspec, structure=np.ones((1))).astype(mean_melspec.dtype)
#     ax[2].plot(mean_melspec)
#     ax[2].set_title('Average of Mel-Spectrogram')
#     ax[2].set_xlabel('Time')
#     ax[2].set_ylabel('Hz (Mel)')
#     # plt.show()
#     fig_name = os.path.basename(filename).split('.')[0] + f'_mean_of_spec_{time_range[0]}_{time_range[1]}' 
#     plt.savefig(os.path.join(save_path, fig_name))


def get_audio_frequency_thrshold(y, signal, sr, duration, frame_size, filename, save_path, time_range, amplitude_factor, first_erosion):
    hop_length = 512
    n_fft = 2048
    # amplitude_factor = 2
    # first_erosion = 17

    # fig = plt.figure(figsize=(16, 12))
    signal = np.float32(signal)
    # signal *= np.abs(signal)
    S = librosa.feature.melspectrogram(signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_mean = S_DB - np.mean(S_DB, 0, keepdims=True)

    mean_melspec = np.mean(S, axis=0)
    a = np.where(np.abs(mean_melspec)<np.mean(mean_melspec)*amplitude_factor, 0, 1)
    a = ndimage.morphology.binary_erosion(a, structure=np.ones((first_erosion))).astype(a.dtype)

    # best = (np.zeros_like(signal), a)
    # a = np.where(np.abs(mean_melspec)<0.25, 0, 1)
    # b = ndimage.morphology.binary_dilation(a, structure=np.ones((13))).astype(a.dtype)
    # c = ndimage.morphology.binary_erosion(b, structure=np.ones((13))).astype(b.dtype)
    # c = np.repeat(c, len(signal)//len(c)+1)
    # print(len(c), len(signal))
    # a = np.repeat(a, len(signal)//len(c))
    
    max_enegry = -5
    # np.repeat(c, len(signal)//len(c))
    for dilation_s in range(1, 51, 6):
        # for erosion_s in range(1, 31, 6):
        b = ndimage.morphology.binary_dilation(a, structure=np.ones((dilation_s))).astype(a.dtype)
        c = b
        # c = ndimage.morphology.binary_erosion(b, structure=np.ones((erosion_s))).astype(b.dtype)
        e = np.sum(mean_melspec*c)
        if e > max_enegry:
            max_enegry = e
            c = np.repeat(c, len(signal)//len(c)+1)[:len(signal)]
            best = (b, c)

    fig, ax = plt.subplots(3,2, figsize=(18, 16))
    # ax[0,0].plot(signal)
    librosa.display.waveplot(signal, sr=sr, x_axis='s', ax=ax[0,0])
    ax[0,0].set_title('Singal')
    ax[0,0].set_xlabel('Time')

    # ax[1].plot(mean_spec)
    ax[1,0].plot(mean_melspec, alpha=0.8)
    ax[1,0].hlines(np.mean(mean_melspec), 0, len(mean_melspec), 'r')
    ax[1,0].set_title('Step 1')
    ax[1,0].set_xlabel('Time')
    ax[1,0].set_ylabel('Hz (Mel)')
    
    ax[2,0].plot(a)
    ax[2,0].set_title('Step 2')
    ax[2,0].set_xlabel('Time')
    ax[2,0].set_ylabel('thrshold')

    ax[0,1].plot(best[0])
    ax[0,1].set_title('Step 3')
    ax[0,1].set_xlabel('Time')
    ax[0,1].set_ylabel('thrshold')

    librosa.display.waveplot(signal, sr=sr, x_axis='s', ax=ax[1,1])
    librosa.display.waveplot(np.max(signal)*best[1], sr=sr, x_axis='s', ax=ax[1,1])
    ax[1,1].set_title('Step 4')
    ax[1,1].set_xlabel('Time')
    ax[1,1].set_ylabel('thrshold')

    # ax[2,1].plot(signal)
    # ax[2,1].plot(np.max(signal)*best[1])
    out_signal = np.float32(signal*best[1])
    librosa.display.waveplot(out_signal, sr=sr, x_axis='s', ax=ax[2,1])
    ax[2,1].set_title('Step 5')
    ax[2,1].set_xlabel('Time')
    ax[2,1].set_ylabel('thrshold')

    name = os.path.basename(filename).split('.')[0] + f'_mean_of_spec_{time_range[0]}_{time_range[1]}' 
    plt.savefig(os.path.join(save_path, name))
    plt.close(fig)
    # plt.show()

    if amplitude_factor==2 and first_erosion==13:
        waveform = array.array(y.array_type, out_signal)
        new_sound = y._spawn(waveform)
        new_sound.export(os.path.join(save_path, name+'.wav'), 'wav')


def first_order_filter():
    # TODO: check https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    f = rf'C:\Users\test\Downloads\AA\1632074953419_NA\1632074953419_34.m4a'
    f = rf'C:\Users\test\Downloads\AA\1631294788806_12_138.70_139.70_018.wav'
    f = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\raw_mono_16k_h\1598482996718_NA\1\1598482996718_47_152.29_153.29_020.wav'
    save_path = rf'C:\Users\test\Downloads\AA'
    y = utils.load_audio_waveform(f, 'wav', channels=1)
    signal = np.float32(np.array(y.get_array_of_samples()))

    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    librosa.display.waveplot(signal)
    plt.show()

    waveform = array.array(y.array_type, emphasized_signal)
    new_sound = y._spawn(waveform)
    new_sound.export(os.path.join(save_path, 'test.wav'), 'wav')
    pass


def get_unconflicted_index():
    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\ASUS_h_train_ASUS_m_test2_2'
    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\ASUS_h_train_ASUS_m_test'
    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\ASUS_h_min_balance'
    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\ASUS_h_train_ASUS_m_test_2sec'
    train_idx = dataset_utils.load_content_from_txt(os.path.join(path, 'train.txt'))
    train_idx.sort()
    valid_idx = dataset_utils.load_content_from_txt(os.path.join(path, 'valid.txt'))
    valid_idx.sort()
    subject_list = []
    new_train_idx = []

    for f in valid_idx:
      subject = os.path.basename(f).split('_')[0]
      if subject not in subject_list:
        subject_list.append(subject)
      
    count = 0
    for f in train_idx:
      for s in subject_list:
        if s in f:
          new_train_idx.append(f)
          count += 1
          print(count, s, f)
          break
        
    new_train_idx = list(set(train_idx)-set(new_train_idx))
    with open(os.path.join(path, 'train_new.txt'), 'w+') as fw:
      for f in new_train_idx:
        fw.write(f)
        fw.write('\n')

    # pprint(new_train_idx)
    pprint(subject_list)


def show_dir_info():
    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\raw_mono_16k_h'
    # path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw'
    # dir_list = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    dir_list = utils.get_dir_list(path)
    acc, acc_p, acc_n, acc_balance = 0, 0, 0, 0
    total_p, total_n, total_balance = [], [], []
    for d in dir_list:
      if os.path.isdir(os.path.join(path, d, '1')):
        p = len(os.listdir(os.path.join(path, d, '1')))
      else:
        p = 0
      if os.path.isdir(os.path.join(path, d, '0')):
        n = len(os.listdir(os.path.join(path, d, '0')))
      else:
        n = 0
      
      acc = acc + p + n
      acc_p += p
      acc_n += n
      acc_balance += 2*min(p, n)
      if p+n == 0:
        balancing = 0
      else:
        balancing = p / (p+n) * 100
      print(f'{d:<30} p: {p:<10} n: {n:<10} p+n: {p+n:<10} balancing: {balancing:0.2f} %')
      total_p.append(p)
      total_n.append(n)
      total_balance.append(balancing)

    print(acc, acc_p, acc_n, acc_balance)

    fig, ax = plt.subplots()
    index = [str(i+1) for i in range(len(total_p))]
    ax.bar(index, total_p, width=0.35, label='Positive')
    ax.bar(index, total_n, width=0.35, bottom=total_p, label='Negative')

    # TODO: alignment problem, working in print function but not ax.text
    dir_list_with_num = [f'{i+1:<5} {f:<30} ({total_balance[i]:0.2f} %)' for i, f in enumerate(dir_list)]
    for d in dir_list_with_num:
        print(d)
    
    # Text box
    textstr = '\n'.join(dir_list_with_num[:len(dir_list_with_num)//2])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    ax.text(0.45, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
    textstr = '\n'.join(dir_list_with_num[len(dir_list_with_num)//2:])
    ax.text(0.67, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

    ax.set_xlabel('Subject')
    ax.set_ylabel('Sample number')
    ax.grid()
    ax.set_title('Data positive/negative')
    ax.legend()
    plt.show()


def stacked_bar_graph(data, data2=None, labels=None, length=None, width=None, x_label=None, y_label=None):
    fig, ax = plt.subplots()

    ax.bar(labels, data, width, label='Men')
    if data2:
        ax.bar(labels, data2, width, bottom=data, label='Women')

    textstr = '\n'.join((
        '1: 163095',
        '2: 163155',
        '3: 163284',
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax.set_ylabel(y_label)
    ax.set_title('Scores by group and gender')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # show_dir_info()
    get_unconflicted_index()
    # first_order_filter()
    # show_frequency()
    # stacked_bar_graph()
    pass