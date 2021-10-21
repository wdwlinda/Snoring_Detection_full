# Detect audio peaks with Librosa (https://librosa.github.io/librosa/)

# imports
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
import random
from scipy.io.wavfile import read
from analysis import data_splitting
import pandas as pd
import test


def get_audio_sample_with_hospital():
    audio_sample_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw2'
    annotation_path = rf'C:\Users\test\Downloads\annotations'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw2_hospital'

    # subject_list = [f.split('_')[0] for f in os.listdir(audio_sample_path)]
    subject_list = os.listdir(audio_sample_path)
    ori_annotation_file_list = os.listdir(annotation_path)
    annotation_file_list = [f.split('.')[0].split('_')[0] for f in ori_annotation_file_list]

    # TODO:
    subject_list = subject_list[:1]
    i = 0
    for subject in subject_list:
        if subject.split('_')[0] in annotation_file_list:
            i += 1
            # print(i, subject)
            samples = os.listdir(os.path.join(audio_sample_path, subject))
            in_range = []
            for j, f in enumerate(samples):
                subject = f.split('_')[0]
                code = f.split('_')[1]
                mapping_f = subject + '_' + code + '.m4a'
                annotation = ori_annotation_file_list[annotation_file_list.index(subject)]
                df = pd.read_csv(os.path.join(annotation_path, annotation))
                x = df.index[df['File'] == mapping_f].tolist()
                # print(f, x)
                # TODO: Not General
                peak_time = float(f.split('_')[2])
                for k in x:
                    start_time, end_time = df['Start time'][k], df['End time'][k]
                    print(f, k, start_time, end_time)
                    if peak_time >= start_time and peak_time <= end_time:
                        print('selected!')
                        in_range.append(1)
                    else:
                        print('xxxxxxxxxxxx')
                        in_range.append(0)
            # df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['#', 'audio clip', 'in range (hospital)'])
            # df2.to_csv(save_path)
    # Get an audio sample

    # Check is sample in time range
    # def check_sample_in_time_range(audio_peak_time, time_range):


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


def show_volume():
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1606921286802_sargo\1606921286802_sargo_15.m4a'
    filename = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1620055140118_ASUS_I002D\1620055140118_ASUS_I002D_5.m4a'
    waveform, sr = get_audio_waveform(filename)
    waveform = waveform.get_array_of_samples()
    waveform = np.array(waveform)
    waveform = get_audio_clip(waveform, [82, 85], sr)
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
    waveform, sr = get_audio_waveform(filename)
    waveform = waveform.get_array_of_samples()
    waveform = np.array(waveform)
    waveform = get_audio_clip(waveform, [82, 85], sr)
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
    y, sr = get_audio_waveform(filename)
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
                ]

    time_ranges = [
                   [82, 91],
                   [38, 47],
                   [115, 120],
                   [101, 110],
                   [14, 18],
                   [2, 6],
                  ]

    save_path = rf'C:\Users\test\Downloads\1018\test\result\mean_spec'
    frame_size = 8

    for filename, time_range in zip(filenames, time_ranges):
        y, sr = get_audio_waveform(filename)
        y = get_audio_clip(y, time_range, 1000)
        waveform = y.get_array_of_samples()
        waveform = np.array(waveform)
        # waveform = get_audio_clip(waveform, time_range, sr)
        # waveform = _1d_median_filtering(waveform, frame_size)
        duration = time_range[1] - time_range[0]
        get_audio_frequency(waveform, sr, duration, frame_size, filename, save_path, time_range)


def get_audio_frequency(signal, sr, duration, frame_size, filename, save_path, time_range):
    signal = np.float32(signal)
    spec = np.abs(librosa.stft(signal))
    mean_spec = np.mean(spec, axis=0)
    melspec = librosa.feature.melspectrogram(signal, sr)
    mean_melspec = np.mean(melspec, axis=0)

    print(f'Shape of Spectrogram: {np.shape(spec)} Shape of Mel-Spectrogram: {np.shape(melspec)}', 
          f'Duration: {time_range[1]-time_range[0]+1} sec',
          f'Sample rate: {sr}')

    
    hop_length = 512
    n_fft = 2048
    
    # D = np.abs(librosa.stft(signal, n_fft=n_fft,  hop_length=hop_length))
    # librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    # plt.colorbar()

    fig = plt.figure(figsize=(16, 12))
    S = librosa.feature.melspectrogram(np.float32(signal), sr=sr, n_fft=n_fft, hop_length=hop_length)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title('Mel-Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    fig_name = os.path.basename(filename).split('.')[0] + f'_melspec.png' 
    fig.savefig(os.path.join(save_path, fig_name))
    plt.close(fig)
    # plt.show()

    # figure(figsize=(8, 6), dpi=80)
    fig, ax = plt.subplots(3,1, figsize=(16, 12))
    ax[0].plot(signal)
    ax[0].set_title('Singal')
    # ax[1].specgram(spec)
    ax[1].set_title('Average of Spectrogram')
    ax[1].plot(mean_spec)
    # ax[1].set_title(f'Frequency')
    # spec_med_filter = _1d_median_filtering(spec, frame_size)
    # ax[2].plot(spec_med_filter)
    ax[2].plot(mean_melspec)
    ax[2].set_title('Average of Mel-Spectrogram')
    # plt.show()
    fig_name = os.path.basename(filename).split('.')[0] + f'_mean_of_spec_{time_range[0]}_{time_range[1]}' 
    plt.savefig(os.path.join(save_path, fig_name))



def get_audio_clip(signal, time_interval, sample_rate):
    signal_interval = [int(time_interval[0]*sample_rate), int(time_interval[1]*sample_rate)]
    return signal[signal_interval[0]:signal_interval[1]]


def save_audio_clips(filelist, save_path, frame_size, hop_length, pre_max, post_max, pre_avg, post_avg, 
                     wait, clip_range_time, offset, time_unit, output_type, use_hospital_condition):
    # Initialize text file
    # with open(os.path.join(save_path, 'valid.txt'), 'w+') as fw:
    #     fw.write('')

    if output_type == 'mono':
        mode = 1
    elif output_type == 'stereo':
        mode = 2
    else:
        mode = None
    
    annotation_path = rf'C:\Users\test\Downloads\annotations'
    ori_annotation_file_list = os.listdir(annotation_path)
    annotation_file_list = [f.split('.')[0].split('_')[0] for f in ori_annotation_file_list]
    sub_dir = ''

    df_dict = {}
    if use_hospital_condition:
        for f in filelist:
            subject = os.path.basename(f).split('_')[0]
            if subject in annotation_file_list:
                if  subject in df_dict:
                    continue
                else:                
                    annotation = ori_annotation_file_list[annotation_file_list.index(subject)]
                    df = pd.read_csv(os.path.join(annotation_path, annotation))
                    df_dict[subject] = df

    for f in filelist:
        idx = 0
        subject = os.path.basename(f).split('_')[0]
        # Get peaks
        print(f)
        y, peak_times = get_peaks(f, frame_size, hop_length, pre_max, post_max, pre_avg, post_avg, wait, None)

        # y, sr = get_audio_waveform(f)
        # waveform = y.get_array_of_samples()
        # waveform = np.array(waveform)
        # # waveform = waveform * np.abs(waveform) * np.abs(waveform)

        # ae = amplitude_envelope(waveform, frame_size, hop_length)
        # # f = os.path.basename(f).split('.')[0]
        # peak_times = pick_peak(
        #     waveform, sr, os.path.basename(f).split('.')[0], hop_length, pre_max, post_max, pre_avg, post_avg, 
        #     delta=2*np.mean(ae), wait=wait, save_path=None)
        # peak_times = np.unique(np.int32(peak_times))
        # # print(len(peak_times))

        # Save audio clips
        # save_dir = f.split('_')[0]
        save_dir = os.path.split(os.path.split(f)[0])[1]
        name = os.path.basename(f).split('.')[0]
        for p in peak_times:
            # Handle boundary value
            # print(p, y.duration_seconds, offset, clip_range_time)
            peak_condition = (p > clip_range_time-offset and p < y.duration_seconds-offset-clip_range_time)
            if use_hospital_condition:
                hospital_condition = False
                if subject in df_dict:
                    df = df_dict[subject]
                    x = df.index[df['File'] == os.path.basename(f)].tolist()
                    if x:
                        for k in x:
                            start_time, end_time, snoring_label = df['Start time'][k], df['End time'][k], df['Label'][k]
                            if p >= start_time and p <= end_time:
                                hospital_condition = True
                                if snoring_label == 'snoring':
                                    sub_dir = '1'
                                elif snoring_label == 'non-snoring':
                                    sub_dir = '0'
                                print(snoring_label, save_dir)
                                break
                peak_condition = (peak_condition and hospital_condition)

                    
            # print(60*'+', peak_condition, hospital_condition, subject)
            if peak_condition:
                signal_clip = get_audio_clip(y, [p+offset-clip_range_time, p+offset+clip_range_time], time_unit)
                
                # Save file name
                if not os.path.isdir(os.path.join(save_path, save_dir, sub_dir)):
                    os.makedirs(os.path.join(save_path, save_dir, sub_dir))
                path = os.path.join(save_path, save_dir, sub_dir, f'{name}_{p+offset-clip_range_time}_{p+offset+clip_range_time}_{idx+1:03d}.wav')
                # with open(os.path.join(save_path, 'valid.txt'), 'a') as fw:
                #     fw.write(path)
                #     fw.write('\n')

                # Save audio clip
                print(f'saving {path}')
                if mode: signal_clip = signal_clip.set_channels(mode)
                signal_clip.export(path, format='wav')
                idx += 1


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
    y3 = 'a'


def split_audio():
    # TODO: record picking parameters in csv 
    # (reason: won't forget every splitting. Can split again very fast in any time any where)
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw'
    data_path = rf'C:\Users\test\Downloads\AA'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw2_mono'
    hop_length = 5120
    frame_size = 10240
    pre_max, post_max, pre_avg, post_avg, wait = 1e5, 1e5, 1e3, 1e3, 10
    clip_range_time = 0.5
    offset = 0.5
    time_unit = 1000
    format = 'm4a'
    mono_or_stereo = 'mono'
    use_hospital_condition = False

    filelist = data_splitting.get_files(data_path, format)
    save_audio_clips(
        filelist, save_path, frame_size, hop_length, pre_max, post_max, pre_avg, post_avg, wait, 
        clip_range_time, offset, time_unit, mono_or_stereo, use_hospital_condition)
    test.save_aLL_files_name(save_path, keyword='wav', name='file_name', shuffle=False)


def amplitude_envelope(signal, frame_size, hop_length):
    amplitude_envelope = np.array([])
    signal = np.array(signal)
    for i in range(0, len(signal), hop_length):
        current_frame_amplitude_envelope = np.max(signal[i:i+frame_size])
        amplitude_envelope = np.append(amplitude_envelope, current_frame_amplitude_envelope)
    return amplitude_envelope


def get_audio_waveform(filename):
    if filename.endswith('wav'):
        y, sr = librosa.load(filename)
    elif filename.endswith('m4a'):
        y = AudioSegment.from_file(filename, format='m4a')
        sr = y.frame_rate
    return y, sr

    
def envelope_plot(signal, frame_size, hop_length):
    # frame_size=10240
    # hop_length=5120
    ae = amplitude_envelope(signal, frame_size, hop_length)
    # frames = range(0, ae.size)
    # t = librosa.frames_to_time(frames, hop_length)
    # print(t)
    # print(ae)
    # print(np.mean(ae))
    # print(len(ae))
    # plt.plot(t, ae, label='envelope', color='g')
    return ae


def save_single_audio_info(filename, data, save_path):
    # TODO: frequency in short time
    with open(os.path.join(save_path, f'{filename}_peak.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['Number', 'Time', 'Time (second)'])
        for idx, s in enumerate(data):
            writer.writerow([idx+1, f'{s//60:02d}:{s%60:02d}', s])
        
        if len(data) > 0:
            start_time = f'start time: {data[0]//60:02d}:{data[0]%60:02d}'
            end_time = f'end time: {data[-1]//60:02d}:{data[-1]%60:02d}'
            duration = data[-1] - data[0]
        else:
            start_time = 'start time: 00:00'
            end_time = 'end time: 00:00'
            duration = 0
        if duration > 0:
            frquency = len(data) / duration
        else:
            frquency = float('nan')

        writer.writerow([])
        writer.writerow(['Attribute'])
        writer.writerow(['start time', start_time])
        writer.writerow(['end time', end_time])
        writer.writerow(['duration', duration])
        writer.writerow(['peak count', len(data)])
        writer.writerow(['frquency', f'{frquency:.02f}'])


def save_audio_info(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def peak_plot(signal, peaks, sample_rate, save_path=None):
    # TODO: plot method for better visualization
    # duration = librosa.get_duration(y)
    signal = np.float32(np.array(signal))
    peaks_in_sec = peaks / sample_rate
    # plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.vlines(peaks_in_sec, 0,
               signal.max()*1.5, color='r', alpha=0.6,
               label='Selected peaks')
    for sec in peaks_in_sec:
        ax.text(sec, signal.max()*1.5+1, f'{int(sec)}',fontsize=10)
    if save_path:
        filename = os.path.basename(save_path).split('.')[0]
    else:
        filename = 'peak'
    ax.set_title(filename)
    ax.xaxis.grid()
    librosa.display.waveplot(signal, sample_rate, x_axis='s', ax=ax)
    # plt.show()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def pick_peak(signal, sample_rate, filename, hop_length, pre_max, post_max, pre_avg, post_avg, delta, wait, save_path=None):
    peaks = librosa.util.peak_pick(signal, pre_max, post_max, pre_avg, post_avg, delta, wait)
    peak_times = np.array(peaks) / sample_rate

    if save_path:
        save_path = os.path.join(save_path, f'{filename}_peak.png')
        peak_plot(signal, peaks, sample_rate, save_path)
    return peak_times


def f_high(y,sr):
    b,a = signal.butter(10, 3000/(sr/2), btype='highpass')
    yf = signal.lfilter(b,a,y)
    return yf


def get_peaks(f, frame_size, hop_length, pre_max, post_max, pre_avg, post_avg, wait, abs_save_path):
    y, sr = get_audio_waveform(f)
    if y.channels == 1:
        waveform = np.float32(np.array(y.get_array_of_samples()))
    elif y.channels == 2:
        left, right = y.split_to_mono()
        waveform = np.array(left.get_array_of_samples()) + np.array(right.get_array_of_samples())
        waveform = np.float32(waveform//2)
    else:
        raise ValueError(f'Unknown Audio channel: {y.channels}')
    
    waveform = waveform * np.abs(waveform) * np.abs(waveform)
    ae = amplitude_envelope(waveform, frame_size, hop_length)
    f = os.path.basename(f).split('.')[0]
    peak_times = pick_peak(
        waveform, sr, f, hop_length, pre_max, post_max, pre_avg, post_avg, 
        delta=2*np.mean(ae), wait=wait, save_path=abs_save_path)
    peak_times = np.unique(np.int32(peak_times))
    return y, peak_times

def main():
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw'
    save_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\infos\peak7'
    hop_length = 5120
    frame_size = 10240
    file_number_threshold = 82
    csv_filename_programming = os.path.join(save_path, 'total_peak_programming.csv')
    csv_filename_reading = os.path.join(save_path, 'total_peak_reading.csv')
    peak_th1 = 10
    peak_th2 = 15
    effective_audio_th = 25
    pre_max, post_max, pre_avg, post_avg, wait = 1e5, 1e5, 1e3, 1e3, 10

    # Select folder with enough number of files
    effective = []
    acc_time = 0
    total_file_count, total_sleeping_time, total_peak_count = 0, 0, 0
    total_count_th1, total_count_th2 = 0, 0
    total_effective_sample1, total_effective_sample2 = 0, 0
    
    
    for f in os.listdir(data_path):
        folder_path = os.path.join(data_path, f)
        if os.path.isdir(folder_path):
            if len(os.listdir(folder_path)) > file_number_threshold:
                effective.append(f)
    
    with open(csv_filename_programming, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Number', 'Name', 'Peak count', f'Thresholding count (>{peak_th1})', f'Thresholding count (>{peak_th2})'])

    with open(csv_filename_reading, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ['Number', 'Name', 'File count', 'Sleeping time (hours)', 'Total peak count',
                f'Effective audio 1 (>{peak_th1} peak)', f'Effective audio 1 (>{peak_th1} peak) (%)', 
                f'Effective sample 1 (>{effective_audio_th} % effective audio)',
                f'Effective audio 2 (>{peak_th2} peak)', f'Effective audio 2 (>{peak_th2} peak) (%)',
                f'Effective sample 2 (>{effective_audio_th} % effective audio)'])

    # TODO:
    # effective = effective[:3]
    num_sample = len(effective)
    for dir_idx, _dir in enumerate(effective):
        start_ = time.time()
        abs_data_path = os.path.join(data_path, _dir)
        abs_save_path = os.path.join(save_path, _dir)
        if not os.path.exists(abs_save_path):
            os.mkdir(abs_save_path)
        file_list = [f for f in os.listdir(abs_data_path) if f.endswith('m4a')]
        file_list.sort(key=len)

        peak_counts, content = [], []
        one_sample_count_th1, one_sample_count_th2 = 0, 0
        effective_sample1, effective_sample2 = 0, 0
        os.chdir(abs_data_path)
        # file_list = file_list[:10] # TODO
        for file_idx, f in enumerate(file_list):
            _, peak_times = get_peaks(f, frame_size, hop_length, pre_max, post_max, pre_avg, post_avg, wait, abs_save_path)
            # y, sr = get_audio_waveform(f)
            # if y.channels == 1:
            #     waveform = np.float32(np.array(y.get_array_of_samples()))
            # elif y.channels == 2:
            #     left, right = y.split_to_mono()
            #     waveform = np.array(left.get_array_of_samples()) + np.array(right.get_array_of_samples())
            #     waveform = np.float32(waveform//2)
            # else:
            #     raise ValueError(f'Unknown Audio channel: {y.channels}')
            
            # waveform = waveform * np.abs(waveform) * np.abs(waveform)
            # ae = amplitude_envelope(waveform, frame_size, hop_length)
            # f = os.path.basename(f).split('.')[0]
            # peak_times = pick_peak(
            #     waveform, sr, f, hop_length, pre_max, post_max, pre_avg, post_avg, 
            #     delta=2*np.mean(ae), wait=wait, save_path=abs_save_path)
            # peak_times = np.unique(np.int32(peak_times))

            print(f'{dir_idx+1}/{len(effective)}', _dir, '||', f'{file_idx+1}/{len(file_list)}', f)
            print(4*'-', peak_times)

            # Save peak information of single audio file
            save_single_audio_info(f, peak_times, save_path=abs_save_path)
            

            # TODO: simplify 
            # +++
            audio_peak_count = len(peak_times)
            count_th1 = 1 if audio_peak_count > peak_th1 else 0
            count_th2 = 1 if audio_peak_count > peak_th2 else 0
            one_sample_count_th1 += count_th1
            one_sample_count_th2 += count_th2
            peak_counts.append(audio_peak_count)
            content.append([file_idx+1, f, audio_peak_count, count_th1, count_th2])

        with open(csv_filename_programming, 'a', newline='') as csvfile:
            effective_audio_percentage1 = one_sample_count_th1/len(file_list)*100
            effective_audio_percentage2 = one_sample_count_th2/len(file_list)*100
            writer = csv.writer(csvfile)
            writer.writerows(content)
            writer.writerow(['', 'mean/sum/sum', np.mean(peak_counts), one_sample_count_th1, one_sample_count_th2])
            writer.writerow(['', 'std/percentage/percentage', np.std(peak_counts), 
                             effective_audio_percentage1, effective_audio_percentage2])
            writer.writerow([])
            # ---

        with open(csv_filename_reading, 'a', newline='') as csvfile:
            if effective_audio_percentage1 > effective_audio_th: effective_sample1 = 1
            if effective_audio_percentage2 > effective_audio_th: effective_sample2 = 1
            total_effective_sample1 += effective_sample1
            total_effective_sample2 += effective_sample2
            file_count = len(file_list)
            one_peak_count = np.sum(peak_counts)
            sleeping_time = len(file_list)/20
            writer = csv.writer(csvfile)
            writer.writerow(
                [dir_idx+1, _dir, file_count, sleeping_time, one_peak_count, 
                one_sample_count_th1, effective_audio_percentage1, effective_sample1, 
                one_sample_count_th2, effective_audio_percentage2, effective_sample2])
            total_file_count += file_count
            total_sleeping_time += sleeping_time
            total_peak_count += one_peak_count
            total_count_th1 += one_sample_count_th1
            total_count_th2 += one_sample_count_th2
        
        end_ = time.time()
        build_time = end_ - start_
        acc_time += build_time
        print(f'  (Spending time)  This round: {build_time} sec || Total: {acc_time} sec' )
    
    with open(csv_filename_reading, 'a', newline='') as csvfile:
        total_file_using_th1 = total_count_th1/total_file_count*100
        total_file_using_th2 = total_count_th2/total_file_count*100
        # if total_file_using_th1 > effective_audio_th: effective_audio_count1 += 1
        # if total_file_using_th2 > effective_audio_th: effective_audio_count2 += 1
        writer = csv.writer(csvfile)
        writer.writerow(
            ['', '', total_file_count, total_sleeping_time/num_sample, total_peak_count,
            f'{total_file_using_th1} %', '', total_effective_sample1, 
            f'{total_file_using_th2} %', '', total_effective_sample2])



    def main2():
        data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw'
        save_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\infos\peak6'
        hop_length = 5120
        frame_size = 10240
        file_number_threshold = 82
        csv_filename_programming = os.path.join(save_path, 'total_peak_programming.csv')
        csv_filename_reading = os.path.join(save_path, 'total_peak_reading.csv')
        peak_th1 = 10
        peak_th2 = 15
        peak_th = [20, 25]
        effective_audio_th = 25
        pre_max, post_max, pre_avg, post_avg, wait = 1e5, 1e5, 1e3, 1e3, 10

        # Select folder with enough number of files
        effective = []
        acc_time = 0
        total_file_count, total_sleeping_time, total_peak_count = 0, 0, 0
        total_count_th1, total_count_th2 = 0, 0
        total_effective_sample1, total_effective_sample2 = 0, 0
        
        programming_properties = ['Number', 'Name', 'Peak count']
        for th in peak_th:
            programming_properties.append(f'Thresholding count (>{th})')
        reading_properties = ['Number', 'Name', 'File count', 'Sleeping time (hours)', 'Total peak count']
        for idx, th in enumerate(peak_th):
            reading_properties.extend([
                f'Effective audio {idx+1} (>{peak_th1} peak)', 
                f'Effective audio {idx+1} (>{peak_th1} peak) (%)', 
                f'Effective sample {idx+1} (>{effective_audio_th} % effective audio)'])

        for f in os.listdir(data_path):
            folder_path = os.path.join(data_path, f)
            if os.path.isdir(folder_path):
                if len(os.listdir(folder_path)) > file_number_threshold:
                    effective.append(f)
                    get_effective_cases()
                else:
                    pass
        
        def get_effective_cases():
            num_sample = len(effective)
            for dir_idx, _dir in enumerate(effective):
                start_ = time.time()
                abs_data_path = os.path.join(data_path, _dir)
                abs_save_path = os.path.join(save_path, _dir)
                if not os.path.exists(abs_save_path):
                    os.mkdir(abs_save_path)
                file_list = [f for f in os.listdir(abs_data_path) if f.endswith('m4a')]
                file_list.sort(key=len)

                peak_counts, content = [], []
                one_sample_count_th1, one_sample_count_th2 = 0, 0
                effective_sample1, effective_sample2 = 0, 0
                os.chdir(abs_data_path)
                # file_list = file_list[:10] # TODO
                for file_idx, f in enumerate(file_list):
                    y, sr = get_audio_waveform(f)
                    # TODO: high pass filtering
                    # waveform = f_high(waveform, sr)
                    waveform = np.float32(np.array(y.get_array_of_samples()))
                    waveform = waveform * np.abs(waveform) * np.abs(waveform)

                    ae = amplitude_envelope(waveform, frame_size, hop_length)
                    f = os.path.basename(f).split('.')[0]
                    peak_times = pick_peak(
                        waveform, sr, f, hop_length, pre_max, post_max, pre_avg, post_avg, 
                        delta=2*np.mean(ae), wait=wait, save_path=abs_save_path)
                    peak_times = np.unique(np.int32(peak_times))

                    print(f'{dir_idx+1}/{len(effective)}', _dir, '||', f'{file_idx+1}/{len(file_list)}', f)
                    print(4*'-', peak_times)

                    # Save peak information of single audio file
                    save_single_audio_info(f, peak_times, save_path=abs_save_path)
                    

                    # TODO: simplify 
                    # +++
                    audio_peak_count = len(peak_times)
                    count_th1 = 1 if audio_peak_count > peak_th1 else 0
                    count_th2 = 1 if audio_peak_count > peak_th2 else 0
                    one_sample_count_th1 += count_th1
                    one_sample_count_th2 += count_th2
                    peak_counts.append(audio_peak_count)
                    content.append([file_idx+1, f, audio_peak_count, count_th1, count_th2])

                with open(csv_filename_programming, 'a', newline='') as csvfile:
                    effective_audio_percentage1 = one_sample_count_th1/len(file_list)*100
                    effective_audio_percentage2 = one_sample_count_th2/len(file_list)*100
                    writer = csv.writer(csvfile)
                    writer.writerows(content)
                    writer.writerow(['', 'mean/sum/sum', np.mean(peak_counts), one_sample_count_th1, one_sample_count_th2])
                    writer.writerow(['', 'std/percentage/percentage', np.std(peak_counts), 
                                    effective_audio_percentage1, effective_audio_percentage2])
                    writer.writerow([])
                    # ---

                with open(csv_filename_reading, 'a', newline='') as csvfile:
                    if effective_audio_percentage1 > effective_audio_th: effective_sample1 = 1
                    if effective_audio_percentage2 > effective_audio_th: effective_sample2 = 1
                    total_effective_sample1 += effective_sample1
                    total_effective_sample2 += effective_sample2
                    file_count = len(file_list)
                    one_peak_count = np.sum(peak_counts)
                    sleeping_time = len(file_list)/20
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [dir_idx+1, _dir, file_count, sleeping_time, one_peak_count, 
                        one_sample_count_th1, effective_audio_percentage1, effective_sample1, 
                        one_sample_count_th2, effective_audio_percentage2, effective_sample2])
                    total_file_count += file_count
                    total_sleeping_time += sleeping_time
                    total_peak_count += one_peak_count
                    total_count_th1 += one_sample_count_th1
                    total_count_th2 += one_sample_count_th2
                
                end_ = time.time()
                build_time = end_ - start_
                acc_time += build_time
                print(f'  (Spending time)  This round: {build_time} sec || Total: {acc_time} sec' )
            
            with open(csv_filename_reading, 'a', newline='') as csvfile:
                total_file_using_th1 = total_count_th1/total_file_count*100
                total_file_using_th2 = total_count_th2/total_file_count*100
                # if total_file_using_th1 > effective_audio_th: effective_audio_count1 += 1
                # if total_file_using_th2 > effective_audio_th: effective_audio_count2 += 1
                writer = csv.writer(csvfile)
                writer.writerow(
                    ['', '', total_file_count, total_sleeping_time/num_sample, total_peak_count,
                    f'{total_file_using_th1} %', '', total_effective_sample1, 
                    f'{total_file_using_th2} %', '', total_effective_sample2])
if __name__ == '__main__':
    # main()
    split_audio()
    # show_volume()
    # show_median_filter()
    # show_frequency()
    # check_two_channels()
    # audio_loading_exp()
    # check_audio_sample_rate()
    # get_audio_sample_with_hospital()
    pass