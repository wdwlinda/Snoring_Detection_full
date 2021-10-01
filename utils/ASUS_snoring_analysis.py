# Detect audio peaks with Librosa (https://librosa.github.io/librosa/)

# imports
from __future__ import print_function
import librosa
from librosa.core import audio
import librosa.display
import soundfile as sf
import numpy as np
import datetime
import time
import csv
from pydub import AudioSegment
import matplotlib.pyplot as plt
import os
from scipy import signal
import random



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
    waveform, sr = get_audio_waveform(filename)
    waveform = waveform.get_array_of_samples()
    waveform = np.float32(np.array(waveform))
    waveform = get_audio_clip(waveform, [162, 165], 44100)
    get_audio_volume(waveform, frame_size=512)


def get_audio_volume(signal, frame_size):
    volume = []
    for i in range(len(signal)):
        if i % 100:
            print(f'{i}/{len(signal)}')
        if i+frame_size < len(signal):
            clip = signal[i:i+frame_size]
        else:
            clip = signal[i:]
        volume.append(np.sum(np.abs(clip)))
    plt.plot(volume)
    plt.show()


def get_audio_clip(signal, time_interval, sample_rate):
    signal_interval = [int(time_interval[0]*sample_rate), int(time_interval[1]*sample_rate)]
    return signal[signal_interval[0]:signal_interval[1]]


def split_audio():
    save_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\infos\test_samples\1'
    filelist = [
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1620055140118_ASUS_I002D\1620055140118_ASUS_I002D_12.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1620055140118_ASUS_I002D\1620055140118_ASUS_I002D_13.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1620055140118_ASUS_I002D\1620055140118_ASUS_I002D_14.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1630345236867_AA0801160\1630345236867_33.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1630345236867_AA0801160\1630345236867_34.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1630345236867_AA0801160\1630345236867_35.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1630866536302_NA\1630866536302_12.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1630866536302_NA\1630866536302_13.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1630866536302_NA\1630866536302_14.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1630949188143_NA\1630949188143_73.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1630949188143_NA\1630949188143_74.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1630949188143_NA\1630949188143_75.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1631119510605_NA\1631119510605_32.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1631119510605_NA\1631119510605_33.m4a',
        rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring\1631119510605_NA\1631119510605_34.m4a',
    ]
    hop_length = 5120
    frame_size = 10240
    pre_max, post_max, pre_avg, post_avg, wait = 1e5, 1e5, 1e3, 1e3, 10
    clip_range_time = 0.5
    offset = 0.5
    time_unit = 1000

    # Initialize text file
    with open(os.path.join(save_path, 'valid.txt'), 'w+') as fw:
        fw.write('')

    idx = 0
    for f in filelist:
        # Get peaks
        y, sr = get_audio_waveform(f)
        waveform = y.get_array_of_samples()
        waveform = np.float32(np.array(waveform))
        ae = amplitude_envelope(waveform, frame_size, hop_length)
        f = os.path.basename(f).split('.')[0]
        peak_times = pick_peak(
            waveform, sr, f, hop_length, pre_max, post_max, pre_avg, post_avg, 
            delta=2*np.mean(ae), wait=wait, save_path=None)
        peak_times = np.unique(np.int32(peak_times))

        # Save audio clips
        for p in peak_times:
            # Handle boundary value
            if p > clip_range_time-offset and p < y.duration_seconds-offset+clip_range_time:
                singal_clip = get_audio_clip(y, [p+offset-clip_range_time, p+offset+clip_range_time], time_unit)
                
                # Save file name
                path = os.path.join(save_path, f'1_{idx}.wav')
                with open(os.path.join(save_path, 'valid.txt'), 'a') as fw:
                    fw.write(path)
                    fw.write('\n')

                # Save audio clip
                print(f'saving {path}')
                singal_clip.export(path, format='wav')
                idx += 1


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
    peaks_in_sec = peaks/sample_rate
    plt.figure(figsize=(14, 5))
    plt.vlines(peaks_in_sec, 0,
               signal.max()*1.5, color='r', alpha=0.6,
               label='Selected peaks')
    for sec in peaks_in_sec:
        plt.text(sec, signal.max()*1.5+1, f'{int(sec)}',fontsize=10)
    librosa.display.waveplot(signal, sample_rate, x_axis='s')
    # plt.show()

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def pick_peak(signal, sample_rate, filename, hop_length, pre_max, post_max, pre_avg, post_avg, delta, wait, save_path=None):
    peaks = librosa.util.peak_pick(signal, pre_max, post_max, pre_avg, post_avg, delta, wait)
    # TODO: Is this divide correct?
    peak_times = np.array(peaks) / sample_rate

    if save_path is not None:
        save_path = os.path.join(save_path, f'{filename}_peak.png')
    peak_plot(signal, peaks, sample_rate, save_path)
    return peak_times


def f_high(y,sr):
    b,a = signal.butter(10, 3000/(sr/2), btype='highpass')
    yf = signal.lfilter(b,a,y)
    return yf


def main():
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring'
    save_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\infos\peak5_4'
    hop_length = 5120
    frame_size = 10240
    file_number_threshold = 82
    csv_filename_programming = os.path.join(save_path, 'total_peak_programming.csv')
    csv_filename_reading = os.path.join(save_path, 'total_peak_reading.csv')
    peak_th1 = 20
    peak_th2 = 25
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
    # effective = effective[9:11] # TODO
    # ppp = [effective[0], effective[9], effective[12], effective[14], effective[20], effective[23], effective[24], effective[27]]
    # effective = list(set(effective)-set(ppp))
    # effective.sort(key=len)
    
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
            waveform, sr = get_audio_waveform(f)
            # TODO: high pass filtering
            # waveform = f_high(waveform, sr)
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


        # with open(os.path.join(save_path, 'total_peak.csv'), 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(['average', np.mean(np.array(peak_count))])

        # def save_peak_statistics_for_programming(save_path, mode):
        #     with open(csv_filename_programming, 'a', newline='') as csvfile:
        #         writer = csv.writer(csvfile)
        #         writer.writerow([])
        #         writer.writerow([_dir, f'[{idx+1}] / [{len(effective)}]'])
                
        # save_peak_statistics_for_reading()
        # save_peak_statistics_for_programming()
        
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
    pass