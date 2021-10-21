# Detect audio peaks with Librosa (https://librosa.github.io/librosa/)

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
from analysis import utils


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
                signal_clip = utils.get_audio_clip(y, [p+offset-clip_range_time, p+offset+clip_range_time], time_unit)
                
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


def split_audio():
    # TODO: record picking parameters in csv 
    # (reason: won't forget every splitting. Can split again very fast in any time any where)
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw'
    # data_path = rf'C:\Users\test\Downloads\AA'
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


def get_peaks(filename, audio_format, frame_size, hop_length, pre_max, post_max, pre_avg, post_avg, wait, save_path=None):
    """Get peaks from audio waveform"""
    # load audio
    y = AudioSegment.from_file(filename, format=audio_format)
    sample_rate = y.frame_rate
    if y.channels == 1:
        waveform = np.float32(np.array(y.get_array_of_samples()))
    elif y.channels == 2:
        left, right = y.split_to_mono()
        waveform = np.array(left.get_array_of_samples()) + np.array(right.get_array_of_samples())
        waveform = np.float32(waveform//2)
    else:
        raise ValueError(f'Unknown Audio channel: {y.channels}')
    
    # peak picking
    waveform_square = waveform * np.abs(waveform) * np.abs(waveform)
    ae = amplitude_envelope(waveform_square, frame_size, hop_length)
    peaks = librosa.util.peak_pick(waveform_square, pre_max, post_max, pre_avg, post_avg, delta=2*np.mean(ae), wait=wait)
    peak_times = np.array(peaks) / sample_rate
    peak_times = np.unique(np.int32(peak_times))

    # Save peak plot
    if save_path:
        filename = os.path.basename(filename).split('.')[0]
        save_path = os.path.join(save_path, f'{filename}_peak.png')
        peak_plot(waveform, peak_times, sample_rate, save_path)
    return y, peak_times


def ASUS_snoring_audio_preprocessing():
    """
    Preprocess the ASUS snoring data, including spitting to clips, plot peak with waveform, ....
    """
    # peak_judge_func = get_peak_judge_functiion()

    # peaks = single_audio_peak_pick(waveform, peak_judge_func)


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
    effective = []
    acc_time = 0
    total_file_count, total_sleeping_time, total_peak_count = 0, 0, 0
    total_count_th1, total_count_th2 = 0, 0
    total_effective_sample1, total_effective_sample2 = 0, 0
    audio_format = 'm4a'
    
    # Select subject which have enough file number (file number > file_number_threshold)
    for f in os.listdir(data_path):
        folder_path = os.path.join(data_path, f)
        if os.path.isdir(folder_path):
            if len(os.listdir(folder_path)) > file_number_threshold:
                effective.append(f)
    
    # Write peak information header
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
        file_list = [f for f in os.listdir(abs_data_path) if f.endswith(audio_format)]
        file_list.sort(key=len)

        peak_counts, content = [], []
        one_sample_count_th1, one_sample_count_th2 = 0, 0
        effective_sample1, effective_sample2 = 0, 0
        os.chdir(abs_data_path)
        # file_list = file_list[:10] # TODO
        for file_idx, f in enumerate(file_list):
            _, peak_times = get_peaks(f, frame_size, hop_length, pre_max, post_max, pre_avg, post_avg, wait, abs_save_path)

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
    save_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\infos\peak7'
    hop_length = 5120
    frame_size = 10240
    file_number_threshold = 82
    csv_filename_programming = os.path.join(save_path, 'total_peak_programming.csv')
    csv_filename_reading = os.path.join(save_path, 'total_peak_reading.csv')
    peak_thresholds = [10, 15]
    peak_th1 = 10
    peak_th2 = 15
    effective_audio_th = 25
    pre_max, post_max, pre_avg, post_avg, wait = 1e5, 1e5, 1e3, 1e3, 10
    effective = []
    acc_time = 0
    total_file_count, total_sleeping_time, total_peak_count = 0, 0, 0
    total_count_th1, total_count_th2 = 0, 0
    total_effective_sample1, total_effective_sample2 = 0, 0
    audio_format = 'm4a'
    
    # Select subject which have enough file number (file number > file_number_threshold)
    for f in os.listdir(data_path):
        folder_path = os.path.join(data_path, f)
        if os.path.isdir(folder_path):
            if len(os.listdir(folder_path)) > file_number_threshold:
                effective.append(f)
    
    # Write peak information header
    with open(csv_filename_programming, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Number', 'Name', 'Peak count']
        header = [header.append(f'Thresholding count (>{peak_th})') for peak_th in peak_thresholds]
        writer.writerow(header)

    with open(csv_filename_reading, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['Number', 'Name', 'File count', 'Sleeping time (hours)', 'Total peak count']
        for idx, peak_th in enumerate(peak_thresholds):
            header.append(f'Effective audio {idx+1} (>{peak_th} peak)', f'Effective audio {idx+1} (>{peak_th} peak) (%)')
            header.append(f'Effective sample {idx+1} (>{effective_audio_th} % effective audio)')
        writer.writerow(header)


    # TODO:
    # effective = effective[:3]
    num_sample = len(effective)
    for dir_idx, _dir in enumerate(effective):
        start_time = time.time()
        abs_data_path = os.path.join(data_path, _dir)
        abs_save_path = os.path.join(save_path, _dir)
        if not os.path.exists(abs_save_path):
            os.mkdir(abs_save_path)
        file_list = [f for f in os.listdir(abs_data_path) if f.endswith(audio_format)]
        file_list.sort(key=len)

        peak_counts, content = [], []
        one_sample_count_th1, one_sample_count_th2 = 0, 0
        one_sample_count_th = 0
        effective_sample1, effective_sample2 = 0, 0
        os.chdir(abs_data_path)
        # file_list = file_list[:10] # TODO
        for file_idx, f in enumerate(file_list):
            _, peak_times = get_peaks(f, frame_size, hop_length, pre_max, post_max, pre_avg, post_avg, wait, abs_save_path)

            print(f'{dir_idx+1}/{len(effective)}', _dir, '||', f'{file_idx+1}/{len(file_list)}', f)
            print(4*'-', peak_times)

            # Save peak information of single audio file
            save_single_audio_info(f, peak_times, save_path=abs_save_path)
            

            # TODO: simplify 

            # +++
            count_th_list, one_sample_count_th_list = [], []
            audio_peak_count = len(peak_times)
            for peak_th in peak_thresholds:
                count_th = 1 if audio_peak_count > peak_th else 0
                one_sample_count_th += count_th
                count_th_list.append(count_th)
                one_sample_count_th_list.append(one_sample_count_th)
            content.append([file_idx+1, f, audio_peak_count, *count_th_list])
            # +++

            # ---
            audio_peak_count = len(peak_times)
            count_th1 = 1 if audio_peak_count > peak_th1 else 0
            count_th2 = 1 if audio_peak_count > peak_th2 else 0
            one_sample_count_th1 += count_th1
            one_sample_count_th2 += count_th2
            peak_counts.append(audio_peak_count)
            content.append([file_idx+1, f, audio_peak_count, count_th1, count_th2])
            # ---

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
        
        end_time = time.time()
        build_time = end_time - start_time
        acc_time += build_time
        print(f'  (Spending time)  This round: {build_time} sec || Total: {acc_time} sec' )
    
    with open(csv_filename_reading, 'a', newline='') as csvfile:
        total_file_using_th1 = total_count_th1/total_file_count*100
        total_file_using_th2 = total_count_th2/total_file_count*100
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