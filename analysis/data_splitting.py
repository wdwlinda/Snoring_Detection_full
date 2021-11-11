import os
import csv
import numpy as np
import logging

from torch.utils.data import dataset
from dataset import dataset_utils
# from test import save_aLL_files_name



def single_sample_recording(path):
    with open(path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        # timestamps for single sample
        timestamps = [[r[0], r[2]] for r in rows if len(r) > 0 and r[0].isdigit()]
        # TODO: file name
        # a = list(iter(rows))
        # timestamps.insert(0, list(iter(rows))[0])
    return timestamps


# TODO: offset, duration
def generate_recording_table(peak_timestamps_path, peak_infos, save_path):
    tags = ['#', 'name', '#', 'peak timestamp', 'offset', 'duration', 'start', 'end', 'label']
    content = [tags]
    with open(peak_infos, 'r', newline='') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        offset_and_duration = next(rows)

    # TODO: create csv if not exist
    # TODO: if already exist? (how to judge the file should be updated?)
    # if not os.path.isfile('record_tabel.csv'):
    #     with open('record_tabel.csv', 'w', newline='') as csvfile:
    #         csv.write('')

    file_list = dataset_utils.get_files(peak_timestamps_path, 'csv')
    count = 0
    for i, f in enumerate(file_list):
        print(f'({i+1}/{len(file_list)}) Processing {os.path.basename(f)}')
        timestamps = single_sample_recording(f)
        # timestamps.insert(0, i+1)
        for j, t in enumerate(timestamps):
            count += 1
            t.insert(0, os.path.basename(f).replace('.csv', ''))
            t.insert(0, count)
            t.extend(offset_and_duration)
            start = float(t[3])+float(t[4])
            end = start + float(t[5])
            t.extend([start, end])
        content.extend(timestamps)
    
    # Write recording table
    with open('record_tabel.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(content)


def continuous_split(path, clip_time, hop_time, sr, channels):
    assert clip_time > 0 and hop_time > 0
    file_name = os.path.basename(path)
    file_format = file_name.split('.')[1]
    y = dataset_utils.load_audio_waveform(path, file_format, sr, channels)
    save_path = os.path.join(os.path.split(path)[0], f'clips_{clip_time}_{hop_time}')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if y.duration_seconds >= clip_time:
        for idx, t in enumerate(range(0, int(y.duration_seconds)-clip_time+1, hop_time), 1):
            start_t, end_t = t, t+clip_time
            # print(idx, start_t, end_t)
            clip = y[1000*start_t:1000*end_t]
            clip.export(os.path.join(save_path, file_name.replace(f'.{file_format}', f'_{idx:03d}.{file_format}')), file_format)
    return save_path


def main():
    # TODO: get_files add recursive option
    files = dataset_utils.get_files(rf'C:\Users\test\Downloads\1112\KC_testing', 'wav')
    for f in files:
        save_path = continuous_split(f, clip_time=2, hop_time=2, sr=16000, channels=1)
    dataset_utils.save_aLL_files_name(save_path, keyword='wav')


if __name__ == '__main__':
    # path = rf'C:\Users\test\Downloads\snoring_test\timestamps\1630345236867_6_peak.csv'
    # path = rf'C:\Users\test\Downloads\1007\env_sounds\x'
    # peak_infos = rf'C:\Users\test\Downloads\peak_infos.csv'
    # generate_recording_table(path, peak_infos, '')

    main()
    
    # path = rf'C:\Users\test\Downloads\1007\env_sounds\x'
    # files = dataset_utils.get_files(path, 'pms')
    # for f in files:
    #     print(f)
    # x = list(set(dataset_utils.get_files(path))-set(dataset_utils.get_files(path, ['m4a', 'm4a'])))
    # print(60*"=")
    # for f in x:
    #     print(f)


        
    pass