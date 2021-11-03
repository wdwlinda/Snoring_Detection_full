
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
from dataset.dataset_utils import get_files, load_content_from_txt, save_content_in_txt
from scipy import signal
from pydub import AudioSegment
import librosa


def list_filtering(content, filter_keys, mode):
    if not filter_keys:
        return content
    if isinstance(filter_keys, str):
        filter_keys = [filter_keys]
    new_content = []

    for k in filter_keys:
        for c in content:
            if mode == 'keep':
                if k in c:
                    new_content.append(c)
            elif mode == 'remove':
                if k not in c:
                    new_content.append(c)
            else:
                raise ValueError('Unknown filtering mode.')
    return new_content


# TODO: replace keyword mode
# TODO: replace original file mode
def process_keyword_in_txt(data_path, filter_keys, mode):
    content = load_content_from_txt(data_path)
    new_content = list_filtering(content, filter_keys, mode)
    new_data_path = os.path.join(os.path.split(data_path)[0], os.path.split(data_path)[1].replace('.', '_new.'))
    save_content_in_txt(new_content, new_data_path, access_mode='w+')


def get_dir_list(data_path):
    dir_list = np.array([], dtype=object)
    for f in os.listdir(data_path):
        folder_path = os.path.join(data_path, f)
        if os.path.isdir(folder_path):
            dir_list = np.append(dir_list, folder_path)
    return list(dir_list)


def balancing_indexing(path):
    dir_list = get_dir_list(path)
    total_p, total_n = sample_dist(path)
    total_p, total_n = total_p.values(), total_n.values()

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


def sample_dist(path):
    dir_list = get_dir_list(path)
    total_p, total_n = np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    total_p, total_n = {}, {}
    for d in dir_list:
        subject = os.path.split(d)[1].split('_')[0]
        path_p = os.path.join(d, '1')
        path_n = os.path.join(d, '0')
        if os.path.isdir(path_p):
            p = len(os.listdir(path_p))
        else:
            p = 0
        total_p[subject] = p
        # total_p = np.append(total_p, p)
        if os.path.isdir(path_n):
            n = len(os.listdir(path_n))
        else:
            n = 0
        total_n[subject] = n
        # total_n = np.append(total_n, n)
    return total_p, total_n


def generate_index(data_path, save_path, subject_list=None):
    # TODO: split automatically if subject list not input
    total_cases = get_dir_list(data_path)
    in_cases, out_cases = [], []
    for s in subject_list:
        for c in total_cases:
            if s in c:
                in_cases.append(c)
                break
    out_cases = list(set(total_cases)-set(in_cases))

    def write_paths(path_list, save_name, data_format):
        total = []
        for p in path_list:
            total.extend(get_files(p, data_format))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_content_in_txt(total, os.path.join(save_path, f'{save_name}.txt'), access_mode='w+')

    write_paths(in_cases, 'train', 'wav')
    write_paths(out_cases, 'valid', 'wav')
    

def indexing(path):
    # TODO: if two cases have same data number
    total_p, total_n = sample_dist(path)
    total_num = {p[0]:p[1]+n[1] for p, n in zip(total_p.items(), total_n.items()) if p[1]+n[1]>0}
    total_num = {k: v for k, v in sorted(total_num.items(), key=lambda item: item[1])}

    test_cases = list(total_num.keys())[::4]
    test_cases = ['1630345236867',
                '1630513297437',
                '1630779176834',
                '1630866536302',
                '1630949188143']
    test_num = [total_p[k]+total_n[k] for k in test_cases]
    test_num = np.sort(np.array(test_num))

    train_cases = list(set(total_num.keys())-set(test_cases))
    train_num = [total_p[k]+total_n[k] for k in train_cases]
    train_num = np.sort(np.array(train_num))

    # p_nums = total_p.values()
    # n_nums = total_n.values()
    # total_num = [p+n for p, n in zip(p_nums, n_nums)]
    # total_num = np.array(total_num)
    # total_num = np.sort(total_num[total_num!=0])
    # print(total_num, len(total_num))
    # valid = total_num[np.argsort(total_num)[::4]]

    # test_600 = ['1630345236867',
    #             '1630513297437',
    #             '1630779176834',
    #             '1630866536302',
    #             '1630949188143']
    # valid = [total_p[k]+total_n[k] for k in test_600]
    # valid = np.array(valid)

    # train = np.setdiff1d(total_num, valid)
    print(train_num, np.sum(train_num), np.max(train_num)/np.sum(train_num), len(train_num))
    print(test_num, np.sum(test_num), np.max(test_num)/np.sum(test_num), len(test_num))
    print(np.sum(train_num)/np.sum(list(total_num.values())))


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
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\freq6_no_limit\4_21\raw_f_h_1_mono_16k'
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\freq6_no_limit\2_13\raw_f_h_2_mono_16k'
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\freq6_no_limit\2_21\raw_f_h_2_mono_16k'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\Freq2\2_21_2s_b'
    subject_list = ['1630779176834', '1620055140118', '1620231545598', '1630345236867', '1630949188143', 
                    '1598482996718', '1630866536302', '1630681292279', '1631119510605', '1633284111726', '1631208670684', 
                    '1631294758253', '1631208559119', '1631902918706', '1631293102954', '1632597921043', '1631551695426', 
                    '1632672580868', '1633540605366', '1631294788806', '1632417619384', '1631037196770', '1633019471084', 
                    '1630600693454', '1632245323932', '1631468777871', '1631810812342', '1631554200509', '1630945379152', 
                    '1630513297437', '1606921286802']
    # indexing(data_path)

    generate_index(data_path, save_path, subject_list)

    # p, n = balancing_indexing(data_path)
    # index_path = rf'C:\Users\test\Downloads\1022'
    # content = np.concatenate([p, n])
    # with open(os.path.join(index_path, 'train.txt'), 'w+') as fw:
    #     for c in content:
    #         fw.write(f'{c}\n')

    # process_keyword_in_txt(
    #     data_path=rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\ASUS_h_train_ASUS_m_test_2sec\train.txt', 
    #     filter_keys='_-', 
    #     mode='remove')


if __name__ == '__main__':
    main()
    pass
