import os
import numpy as np
import torchaudio
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import random
from analysis import data_splitting
from dataset import dataset_utils
from analysis import utils
import shutil

# def get_file_names(path, keyword=None, filtering_mode='in', is_fullpath=True, shuffle=True):
#     files = os.listdir(path)
#     # files.sort(key=len)
#     file_names = []
#     for f in files:
#         if keyword:
#             if filtering_mode == 'in':
#                 if keyword not in f: continue
#             elif filtering_mode == 'out':
#                 if keyword in f: continue

#         if is_fullpath:
#             file_names.append(os.path.join(path, f))
#         else:
#             file_names.append(f)
#     if shuffle: random.shuffle(file_names)
#     return file_names


def save_aLL_files_name(path, name='file_names', keyword=None, filtering_mode='in', is_fullpath=True, shuffle=True, save_path=None):
    # file_names = get_file_names(path, keyword, filtering_mode, is_fullpath, shuffle)
    file_names = data_splitting.get_files(path, keys=keyword, is_fullpath=True, sort=True)
    if not save_path: save_path = path
    dataset_utils.save_content_in_txt(
        file_names, os.path.join(save_path, f'{name}.txt'), filter_bank=[], access_mode='w+', dir=None)
    # with open(os.path.join(save_path, f'{name}.txt'), 'w+') as fw:
    #     for f in file_names:
    #         fw.write(f)    
    #         fw.write('\n')


def generate_index_for_subject():
    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw2_mono_hospital'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\ASUS_subject_training'
    valid_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\ASUS_h_train_ASUS_m_test\valid.txt'
    dir_list = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    val_content = dataset_utils.load_content_from_txt(valid_path)

    for d in dir_list:
        print(f'[INFO] Generating training index for subject {d}')
        if not os.path.isdir(os.path.join(save_path, d)):
            os.mkdir(os.path.join(save_path, d))
        # save_aLL_files_name(
        #     os.path.join(path, d), keyword='wav', name='train', shuffle=False, save_path=os.path.join(save_path, d))
        file_names = data_splitting.get_files(os.path.join(path, d), keys='wav', is_fullpath=True, sort=True)
        # TODO: redundency code, change it ASAP
        for f in file_names:
            for valid_f in val_content:
                v_f = os.path.basename(valid_f).split('_')[0] + '_' + os.path.basename(valid_f).split('_')[1]
                if v_f in f:
                    print(f'remove {f}')
                    if f in file_names:
                        file_names.remove(f)
        dataset_utils.save_content_in_txt(
            file_names, os.path.join(os.path.join(save_path, d), f'train.txt'), filter_bank=[], access_mode='w+', dir=None)
        dataset_utils.save_content_in_txt(
            val_content, os.path.join(os.path.join(save_path, d), 'valid.txt'), filter_bank=[], access_mode='w+', dir=None)
    dir_list_full = [os.path.join(save_path, f) for f in dir_list]
    dataset_utils.save_content_in_txt(
        dir_list_full, os.path.join(save_path, 'dir_name.txt'), filter_bank=[], access_mode='w+', dir=None)
    # print(dir_list)


def save_files_in_csv():
    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\ASUS_subject_training'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\ASUS_subject_training'
    valid_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\ASUS_h_train_ASUS_m_test\valid.txt'
    dir_list = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    dir_list = dir_list[:4]
    for d in dir_list:
        print(f'[INFO] Saving file name for subject {d}')
        if not os.path.isdir(os.path.join(save_path, d)):
            os.mkdir(os.path.join(save_path, d))

        file_names = dataset_utils.load_content_from_txt(os.path.join(path, d, 'train.txt'))
        file_names.sort()
        pred_label = [int(os.path.split(os.path.split(f)[0])[1]) for f in file_names]
        df = pd.DataFrame({'file_name': file_names})
        df.index += 1
        df['predict_label'] = pred_label
        df.to_csv(os.path.join(save_path, d, f'{d}_label.csv'))


def string_process(_str, keyword_pair, keep_remain=True):
    assert isinstance(keyword_pair, (list, tuple))
    if keep_remain:
        return _str.replace(keyword_pair[0], keyword_pair[1])
    else:
        return keyword_pair[1]


# TODO: coding start in 0 or 1
# TODO: format filtering
# TODO: optional zfill number?
def change_all_file_names(path, keyword_pair, keep_remain=False, recode=True):
    files = os.listdir(path)
    files.sort(key=len)
    os.chdir(path)
    for i, f in enumerate(files):
        
        old_name, suffix = f.split('.')
        new_name = string_process(old_name, keyword_pair, keep_remain)
        if recode:
            new_name = f'{new_name}_{i+1:03d}.{suffix}'
        print(f'Changing file name from {f} to {new_name}')
        os.rename(f, new_name)


def try_noisereduce():
    from scipy.io import wavfile
    import noisereduce as nr
    
    # load data
    file = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw2_mono_hospital\1630513297437_AA1700268\1\1630513297437_16_141.0_142.0_002.wav'
    rate, data = wavfile.read(file)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write('noisereduce_test.wav', rate, reduced_noise)


def Snoring_data_analysis():
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring'
    threshold = 82
    single_length = 3
    data_format = 'm4a'
    test_amplitude = False
    data_analysis(data_path, threshold, single_length, data_format, test_amplitude)


def data_analysis(path, threshold, single_length, data_format, test_amplitude):
    sample_list = os.listdir(path)
    sample_nums = len(sample_list)
    sample_length, sample_length_threshold, data_length_threshold, max_amplitude, min_amplitude = [], [], [], [], []
    min_length, max_length, sample_names, non_effective = [], [], [], []
    persons = {}
    os.chdir(path)
    for idx, dir in enumerate(sample_list):
        if os.path.isdir(dir):
            audio_list = os.listdir(dir)
            file_length = len(audio_list)
            sample_length.append(file_length)
            if file_length > threshold:
                sample_length_threshold.append(file_length)
                data_length_threshold.append(file_length*single_length)
                sample_names.append(dir)
                working_number = dir.split('_')[1]
                if working_number in persons:
                    persons[working_number] += 1
                else:
                    persons[working_number] = 1

                if test_amplitude:
                    max_temp_amplitude, min_temp_amplitude = [], []
                    for f in audio_list:
                        if f.endswith(data_format):
                            print(f'{idx+1}/{len(sample_list)}', dir, f)
                            x = AudioSegment.from_file(os.path.join(dir, f), format='m4a').get_array_of_samples()
                            x = np.float32(np.array(x))
                            max_temp_amplitude.append(np.max(x))
                            min_temp_amplitude.append(np.min(x))
                    max_amplitude.append(np.max(max_temp_amplitude))
                    min_amplitude.append(np.min(min_temp_amplitude))
            else:
                non_effective.append(dir)
    print('Sample Number: ', len(sample_length))
    print('Effective Sample Number: ', len(sample_length_threshold))
    print(f"Total data duration: {np.sum(data_length_threshold)} (sec) ")
    print(f"Mean data duration: {np.mean(data_length_threshold)} (sec) ")
    print(f"Std data duration: {np.std(data_length_threshold)} (sec) ")
    print(f"Max data duration: {np.max(data_length_threshold)} (sec) ")
    print(f"Min data duration: {np.min(data_length_threshold)} (sec) ")
    print(sample_length)
    if test_amplitude:
        print(f"Max data amplitude: {np.max(max_amplitude)}")
        print(f"Min data amplitude: {np.min(min_amplitude)}")

    
    # Write effective sample names to text file.
    # print(sample_names)
    with open('effective_samples.txt', 'w+') as fw:
        fw.write("effective\n")
        for item in sample_names:
            fw.write(f"{item}\n")
        fw.write("\n")
        fw.write("non-effective\n")
        for item in non_effective:
            fw.write(f"{item}\n")

    # Write audio belonging information
    # print(persons)
    total_person = sum(list(persons.values()))
    with open('persons.txt', 'w+') as fw:
        for k, v in persons.items():
            fw.write(f'{k}: {v}\n')
        fw.write(30*'-')
        fw.write(f'\nTotal: {total_person}')


def thresholding(filename, threshold):
    df = pd.read_csv(filename)
    data = df[df.columns[1]]
    data = data.to_numpy()
    data = np.int32(data)

    for i in data:
        if isinstance(i, int):
            if i >= threshold:
                df = 1
    print([156])


def peak_analysis():
    filename = rf'C:\Users\test\Downloads\total_peak2.csv'
    thresholding(filename, threshold=25)


def re_split():
    """ A test code which is for temporally purpose to extend audio clip from 1 sec to 2 sec. 
    Not a great example, please define everything clearly at first"""
    peak_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\infos\peak3_2'
    audio_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_raw'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\subset2_new'
    times = [1, 2]
    sr = 16000
    channels = 1
    load_format = 'm4a'
    save_format = 'wav'

    # dir_list = utils.get_dir_list(peak_path)
    # for d in dir_list:
    #     os.rename(d, os.path.join(peak_path, os.path.basename(d).split('_')[0]))

    c = '0'
    for c in ['0', '1']:
        clip_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\subset2\{c}'
        filenames = data_splitting.get_files(clip_path, keys='wav', is_fullpath=True, sort=True)
        for i, f in enumerate(filenames):
            # if f == rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\subset2\0\1630345236867_6_034.wav':
            # Get subject, peak number, and peak time
            subject = os.path.split(f)[1].split('_')[0]
            number = os.path.split(f)[1].split('_')[1]
            peak_number = os.path.split(f)[1].split('_')[2].split('.')[0]
            df = pd.read_csv(os.path.join(peak_path, subject, f'{subject}_{number}_peak.csv'))
            if df['Time (second)'][0] == 0:
                index = int(peak_number)
            else:
                index = int(peak_number) - 1
            peak_time = df['Time (second)'][index]
            
            #  Load raw audio
            data_path = os.path.join(audio_path, subject, f'{subject}_{number}.{load_format}')
            # y = utils.load_audio_waveform(data_path, load_format)
            y = utils.load_audio_waveform(data_path, load_format, sr, channels)

            # if i+1==158:
            #     print(i)
            print(i+1, f, c)
            for t in times:
                # pick_next = utils.get_audio_clip(y, [peak_time, peak_time+t], 1000)
                # pick_cent = utils.get_audio_clip(y, [peak_time-0.5*t, peak_time+0.5*t], 1000)

                # # make dir
                # path_cent = os.path.join(save_path, f'test_{t}sec_cent_mono', c)
                # path_next = os.path.join(save_path, f'test_{t}sec_next_mono', c)
                # if not os.path.isdir(path_cent): os.makedirs(path_cent)
                # if not os.path.isdir(path_next): os.makedirs(path_next)

                # # save audio clip
                # save_name = os.path.basename(f)
                # pick_cent.export(os.path.join(path_cent, save_name), format=save_format)
                # pick_next.export(os.path.join(path_next, save_name), format=save_format)


                range_next = [peak_time, peak_time+t]
                if range_next[0] > 0 and range_next[1] < y.duration_seconds:
                    pick_next = utils.get_audio_clip(y, range_next, 1000)
                    path_next = os.path.join(save_path, f'test_{t}sec_next_mono_16k', c)
                    # make dir
                    if not os.path.isdir(path_next): os.makedirs(path_next)
                    # save audio clip
                    save_name = os.path.basename(f)
                    pick_next.export(os.path.join(path_next, save_name), format=save_format)

                range_cent = [peak_time-0.5*t, peak_time+0.5*t]
                if range_cent[0] > 0 and range_cent[1] < y.duration_seconds:
                    pick_cent = utils.get_audio_clip(y, range_cent, 1000)
                    path_cent = os.path.join(save_path, f'test_{t}sec_cent_mono_16k', c)
                    # make dir
                    if not os.path.isdir(path_cent): os.makedirs(path_cent)
                    # save audio clip
                    save_name = os.path.basename(f)
                    pick_cent.export(os.path.join(path_cent, save_name), format=save_format)


def save_audio_fig():
    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring'
    save_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\imgs\ASUS_snoring'
    sample_list = os.listdir(path)
    threshold = 82
    data_format = 'm4a'
    os.chdir(path)
    for idx, dir in enumerate(sample_list):
        if idx > 37:
            audio_list = os.listdir(dir)
            file_length = len(audio_list)
            if file_length > threshold:
                if not os.path.isdir(os.path.join(save_path, dir)):
                    os.mkdir(os.path.join(save_path, dir))
                for f in audio_list:
                    if f.endswith(data_format):
                        print(f'{idx+1}/{len(sample_list)}', dir, f)
                        x = AudioSegment.from_file(os.path.join(dir, f), format='m4a').get_array_of_samples()
                        x = np.float32(np.array(x))
                        plt.figure(figsize=(14, 5))
                        librosa.display.waveplot(x, 44100)
                        plt.savefig(os.path.join(save_path, dir, f'{f}.png'))
                        plt.close()


def get_diff_files(src1, src2, dst, data_format='wav'):
    files1 = data_splitting.get_files(src1, data_format, is_fullpath=False)
    files2 = data_splitting.get_files(src2, data_format, is_fullpath=False)
    # TODO: fix above lines for generalization
    dst_files = list(set(files1)-set(files2))
    dst_files = [os.path.join(src1, f.split('_')[0], f) for f in dst_files]
    # 


    for idx, f in enumerate(dst_files):
        print(f'{idx+1}/{len(dst_files)}', f)
        new_f = f.replace(src1, dst)
        if not os.path.isdir(os.path.split(new_f)[0]):
            os.makedirs(os.path.split(new_f)[0])
        shutil.copyfile(f, new_f)


def convert_testing_data(path, src, dst, add_dB):
    y1 = dataset_utils.load_audio_waveform(path, src, 16000, 1)
    y2 = y1 + add_dB
    y1.export(path.replace(src, dst), dst)
    y2.export(path.replace(f'.{src}', f'_6dB.{dst}'), dst)


def convert_KC_testing():
    file_list = data_splitting.get_files(rf'C:\Users\test\Downloads\1112\KC_testing', 'm4a')
    for f in file_list:
        convert_testing_data(f, 'm4a', 'wav', 6)


if __name__ == '__main__':
    # Snoring_data_analysis()
    # save_audio_fig()
    # peak_analysis()
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\subset1\1')
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset\0', keyword=None, filtering_mode='in', shuffle=False, is_fullpath=False)
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset\0', keyword='wav', name='0')
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset\1', keyword='wav', name='1')
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\subset2\1', keyword='wav', name='1', shuffle=False)
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw2_mono_hospital', keyword='wav', name='filename', shuffle=True)
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\raw_mono', keyword='wav', name='filenames', shuffle=False)
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50_process\ecs50\ecs50_1', keyword='wav', name='train', shuffle=False)
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50_process\ecs50\ecs50_2', keyword='wav', name='train', shuffle=False)
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\raw_mono_MFCC', keyword='npy', name='filenames', shuffle=False)
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\raw_mono_h_MFCC', keyword='npy', name='filenames', shuffle=False)
    # save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\raw_mono_h_MFCC', keyword='npy', name='filenames', shuffle=False)
    # change_all_file_names(rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\infos\test_samples\0', 
    #                       keyword_pair=['1', '0'], 
    #                       keep_remain=False, 
    #                       recode=True)
    # generate_index_for_subject()
    # save_files_in_csv()
    # try_noisereduce()
    # re_split()

    save_aLL_files_name(rf'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50_process\esc50_16k\esc50_16k_2', keyword='wav', name='file_names', shuffle=False)
    
    # convert_KC_testing()

    # for i in [2, 4]:
    #     for j in [13,21]:
    #         for k in [1,2,3]:
    #             src1 = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\freq7_no_limit\{i}_{j}\raw_f_{k}_mono_16k'
    #             src2 = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\freq7_no_limit\{i}_{j}\raw_f_h_{k}_mono_16k'
    #             dst = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\freq7_no_limit\{i}_{j}\raw_f_unlabeled_{k}_mono_16k_'
    #             get_diff_files(src1, src2, dst, data_format='wav')
    
    pass