
import importlib
from pathlib import Path
import os
import re

import numpy as np
import logging
from pydub import AudioSegment
import pandas as pd
import glob
import cv2



# TODO: General solution
# TODO: Can string_filtering be called by generate_filenames?
# TODO: output only benign file?
# TODO: think about input output design of generate_filenames
# def generate_filename_list(path, file_key, dir_key='', only_filename=False):
#     input_paths, gt_paths = [], []
#     for root, dirs, files in os.walk(path):
#         for f in files:
#             if not only_filename:
#                 fullpath = os.path.join(root, f)
#             else:
#                 fullpath = f
#             if dir_key in fullpath:
#                 if file_key in fullpath:
#                     gt_paths.append(fullpath)
#                 else:
#                     input_paths.append(fullpath)
#     return input_paths, gt_paths


def get_melspec_from_cpp(wav_list_path, out_dir, sampling_rate=None):
    # Generate output path list (wav -> csv)
    out_path_list = []
    wav_list_dir, wav_list_filename = os.path.split(wav_list_path)
    with open(wav_list_path, 'r') as fw:
        in_path_list = fw.readlines()
    in_path_list = [f if f[-3:] == 'wav' else f[:-1] for f in in_path_list]

    csv_out_dir = os.path.join(out_dir, 'csv', wav_list_filename[:-4])
    os.makedirs(csv_out_dir, exist_ok=True)
    data_pair = {}
    for in_path in in_path_list:
        in_dir, in_file = os.path.split(in_path)
        out_file = in_file.replace('wav', 'csv')
        out_path = os.path.join(csv_out_dir, out_file)
        out_path_list.append(out_path)

        key = out_file[:-4]
        # label = int(os.path.split(in_dir)[1])
        label = 0
        data_pair[key] = {'label': label}

    csv_list_path = os.path.join(csv_out_dir, wav_list_filename)
    with open(csv_list_path, 'w+') as fw:
        for path in out_path_list:
            fw.write(f'{path}\n')

    # Cpp MelSpectrogram
    exe_file = Path(r'compute-mfcc.exe')
    inputlist = Path(wav_list_path)
    outputlist = Path(csv_list_path)
    command = (
        f'{exe_file.as_posix()} '
        f'--inputlist "{inputlist.as_posix()}" '
        f'--outputlist "{outputlist.as_posix()}" '
    )
    if sampling_rate is not None:
        command += f'--samplingrate {sampling_rate} '
    os.system(command)

    # Save CPP feature (csv -> npy)
    # csv_list = glob.glob(os.path.join(csv_out_dir, '*.csv'))
    csv_list = out_path_list
    img_out_dir = os.path.join(out_dir, 'img', wav_list_filename[:-4])
    os.makedirs(img_out_dir, exist_ok=True)
    img_save_paths = []
    for idx, csv_f in enumerate(csv_list):
        # if idx>10:break
        _, filename = os.path.split(csv_f)
        try:
            df = pd.read_csv(csv_f)
            # XXX: wrong data extraction for temp using
            # df = pd.read_csv(csv_f, header=None)
            data = df.to_numpy().T
        except pd.errors.EmptyDataError:
            print(f'- Empty pandas data {csv_f}')
            data = np.zeros(1)

        save_path = os.path.join(img_out_dir, filename.replace('csv', 'npy'))
        img_save_paths.append(save_path)
        data_pair[filename[:-4]]['path'] = save_path
        np.save(save_path, data)
        
    with open(os.path.join(out_dir, wav_list_filename), 'w+') as fw:
        fw.writelines(img_save_paths)

    inputs, paths, labels = [], [], []
    for k, v in data_pair.items():
        inputs.append(k[:-4])
        if 'path' in v:
            paths.append(v['path'])
        else:
            paths.append('')
        if 'label' in v:
            labels.append(v['label'])
        else:
            labels.append('')
        # labels.append(v['label'])
    df = pd.DataFrame({
        'input': inputs, 'img_path': paths, 'label': labels
    })
    df.to_csv(os.path.join(out_dir, wav_list_filename).replace('txt', 'csv'))


def get_dir_list(data_path, full_path=True):
    dir_list = np.array([], dtype=object)
    for f in os.listdir(data_path):
        folder_path = os.path.join(data_path, f)
        if os.path.isdir(folder_path):
            if full_path:
                dir_list = np.append(dir_list, folder_path)
            else:
                dir_list = np.append(dir_list, os.path.split(folder_path)[1])
    return list(dir_list)


def get_clips_from_audio(y, clip_time, hop_time):
    clips = []
    if y.duration_seconds >= clip_time:
        for t in range(0, int(y.duration_seconds)-clip_time+1, hop_time):
            start_t, end_t = t, t+clip_time
            clip = y[1000*start_t:1000*end_t]
            clips.append(clip)
    return clips


def continuous_split(path, clip_time, hop_time, sr, channels, add_volume, output_format='wav'):
    assert clip_time > 0 and hop_time > 0
    file_name = os.path.basename(path)
    file_format = file_name.split('.')[1]
    y = load_audio_waveform(path, file_format, sr, channels)
    save_path = os.path.join(os.path.split(path)[0], f'clips_{clip_time}_{hop_time}_{add_volume}dB')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    y += add_volume
    clips = get_clips_from_audio(y, clip_time, hop_time)
    for idx, clip in enumerate(clips, 1):
        clip.export(os.path.join(save_path, file_name.replace(f'.{file_format}', f'_{idx:03d}.{output_format}')), output_format)
    print('Finish spitting.')
    return save_path


def save_aLL_files_name(path, name='file_names', keyword=[], filtering_mode='in', is_fullpath=True, shuffle=True, save_path=None):
    # file_names = get_file_names(path, keyword, filtering_mode, is_fullpath, shuffle)
    file_names = get_files(path, keys=keyword, return_fullpath=True, sort=True)
    if not save_path: save_path = path
    save_content_in_txt(
        file_names, os.path.join(save_path, f'{name}.txt'), filter_bank=[], access_mode='w+', dir=None)
    # with open(os.path.join(save_path, f'{name}.txt'), 'w+') as fw:
    #     for f in file_names:
    #         fw.write(f)    
    #         fw.write('\n')


# TODO: recursively get the files should be a option
# TODO: input soring function, condition
# TODO: think about the design of shuffling and sorting, in-function or separated
# def get_files(path, keys=[], is_fullpath=True, sort=True):
#     """Get all the file name under the given path with assigned keys"""
#     file_list = []
#     assert isinstance(keys, (list, str))
#     if isinstance(keys, str): keys = [keys]
#     # Rmove repeated key
#     keys = list(set(keys))

#     def func(root, f, file_list, is_fullpath):
#         if is_fullpath:
#             file_list.append(os.path.join(root, f))
#         else:
#             file_list.append(f)

#     for i, (root, dirs, files) in enumerate(os.walk(path)):
#         for j, f in enumerate(files):
#             if keys:
#                 for key in keys:
#                     if key in f:
#                         func(root, f, file_list, is_fullpath)
#             else:
#                 func(root, f, file_list, is_fullpath)

#     if file_list:
#         if sort: file_list.sort()
#         # if sort: file_list.sort(key=len)
#     else:
#         if keys: 
#             logging.warning(f'No file exist with key {keys}.') 
#         else: 
#             logging.warning(f'No file exist.') 
#     return file_list


def get_ASUS_snoring_index(save_path, split, balancing=True):
    # TODO: Multiple class (miner)
    # Load and check dataset
    subject_list = []
    data_pair = {}
    train_pairs, valid_pairs = [], []
    df = pd.read_csv(save_path)
    
    for index, row in df.iterrows():
        pair = (row['path'], row['label'])
        if row['subject'] in data_pair:
            data_pair[row['subject']].append(pair)
        else:
            subject_list.append(row['subject'])
            data_pair[row['subject']] = [pair]
        # TODO: balancing
        
    # indexing
    if balancing:
        sample_count = {}
        for subject in subject_list:
            for sample in data_pair[subject]:
                if sample[1] in sample_count[subject]:
                    sample_count[subject][sample[1]] += 1
                else:
                    sample_count[subject][sample[1]] = 1
        
    else:
        sample_count = []
        for subject in subject_list:
            sample_count.append(len(data_pair[subject]))
        sorted_idx = np.argsort(np.array(sample_count))
        subject_list = np.take(subject_list, sorted_idx)
        # TODO: subject_list[::3]
        valid_subjects = subject_list[::3]
        train_subjects = list(set(subject_list)-set(valid_subjects))
        
        for subject in train_subjects:
            train_pairs.extend(data_pair[subject])
            
        for subject in valid_subjects:
            valid_pairs.extend(data_pair[subject])    
        
    
    
    return train_pairs, valid_pairs
        
            
        
            
    return train_pairs, valid_pairs


def save_ASUS_snoring_index(data_path, data_suffixs):
    # pass if index has existed
    
    # save dataset information
    
    # save dataset index in csv format
    return save_path


def load_input_data(config):
    assert isinstance(config.dataset.data_path, dict)
    total_train, total_valid = [], []
    for key in config.dataset.data_path:
        if key == 'ASUS_snoring':
            save_path = save_ASUS_snoring_index(config.dataset.data_path[key])
            train, valid = get_ASUS_snoring_index(save_path)
        elif key == 'esc-50':
            pass
        elif key == 'Kaggle_snoring':
            pass
        total_train.extend(train)
        total_valid.extend(valid)
            
    return total_train, total_valid
    
    
    return datasets_indexed


def load_audio_waveform(filename, audio_format, sr=None, channels=None):
    """Pydub based audio waveform loading function"""
    y = AudioSegment.from_file(filename, audio_format)
    if sr: y = y.set_frame_rate(sr)
    if channels: y = y.set_channels(channels)
    return y    


def get_files(path, keys=[], return_fullpath=True, sort=True, sorting_key=None):
    """Get all the file name under the given path with assigned keys
    Args:
        path: (str)
        keys: (list, str)
        return_fullpath: (bool)
        sort: (bool)
        sorting_key: (func)
    Return:
        file_list: (list)
    """
    file_list = []
    assert isinstance(keys, (list, str))
    if isinstance(keys, str): keys = [keys]
    # Rmove repeated keys
    keys = list(set(keys))

    def push_into_filelist(root, f, file_list, is_fullpath):
        if is_fullpath:
            file_list.append(os.path.join(root, f))
        else:
            file_list.append(f)

    for i, (root, dirs, files) in enumerate(os.walk(path)):
        for j, f in enumerate(files):
            if keys:
                for key in keys:
                    if key in f:
                        push_into_filelist(root, f, file_list, return_fullpath)
            else:
                push_into_filelist(root, f, file_list, return_fullpath)

    if file_list:
        if sort: file_list.sort(key=sorting_key)
    else:
        if keys: 
            logging.warning(f'No file exist with key {keys}.') 
        else: 
            logging.warning(f'No file exist.') 
    return file_list



def generate_filenames(path, keys=None, include_or_exclude=None, is_fullpath=True, loading_formats=None):
    """Get all the file name under the given path with assigned keys and including condition"""
    # TODO: index error when keys=['a','a','a'] include_or_exclude=['include', 'exclude', 'exclude']
    if len(keys) == 0: keys = None
    if len(include_or_exclude) == 0: include_or_exclude = None
    if keys and include_or_exclude:
        assert len(keys) == len(include_or_exclude)
        full_keys = [f'{condition}_{k}' for k, condition in zip(keys, include_or_exclude)]
        # TODO: logging instead of print for repeat key (think about raise error or not)
        if len(full_keys) != len(list(set(full_keys))):
            print('Warning: Repeat keys')
            full_keys = list(set(full_keys))
    if keys:
        filenames = {k: [] for k in full_keys}
    else:
        filenames = []

    for root, dirs, files in os.walk(path):
        for f in files:
            if loading_formats:
                process = False
                for format in loading_formats:
                    if format in f:
                        process = True
                        break
            else:
                process = True

            if process:
                if is_fullpath:
                    final_path = os.path.join(root, f)
                else:
                    final_path = f
                if keys:
                    for idx, k in enumerate(keys):
                        if include_or_exclude[idx] == 'include':
                            if k in final_path:
                                filenames[full_keys[idx]].append(final_path)
                        elif include_or_exclude[idx] == 'exclude': 
                            if k not in final_path:
                                filenames[full_keys[idx]].append(final_path)
                        else:
                            raise ValueError('Unknown key condition')
                else:
                    filenames.append(final_path)
    return filenames

    
def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')


def load_content_from_txt(path, access_mode='r'):
    with open(path, access_mode) as fw:
        content = fw.read().splitlines()
    return content


def inspect_data_split(data_split):
    # TODO: if split in 0.725 0.275
    train_split = data_split.get('train', 0)
    valid_split = data_split.get('valid', 0)
    test_split = data_split.get('test', 0)
    if train_split+valid_split+test_split != 1:
        raise ValueError('Incorrect splitting of dataset.')
    split_code = f'{10*train_split:.0f}{10*valid_split:.0f}{10*test_split:.0f}'
    return split_code


def get_data_path(data_path, index_root, data_split, keywords=[]):
    # TODO: save in csv
    # TODO: save with label
    split_code = inspect_data_split(data_split)
    dataset_name = os.path.split(data_path)[1]
    
    # Check if an index already exists, create one if not.
    index_path = os.path.join(index_root, dataset_name)
    if not os.path.isdir(index_path):
        os.mkdir(index_path)
        file_path_list = get_files(data_path, keys=keywords, return_fullpath=True, sort=True)
        train_split = int(data_split.get('train', 0)*len(file_path_list))
        valid_split = int(data_split.get('valid', 0)*len(file_path_list))
        test_split = int(data_split.get('test', 0)*len(file_path_list))

        train_path_list = file_path_list[:train_split]
        valid_path_list = file_path_list[train_split:train_split+valid_split]
        test_path_list = file_path_list[train_split+valid_split:train_split+valid_split+test_split]

        train_path = os.path.join(index_path, f'{dataset_name}_train_{split_code}.txt')
        valid_path = os.path.join(index_path, f'{dataset_name}_valid_{split_code}.txt')
        test_path = os.path.join(index_path, f'{dataset_name}_test_{split_code}.txt')

        save_content_in_txt(train_path_list, train_path)
        save_content_in_txt(valid_path_list, valid_path)
        save_content_in_txt(test_path_list, test_path)

        data_path_dict = {
            'train': train_path,
            'valid': valid_path,
            'test': test_path}
    else:
        file_path_list = get_files(index_root, return_fullpath=True, sort=True)
        data_path_dict
    return data_path_dict


def get_data_indices(data_name, data_path, save_path, data_split, mode, generate_index_func):
    """"Get dataset indices and create if not exist"""
    # # TODO: gt not exist condition
    # create index folder and sub-folder if not exist
    os.chdir(save_path)
    index_dir_name = f'{data_name}_data_index'
    sub_index_dir_name = f'{data_name}_{data_split[0]}_{data_split[1]}'
    input_data_path = os.path.join(save_path, index_dir_name, sub_index_dir_name)
    if not os.path.isdir(input_data_path): os.makedirs(input_data_path)
    
    # generate index list and save in txt file
    generate_index_func(data_path, data_split, input_data_path)
        
    # load index list from saved txt
    os.chdir(input_data_path)
    if os.path.isfile(f'{mode}.txt'):
        input_data_indices = load_content_from_txt(f'{mode}.txt')
        input_data_indices.sort()
    else:
        input_data_indices = None

    if os.path.isfile(f'{mode}_gt.txt'):
        ground_truth_indices = load_content_from_txt(f'{mode}_gt.txt')
        ground_truth_indices.sort()
    else:
        ground_truth_indices = None
    return input_data_indices, ground_truth_indices


def generate_kaggle_breast_ultrasound_index(data_path, save_path, data_split):
    data_keys = {'input': 'exclude_mask', 'ground_truth': 'include_mask'}
    save_input_and_label_index(data_path, save_path, data_split, data_keys, loading_format=['png', 'jpg'])


def generate_kaggle_snoring_index(data_path, save_path, data_split):
    # TODO: no ground truth case
    # TODO: keys=None, include_or_exclude=None case
    data_keys = {'input': None}
    save_input_and_label_index(data_path, save_path, data_split, data_keys, loading_format=['wav', 'm4a'])


def save_input_and_label_index(data_path, save_path, data_split, data_keys=None, loading_format=None):
    # TODO: test input output
    # assert 'input' in data_keys, 'Undefined input data key'
    # assert 'ground_truth' in data_keys, 'Undefined ground truth data key'
    class_name = os.listdir(data_path)
    os.chdir(save_path)
    include_or_exclude, keys = [], []
    if data_keys:
        for v in data_keys.values():
            if v:
                include_or_exclude.append(v.split('_')[0])
                keys.append(v.split('_')[1]) 
    data_dict = generate_filenames(
        data_path, keys=keys, include_or_exclude=include_or_exclude, is_fullpath=True, loading_formats=loading_format)

    def save_content_ops(data, train_name, valid_name):
        # TODO: Is this a solid solution?
        data.sort(key=len)
        split = int(len(data)*data_split[0])
        train_input_data, val_input_data = data[:split], data[split:]
        save_content_in_txt(train_input_data,  train_name, filter_bank=class_name, access_mode="w+", dir=save_path)
        save_content_in_txt(val_input_data, valid_name, filter_bank=class_name, access_mode="w+", dir=save_path)


    # if data_keys['input']:    
    #     input_data = data_dict[data_keys['input']]
    # else:
    #     input_data = data_dict
    
    if data_keys:
        if 'input' in data_keys:
            if data_keys['input']:
                input_data = data_dict[data_keys['input']]
            else:
                input_data = data_dict
            save_content_ops(input_data, 'train.txt', 'valid.txt')
        if 'ground_truth' in data_keys:
            if data_keys['ground_truth']:
                ground_truth = data_dict[data_keys['ground_truth']]
            else:
                ground_truth = data_dict
            save_content_ops(ground_truth, 'train_gt.txt', 'valid_gt.txt')
    else:
        input_data = data_dict
        save_content_ops(input_data, 'train.txt', 'valid.txt')
        
    # input_data.sort()
    # split = int(len(input_data)*data_split[0])
    # train_input_data, val_input_data = input_data[:split], input_data[:split]
    # save_content_in_txt(train_input_data, 'train.txt', filter_bank=class_name, access_mode="w+", dir=save_path)
    # save_content_in_txt(val_input_data, 'valid.txt', filter_bank=class_name, access_mode="w+", dir=save_path)
    # if 'ground_truth' in data_keys:
    #     ground_truth = data_dict[data_keys['ground_truth']]
    #     ground_truth.sort()
    #     train_ground_truth, valid_ground_truth = ground_truth[split:], ground_truth[split:]
    #     save_content_in_txt(train_ground_truth, 'train_gt.txt', filter_bank=class_name, access_mode="w+", dir=save_path)
    #     save_content_in_txt(valid_ground_truth, 'valid_gt.txt', filter_bank=class_name, access_mode="w+", dir=save_path)


# def string_filtering(s, filter):
#     filtered_s = {}
#     for f in filter:
#         if f in s:
#             filtered_s[f] = s
#     if len(filtered_s) > 0:
#         return filtered_s
#     else:
#         return None

# TODO: mkdir?
def save_content_in_txt(content, path, filter_bank=None, access_mode='a+', dir=None):
    # assert isinstance(content, (str, list, tuple, dict))
    # TODO: overwrite warning
    with open(path, access_mode) as fw:
        # def string_ops(s, dir, filter):
        #     pair = string_filtering(s, filter)
        #     return os.path.join(dir, list(pair.keys())[0], list(pair.values())[0])

        if isinstance(content, str):
            # if dir:
            #     content = string_ops(content, dir, filter=filter_bank)
            #     # content = os.path.join(dir, content)
            fw.write(content)
        else:
            for c in content:
                # if dir:
                #     c = string_ops(c, dir, filter=filter_bank)
                #     # c = os.path.join(dir, c)
                fw.write(f'{c}\n')


def get_path_generator_in_case_order(data_path, return_fullpath, load_format=[]):
    dir_list = get_dir_list(data_path)
    for d in dir_list:
        file_list = get_files(d, keys=load_format, return_fullpath=return_fullpath)
        for file_idx, f in enumerate(file_list):
            yield (file_idx, f)


def save_data_label_pair_in_csv(data_path, save_path=None, save_name=None, load_format='wav', return_fullpath=True):
    path_loader = get_path_generator_in_case_order(data_path, return_fullpath, load_format=load_format)
    nums, ids, labels = [], [], []
    for idx, file_and_idx in enumerate(path_loader, 1):
        file_idx, f = file_and_idx
        file_path, file_name = os.path.split(f)
        label = int(file_path[-1])
        file_name = file_name.split('.')[0]
        print(idx, file_name, label)
        nums.append(file_idx)
        ids.append(file_name)
        labels.append(label)
        
    pair_dict = {'case_index': nums,
                 'id': ids,
                 'label': labels}
    pair_df = pd.DataFrame(pair_dict)
    if not save_name:
       save_name = 'train.csv' 
    if save_path is not None:
        pair_df.to_csv(os.path.join(save_path, save_name))
    else:
        pair_df.to_csv(save_name)


def save_fileanmes_in_txt(glob_path, save_path=None, recursive=True):
    files = glob.glob(glob_path, recursive=recursive)
    if not save_path:
        save_path = 'filenames.txt'

    with open(save_path, 'w') as fw:
        for file in files:
            fw.write(f'{file}\n')
    return files


def generate_gt_csv_for_data(id_to_label, save_path):
    pass
    


if __name__ == "__main__":
    from dataset.melspec import melspec
    melspec()

    # data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\freq6_no_limit\2_21\raw_f_h_2_mono_16k'
    # save_data_label_pair_in_csv(data_path, save_name='train1.csv')
    # data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\freq6_no_limit_shift\2_21\raw_f_h_2_mono_16k'
    # save_data_label_pair_in_csv(data_path, save_name='train2.csv')
    # data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50_process\esc50_16k\esc50_16k_2'
    # save_data_label_pair_in_csv(data_path, save_name='train3.csv')


    # # ASUS snoring
    # train_wav_list_path = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\Freq2\2_21_2s_my2\train.txt'
    # test_wav_list_path = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\Freq2\2_21_2s_my2\test.txt'
    # out_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\2_21_2s_my2'
    # # out_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\2_21_2s_my2'
    # get_melspec_from_cpp(train_wav_list_path, out_dir)
    # get_melspec_from_cpp(test_wav_list_path, out_dir)


    # # ESC-50
    # # wav_list_path = r'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50_process\esc50\esc50_2\test.txt'
    # wav_list_path = r'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50_process\esc50\esc50_2\file_names.txt'
    # out_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\esc50\44100'
    # # out_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\esc50\44100'
    # get_melspec_from_cpp(wav_list_path, out_dir, sampling_rate=44100)


    # Kaggle snoring
    # wav_list_path = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\Kaggle_snoring_full\valid.txt'
    # out_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\kaggle'
    # get_melspec_from_cpp(wav_list_path, out_dir, sampling_rate=48000)


    # # ASUS new
    # redmi = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\wave_split\1658889529250_RedmiNote8\0'
    # pixel = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\wave_split\1658889531056_Pixel4XL\0'
    # iphone = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\wave_split\1658889531172_iPhone11\0'
    # sr = 16000

    # # for data_path in [redmi, pixel, iphone]:
    # for data_path in [redmi]:
    #     split = os.path.split(os.path.split(data_path)[0])[1]
    #     glob_path = os.path.join(data_path, '*.wav')
    #     save_path = os.path.join(data_path, 'filenames.txt')
    #     save_fileanmes_in_txt(glob_path, recursive=True, save_path=save_path)

    #     wav_list_path = save_path
    #     out_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp'
    #     out_dir = os.path.join(out_dir, split, str(sr))
    #     get_melspec_from_cpp(wav_list_path, out_dir, sampling_rate=sr)


