import os
import logging
from pydub import AudioSegment
import numpy as np

MAJOR_CATEGORIES = [
    'Animals',
    'Natural',
    'Human-non-speech',
    'Domestic',
    'Urban',
]


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


def ecs50_process(data_path, save_path, base_folder_name='ecs50'):
    """
    Split ecs50 to 1 sec, 2 sec sample for snoring detection task
    Args:
        data_path: Input data path (str)
        save_path: Saving data path (str)
        base_folder_name: The base folder name(str)
    """
    file_list = get_files(data_path, 'wav')
    base_save_path1 = os.path.join(save_path, base_folder_name, f'{base_folder_name}_1')
    base_save_path2 = os.path.join(save_path, base_folder_name, f'{base_folder_name}_2')

    for file_idx, f in enumerate(file_list, 1):
        print(f'{file_idx}/{len(file_list)}', f)
        y = AudioSegment.from_file(f, format='wav')
        basename = os.path.basename(f)
        categories = int(basename.split('-')[-1].split('.')[0])
        major_c = MAJOR_CATEGORIES[categories//10]
        duration = int(y.duration_seconds)

        # label snoring (categories=28) as '1'
        if categories == 28:
            sub_dir = '1'
        else:
            sub_dir = '0'
        save_path1 = os.path.join(base_save_path1, major_c, sub_dir)
        save_path2 = os.path.join(base_save_path2, major_c, sub_dir)
        if not os.path.exists(save_path1): os.makedirs(save_path1)
        if not os.path.exists(save_path2): os.makedirs(save_path2)

        # split to 1 sec samples
        for idx, t in enumerate(range(duration), 1):
            clip = y[1000*t:1000*(t+1)]
            # Filter out no sound clips (all zeros)
            if np.sum(clip.get_array_of_samples()) != 0:
                clip.export(os.path.join(save_path1, basename).replace('.wav', f'-{idx}.wav'), 'wav')

        # split to 2 sec samples
        for idx, t in enumerate(range(duration-1), 1):
            clip = y[1000*t:1000*(t+2)]
            # Filter out no sound clips (all zeros)
            if np.sum(clip.get_array_of_samples()) != 0:
                clip.export(os.path.join(save_path2, basename).replace('.wav', f'-{idx}.wav'), 'wav')


if __name__ == '__main__':
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50-master'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50_process'
    ecs50_process(data_path, save_path, base_folder_name='esc50')