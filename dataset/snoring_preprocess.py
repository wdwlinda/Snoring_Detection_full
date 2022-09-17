# Process snoring raw data to the format can be used for training.
import os
import glob

import numpy as np
import pandas as pd
import wave
from pydub import AudioSegment


# TODO: drop_last args for splitting process
# TODO: structure
# TODO: docstring
# TODO: consider input labeling
# TODO: splitting?
# TODO: data_refs to path_refs is slow


def wav_data_split(wav_file, split_duration, dist_dir=None):
    """AI is creating summary for wav_data_split

    Args:
        wav_file (str): 
        split_duration (ms): Splitting duration in minisecond
        dist_dir (str): 
    """
    sound = AudioSegment.from_file(wav_file, 'wav')
    src_dir, filename = os.path.split(wav_file)
    filename= filename[:-4]
    sound_duration = int(sound.duration_seconds*1000)
    if not dist_dir:
        dist_dir = src_dir

    wav_clips = []
    for idx, start_time in enumerate(range(0, sound_duration, split_duration)):
        end_time = start_time + split_duration
        if end_time > sound_duration:
            start_time = start_time - (end_time - sound_duration)
        # print(idx, start_time)
        clip = sound[start_time:start_time+split_duration]
        clip_data = np.array(clip.get_array_of_samples(), np.float32)
        split_filename = f'{filename}-{idx:03d}.wav'
        save_path = os.path.join(dist_dir, split_filename)
        clip.export(save_path, format='wav')
        wav_clips.append(save_path)

    return wav_clips



def pcm2wave(pcm_path, sr=16000, dist_dir=None):
    with open(pcm_path, 'rb') as pcmfile:
        pcmdata = pcmfile.read()

    pcm_dir, pcm_file = os.path.split(pcm_path)
    if dist_dir is not None:
        os.makedirs(dist_dir, exist_ok=True)
    else:
        dist_dir = pcm_dir

    with wave.open(os.path.join(dist_dir, pcm_file.replace('.pcm', '.wav')), 'wb') as wavfile:
        # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        wavfile.setparams((1, 2, sr, 0, 'NONE', 'NONE')) 
        wavfile.writeframes(pcmdata)


def get_path_refs(data_root, data_refs, suffix):
    files = glob.glob(os.path.join(data_root, '**', f'**.{suffix}'), recursive=True)
    path_refs = {}
    for path in files:
        filename = path_process(path)
        df_row = data_refs.loc[data_refs['id'] == filename]
        label = df_row['label'].values
        path_refs[path] = label[0]
    return path_refs


def get_refs_by_assign(data_root, assign_label, suffix='wav', fullpath=False, keep_suffix=False):
    files = glob.glob(os.path.join(data_root, '**', f'**.{suffix}'), recursive=True)
    data_refs = {}
    for f in files:
        f = path_process(f, fullpath, keep_suffix)
        data_refs[f] = assign_label
    return data_refs


def get_refs_by_dirname(data_root, suffix='wav', fullpath=False, keep_suffix=False):
    files = glob.glob(os.path.join(data_root, '**', f'**.{suffix}'), recursive=True)
    data_refs = {}
    for f in files:
        dirname = os.path.split(os.path.split(f)[0])[1]
        try:
            label = int(dirname)
            f = path_process(f, fullpath, keep_suffix)
            data_refs[f] = label
        except ValueError:
            print(f'Directory "{dirname}" cannot convert to Int')
    return data_refs


def path_process(path, fullpath=False, keep_suffix=False):
    if not fullpath:
        path = os.path.split(path)[1]
        if not keep_suffix:
            path = path.split('.')[0]
    return path


def sound_to_wav_clips(data_root, save_root, recursive=True, sr=16000, split_duration=2000):
    # get wave list, pcm list
    wav_list = glob.glob(os.path.join(data_root, '**.wav'), recursive=recursive)
    pcm_list = glob.glob(os.path.join(data_root, '**.pcm'), recursive=recursive)
    
    # convert all pcm to wav
    wav_dir = os.path.join(save_root, 'wave')
    os.makedirs(wav_dir, exist_ok=True)
    for pcm_f in pcm_list:
        pcm_name = os.path.split(pcm_f)[1][:-4]
        wav_f = os.path.join(wav_dir, f'{pcm_name}.wav')
        pcm2wave(pcm_f, sr, dist_dir=wav_dir)
        wav_list.append(wav_f)

    # split all the files in wav list
    wav_split_dir = os.path.join(save_root, 'wave_split')
    os.makedirs(wav_split_dir, exist_ok=True)
    total_wav_clips = []
    for wav_f in wav_list:
        wav_clips = wav_data_split(wav_f, split_duration, dist_dir=wav_split_dir)
        total_wav_clips.extend(wav_clips)

    return total_wav_clips


def get_snoring_data_references(snoring_dataset_roots, save_root, split_datasets):
    """"""
    # get data reference
    for data_name, data_root in snoring_dataset_roots.items():
        print(data_name)
        data_save_root = os.path.join(save_root, data_name)
        os.makedirs(data_save_root, exist_ok=True)
        if data_name in split_datasets:
            wav_clip_paths = sound_to_wav_clips(data_root, data_save_root)
            wav_split_root = os.path.join(data_save_root, 'wave_split')
            data_refs = get_refs_by_assign(wav_split_root, assign_label=0, suffix='wav', fullpath=False)
        else:
            data_refs = get_refs_by_dirname(data_root, suffix='wav')

        # save data reference in csv
        df_path = os.path.join(data_save_root, 'data.csv')
        save_refs_in_csv(data_refs, df_path)


def save_refs_in_csv(data, save_root):
    data_csv = {
        'id': [], 
        'label': []
    }
    for k, v in data.items():
        data_csv['id'].append(k)
        data_csv['label'].append(v)

    df = pd.DataFrame.from_dict(data_csv)
    df.to_csv(save_root)


def main():
    # ASUS_snoring
    ASUS_snoring = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\freq6_no_limit\2_21\raw_f_h_2_mono_16k'

    # ESC50
    ESC50 = r'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50_process\esc50\esc50_2'

    # 0727 data
    redmi = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\1658889529250_RedmiNote8'
    pixel = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\1658889531056_Pixel4XL'
    iphone = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\1658889531172_iPhone11'

    # 0811 data
    Mi11_night = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0811\20220811_testing\Mi11_night\1660060671210_thres0.55'
    Mi11_office = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0811\20220811_testing\Mi11_office\1660109492745'
    Redmi_Note8_night = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0811\20220811_testing\Redmi_Note8_night\1659630126152_thres0.55'
    Samsung_Note10Plus_night = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0811\20220811_testing\Samsung_Note10Plus_night\1659630157589_thres0.55'

    ASUS_snoring_data = {
        'ASUS_snoring': ASUS_snoring
    }

    ESC50_data = {
        'ESC50': ESC50
    }

    Kaggle_snoring_data = {
        'Kaggle_snoring': r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset'
    }

    _0727_data = {
        'redmi': redmi, 
        'pixel': pixel, 
        'iphone': iphone
    }

    _0811_data = {
        'Mi11_night': Mi11_night, 
        'Mi11_office': Mi11_office, 
        'Redmi_Note8_night': Redmi_Note8_night, 
        'Samsung_Note10Plus_night': Samsung_Note10Plus_night
    }
    
    _0908_data = {
        'pixel_0908': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0908\1662605121213_pixel4xl',
        'iphone11_0908': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0908\1662604080863_iphone11'
    }

    _0908_data_2 = {
        'pixel_0908_2': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0908_2\1662633178747_pixel4xl',
        'iphone11_0908_2': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0908_2\1662632007621_iphone11_2'
    }
    
    # data_paths = _0727_data.update(_0811_data)
    preprocess_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess'
    save_root = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\pp'

    data_paths = {}
    data_paths.update(_0727_data)
    data_paths.update(_0811_data)
    data_paths.update(ASUS_snoring_data)
    data_paths.update(ESC50_data)
    data_paths.update(Kaggle_snoring_data)
    # data_paths.update(_0908_data)
    # data_paths.update(_0908_data_2)
    split_datasets = list(_0727_data.keys()) + list(_0811_data.keys())
    # split_datasets = ['pixel_0908', 'iphone11_0908']
    get_snoring_data_references(data_paths, save_root, split_datasets=split_datasets)
    # data_paths = _0811_data
    


if __name__ == '__main__':
    main()
    r = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\pp\pixel'
    f = glob.glob(os.path.join(r, '**', '*.csv'))
    print(f)