import os

import glob
from pydub import AudioSegment
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

from dataset.transformations import pcm2wave
from inference import pred, pred_from_feature, test, plot_confusion_matrix
from dataset.dataset_utils import save_fileanmes_in_txt, get_melspec_from_cpp


def pcm_data_convert(data_dir, sr=16000, dist_dir=None):
    path, dir_name = os.path.split(data_dir)
    if not dist_dir:
        dist_dir = os.path.join(path, 'wave', dir_name)
    
    # process pcm
    pcm_list = glob.glob(os.path.join(data_dir, '*.pcm'))
    for f in pcm_list:
        pcm2wave(f, sr, dist_dir=dist_dir)


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

    for idx, start_time in enumerate(range(0, sound_duration, split_duration)):
        end_time = start_time + split_duration
        if end_time > sound_duration:
            start_time = start_time - (end_time - sound_duration)
        # print(idx, start_time)
        clip = sound[start_time:start_time+split_duration]
        clip_data = np.array(clip.get_array_of_samples(), np.float32)
        split_filename = f'{filename}-{idx:03d}.wav'
        clip.export(os.path.join(dist_dir, split_filename), format='wav')


class Waveform_to_Clips():
    def __init__(self, split_duration, sample_rate, source, overlapping):
        assert source in ['pcm', 'wav']
        assert overlapping >= 0.0 and overlapping < 1.0 # overlapping level
        self.split_duration = split_duration
        self.sample_rate = sample_rate
        self.source = source
        self.overlapping = overlapping

    def process(self, dataset_name, data_pair, out_dir):
        assert isinstance(data_pair, (list, dict)), \
            'The data_pair should be the format like [path1 (str), path2, ...pathN] or \
            {path1 (str): label1 (int), path2: label2, ... pathN: labelN'
        # print(f'Processing {in_dir}')

        # pcm files to wav files
        if self.source == 'pcm':
            wav_data_path = os.path.join(out_dir, 'wav')
            pcm_data_convert(in_dir, sr=self.sample_rate, wav_data_path=wav_data_path)
        else:
            wav_data_path = in_dir

        # split wav files
        # TODO: overlapping
        # TODO: option to save the middle processing data
        clip_data_path = os.path.join(out_dir, 'wave_split')
        os.makedirs(clip_data_path, exist_ok=True)
        wave_list = glob.glob(os.path.join(wav_data_path, '*.wav'))
        for wav_file in wave_list:
            wav_data_split(wav_file, split_duration=self.split_duration, dist_dir=clip_data_path)

        # TODO: CSV
        # Save all the file names
        glob_path = os.path.join(clip_data_path, '*.wav')
        save_path = os.path.join(out_dir, 'filenames.txt')
        save_fileanmes_in_txt(glob_path, save_path=save_path, recursive=True)


def data_preprocess(data_paths, preprocess_dir, source, sr=16000, split_duration=2000):
    for dataset, raw_data_path in data_paths.items():
        print(f'Processing {dataset} in {raw_data_path}')
        out_dir = os.path.join(preprocess_dir, dataset)

        # pcm files to wav files
        if source == 'pcm':
            wav_data_path = os.path.join(out_dir, 'wave')
            pcm_data_convert(raw_data_path, dist_dir=wav_data_path)
        elif source in ['wav', 'wave']:
            wav_data_path = raw_data_path

        # split wav files
        split_data_path = os.path.join(out_dir, 'wave_split')
        os.makedirs(split_data_path, exist_ok=True)
        wave_list = glob.glob(os.path.join(wav_data_path, '*.wav'))
        for wav_file in wave_list:
            wav_data_split(wav_file, split_duration=split_duration, dist_dir=split_data_path)

        # Save all the file names
        glob_path = os.path.join(split_data_path, '*.wav')
        save_path = os.path.join(out_dir, 'filenames.txt')
        save_fileanmes_in_txt(glob_path, save_path=save_path, recursive=True)

        # Get MelSpectrogarm from C++
        wav_list_path = save_path
        melspec_dir = os.path.join(out_dir, 'melspec')
        get_melspec_from_cpp(wav_list_path, melspec_dir, sampling_rate=sr)


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

    data_paths = {}
    # data_paths.update(_0727_data)
    # data_paths.update(_0811_data)
    data_paths.update(_0908_data)
    data_paths.update(_0908_data_2)
    data_preprocess(data_paths, preprocess_dir, source='pcm')

    # data_paths = _0811_data
    


if __name__ == '__main__':
    main()
    # pred_data()