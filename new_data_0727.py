import os

import glob
from pydub import AudioSegment
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

from dataset.transformations import pcm2wave
from inference import pred, pred_from_feature
from dataset.dataset_utils import save_fileanmes_in_txt, get_melspec_from_cpp


def pcm_data_convert(data_dir, sr=16000, dist_dir=None):
    path, dir_name = os.path.split(data_dir)
    if not dist_dir:
        dist_dir = os.path.join(path, 'wave', dir_name)
    f_list = glob.glob(os.path.join(data_dir, '*.pcm'))
    for f in f_list:
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


def data_preprocess(data_paths, preprocess_dir):
    sr = 16000
    split_duration = 2000
    for dataset, raw_data_path in data_paths.items():
        print(f'Processing {dataset} in {raw_data_path}')
        out_dir = os.path.join(preprocess_dir, dataset)

        # pcm files to wav files
        wav_data_path = os.path.join(out_dir, 'wave')
        pcm_data_convert(raw_data_path, wav_data_path=wav_data_path)

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
    # 0727 data
    redmi = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\wave\1658889529250_RedmiNote8'
    pixel = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\wave\1658889531056_Pixel4XL'
    iphone = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\wave\1658889531172_iPhone11'

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
    # data_paths = _0727_data.update(_0811_data)
    data_paths = _0811_data
    preprocess_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess'
    data_preprocess(data_paths, preprocess_dir)


def pred_data():
    total_data_info = {}

    preprocess_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp'
    _0727_data = ['1658889529250_RedmiNote8', '1658889531056_Pixel4XL', '1658889531172_iPhone11']
    for dataset in _0727_data:
        src_dir = os.path.join(preprocess_dir, dataset, '16000', 'img', 'filenames')
        dist_dir = os.path.join(preprocess_dir, dataset, '16000', 'pred')
        gt_dir = os.path.join(preprocess_dir, dataset, '16000', 'filenames.csv')
        total_data_info[dataset] = {'src': src_dir, 'dist': dist_dir, 'gt': gt_dir}

    # preprocess_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess'
    # _0811_data = ['Mi11_night', 'Mi11_office', 'Redmi_Note8_night', 'Samsung_Note10Plus_night']
    # for dataset in _0811_data:
    #     src_dir = os.path.join(preprocess_dir, dataset, 'melspec', 'img', 'filenames')
    #     dist_dir = os.path.join(preprocess_dir, dataset, 'pred')
    #     gt_dir = os.path.join(preprocess_dir, dataset, 'melspec', 'filenames.csv')
    #     total_data_info[dataset] = {'src': src_dir, 'dist': dist_dir, 'gt': gt_dir}

    # test = {'Test': {
    #     'src': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\temp_test',
    #     'dist': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\temp_test',
    #     'gt': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\2_21_2s_my2\test.csv',
    #     }
    # }
    # total_data_info.update(test)

    # esc50 = {'ESC-50': {
    #     'src': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\esc50\44100\img\file_names',
    #     'dist': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\esc50\44100\pred',
    #     'gt': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\esc50\44100\file_names.csv',
    #     }
    # }
    # total_data_info.update(esc50)

    # asus_snoring = {'ASUS_snoring': {
    #     'src': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\2_21_2s_my2\img\test',
    #     'dist': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\2_21_2s_my2\pred',
    #     'gt': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\2_21_2s_my2\test.csv',
    #     }
    # }
    # total_data_info.update(asus_snoring)

    total_confidence = {}
    data_names = []
    for dataset, data_info in total_data_info.items():
        data_names.append(dataset)
        src_dir = data_info['src']
        dist_dir = data_info['dist']
        prediction = pred_from_feature(src_dir, dist_dir)

        y_true, y_pred, confidence = [], [], []
    
        # df = pd.read_csv(data_info['gt'])
        # for index, sample_gt in df.iterrows():
        #     if prediction.get(sample_gt['input'], None):
        #         true_val = sample_gt['label']
        #         y_true.append(true_val)
        #         y_pred.append(prediction[sample_gt['input']]['pred'])
        #         confidence.append(sample['prob'][0, true_val])

        true_val = 0
        for index, sample in prediction.items():
            y_pred.append(sample['pred'])
            y_true.append(true_val)
            confidence.append(sample['prob'][0, true_val])

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        with open(os.path.join(data_info['dist'], 'result.txt'), 'w') as fw:
            fw.write(f'Precision {precision:.4f}\n')
            fw.write(f'Recall {recall:.4f}\n')
            fw.write(f'Accuracy {acc:.4f}\n')
        # print(acc)

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(os.path.join(data_info['dist'], 'cm.png'))

        total_confidence[dataset] = confidence

    fig, ax = plt.subplots(1, 1)
    for idx, (dataset, confidence) in enumerate(total_confidence.items(), 1):
        ax.scatter(np.ones_like(confidence, dtype=np.int32)*idx, confidence, s=0.5)
    ax.set_xlabel('dataset')
    ax.set_ylabel('probability')
    ax.set_title('Prediction confidence comparision')
    ax.set_xticks(np.arange(1, len(total_confidence)+1), data_names)
    ax.plot([1, len(total_confidence)+1], [0.5, 0.5], 'k--')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(os.path.join(data_info['dist'], 'confidence_comp.png'))


if __name__ == '__main__':
    # main()
    pred_data()