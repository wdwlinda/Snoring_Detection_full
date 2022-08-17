import os

import glob
from pydub import AudioSegment

from dataset.transformations import pcm2wave
from inference import pred, pred_from_feature
from dataset.dataset_utils import save_fileanmes_in_txt, get_melspec_from_cpp


def data_convert(data_dir, dist_dir=None):
    path, dir_name = os.path.split(data_dir)
    if not dist_dir:
        dist_dir = os.path.join(path, 'wave', dir_name)
    f_list = glob.glob(os.path.join(data_dir, '*.pcm'))
    for f in f_list:
        pcm2wave(f, sr=16000, dist_dir=dist_dir)


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


def data_preprocess(data_paths, preprocess_dir):
    sr = 16000
    split_duration = 2000
    for dataset, raw_data_path in data_paths.items():
        print(f'Processing {dataset} in {raw_data_path}')
        out_dir = os.path.join(preprocess_dir, dataset)

        # pcm files to wav files
        wav_data_path = os.path.join(out_dir, 'wave')
        data_convert(raw_data_path, wav_data_path)

        # split wav files
        # split_dirs = []
        # dataset = os.path.split(raw_data_path)[1]
        split_data_path = os.path.join(out_dir, 'wave_split')
        os.makedirs(split_data_path, exist_ok=True)
        wave_list = glob.glob(os.path.join(wav_data_path, '*.wav'))
        for wav_file in wave_list:
            wav_data_split(wav_file, split_duration=split_duration, dist_dir=split_data_path)

        # Save all the file names
        # split = os.path.split(os.path.split(split_data_path)[0])[1]
        glob_path = os.path.join(split_data_path, '*.wav')
        save_path = os.path.join(out_dir, 'filenames.txt')
        save_fileanmes_in_txt(glob_path, save_path=save_path, recursive=True)

        # Get MelSpectrogarm from C++
        wav_list_path = save_path
        # out_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp'
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
    preprocess_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess'
    _0811_data = ['Mi11_night', 'Mi11_office', 'Redmi_Note8_night', 'Samsung_Note10Plus_night']
    for dataset in _0811_data:
        src_dir = os.path.join(preprocess_dir, dataset, 'melspec', 'img', 'filenames')
        dist_dir = os.path.join(preprocess_dir, dataset, 'pred')
        # src_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\Mi11_night\melspec\img\filenames'
        # dist_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\Mi11_night\pred'
        pred_from_feature(src_dir, dist_dir)


if __name__ == '__main__':
    # main()
    pred_data()