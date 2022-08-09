import os

import glob
from pydub import AudioSegment

from dataset.transformations import pcm2wave
from inference import pred


def data_convert():
    redmi = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\1658889529250_RedmiNote8'
    pixel = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\1658889531056_Pixel4XL'
    iphone = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\1658889531172_iPhone11'
    for data_dir in [pixel, redmi, iphone]:
        path, dir_name = os.path.split(data_dir)
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
        print(idx, start_time)
        clip = sound[start_time:start_time+split_duration]
        split_filename = f'{filename}_{idx:03d}.wav'
        clip.export(os.path.join(dist_dir, split_filename), format='wav')

def main():
    redmi = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\wave\1658889529250_RedmiNote8'
    pixel = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\wave\1658889531056_Pixel4XL'
    iphone = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\wave\1658889531172_iPhone11'

    # data_convert()

    split_dirs = []
    for src_dir in [redmi, pixel, iphone]:
        dist_dir = src_dir.replace('wave', 'wave_split')
        split_dirs.append(dist_dir)
        os.makedirs(dist_dir, exist_ok=True)
        wave_list = glob.glob(os.path.join(src_dir, '*.wav'))
        for wav_file in wave_list:
            wav_data_split(wav_file, split_duration=2000, dist_dir=dist_dir)


    pixel_split = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_0727\wave_split'
    for src_dir in split_dirs:
        dist_dir = src_dir.replace('wave_split', 'pred')
        pred(src_dir, dist_dir)


if __name__ == '__main__':
    main()