import os
import numpy as np
import torchaudio
from pydub import AudioSegment


def Snoring_data_analysis():
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring'
    data_analysis(data_path)


def data_analysis(path):
    sample_list = os.listdir(path)
    sample_nums = len(sample_list)
    sample_length, sample_length_threshold, data_length_threshold, max_amplitude, min_amplitude = [], [], [], [], []
    min_length, max_length = [], []
    threshold = 82
    single_length = 3
    data_format = 'm4a'
    os.chdir(path)
    for idx, dir in enumerate(sample_list):
        audio_list = os.listdir(dir)
        file_length = len(audio_list)
        sample_length.append(file_length)
        if file_length > threshold:
            sample_length_threshold.append(file_length)
            data_length_threshold.append(file_length*single_length)
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
    print('Sample Number: ', len(sample_length))
    print('Effective Sample Number: ', len(sample_length_threshold))
    print(f"Total data length: {np.sum(data_length_threshold)}")
    print(f"Mean data length: {np.mean(data_length_threshold)}")
    print(f"Std data length: {np.std(data_length_threshold)}")
    print(f"Max data length: {np.max(data_length_threshold)}")
    print(f"Min data length: {np.min(data_length_threshold)}")
    print(f"Max data amplitude: {np.max(max_amplitude)}")
    print(f"Min data amplitude: {np.min(min_amplitude)}")
    print(sample_length)

if __name__ == '__main__':
    Snoring_data_analysis()