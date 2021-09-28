import os
import numpy as np
import torchaudio
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd


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
    print(f"Total data length: {np.sum(data_length_threshold)}")
    print(f"Mean data length: {np.mean(data_length_threshold)}")
    print(f"Std data length: {np.std(data_length_threshold)}")
    print(f"Max data length: {np.max(data_length_threshold)}")
    print(f"Min data length: {np.min(data_length_threshold)}")
    print(sample_length)
    if test_amplitude:
        print(f"Max data amplitude: {np.max(max_amplitude)}")
        print(f"Min data amplitude: {np.min(min_amplitude)}")

    
    # Write effective sample names to text file.
    print(sample_names)
    with open('effective_samples.txt', 'w+') as fw:
        fw.write("effective\n")
        for item in sample_names:
            fw.write(f"{item}\n")
        fw.write("\n")
        fw.write("non-effective\n")
        for item in non_effective:
            fw.write(f"{item}\n")

    # Write audio belonging information
    print(persons)
    total_person = sum(list(persons.values()))
    with open('persons.txt', 'w+') as fw:
        for k, v in persons.items():
            fw.write(f'{k}: {v}\n')
        fw.write(30*'-')
        fw.write(f'\nTotal: {total_person}')


def thrsholding(filename, threshold):
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
    thrsholding(filename, threshold=25)


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


if __name__ == '__main__':
    # Snoring_data_analysis()
    # save_audio_fig()
    peak_analysis()