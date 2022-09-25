import os
import logging
import glob

from pydub import AudioSegment
import numpy as np

MAJOR_CATEGORIES = [
    'Animals',
    'Natural',
    'Human-non-speech',
    'Domestic',
    'Urban',
]


def ecs50_process(data_path, save_path, base_folder_name='ecs50'):
    """
    Split ecs50 to 1 sec, 2 sec sample for snoring detection task
    Args:
        data_path: Input data path (str)
        save_path: Saving data path (str)
        base_folder_name: The base folder name(str)
    """
    file_list = glob.glob(os.path.join(data_path, '**', '*.wav'), recursive=True)
    base_save_path1 = os.path.join(save_path, base_folder_name, f'{base_folder_name}_1')
    base_save_path2 = os.path.join(save_path, base_folder_name, f'{base_folder_name}_2')

    for file_idx, f in enumerate(file_list, 1):
        print(f'{file_idx}/{len(file_list)}', f)
        y = AudioSegment.from_file(f, format='wav')
        y = y.set_frame_rate(16000)
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
            if np.sum(np.float32(clip.get_array_of_samples())) != 0:
                clip.export(os.path.join(save_path1, basename).replace('.wav', f'-{idx}.wav'), 'wav')

        # split to 2 sec samples
        for idx, t in enumerate(range(duration-1), 1):
            clip = y[1000*t:1000*(t+2)]
            # Filter out no sound clips (all zeros)
            # if basename == rf'4-188595-A-29.wav':
            #     aa = np.float32(clip.get_array_of_samples())
            #     bb = np.sum(aa)
            #     print(10)
            #     print(np.max(aa), np.min(aa), np.sum(aa))
            if np.sum(np.float32(clip.get_array_of_samples())) != 0:
                clip.export(os.path.join(save_path2, basename).replace('.wav', f'-{idx}.wav'), 'wav')


if __name__ == '__main__':
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50-master'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50_process'
    ecs50_process(data_path, save_path, base_folder_name='esc50_16k')