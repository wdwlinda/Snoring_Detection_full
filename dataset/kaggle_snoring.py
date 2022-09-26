
from pathlib import Path
from abc import ABC
from typing import Union

import pandas as pd

from dataset import dataset_utils

# TODO: splitting might return not only one sound clip
# TODO: doc str
# TODO: type hints (str == Path?)
# TODO: split json
# TODO: accept multiple format

# TODO: alert if upsampling
# TODO: logging


class Preprocesser():
    def __init__(self, 
                 suffix: str = 'wav', 
                 target_sr: int = 16000, 
                 target_channel: int = 1, 
                 target_duration: Union[int, float] = 2):
        self.suffix = suffix
        self.target_sr = target_sr
        self.target_channel = target_channel
        self.target_duration = target_duration

    def __call__(self, dataset_name: str, data_root: str, save_root: str) -> None:
        data_refs = self.get_data_refs(dataset_name, data_root, save_root)
        for input_path, save_path in zip(data_refs['input_path'], data_refs['save_path']):
            sound = dataset_utils.get_pydub_sound(
                input_path, self.suffix, sr=self.target_sr, channels=self.target_channel)
            
            new_sound = self.sound_preprocess(sound)

            new_sound.export(save_path, format=self.suffix)
        
        # Save reference table
        data_save_root = Path(save_root) / Path(dataset_name)
        self.ref_to_csv(data_refs, data_save_root)
        self.ref_to_text(data_refs, data_save_root)
   
    def get_data_refs(self, dataset_name: str, data_root: str, save_root: str) -> dict:
        # Get files
        data_save_root = Path(save_root) / Path(dataset_name)
        files = list(data_root.rglob(f'*.{self.suffix}'))

        # Preprocessing
        ref = {'input': [], 'target': [], 'input_path': [], 'save_path': []}
        for f in files:
            label = f.parent.name # TODO:
            save_dir = data_save_root.joinpath(label)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_name = f.name

            ref['input'].append(save_name)
            ref['target'].append(int(label))
            ref['input_path'].append(str(f))
            ref['save_path'].append(str(save_dir / save_name))
        return ref

    def ref_to_csv(self, data_refs: dict, data_save_root: str):
        ref_to_save = {}
        ref_to_save['input'] = data_refs['input']
        ref_to_save['target'] = data_refs['target']
        df = pd.DataFrame(ref_to_save)
        df.to_csv(data_save_root.joinpath('data.csv'))

    def ref_to_text(self, data_refs: dict, data_save_root: str):
        with open(data_save_root.joinpath('filenames.txt'), 'w') as fw:
            for file in data_refs['save_path']:
                fw.write(f'{file}\n')

    def sound_preprocess(self, sound):
        """ 
        Simple preprocessing for input datato have unify duration, channel, sr.
        Apply spitting if the raw data is to long, and apply repeat, padding to
        extend thedata length.
        """
        if sound.duration_seconds < self.target_duration:
            # repeat
            # padding
            new_sound = sound * 2
            pass
        else:
            # continuous spitting
            pass
        return new_sound


def run(data_root):
    # args
    dataset_name = 'kaggle_snoring'
    save_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring')
    data_save_root = Path(save_root) / Path(dataset_name)
    target_sr = 16000
    target_channel = 1
    target_duration = 2
    suffix = 'wav'

    # Get files
    files = list(data_root.rglob(f'*.{suffix}'))
    files = files[1:] # 0_0.wav cannot be read

    # Preprocessing
    ref = {'input': [], 'target': []}
    for f in files:
        sound = dataset_utils.get_pydub_sound(
            str(f), suffix, sr=target_sr, channels=target_channel)
        repeat_sound = sound * 2 # TODO:

        label = f.parent.name # TODO:
        save_dir = data_save_root.joinpath(label)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = f.name

        repeat_sound.export(str(save_dir / save_name), format=suffix)
        ref['input'].append(save_name)
        ref['target'].append(int(label))

    # Save reference table
    df = pd.DataFrame(ref)
    df.to_csv(data_save_root.joinpath('data.csv'))

    # to_txt
    # Save split in json


def run_class():
    processer = Preprocesser()
    dataset_name = 'kaggle_snoring'
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset')
    save_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring')
    processer(dataset_name, data_root, save_root)


if __name__ == '__main__':
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset')
    # run(data_root)
    run_class()