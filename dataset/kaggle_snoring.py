
from email.mime import base
from pathlib import Path
from abc import ABC
from typing import Union

import pandas as pd
import numpy as np

from dataset import dataset_utils

# TODO: doc str
# TODO: split json

# TODO: splitting might return not only one sound clip
# TODO: general sound preprocessing
# TODO: accept multiple format
# TODO: alert if upsampling
# TODO: logging


def get_splitting(seq_length, split_length):
    assert seq_length > split_length, ''
    indices = np.arange(0, seq_length, split_length)
    return indices

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
        # FIXME: Incorrect data_refs
        data_refs = self.get_data_refs(dataset_name, data_root, save_root)
        for input_path, save_path in zip(data_refs['input_path'], data_refs['save_path']):
            # Load raw audio
            sound = dataset_utils.get_pydub_sound(
                str(input_path), self.suffix, sr=self.target_sr, channels=self.target_channel)
            
            # Audio preprocessing
            new_sounds = self.sound_preprocess(sound)

            # Save preprocessed audio
            for idx, new_sound in enumerate(new_sounds, 1):
                if len(new_sounds) > 1:
                    basename = save_path.stem
                    new_basename = f'{basename}_{idx:03d}'
                    new_save_path = save_path.with_stem(new_basename)
                else:
                    new_save_path = save_path
                new_sound.export(new_save_path, format=self.suffix)
        
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
            # FIXME: 
            ref['target'].append(1)
            # ref['target'].append(int(label))
            ref['input_path'].append(f)
            ref['save_path'].append(save_dir / save_name)
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
        # TODO: overlapping
        # TODO: drop_Last or keep_last for splitting? (roll_back, padding , overlapping)
        # TODO: repeat, padding for extend
        if sound.duration_seconds < self.target_duration:
            # repeat
            repeat_ratio = int(self.target_duration//sound.duration_seconds)
            new_sound = sound*repeat_ratio + sound
            new_sounds = [new_sound[:1000*self.target_duration]]
        else:
            # continuous spitting
            new_sounds = []
            split_indics = get_splitting(sound.duration_seconds, self.target_duration)
            # This implementation will drop the last unqualified sequence
            for idx in range(len(split_indics)-1):
                new_sounds.append(sound[split_indics[idx]*1000:split_indics[idx+1]*1000])
        return new_sounds

    def to_coco(self):
        pass


def run_class():
    processer = Preprocesser()

    # dataset_name = 'kaggle_snoring'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset')
    # save_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring')
    # processer(dataset_name, data_root, save_root)

    dataset_name = 'web_snoring'
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring\yt_snoring')
    save_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring')
    processer(dataset_name, data_root, save_root)


if __name__ == '__main__':
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset')
    run_class()