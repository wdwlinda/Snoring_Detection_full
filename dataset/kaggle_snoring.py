from pathlib import Path
import abc
from typing import Union
import json

import pandas as pd
import numpy as np

from dataset import dataset_utils
from dataset import transformations

# TODO: doc str
# TODO: split json

# TODO: general sound preprocessing
# TODO: general splitting

# TODO: pcm to wav?
# TODO: accept multiple format
# TODO: alert if upsampling
# TODO: logging

# TODO: inherit problem
# TODO: this processer is mainly for audio classification, think about the generbility


class ClsPreprocess(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_data_refs(self):
        """Get the data reference"""
        return NotImplemented

    @abc.abstractmethod
    def get_class_label(self):
        """Get the class label of dataset"""
        return NotImplemented


def continuous_split_sound(sound, target_duration):
    new_sounds = []
    split_indics = get_splitting(sound.duration_seconds, target_duration)
    # This implementation will drop the last unqualified sequence
    for idx in range(len(split_indics)-1):
        new_sounds.append(sound[split_indics[idx]*1000:split_indics[idx+1]*1000])
    return new_sounds


def get_splitting(seq_length, split_length):
    assert seq_length > split_length, ''
    indices = np.arange(0, seq_length, split_length)
    return indices


class SnoringPreprocess(ClsPreprocess):
    def __init__(self, 
                 suffix: str = 'wav', 
                 target_sr: int = 16000, 
                 target_channel: int = 1, 
                 target_duration: Union[int, float] = 2,
                 split_ratio: dict ={'train': 0.7, 'valid': 0.1, 'test': 0.2}):
        assert sum([ratio for ratio in list(split_ratio.values())]) == 1.0
        self.suffix = suffix
        self.target_sr = target_sr
        self.target_channel = target_channel
        self.target_duration = target_duration
        self.split_ratio = split_ratio

    def __call__(self, dataset_name: str, data_root: str, save_root: str) -> None:
        files = self.get_data_refs(data_root)
        preprocess_data_refs = {'input': [], 'target': [], 'process_path': []}
        for raw_path in files:
            # Load raw audio
            sound = dataset_utils.get_pydub_sound(
                str(raw_path), self.suffix, sr=self.target_sr, channels=self.target_channel)
            
            # Audio preprocessing
            new_sounds = self.sound_preprocess(sound, raw_path)

            # Save preprocessed audio
            for idx, new_sound in enumerate(new_sounds, 1):
                raw_root = raw_path.parent
                part_dirs = raw_root.relative_to(data_root)
                data_save_root = save_root.joinpath(dataset_name, 'data', part_dirs)
                data_save_root.mkdir(parents=True, exist_ok=True)
                basename = raw_path.stem
                if len(new_sounds) > 1:
                    new_basename = f'{basename}-{idx:03d}'
                else:
                    new_basename = basename
                new_save_path = data_save_root.joinpath(new_basename)
                
                new_save_path = new_save_path.with_suffix(
                    self.suffix if self.suffix.startswith('.') else f'.{self.suffix}'
                )

                preprocess_data_refs['input'].append(new_basename)
                label = self.get_class_label(new_save_path)
                preprocess_data_refs['target'].append(label)
                preprocess_data_refs['process_path'].append(new_save_path)
                new_sound.export(new_save_path, format=self.suffix)
        
        # Save reference table
        data_save_root = Path(save_root) / Path(dataset_name)
        self.ref_to_csv(preprocess_data_refs, data_save_root)
        self.ref_to_text(preprocess_data_refs, data_save_root)

        # Save dataset split (COCO annotation)
        self.ref_to_coco(preprocess_data_refs, data_save_root)
   
    def get_data_refs(self, data_root: str) -> dict:
        files = list(data_root.rglob(f'*.{self.suffix}'))
        return files

    def get_class_label(self, save_path):
        return int(save_path.parent.name)

    def ref_to_csv(self, data_refs: dict, data_save_root: str) -> None:
        ref_to_save = {}
        ref_to_save['input'] = data_refs['input']
        ref_to_save['target'] = data_refs['target']
        df = pd.DataFrame(ref_to_save)
        df.to_csv(data_save_root.joinpath('data.csv'))

    def ref_to_text(self, data_refs: dict, data_save_root: str) -> None:
        with open(data_save_root.joinpath('filenames.txt'), 'w') as fw:
            for file in data_refs['process_path']:
                fw.write(f'{file}\n')

    def split_data_reference(self, data_refs: dict) -> dict:
        split_data_refs = {}
        labels = np.unique(data_refs['target'])
        def assign_split_indices(split_data_refs, split_name, label, label_idx):
            if split_name not in split_data_refs:
                split_data_refs[split_name] = {}
            split_data_refs[split_name][label] = label_idx

        for label in labels:
            label_idx = np.asarray(data_refs['target']==label).nonzero()[0]
            label_num = len(label_idx)
            for idx, (split_name, split_ratio) in enumerate(self.split_ratio.items()):
                if idx == len(self.split_ratio)-1:
                    break
                split_label_num = int(split_ratio*label_num)
                split_idx = np.random.choice(label_idx, split_label_num, replace=False)
                assign_split_indices(split_data_refs, split_name, label, label_idx)

                # Remove selected indices
                label_idx = np.setdiff1d(label_idx, split_idx, assume_unique=True)
            
            # Assign the last name of data split (assign remain indices)
            assign_split_indices(split_data_refs, split_name, label, label_idx)
        return split_data_refs

    def ref_to_coco(self, data_refs: dict, data_save_root: str) -> None:
        split_data_refs = self.split_data_reference(data_refs)

        info = []
        annotation = []
        for split_name, split_labels in split_data_refs.items():
            data = []
            coco_data = {}
            for label, label_indices in split_labels.items():
                for label_idx in label_indices:
                    sample = {
                        'id': int(label_idx),
                        'path': str(data_refs['process_path'][label_idx]),
                        'file_name': data_refs['process_path'][label_idx].name
                    }
                    data.append(sample)

            coco_data['waveform'] = data
            json_path = data_save_root.joinpath(f'{split_name}.json')
            with open(json_path, 'wt', encoding='UTF-8') as jsonfile:
                json.dump(coco_data, jsonfile, ensure_ascii=True, indent=4)

    def sound_preprocess(self, sound, *args, **kwargs):
        """ 
        Simple preprocessing for input datato have unify duration, channel, sr.
        Apply spitting if the raw data is to long, and apply repeat, padding to
        extend thedata length.
        """
        # TODO: overlapping
        # TODO: drop_Last or keep_last for splitting? (roll_back, padding , overlapping)
        # TODO: repeat, padding for extend
        # TODO: modulize
        if sound.duration_seconds < self.target_duration:
            # repeat
            repeat_ratio = int(self.target_duration//sound.duration_seconds)
            new_sound = sound*repeat_ratio + sound
            new_sounds = [new_sound[:1000*self.target_duration]]
        elif sound.duration_seconds > self.target_duration:
            # continuous spitting
            new_sounds = continuous_split_sound(sound, self.target_duration)
        else:
            new_sounds = [sound]
        return new_sounds

    def to_coco(self):
        pass


class AssignLabelPreprocess(SnoringPreprocess):
    def __init__(self, 
                 assign_label: int,
                 suffix: str = 'wav', 
                 target_sr: int = 16000, 
                 target_channel: int = 1, 
                 target_duration: Union[int, float] = 2):

        super().__init__(suffix, target_sr, target_channel, target_duration)
        self.assign_label = assign_label

    def get_class_label(self, *args):
        return self.assign_label


class ESC50Preprocess(SnoringPreprocess):
    def __init__(self, 
                 suffix: str = 'wav', 
                 target_sr: int = 16000, 
                 target_channel: int = 1, 
                 target_duration: Union[int, float] = 2):

        super().__init__(suffix, target_sr, target_channel, target_duration)

    def get_class_label(self, save_path):
        filename = save_path.stem
        esc50_label = filename.split('-')[-1]
        if esc50_label == '28':
            return 1
        else:
            return 0

    def sound_preprocess(self, sound, *args, **kwargs):
        raw_path = args[0]
        new_sounds = continuous_split_sound(sound, self.target_duration)
        if raw_path.stem.split('-')[-1] == '28':
            wav_mean = []
            for new_sound in new_sounds:
                waveform = np.array(new_sound.get_array_of_samples(), np.float32)
                wav_mean.append(np.mean(np.abs(waveform)))
            new_sounds = [new_sounds[np.argmax(wav_mean)]]
        return new_sounds


def run_class():
    save_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\test\web_snoring_pre')

    # processer = SnoringPreprocess()
    # dataset_name = 'kaggle_snoring'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Kaggle_snoring\Snoring Dataset')
    # processer(dataset_name, data_root, save_root)

    processer = AssignLabelPreprocess(assign_label=1)
    dataset_name = 'web_snoring'
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\test\web_snoring')
    processer(dataset_name, data_root, save_root)

    # processer = ESC50Preprocess()
    # dataset_name = 'ESC50'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50-master\audio')
    # processer(dataset_name, data_root, save_root)

    # processer = AssignLabelPreprocess(assign_label=0)
    # dataset_name = 'Mi11_office'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\ASUS_snoring\ASUS_snoring_0811\20220811_testing\Mi11_office_wav')
    # processer(dataset_name, data_root, save_root)

if __name__ == '__main__':
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset')
    run_class()