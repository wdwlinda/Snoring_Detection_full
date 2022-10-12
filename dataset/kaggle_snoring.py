from pathlib import Path
import abc
from typing import Union
import json
from datetime import date
import random

import pandas as pd
import numpy as np
from pydub import AudioSegment

from dataset import dataset_utils
from dataset import transformations

# TODO: doc str
# TODO: general coco part
# TODO: duration, sr, channel for info?

# TODO: general sound preprocessing

# TODO: pcm to wav?
# TODO: accept multiple format
# TODO: alert if upsampling
# TODO: logging

# TODO: inherit problem
# TODO: this processer is mainly for audio classification, think about the generbility
# TODO: Non-label class
# TODO: set up input format internel and output_suffix
"""
Current functions
    - Simple preprocessing for duration, channel, sr. (Align input waveform format)
    - Simple label generating (from directory name, from single label asign)
    - Generate coco anntations
TODO:
    - Data format conversion (pcm, m4a)
    - Label generated from csv
    - DVC
"""


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
                 out_suffix: str = 'wav', 
                 target_sr: int = 16000, 
                 target_channel: int = 1, 
                 target_duration: Union[int, float] = 2,
                 split_ratio: dict = {'train': 0.7, 'valid': 0.1, 'test': 0.2},
                 sample_ratio: float = 1.0):
        assert sum([ratio for ratio in list(split_ratio.values())]) == 1.0
        self.suffix = suffix
        self.out_suffix = out_suffix
        self.target_sr = target_sr
        self.target_channel = target_channel
        self.target_duration = target_duration
        self.split_ratio = split_ratio
        self.sample_ratio = sample_ratio

    def __call__(self, dataset_name: str, data_root: str, save_root: str) -> None:
        self.dataset_name = dataset_name
        files = self.get_data_refs(data_root)
        preprocess_data_refs = {'input': [], 'target': [], 'process_path': []}
        print(f'Preprocessing -- {dataset_name}')
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
                    self.out_suffix if self.out_suffix.startswith('.') else f'.{self.out_suffix}'
                )

                preprocess_data_refs['input'].append(new_basename)
                label = self.get_class_label(new_save_path)
                preprocess_data_refs['target'].append(label)
                preprocess_data_refs['process_path'].append(new_save_path)
                new_sound.export(new_save_path, format=self.out_suffix)
        
        # Save reference table
        data_save_root = Path(save_root) / Path(dataset_name)
        self.ref_to_csv(preprocess_data_refs, data_save_root)
        self.ref_to_text(preprocess_data_refs, data_save_root)

        # Save dataset split (COCO annotation)
        self.ref_to_coco(preprocess_data_refs, data_save_root)
   
    def get_data_refs(self, data_root: str) -> list:
        files = list(data_root.rglob(f'*.{self.suffix}'))
        files = np.random.choice(files, int(len(files)*self.sample_ratio), replace=False)
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
        """
        This function helps to split the data reference with initilize split_ratio.

        Args:
            data_refs (dict): input data filename, local path and target

        Returns:
            split_data_refs (dict): splitted data reference
        """
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
                assign_split_indices(split_data_refs, split_name, label, split_idx)

                # Remove selected indices
                label_idx = np.setdiff1d(label_idx, split_idx, assume_unique=True)
            
            # Assign the last name of data split (assign remain indices)
            assign_split_indices(split_data_refs, split_name, label, label_idx)
        return split_data_refs

    def ref_to_coco(self, data_refs: dict, data_save_root: str) -> None:
        split_data_refs = self.split_data_reference(data_refs)

        info = {
            "description": f'{self.dataset_name}',
            "url": "",
            "version": "1.0",
            "contributor": "ASUS_DIT",
            "date_created": f'{date.today().isoformat()}',
            "dataset_number": len(data_refs['input'])
        }
        cat_ids = [{'id': 1, 'name': 'snoring'}]
        for split_name, split_labels in split_data_refs.items():
            data = []
            annots = []
            coco_data = {}
            for label, label_indices in split_labels.items():
                for label_idx in label_indices:
                    sample = {
                        'id': int(label_idx),
                        'path': str(data_refs['process_path'][label_idx]),
                        'file_name': data_refs['process_path'][label_idx].name
                    }
                    sample_annot = {
                        'image_id': int(label_idx),
                        'id': int(label_idx),
                        'category_id': int(label)
                    }
                    data.append(sample)
                    annots.append(sample_annot)

            info['split_number'] = len(data)
            coco_data['info'] = info
            coco_data['waveform'] = data
            coco_data['categories'] = cat_ids
            coco_data['annotations'] = annots
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
                 out_suffix: str = 'wav', 
                 target_sr: int = 16000, 
                 target_channel: int = 1, 
                 target_duration: Union[int, float] = 2,
                 sample_ratio: float = 1.0):

        super().__init__(
            suffix, out_suffix, target_sr, target_channel, target_duration, 
            sample_ratio=sample_ratio)
        self.assign_label = assign_label

    def get_class_label(self, *args):
        return self.assign_label


class KagglePadPreprocess(SnoringPreprocess):
    def __init__(self, 
                 suffix: str = 'wav', 
                 out_suffix: str = 'wav', 
                 target_sr: int = 16000, 
                 target_channel: int = 1, 
                 target_duration: Union[int, float] = 2):
        super().__init__(
            suffix, out_suffix, target_sr, target_channel, target_duration)

    def sound_preprocess(self, sound, *args, **kwargs):
        """ 
        Simple preprocessing for input datato have unify duration, channel, sr.
        Apply spitting if the raw data is to long, and apply repeat, padding to
        extend thedata length.
        """
        if sound.duration_seconds < self.target_duration:
            # side padding
            pad_ms = (self.target_duration-sound.duration_seconds)/2 * 1000
            silence = AudioSegment.silent(duration=pad_ms)
            new_sound = silence + sound + silence
            new_sounds = [new_sound[:1000*self.target_duration]]
        elif sound.duration_seconds > self.target_duration:
            # continuous spitting
            new_sounds = continuous_split_sound(sound, self.target_duration)
        else:
            new_sounds = [sound]
        return new_sounds


class ESC50Preprocess(SnoringPreprocess):
    def __init__(self, 
                 suffix: str = 'wav', 
                 target_sr: int = 16000, 
                 target_channel: int = 1, 
                 target_duration: Union[int, float] = 2):

        super().__init__(
            suffix, target_sr, target_channel, target_duration)

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


class AsusSnoring0Preprocess(SnoringPreprocess):
    def __init__(self,
                 split_files_root: str,
                 suffix: str = 'wav', 
                 target_sr: int = 16000, 
                 target_channel: int = 1, 
                 target_duration: Union[int, float] = 2):

        super().__init__(
            suffix, target_sr, target_channel, target_duration)
        self.split_files_root = Path(split_files_root)

    def get_data_refs(self, *args) -> list:
        files = self.split_files_root.glob('*.txt')
        self.split_refs = {}
        data_refs = []
        for file in files:
            with open(file, 'r') as fw:
                data_paths = fw.readlines()
            split_name = file.stem
            split_data_refs = []
            for data_path in data_paths:
                data_path = Path(data_path.rstrip('\n'))
                split_data_refs.append(data_path)
            data_refs.extend(split_data_refs)
            self.split_refs[split_name] = split_data_refs
        return data_refs

    def split_data_reference(self, data_refs: dict) -> dict:
        """
        This function helps to split the data reference with initilize split_ratio.

        Args:
            data_refs (dict): input data filename, local path and target

        Returns:
            split_data_refs (dict): splitted data reference
        """
        split_data_refs = {}
        for split_name, split_paths in self.split_refs.items():
            if split_name not in split_data_refs:
                split_data_refs[split_name] = {}
            for split_path in split_paths:
                label = self.get_class_label(split_path)
                idx = data_refs['input'].index(split_path.stem)
                if label not in split_data_refs[split_name]:
                    split_data_refs[split_name][label] = [idx]
                else:
                    split_data_refs[split_name][label].append(idx)

        # split valid from train
        train_refs = {}
        valid_refs = {}
        for label in split_data_refs['train']:
            train_split = split_data_refs['train'][label]
            train_num = len(train_split)
            train_label_num = int(self.split_ratio['train']*train_num)
            new_train_split = np.random.choice(train_split, train_label_num, replace=False)
            valid_split = np.setdiff1d(train_split, new_train_split, assume_unique=True)
            train_refs[label] = new_train_split
            valid_refs[label] = valid_split
        split_data_refs['train'] = train_refs
        split_data_refs['valid'] = valid_refs
        return split_data_refs


def run_class():
    save_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\test\web_snoring_pre')

    # processer = KagglePadPreprocess()
    # dataset_name = 'kaggle_snoring_pad'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Kaggle_snoring\Snoring Dataset')
    # processer(dataset_name, data_root, save_root)

    # processer = SnoringPreprocess()
    # dataset_name = 'kaggle_snoring'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Kaggle_snoring\Snoring Dataset')
    # processer(dataset_name, data_root, save_root)

    # processer = AssignLabelPreprocess(assign_label=1)
    # dataset_name = 'web_snoring'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\test\web_snoring')
    # processer(dataset_name, data_root, save_root)

    # processer = ESC50Preprocess()
    # dataset_name = 'ESC50'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50-master\audio')
    # processer(dataset_name, data_root, save_root)

    # processer = AssignLabelPreprocess(assign_label=0)
    # dataset_name = 'Mi11_office'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\Mi11_office\wave_split')
    # processer(dataset_name, data_root, save_root)

    # processer = AssignLabelPreprocess(assign_label=0)
    # dataset_name = 'iphone11_0908'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\iphone11_0908\wave_split')
    # processer(dataset_name, data_root, save_root)

    # processer = AssignLabelPreprocess(assign_label=0)
    # dataset_name = 'pixel_0908'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\pixel_0908\wave_split')
    # processer(dataset_name, data_root, save_root)

    # split_data_root = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\Freq2\2_21_2s_my2'
    # processer = AsusSnoring0Preprocess(split_data_root)
    # dataset_name = 'ASUS_snoring_case_split'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\freq6_no_limit\2_21\raw_f_h_2_mono_16k')
    # processer(dataset_name, data_root, save_root)

    # processer = AssignLabelPreprocess(assign_label=1)
    # dataset_name = 'pixel_0908'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\pixel_0908\wave_split')
    # processer(dataset_name, data_root, save_root)

    # processer = AssignLabelPreprocess(assign_label=1)
    # dataset_name = 'iphone11_0908'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\iphone11_0908\wave_split')
    # processer(dataset_name, data_root, save_root)

    # processer = AssignLabelPreprocess(assign_label=1, suffix='mp3')
    # dataset_name = '0908_ori'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\0908_ori')
    # processer(dataset_name, data_root, save_root)

    # processer = AssignLabelPreprocess(assign_label=1)
    # dataset_name = 'pixel_0908_2'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\pixel_0908_2\wave_split')
    # processer(dataset_name, data_root, save_root)

    # processer = AssignLabelPreprocess(assign_label=1)
    # dataset_name = 'iphone11_0908_2'
    # data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\iphone11_0908_2\wave_split')
    # processer(dataset_name, data_root, save_root)

    processer = AssignLabelPreprocess(assign_label=0, sample_ratio=0.25)
    dataset_name = 'Redmi_Note8_night'
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\Redmi_Note8_night\wave_split')
    processer(dataset_name, data_root, save_root)

    processer = AssignLabelPreprocess(assign_label=0)
    dataset_name = 'redmi'
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\redmi\wave_split')
    processer(dataset_name, data_root, save_root)

    processer = AssignLabelPreprocess(assign_label=0)
    dataset_name = 'pixel'
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\pixel\wave_split')
    processer(dataset_name, data_root, save_root)

    processer = AssignLabelPreprocess(assign_label=0, sample_ratio=0.25)
    dataset_name = 'Mi11_night'
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\Mi11_night\wave_split')
    processer(dataset_name, data_root, save_root)

    processer = AssignLabelPreprocess(assign_label=0)
    dataset_name = 'iphone'
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\iphone\wave_split')
    processer(dataset_name, data_root, save_root)

    processer = AssignLabelPreprocess(assign_label=0, sample_ratio=0.25)
    dataset_name = 'Samsung_Note10Plus_night'
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\Samsung_Note10Plus_night\wave_split')
    processer(dataset_name, data_root, save_root)


if __name__ == '__main__':
    data_root = Path(r'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset')
    run_class()