
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from cfg import DATA_PATH
from typing import AbstractSet
import random
import wave

# import librosa
from scipy.ndimage.measurements import label
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import csv
import pandas as df
import os
import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd

# from analysis.data_splitting import get_files
from dataset import preprocess_utils
from dataset import input_preprocess
from dataset import dataset_utils
from dataset import transformations
from utils import configuration
from analysis.resample_test import resample
from analysis import utils
from dataset.data_transform import get_wav_transform


class ImageDataset(Dataset):
    def __init__(self, config, loader, transform=None, target_loader=None):
        self.check_config(config)
        self.input_indices = config['indices']
        self.loader = loader
        self.transform = transform if transform is not None else None
        self.target_loader = target_loader if target_loader is not None else None
        self.log_dataset()

    def __len__(self):
        return len(self.input_indices)
            
    def __getitem__(self, idx):
        input_data = self.loader(self.input_indices[idx]['raw'])

        if self.target_loader is not None:
            target = self.target_loader(self.input_indices[idx]['target'])
            if self.transform is not None:
                input_data, target = self.transform(input_data, target)
            return {'input': input_data, 'target': target}
        else:
            if self.transform is not None:
                input_data = self.transform(input_data)
            return {'input': input_data}


def make_index_dict(data_path):
    if not os.path.exists(data_path):
        return None

    index_lookup = {}
    with open(data_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['File']] = row['Label']
            line_count += 1
    return index_lookup


def convert_value(image, value_pair=None):
    # TODO: one-to-one mapping, non-repeat
    # TODO: label pair convert
    for k in value_pair:
        image[image==k] = value_pair[k]
    return image


# def create_kaggle_snoring_dataloader():
#     load_func = None
#     gen_index_func = dataset_utils.generate_kaggle_snoring_index

#     kaggle_snoring_dataloader = KaggleSnoringDataset(load_func,
#                                                      gen_index_func,)
#     return kaggle_snoring_dataloader
    

# TODO: check data shape and type
# TODO: check whether python abc helpful?
class AbstractDastaset(Dataset):
    def __init__(self, config, mode):
        assert (mode=='train' or mode=='valid' or mode=='test'), f'Unknown executing mode [{mode}].'
        self.dataset_config = config.dataset
        self.model_config = config.model
        self.preprocess_config = self.dataset_config.preprocess_config
        if mode == 'train':
            self.is_data_augmentation = self.dataset_config.is_data_augmentation
        else:
            self.is_data_augmentation = False
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])
        # self.load_func = load_func

        self.check_dataset_split(config.dataset.data_split)
        # TODO: General solution
        self.input_data_indices, self.ground_truth_indices = [], []
        # self.input_data_indices, self.ground_truth_indices = dataset_utils.get_data_indices(
        #     data_name=config.dataset.data_name,
        #     data_path=config.dataset.data_path,
        #     save_path=config.dataset.index_path,
        #     mode=self.mode,
        #     data_split=config.dataset.data_split,
        #     generate_index_func=self.generate_index_func)

        # TODO: logger
        # print(f"{self.mode}  Samples: {len(self.input_data_indices)}")

    def __len__(self):
        return len(self.input_data_indices)

    def __getitem__(self, idx):
        input_data = self.data_loading_function(self.input_data_indices[idx])
        input_data = self.preprocess(input_data)
        input_data = self.transform(input_data)
        if self.ground_truth_indices:
            ground_truth = self.data_loading_function(self.ground_truth_indices[idx])
            ground_truth = self.preprocess(ground_truth)
            ground_truth = self.transform(ground_truth)
        else:
            ground_truth = None
        return {'input': input_data, 'gt': ground_truth}

    def data_loading_function(self, fielname):
        return fielname

    def preprocess(self, data):
        return data

    def check_dataset_split(self, data_split):
        assert (isinstance(data_split, list) or isinstance(data_split, tuple))
        assert data_split[0] + data_split[1] == 1.0
        assert data_split[0] >= 0.0 and data_split[0] <= 1.0
        assert data_split[1] >= 0.0 and data_split[1] <= 1.0
    # def generate_index_func(self):
    #     return dataset_utils.generate_kaggle_breast_ultrasound_index


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    # waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        spectrum, _, _, im = axes[c].specgram(waveform[c], Fs=sample_rate, NFFT=512, noverlap=256)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)
    return spectrum




# TODO: Varing audio length --> cut and pad
class AudioDataset(AbstractDastaset):
    def __init__(self, config, mode, eval_mode=True):
        super().__init__(config, mode)
        self.data_suffix = self.dataset_config.data_suffix
        self.in_channels = config.model.in_channels
        # XXX: define duration and sample rate
        self.wav_length = 2 * 16000 # duration * sample_ratre
        self.mixup = config.TRAIN.MIXUP
        self.is_wav_transform = config.dataset.wav_transform
        self.mean_sub = config.dataset.mean_sub
        if self.is_wav_transform:
            self.wav_transform = get_wav_transform()

        # if wav_transform is not None:
        #     self.wav_transform = wav_transform
        # else:
        #     self.wav_transform = None

        # if mode in ('train', 'valid'):
        #     self.input_data_indices = dataset_utils.load_content_from_txt(
        #             os.path.join(config.dataset.index_path, 'train.txt'))
        #     # TODO:
        #     np.random.shuffle(self.input_data_indices)
        #     if mode == 'train':
        #         # self.input_data_indices = self.input_data_indices[:int(len(self.input_data_indices)*self.dataset_config.data_split[0])]
        #         self.input_data_indices = self.input_data_indices
        #     else:
        #         # self.input_data_indices = self.input_data_indices[int(len(self.input_data_indices)*self.dataset_config.data_split[0]):]
        #         self.input_data_indices = dataset_utils.load_content_from_txt(
        #             os.path.join(config.dataset.index_path, 'test.txt'))
        # elif mode == 'test':
        #     self.input_data_indices = dataset_utils.load_content_from_txt(
        #             os.path.join(config.dataset.index_path, 'test.txt'))
        # else:
        #     raise ValueError('Unknown mode.')

        self.input_data_indices = []
        for dataset_name, index_path in config.dataset.index_path[mode].items():
            files = dataset_utils.load_content_from_txt(index_path)
            self.input_data_indices.extend(files)

        # TODO: gt
        # judge return (data) or (data, label), data_split, use two dataset together?
        # XXX: change the ground truth getting way
        self.eval_mode = eval_mode
        self.ground_truth_indices = []
        if self.eval_mode:
            for f in self.input_data_indices:
                if '2_21' in f or 'esc50' in f:
                    self.ground_truth_indices.append(int(os.path.split(os.path.split(f)[0])[1]))
                else:
                    self.ground_truth_indices.append(0)
        else:
            self.ground_truth_indices = None
            
        self.transform_methods = config.dataset.transform_methods
        self.transform_config = self.dataset_config.transform_config
        print(f"Samples: {len(self.input_data_indices)}")
        self.transform = transforms.Compose([transforms.ToTensor()])

    def data_loading_function(self, filename):
        y = dataset_utils.get_pydub_sound(filename, self.data_suffix, self.dataset_config.sample_rate, channels=1)
        sr = y.frame_rate
        waveform = np.array(y.get_array_of_samples(), np.float32)
        return waveform, sr

    def data_loading_function_torchaudio(self, filename):
        waveform, sr = torchaudio.load(filename, normalize=True)
        return waveform, sr

    def preprocess(self, waveform, sample_rate, mix_waveform=None):
        if len(waveform.shape) == 1:
            waveform = np.expand_dims(waveform, axis=0)
            if mix_waveform is not None:
                mix_waveform = np.expand_dims(mix_waveform, axis=0)

        if mix_waveform is not None:
            waveform, mix_lambda = input_preprocess.mix_up(waveform, mix_waveform)
        else:
            mix_lambda = None

        features = transformations.get_audio_features(
            waveform, sample_rate, self.transform_methods, self.transform_config)
        audio_feature = self.merge_audio_features(features)
        audio_feature = np.transpose(audio_feature, (1, 2, 0))

        audio_feature = self.transform(audio_feature)
        if self.is_data_augmentation:
            audio_feature = input_preprocess.spectrogram_augmentation(audio_feature, **self.preprocess_config)

        return audio_feature, mix_lambda

    def __getitem__(self, idx):
        # waveform, sr = self.data_loading_function(
        #     self.input_data_indices[idx])
        waveform, sr = self.data_loading_function_torchaudio(
            self.input_data_indices[idx])
        # XXX:
        if self.mean_sub:
            waveform = waveform - waveform.mean()

        # XXX: modeulize and check the splitting at first
        # TODO: repeat
        if waveform.shape[1] < self.wav_length:
            pad_lenth = self.wav_length - waveform.shape[1]
            left_pad = pad_lenth // 2
            right_pad = pad_lenth - left_pad
            waveform = torch.nn.functional.pad(waveform, pad=(left_pad, right_pad), mode='constant')
            # waveform = np.pad(waveform, pad_width=(left_pad, right_pad), mode='mean')
        if waveform.shape[1] > self.wav_length:
            waveform = waveform[:, :self.wav_length]

        # XXX: 
        # waveform augmentation
        # waveform = time_transform.augmentation(waveform)
        # if self.is_wav_transform:
        #     waveform = waveform[None]
        #     waveform = self.wav_transform(waveform, self.dataset_config.sample_rate)
        #     waveform = waveform[0]

        input_data = waveform
        # input_data, mix_lambda = self.preprocess(waveform, sr, mix_waveform)
        if input_data.shape[-1] == 1: 
            print(self.input_data_indices[idx])
            
        # # TODO: bad implementation
        # if self.in_channels == 3:
        #     input_data = torch.tile(input_data, (3, 1, 1))
        if self.ground_truth_indices:
            ground_truth = self.ground_truth_indices[idx]
            # TODO: binary to multi np.eye(2)
            ground_truth = np.eye(2)[ground_truth]
        else:
            ground_truth = None

        if ground_truth is not None:
            return {'input': input_data, 'target': ground_truth, 'sr': sr}
        else:
            return {'input': input_data, 'sr': sr}

    # def __getitem__(self, idx):
    #     waveform, sr = self.data_loading_function(self.input_data_indices[idx])
    #     # print(idx, self.input_data_indices[idx])

    #     mix_waveform = None
    #     mix_up = self.dataset_config.preprocess_config.mix_up
    #     if mix_up and self.is_data_augmentation:
    #         if mix_up > random.random():
    #             mix_idx = random.randint(0, len(self.input_data_indices)-1)
    #             mix_waveform, sr = self.data_loading_function(self.input_data_indices[mix_idx])

    #     # XXX: 
    #     # waveform augmentation
    #     # waveform = time_transform.augmentation(waveform)
    #     if self.wav_transform is not None:
    #         waveform = waveform[None]
    #         waveform = self.wav_transform(waveform, self.dataset_config.sample_rate)
    #         waveform = waveform[0]
        
    #     input_data, mix_lambda = self.preprocess(waveform, sr, mix_waveform)
    #     if input_data.shape[-1] == 1: 
    #         print(self.input_data_indices[idx])
            
    #     # TODO: bad implementation
    #     if self.in_channels == 3:
    #         input_data = torch.tile(input_data, (3, 1, 1))
    #     if self.ground_truth_indices:
    #         ground_truth = self.ground_truth_indices[idx]
    #         # TODO: binary to multi np.eye(2)
    #         ground_truth = np.eye(2)[ground_truth]
    #         if mix_up and self.is_data_augmentation:
    #             if mix_lambda:
    #                 mix_ground_truth = np.eye(2)[self.ground_truth_indices[mix_idx]]
    #                 ground_truth = mix_lambda*ground_truth + (1-mix_lambda)*mix_ground_truth
    #     else:
    #         ground_truth = None

    #     if ground_truth is not None:
    #         return {'input': input_data, 'target': ground_truth}
    #     else:
    #         return {'input': input_data}

    def merge_audio_features(self, features):
        if not isinstance(features, dict):
            return features

        reshape_f = []
        for f in features.values():
            # Average of channel
            # print(f.size())
            f = preprocess_utils.channel_fusion(
                f, method=self.dataset_config.fuse_method, dim=0, reapeat_times=self.model_config.in_channels)
            reshape_f.append(f)
            # reshape_f.append(f)
                
        audio_feature = np.concatenate(reshape_f, axis=0)
        # self.model_config.in_channels = self.model_config.in_channels * len(features)
        return audio_feature


class AudioDatasetfromNumpy(Dataset):
    def __init__(self, config, mode, eval_mode=True):
        self.dataset_config = config.dataset
        self.model_config = config.model
        self.data_suffix = self.dataset_config.data_suffix
        self.in_channels = config.model.in_channels
        # self.input_data_indices = dataset_utils.load_content_from_txt(
        #         os.path.join(config.dataset.index_path, f'{mode}.txt'))
        self.preprocess_config = self.dataset_config.preprocess_config
        if mode == 'train':
            self.is_data_augmentation = self.dataset_config.is_data_augmentation
        else:
            self.is_data_augmentation = False

        all_dfs = []
        index_paths = config.dataset.index_path[mode]
        for dataset_name, index_path in index_paths.items():
            df = pd.read_csv(index_path)
            all_dfs.append(df)
        self.input_data_indices = pd.concat(all_dfs, axis=0)

        self.eval_mode = eval_mode
        self.transform_methods = config.dataset.transform_methods
        self.transform_config = self.dataset_config.transform_config
        print(f"Samples: {self.input_data_indices.shape[0]}")
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.input_data_indices.shape[0]

    def __getitem__(self, idx):
        df = self.input_data_indices.iloc[idx]
        input_data = np.load(df['img_path'])
        input_data = input_data[np.newaxis]
        
        input_data = np.transpose(input_data, (1, 2, 0))
        input_data = self.transform(input_data)
        if self.is_data_augmentation:
            input_data = input_preprocess.spectrogram_augmentation(
                input_data, **self.preprocess_config)

        if self.in_channels == 3:
            input_data = torch.tile(input_data, (3, 1, 1))
        
        if 'label' in df:
            ground_truth = df['label']
            ground_truth = np.eye(2)[ground_truth]
        else:
            ground_truth = None

        if ground_truth is not None:
            return {'input': input_data, 'target': ground_truth}
        else:
            return {'input': input_data}




class SimpleAudioDatasetfromNumpy_csv(Dataset):
    def __init__(self, config, path):
        self.dataset_config = config.dataset
        self.model_config = config.model
        self.in_channels = self.model_config.in_channels
        self.data_suffix = config.dataset.data_suffix
        self.input_data_indices = pd.read_csv(path)
        self.input_data_indices = self.input_data_indices['img_path'].tolist()
        
        # self.features = self.get_waveforms_from_path(path)
        self.transform_methods = config.dataset.transform_methods
        self.transform_config = self.dataset_config.transform_config
        print(f"Samples: {len(self.input_data_indices)}")
        self.transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return len(self.input_data_indices)

    def __getitem__(self, idx):
        # df = self.input_data_indices.iloc[idx]
        # input_data = np.load(df['img_path'])
        input_data = np.load(self.input_data_indices[idx])
        input_data = input_data[...,np.newaxis]
        input_data = self.transform(input_data)
        if self.in_channels == 3:
            input_data = torch.tile(input_data, (3, 1, 1))
        input_data = np.float32(input_data)
        return {'input': input_data}


class SimpleAudioDatasetfromNumpy(Dataset):
    def __init__(self, config, path):
        self.dataset_config = config.dataset
        self.model_config = config.model
        self.in_channels = self.model_config.in_channels
        self.data_suffix = config.dataset.data_suffix
        
        self.features = self.get_waveforms_from_path(path)
        self.transform_methods = config.dataset.transform_methods
        self.transform_config = self.dataset_config.transform_config
        # print(f"Samples: {len(self.input_data_indices)}")
        self.transform = transforms.Compose([transforms.ToTensor()])

    def get_waveforms_from_path(self, data_path):
        features = []
        audio_format = 'npy'
        self.input_data_indices = dataset_utils.get_files(data_path, keys=audio_format)
        for f in self.input_data_indices:
            y = np.load(f)
            features.append(np.float32(y))
        return features

    def __len__(self):
        return len(self.input_data_indices)

    def __getitem__(self, idx):
        input_data = self.features[idx]
        input_data = input_data[...,np.newaxis]
        input_data = self.transform(input_data)
        if self.in_channels == 3:
            input_data = torch.tile(input_data, (3, 1, 1))
        return {'input': input_data}


class SimpleAudioDataset(Dataset):
    def __init__(self, config, path):
        self.dataset_config = config.dataset
        self.model_config = config.model
        self.in_channels = self.model_config.in_channels
        self.data_suffix = config.dataset.data_suffix
        self.waveforms = self.get_waveforms_from_path(path)
        self.transform_methods = config.dataset.transform_methods
        self.transform_config = self.dataset_config.transform_config
        print(f"Samples: {len(self.input_data_indices)}")
        self.transform = transforms.Compose([transforms.ToTensor()])

    def get_waveforms_from_path(self, data_path):
        waveforms = []
        audio_format = 'wav'
        self.input_data_indices = glob.glob(os.path.join(data_path, '*.wav'))
        for f in self.input_data_indices:
            y = dataset_utils.get_pydub_sound(f, audio_format, self.dataset_config.sample_rate, channels=1)
            waveforms.append((np.float32(y.get_array_of_samples()), y.frame_rate))
        return waveforms

    def __len__(self):
        return len(self.input_data_indices)

    def __getitem__(self, idx):
        waveform, sr = self.waveforms[idx]
        input_data = self.preprocess(waveform, sr)
        if self.in_channels == 3:
            input_data = torch.tile(input_data, (3, 1, 1))
        return {'input': input_data}

    def preprocess(self, waveform, sample_rate):
        if len(waveform.shape) == 1:
            waveform = np.expand_dims(waveform, axis=0)
        features = transformations.get_audio_features(waveform, sample_rate, self.transform_methods, self.transform_config)
        audio_feature = self.merge_audio_features(features)
        audio_feature = np.swapaxes(np.swapaxes(audio_feature, 0, 1), 1, 2)
        audio_feature = self.transform(audio_feature)
        return audio_feature

    def merge_audio_features(self, features):
        if not isinstance(features, dict):
            return features

        reshape_f = []
        for f in features.values():
            # Average of channel
            # print(f.size())
            f = preprocess_utils.channel_fusion(
                f, method=self.dataset_config.fuse_method, dim=0, reapeat_times=self.model_config.in_channels)
            reshape_f.append(f)
                
        audio_feature = np.concatenate(reshape_f, axis=0)
        return audio_feature



if __name__ == "__main__":
    pass
        