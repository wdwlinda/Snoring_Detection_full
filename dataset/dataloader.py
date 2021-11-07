
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from cfg import DATA_PATH
from typing import AbstractSet
import random
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
import matplotlib.pyplot as plt
from analysis.data_splitting import get_files
from dataset import preprocess_utils
from dataset import input_preprocess
from dataset import dataset_utils
from dataset import transformations
from utils import configuration
from analysis.resample_test import resample
from analysis import utils


# def load_input_data(index_path, data_path, keys, data_split, label_csv=None):
    
#     # Load data indices if exist
#     dataset_name = '-'.join([os.path.basename(data_path), data_split[0], data_split[1]])

#     # Create and save data indices
#     data_list = get_files(data_path, keys)
#     data_list.sort()
#     if os.path.exists(label_csv):
#         df_label = df.read_csv(label_csv)
#         labels = {f: df_label['full_path'][f] for f in data_list}
#     else:
#         labels = None
    
#     return data_list, labels


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
        assert (mode=='train' or mode=='valid'), f'Unknown executing mode [{mode}].'
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


# TODO: Varing audio length --> cut and pad
class AudioDataset(AbstractDastaset):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        # self.input_data_indices, self.ground_truth_indices = dataset_utils.load_input_data(
        #     config.dataset.data_path, config.dataset.data_suffix, label_csv=os.path.join(config.dataset.data_path, 'label.csv'))
        # # TODO: validation dataset_split
        # # self.ground_truth_indices = make_index_dict(os.path.join(config.dataset.data_path, 'label.csv'))

        self.input_data_indices, self.ground_truth_indices = dataset_utils.load_input_data()
        
        self.data_suffix = config.dataset.data_suffix
        self.input_data_indices = dataset_utils.load_content_from_txt(
                os.path.join(config.dataset.index_path, f'{mode}.txt'))
                
        # TODO: gt
        # judge return (data) or (data, label), data_split, use two dataset together?
        self.ground_truth_indices = [int(os.path.split(os.path.split(f)[0])[1]) for f in self.input_data_indices]
        self.transform_methods = config.dataset.transform_methods
        self.transform_config = self.dataset_config.transform_config
        print(f"Samples: {len(self.input_data_indices)}")
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def data_loading_function(self, filename):
        y = dataset_utils.load_audio_waveform(filename, self.data_suffix, self.dataset_config.sample_rate, channels=1)
        sr = y.frame_rate
        waveform = np.float32(np.array(y.get_array_of_samples()))
        waveform = torch.from_numpy(waveform)
        if self.dataset_config.sample_rate:
            waveform = resample('transforms', waveform, sr, self.dataset_config.sample_rate)
            sr = self.dataset_config.sample_rate
        return waveform, sr

    def preprocess(self, waveform, sample_rate, mix_waveform=None):
        # input_preprocess.audio_preprocess(
        #     waveform, sample_rate, mix_waveform, self.transform_methods, self.transform_config, **self.preprocess_config)
        # print(waveform.max(), waveform.min())
        if len(waveform.shape) == 1:
            waveform = torch.unsqueeze(waveform, dim=0)
            if mix_waveform is not None:
                mix_waveform = torch.unsqueeze(mix_waveform, dim=0)

        if mix_waveform is not None:
            waveform, mix_lambda = input_preprocess.mix_up(waveform, mix_waveform)
        else:
            mix_lambda = None
        features = transformations.get_audio_features(waveform, sample_rate, self.transform_methods, self.transform_config)
        audio_feature = self.merge_audio_features(features)

        if self.is_data_augmentation:
            audio_feature = input_preprocess.spectrogram_augmentation(audio_feature, **self.preprocess_config)
        return audio_feature, mix_lambda

    def __getitem__(self, idx):
        waveform, sr = self.data_loading_function(self.input_data_indices[idx])
        
        # TODO: related to is_augmentation?
        mix_waveform = None
        # if self.mode == 'train':
        mix_up = self.dataset_config.preprocess_config.mix_up
        if mix_up:
            if mix_up > random.random():
                mix_idx = random.randint(0, len(self.input_data_indices)-1)
                mix_waveform, sr = self.data_loading_function(self.input_data_indices[mix_idx])
        input_data, mix_lambda = self.preprocess(waveform, sr, mix_waveform)

        if self.ground_truth_indices:
            ground_truth = self.ground_truth_indices[idx]
            # TODO: binary to multi np.eye(2)
            ground_truth = torch.eye(2)[ground_truth]
            if mix_up:
                if mix_lambda:
                    mix_ground_truth = torch.eye(2)[self.ground_truth_indices[mix_idx]]
                    ground_truth = mix_lambda*ground_truth + (1-mix_lambda)*mix_ground_truth
        else:
            ground_truth = None
        # TODO: general solution for channel issue
        # input_data = torch.unsqueeze(input_data, 0)
        # input_data = input_data.repeat(3, 1, 1)

        # input_data = input_data.repeat(2, 1, 1)
        # input_data = input_data[:2]
        # # input_data = input_data[:, :98, :128]
        # print('input size', input_data.size())

        # print(input_data.max(), input_data.min())
        return {'input': input_data, 'gt': ground_truth}

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
                
        audio_feature = torch.cat(reshape_f, axis=0)
        # self.model_config.in_channels = self.model_config.in_channels * len(features)
        return audio_feature


if __name__ == "__main__":
    pass
        