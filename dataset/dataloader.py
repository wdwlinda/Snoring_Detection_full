
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from cfg import DATA_PATH
from typing import AbstractSet
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from dataset import preprocess_utils
from dataset import input_preprocess
from dataset import dataset_utils
from dataset import transformations
from utils import configuration
from analysis.resample_test import resample
from analysis import utils


def minmax_normalization(image):
    img_max = np.max(image)
    img_min = np.min(image)
    return (image-img_min) / (img_max-img_min)


def convert_value(image, value_pair=None):
    # TODO: one-to-one mapping, non-repeat
    # TODO: label pair convert
    for k in value_pair:
        image[image==k] = value_pair[k]
    return image


def create_kaggle_snoring_dataloader():
    load_func = None
    gen_index_func = dataset_utils.generate_kaggle_snoring_index

    kaggle_snoring_dataloader = KaggleSnoringDataset(load_func,
                                                     gen_index_func,)
    return kaggle_snoring_dataloader
    

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
        assert data_split[0] + data_split[1] == 1

    # def generate_index_func(self):
    #     return dataset_utils.generate_kaggle_breast_ultrasound_index

# TODO: Varing audio length --> cut and pad
class AudioDataset(AbstractDastaset):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.input_data_indices = dataset_utils.load_content_from_txt(
                os.path.join(config.dataset.index_path, f'{mode}.txt'))
                
        self.ground_truth_indices = [int(os.path.split(os.path.split(f)[0])[1]) for f in self.input_data_indices]
        # self.ground_truth_indices = [int(os.path.basename(f)[0]) for f in self.input_data_indices]
        self.transform_methods = config.dataset.transform_methods
        self.transform_config = self.dataset_config.transform_config
        print(f"{self.mode}  Samples: {len(self.input_data_indices)}")
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def data_loading_function(self, filename):
        waveform, sr = torchaudio.load(filename)
        if self.dataset_config.sample_rate:
            waveform = resample('transforms', waveform, sr, self.dataset_config.sample_rate)
            sr = self.dataset_config.sample_rate
        return waveform, sr
    
    # def data_loading_function(self, filename):
    #     y = utils.load_audio_waveform(filename, 'wav', self.dataset_config.sample_rate, channels=1)
    #     sr = y.frame_rate
    #     waveform = np.float32(np.array(y.get_array_of_samples()))
    #     waveform = torch.from_numpy(waveform)
    #     # waveform, sr = torchaudio.load(filename)
    #     if self.dataset_config.sample_rate:
    #         waveform = resample('transforms', waveform, sr, self.dataset_config.sample_rate)
    #         sr = self.dataset_config.sample_rate
    #     return waveform, sr

    def preprocess(self, waveform, sample_rate, mix_waveform=None):
        # input_preprocess.audio_preprocess(
        #     waveform, sample_rate, mix_waveform, self.transform_methods, self.transform_config, **self.preprocess_config)
        if mix_waveform is not None:
            waveform = input_preprocess.mix_up(waveform, mix_waveform)
        features = transformations.get_audio_features(waveform, sample_rate, self.transform_methods, self.transform_config)
        audio_feature = self.merge_audio_features(features)

        if self.is_data_augmentation:
            audio_feature = input_preprocess.spectrogram_augmentation(audio_feature, **self.preprocess_config)
        return audio_feature

    def __getitem__(self, idx):
        waveform, sr = self.data_loading_function(self.input_data_indices[idx])
        
        # TODO: related to is_augmentation?
        mix_waveform = None
        if self.mode == 'train':
            mix_up = self.dataset_config.preprocess_config.mix_up
            if mix_up:
                if mix_up > random.random():
                    mix_idx = random.randint(0, len(self.input_data_indices)-1)
                    mix_waveform, sr = self.data_loading_function(self.input_data_indices[mix_idx])

        input_data = self.preprocess(waveform, sr, mix_waveform)

        def log(data):
            # factor = torch.max(data)
            # data = (data - torch.min(data)) / torch.max(data)
            # factor /= torch.max(data)
            # data *= factor
            # data = torch.where(data == 0, np.finfo(float).eps, data)
            data2 = 20 * torch.log10(data + 1)
            # data2 = T.AmplitudeToDB()(data)
            # data2 = (data + 1)
            torch.set_printoptions(precision=12)
            print(data.max(), data.min(), data2.max(), data2.min())
            return data2
        # input_data = log(input_data)

        def check_input_data(idx1=22, idx2=23):
            import librosa.display
            f1, f2 = self.input_data_indices[idx1], self.input_data_indices[idx2]
            w1, sr1 = self.data_loading_function(f1)
            w2, sr2 = self.data_loading_function(f2)

            spec1 = self.preprocess(w1, sr1, mix_waveform=None)
            spec2 = self.preprocess(w2, sr2, mix_waveform=None)

            # spec1 = T.AmplitudeToDB()(spec1)
            # spec2 = T.AmplitudeToDB()(spec2)

            fig, ax = plt.subplots(2,1)
            ax[0].set_title(f'{os.path.split(f1)[1]}_{self.ground_truth_indices[idx1]}')
            ax[1].set_title(f'{os.path.split(f2)[1]}_{self.ground_truth_indices[idx2]}')
            img = librosa.display.specshow(spec1[0].cpu().numpy(), x_axis='time', y_axis='mel', ax=ax[0])
            librosa.display.specshow(spec2[0].cpu().numpy(), x_axis='time', y_axis='mel', ax=ax[1])
            fig.colorbar(img, ax=ax, format="%+2.f dB")
            # ax[0].imshow(spec1[0].cpu().numpy())
            # ax[1].imshow(spec2[0].cpu().numpy())

            plt.show()

        # check_input_data()
        # +++
        # plt.imshow(mfcc)
        # plt.show()
        # print(self.input_data_indices[idx])
        # factor = torch.max(input_data)
        # input_data = (input_data - torch.min(input_data)) / torch.max(input_data)
        # factor /= torch.max(input_data)
        # input_data *= factor
        # # input_data = torch.where(input_data == 0, np.finfo(float).eps, input_data)
        # input_data = 20 * np.log10(input_data + 1)
        # plt.imshow(input_data[0])
        # import librosa.display
        # librosa.display.specshow(input_data[0].cpu().numpy())
        # plt.show()
        # print(mfcc == input_data)
        # +++
        if self.ground_truth_indices:
            ground_truth = self.ground_truth_indices[idx]
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


class KaggleSnoringDataset(AudioDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)

    def preprocess(self, data):
        # data = self.audio_preprocess(data)
        return data


if __name__ == "__main__":
    pass
        