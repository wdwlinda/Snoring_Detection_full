
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from cfg import DATA_PATH
from typing import AbstractSet
import torch
import torchaudio
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



def data_analysis(path, dir_key, file_key):
    """check image and mask value range"""
    input_paths, gt_paths = dataset_utils.generate_filename_list(path, file_key, dir_key)
    # print(len(input_paths), len(gt_paths))
    # assert len(input_paths) == len(gt_paths)
    image_height, image_width = [], []
    for path, gt_path in zip(input_paths, gt_paths):
        image = cv2.imread(path)
        gt = cv2.imread(gt_path)
        # assert image.shape() == gt.shape()
        image_height.append(image.shape[0])
        image_width.append(image.shape[1])
    def mean(data):
        return sum(data)/len(data)
    print("Height Information (Min: {}, Max:{} Mean:{} Std:{})".
        format(min(image_height), max(image_height), mean(image_height), mean(image_height)))
    print("Width Information (Min: {}, Max:{} Mean:{} Std:{})".
        format(min(image_width), max(image_width), mean(image_width), mean(image_width)))

# TODO: Rewrite
# TODO: general data analysis tool
def data_preprocessing(path, file_key, dir_key):
    """merge mask"""
    input_paths, gt_paths = dataset_utils.generate_filename_list(path, file_key, dir_key)
    for idx, gt_path in enumerate(gt_paths):
        if 'mask_1' in gt_path:
            print(idx)
            keyword = gt_path.split('mask_')[0]
            start = min(0, idx-2)
            end = max(len(gt_paths)-1, idx+2)
            mask_path_list = []
            for i in range(start, end):
                if keyword in gt_paths[i]:
                    mask_path_list.append(gt_paths[i])
            # Merge masks and save
            mask = 0
            for mask_path in mask_path_list:
                mask = mask + cv2.imread(mask_path)
                os.remove(mask_path)
            filename = mask_path_list[0]
            cv2.imwrite(filename, mask)


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
        self.dataset_config = config.dataset.train if mode == 'train' else config.dataset.val
        self.model_config = config.model
        self.preprocess_config = self.dataset_config.preprocess_config
        self.is_data_augmentation = self.dataset_config.is_data_augmentation
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
        print(f"{self.mode}  Samples: {len(self.input_data_indices)}")

    def __len__(self):
        return len(self.input_data_indices)

    def __getitem__(self, idx):
        input_data = self.data_loading_function(self.input_data_indices[idx])
        input_data = self.preprocess(input_data)
        input_data = self.transform(input_data)
        if self.ground_truth_indices is not None:
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


# TODO: Write in inherit form
# TODO: inference mode (no gt exist)
# TODO: dir_key to select benign or malignant
class ImageDataset(Dataset):
    def __init__(self, config, mode):
        # TODO: dynamic
        assert (mode=='train' or mode=='test'), f'Unknown executing mode [{mode}].'
        self.dataset_config = config.dataset.train if mode == 'train' else config.dataset.val
        self.model_config = config.model
        
        self.preprocess_config = self.dataset_config.preprocess_config
        self.is_data_augmentation = self.dataset_config.is_data_augmentation
        self.min_resize_value = self.preprocess_config.min_resize_value
        self.max_resize_value = self.preprocess_config.max_resize_value
        self.scale_factor_step_size = self.preprocess_config.scale_factor_step_size
        self.crop_size = self.preprocess_config.crop_size
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])

        data_path = os.path.join(config.dataset.index_path, f'{mode}.txt')
        self.input_data = dataset_utils.load_content_from_txt(data_path)
        self.input_data.sort()
        self.ground_truth = [f.replace('.png', '_mask.png') for f in self.input_data] 
        # self.ground_truth = [f.split('.png')[0]+'_mask.png' for f in self.input_data] 
        print(f"{self.mode}  Samples: {len(self.input_data)}")

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        # TODO: assert for the situation that in_channels=1 but value different between channels
        # Load images
        self.original_image = cv2.imread(self.input_data[idx])[...,0:self.model_config.in_channels]
        # TODO: dynamic value pair
        self.original_label = convert_value(
            image=cv2.imread(self.ground_truth[idx])[...,0:self.model_config.in_channels], value_pair={255: 1})

        input_image, gt_image = preprocessing.resize_to_range(self.original_image, self.original_label,
            min_size=self.min_resize_value, max_size=self.max_resize_value, factor=self.scale_factor_step_size)

        if self.is_data_augmentation:
            preprocessor = input_preprocess.DataPreprocessing(self.dataset_config['preprocess_config'])
            input_image, gt_image = preprocessor(input_image, gt_image)

        # Pad image and label to have dimensions >= [crop_height, crop_width]
        image_shape = input_image.shape
        image_height = image_shape[0]
        image_width = image_shape[1]

        target_height = image_height + max(self.crop_size[0] - image_height, 0)
        target_width = image_width + max(self.crop_size[1] - image_width, 0)
        
        input_image = preprocessing.pad_to_bounding_box(
            input_image, 0, 0, target_height, target_width, pad_value=0)
        if gt_image is not None:
            gt_image = preprocessing.pad_to_bounding_box(
                gt_image, 0, 0, target_height, target_width, pad_value=0)

        if self.is_data_augmentation:
            input_image, gt_image = preprocessing.random_crop(input_image, gt_image, self.crop_size)
            input_image, gt_image = preprocessing.random_flip(input_image, gt_image, 
                flip_prob=self.preprocess_config.flip_prob, flip_mode=self.preprocess_config.flip_mode)

        # Transform to Torch tensor
        input_image = self.transform(input_image)
        gt_image = self.transform(gt_image)
        return {'input': input_image, 'gt': gt_image}

  
class AudioDataset(AbstractDastaset):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        # self.preprocess_method = config.dataset_config.preprocess_config
        self.input_data_indices = dataset_utils.load_content_from_txt(
                os.path.join(config.dataset.index_path, f'{mode}.txt'))
        self.ground_truth_indices = [int(os.path.basename(f)[0]) for f in self.input_data_indices]
        self.transform_methods = config.dataset.transform
        print(f"{self.mode}  Samples: {len(self.input_data_indices)}")

    def data_loading_function(self, filename):
        return torchaudio.load(filename)

    def preprocess(self, data):
        features = self.audio_trasform(data)
        if len(features) > 1:
            audio_feature = self.merge_audio_features(features)
        else:
            audio_feature = features.values[0]
        return audio_feature

    def audio_trasform(self, data):
        # TODO: different method, e.g., mel-spectogram, MFCC, time-domain
        waveform, sample_rate = data
        # TODO: how to use time-domain data, split to clips?
        # if len(self.transform_methods) == 0:
        #     return waveform
        features = {}
        for method in self.transform_methods:
            if method == 'fbank':
                features[method] = transformations.fbank(waveform, sample_rate)
            elif method == 'spectrogram':
                features[method] = transformations.spectrogram(waveform, sample_rate)
            elif method == 'mel-spectrogram':
                features[method] = transformations.mel_spec(waveform, sample_rate)
            elif method == 'MFCC':
                features[method] = transformations.MFCC(waveform, sample_rate)
            else:
                raise ValueError('Unknown audio transformations')
        return features

    def __getitem__(self, idx):
        input_data = self.data_loading_function(self.input_data_indices[idx])
        input_data = self.preprocess(input_data)
        # input_data = self.transform(input_data)
        if self.ground_truth_indices is not None:
            ground_truth = self.ground_truth_indices[idx]
        else:
            ground_truth = None
        # TODO: general solution for channel issue
        input_data = torch.unsqueeze(input_data, 0)
        input_data = input_data.repeat(3, 1, 1)
        return {'input': input_data, 'gt': ground_truth}

    def merge_audio_features(self):
        pass


class KaggleSnoringDataset(AudioDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)

    def preprocess(self, data):
        # data = self.audio_preprocess(data)
        return data


class ClassificationImageDataset(ImageDataset):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        # dataset_config, model_config = config.dataset, config.model
        self.ground_truth = []
        for f in self.input_data:
            if 'benign' in f:
                self.ground_truth.append(0)
            if 'malignant' in f:
                self.ground_truth.append(1)
            if 'normal' in f:
                self.ground_truth.append(2)

    def __getitem__(self, idx):
        # Load images
        # input_image = cv2.imread(self.input_data[idx])[...,0:1]
        self.original_image = cv2.imread(self.input_data[idx])
        self.original_label = self.ground_truth[idx]

        input_image, _ = preprocessing.resize_to_range(self.original_image, label=None,
            min_size=self.min_resize_value, max_size=self.max_resize_value, factor=self.scale_factor_step_size)

        # Data preprocessing
        if self.is_data_augmentation:
            preprocessor = input_preprocess.DataPreprocessing(self.preprocess_config)
            input_image, _ = preprocessor(input_image)
        
        # Pad image and label to have dimensions >= [crop_height, crop_width]
        image_shape = input_image.shape
        image_height = image_shape[0]
        image_width = image_shape[1]

        target_height = image_height + max(self.crop_size[0] - image_height, 0)
        target_width = image_width + max(self.crop_size[1] - image_width, 0)
        
        input_image = preprocessing.pad_to_bounding_box(
            input_image, 0, 0, target_height, target_width, pad_value=0)

        if self.is_data_augmentation:
            input_image, _ = preprocessing.random_crop(input_image, label=None, crop_size=self.crop_size)
            input_image, _ = preprocessing.random_flip(input_image, label=None, 
                flip_prob=self.preprocess_config.flip_prob, flip_mode=self.preprocess_config.flip_mode)

        # Transform to Torch tensor
        input_image = self.transform(input_image)
        return {'input': input_image, 'gt': self.original_label}

    # def __getitem__(self, idx):
        # # Load images
        # # input_image = cv2.imread(self.input_data[idx])[...,0:1]
        # input_image = cv2.imread(self.input_data[idx])
        # gt = self.ground_truth[idx]

        # # Data preprocessing
        # output_strides = self.model_config.output_strides
        # input_image, _ = preprocessing.output_strides_align(input_image, output_strides)
        # if self.dataset_config.is_data_augmentation:
        #     preprocessor = input_preprocess.DataPreprocessing(self.dataset_config['preprocess_config'])
        #     input_image, _ = preprocessor(input_image)
        
        # # Standardize
        # # input_image = preprocessing.standardize(input_image)

        # # Transform to Torch tensor
        # input_image = self.transform(input_image)
        # return {'input': input_image, 'gt': gt}

    """
    BreastUltrasoundDataset(ImagaDataset)
    BU_segmentation_dataset = dataloader.BreastUltrasoundDataset(params, label_type='pixel-wise-label')
    BU_classification_dataset = dataloader.BreastUltrasoundDataset(params, label_type='image-wise-label')
    """
if __name__ == "__main__":
    # CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\config\_2dunet_cls_train_config.yml'
    # config = configuration.load_config(CONFIG_PATH)
    # dataset = AbstractDastaset(
    #     config, 
    #     mode='valid',
    #     generate_index_func=dataset_utils.generate_kaggle_breast_ultrasound_index)

    # CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\_cnn_train_config.yml'
    # config = configuration.load_config(CONFIG_PATH)
    # dataset = AudioDataset(
    #     config, 
    #     mode='valid')
    # train_dataloader = DataLoader(
    #     dataset, batch_size=2, shuffle=True)
    # # i, data = next(iter(train_dataloader))
    # for i, data in enumerate(train_dataloader):
    #     print(i, data['input'].shape, data['gt'])
    pass
        