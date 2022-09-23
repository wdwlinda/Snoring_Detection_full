import random
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import pandas as pd
import os

import torchaudio

from dataset import snoring_preprocess
from dataset import dataset_utils


# TODO: some inspections for arguments
# TODO: ABC? (to a meta dataset)
# TODO: split rate for testing --> train and testing dataset building need to be separated
# TODO: Considering a reusing dataset builder (Can we minimize the building effort?)

class MetaDataset(Dataset):
    def __init__(self, loader, data_refs, dataset_name=None, transform=None, target_loader=None):
        self.loader = loader
        self.data_refs = data_refs
        self.dataset_name = dataset_name
        self.transform = transform
        self.target_loader = target_loader
        self.input_key = 'input'
        self.target_key = 'target'
        self.log_dataset()
        self.dataflow_func = self.get_dataflow_format()

    def __len__(self):
        return len(self.data_refs)
            
    def __getitem__(self, idx):
        return self.dataflow_func(idx)

    def get_dataflow_format(self):
        first_sample = self.data_refs[0]
        if isinstance(first_sample, dict):
            assert self.input_key in first_sample, 'No input data reference'
            if self.target_key in first_sample:
                return self.get_datapair_from_dict_seq
            else:
                return self.get_input_from_dict_seq
        elif isinstance(first_sample, str):
            return self.get_input_from_str_seq
        else:
            raise ValueError('Unknown reference format.')
            
    def get_input_from_str_seq(self, idx):
        input_data = self.loader(self.data_refs[idx])
        if self.transform is not None:
            input_data = self.transform(input_data)
        return input_data
    
    def get_input_from_dict_seq(self, idx):
        input_data = self.loader(self.data_refs[idx]['input'])
        if self.transform is not None:
            input_data = self.transform(input_data)
        return input_data
    
    def get_datapair_from_dict_seq(self, idx):
        input_data = self.loader(self.data_refs[idx]['input'])
        if self.target_loader is not None:
            target = self.target_loader(self.data_refs[idx]['target'])
        else:
            target = self.data_refs[idx]['target']
        if self.transform is not None:
            input_data, target = self.transform(input_data, target)
        return {'input': input_data, 'target': target}
        
    def log_dataset(self):
        pass

    def get_ref_mapping():
        pass


def wav_loader(filename, sr=None, channels=1):
    y = dataset_utils.get_pydub_sound(
        filename, 'wav', sr, channels)
    sr = y.frame_rate
    waveform = np.array(y.get_array_of_samples(), np.float32)
    return waveform, sr


# def build_single_snoring_dataset(
#     data, data_refs, loader, data_transform=None, dataset_name=None):
#     data_config = {
#         'data': data,
#         'data_refs': data_refs,
#         'loader': loader,
#         'dataset_name': dataset_name,
#         'transform': data_transform,
#         'target_loader': None,
#     }

#     data_config['data_refs'] = train_refs
#     train_dataset = SnoringDataset(**data_config)

#     data_config['data_refs'] = valid_refs
#     data_config['transform'] = None
#     valid_dataset = SnoringDataset(**data_config)
#     return train_dataset, valid_dataset
    

# def get_snoring_refs(label_path, split_json):
#     local_ref = json.load(split_json)
#     label_table = pd.read_csv(label_path)
#     return path_refs

# XXX: change to JSON format
def get_snoring_refs(data_ref_path, data_root):
    data_refs = pd.read_csv(data_ref_path)
    path_refs = snoring_preprocess.get_path_refs_fast(data_root, data_refs, suffix='wav')
    return path_refs


# XXX: Put in here currently
def snoring_transform(waveform):
    waveform = wavform_transform(waveform)
    spec = wavform_to_spec(waveform)
    spec = spec_transform(spec)
    return spec


# XXX: return sr?
def torchaudio_loader(path):
    waveform, sr = torchaudio.load(path, normalize=True)
    return waveform


def build(data_csv, train_json, valid_json, dataset_name):
    train_data_refs = get_snoring_refs(data_csv, train_json)
    valid_data_refs = get_snoring_refs(data_csv, valid_json)
    loader = torchaudio_loader

    train_dataset = MetaDataset(
        loader, train_data_refs, dataset_name, snoring_transform, target_loader=None)
    valid_dataset = MetaDataset(
        loader, valid_data_refs, dataset_name, snoring_transform, target_loader=None)
    return train_dataset, valid_dataset


def build_dataloader(total_data_config, train_batch_size):
    total_train_dataset = []
    total_valid_dataset = []
    for data_config in total_data_config:
        train_dataset, valid_dataset = build(**data_config)
        total_train_dataset.append(train_dataset)
        total_valid_dataset.append(valid_dataset)
        
    concat_train_dataset = ConcatDataset(total_train_dataset)
    concat_valid_dataset = ConcatDataset(total_valid_dataset)
    
    train_loader = DataLoader(concat_train_dataset, train_batch_size, shuffle=True)
    valid_loader = DataLoader(concat_valid_dataset, 1, shuffle=False)
    return train_loader, valid_loader


def main():
    data_root = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\pp\Samsung_Note10Plus_night\wave_split'
    data_ref_path = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\pp\Samsung_Note10Plus_night\data.csv'
    data_refs = pd.read_csv(data_ref_path)
    path_refs = snoring_preprocess.get_path_refs_fast(data_root, data_refs, suffix='wav')
    pass
    # build_snoring_dataset(data_roots, data_ref_paths, train_batch_size, data_transform=None)


if __name__ == '__main__':
    main()