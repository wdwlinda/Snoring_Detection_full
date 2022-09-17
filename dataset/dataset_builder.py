import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import pandas as pd
import os

from dataset import snoring_preprocess
from dataset import dataset_utils


# TODO: some inspections for arguments
# TODO: ABC? (to a meta dataset)
# TODO: split rate for testing --> train and testing dataset building need to be separated
# TODO: Considering a reusing dataset builder (Can we minimize the building effort?)

class SnoringDataset(Dataset):
    def __init__(self, loader, data_refs, dataset_name=None, transform=None, target_loader=None):
        self.loader = loader
        self.data_refs = data_refs
        self.dataset_name = dataset_name
        self.transform = transform
        self.target_loader = target_loader
        self.log_dataset()
        self.dataflow_func = self.get_dataflow()

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
    y = dataset_utils.load_audio_waveform(
        filename, 'wav', sr, channels)
    sr = y.frame_rate
    waveform = np.array(y.get_array_of_samples(), np.float32)
    return waveform, sr


def build_snoring_dataset(data_roots, data_ref_paths, train_batch_size, data_transform=None):
    total_train_dataset = []
    total_valid_dataset = []
    total_test_dataset = []
    for dataset_name in zip(data_roots, data_ref_paths):
        data_root = data_roots[dataset_name]
        data_ref_path = data_ref_paths[dataset_name]
        train_dataset, valid_dataset, test_dataset = build_single_snoring_dataset(
            data_root, data_ref_path, data_transform, dataset_name)
        total_train_dataset.append(train_dataset)
        total_valid_dataset.append(valid_dataset)
        total_test_dataset.append(test_dataset)
        
    concat_train_dataset = ConcatDataset(total_train_dataset)
    concat_valid_dataset = ConcatDataset(total_valid_dataset)
    concat_test_dataset = ConcatDataset(total_test_dataset)
    
    train_loader = DataLoader(concat_train_dataset, train_batch_size, shuffle=True)
    valid_loader = DataLoader(concat_valid_dataset, train_batch_size, shuffle=True)
    test_loader = DataLoader(concat_test_dataset, 1, shuffle=False)
    return train_loader, valid_loader, test_loader
    
    
def build_single_snoring_dataset(data_root, data_ref_path, data_transform=None, dataset_name=None):
    data_refs = pd.read_csv(data_ref_path)
    path_refs = snoring_preprocess.get_path_refs(data_root, data_refs, suffix='wav')
    data_config = {
        'loader': wav_loader,
        'data_refs': path_refs,
        'dataset_name': dataset_name,
        'transform': data_transform,
        'target_loader': None,
    }

    full_refs = path_refs
    full_size = len(full_refs)
    train_refs = random.sample(full_refs, int(0.9*full_size))
    valid_refs = list(set(full_refs)-set(train_refs))
    test_refs = None

    data_config['data_refs'] = train_refs
    train_dataset = SnoringDataset(**data_config)

    data_config['data_refs'] = valid_refs
    data_config['transform'] = None
    valid_dataset = SnoringDataset(**data_config)

    data_config['data_refs'] = test_refs
    test_dataset = SnoringDataset(**data_config)
    return train_dataset, valid_dataset, test_dataset
    

def main():
    pass
    # build_snoring_dataset(data_roots, data_ref_paths, train_batch_size, data_transform=None)


if __name__ == '__main__':
    main()