import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os

from dataset import snoring_preprocess
from dataset import dataset_utils


class SnoringDataset(Dataset):
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

    def check_config(self, config):
        pass

    def log_dataset(self):
        pass

    def get_ref_mapping():
        pass


def npy_loader(ref):
    return np.load(ref)


def get_data_refs(indices_path):
    input_indices = pd.read_csv(indices_path)
    return input_indices


def wav_loader(filename, sr=None, channels=1):
    y = dataset_utils.load_audio_waveform(
        filename, 'wav', sr, channels)
    sr = y.frame_rate
    waveform = np.array(y.get_array_of_samples(), np.float32)
    return waveform, sr


# def builder():
#     # TODO: Come from a YAML
#     config = {
#         'name': 'pixel_0908',
#         'data_root': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\pixel_0908\melspec\img\filenames',
#         'indices': None,
#         'split': (7, 2, 1),
#     }

#     # name, data_refs
#     img_dataset_infer = ImageDataset(config, data_loading_func=npy_loader)
#     img_dataset_train = ImageDataset(config, data_loading_func=npy_loader, target_loader=identity_mapping)
#     data_builder = DataBuilder(name, data_loading_func=npy_loader, target_loader=identity_mapping)
#     dataset = data_builder.build_from_csv(csv_path, input_header='img', target_header='label')
#     dataset = data_builder.build_from_dir(input_root, target_root)


def builder_test():
    data_transform = None
    full_refs = [{'data': '', 'target': ''}, {'data': '', 'target': ''}]
    
    data_roots = {
        'ASUS_snoring': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw_final_test\freq6_no_limit\2_21\raw_f_h_2_mono_16k',
        'ESC50': r'C:\Users\test\Desktop\Leon\Datasets\ESC-50\ESC-50_process\esc50\esc50_2',
    }
    esc50_df_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\pp\ESC50'

    data_refs = pd.read_csv(os.path.join(esc50_df_dir, 'data.csv'))
    path_refs = snoring_preprocess.get_path_refs(data_roots['ESC50'], data_refs, suffix='wav')
    data_config = {
        'dataset_name': 'test',
        'data_loading_func': wav_loader,
        'target_loader': None,
        'transform': data_transform,
        'data_refs': path_refs
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
    data_config['target_loader'] = None
    test_dataset = SnoringDataset(**data_config)
    

def main():
    # builder()
    builder_test()


if __name__ == '__main__':
    main()