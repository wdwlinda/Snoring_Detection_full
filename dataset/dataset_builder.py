import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class ImageDataset(Dataset):
    def __init__(self, config, loader, transform=None, target_loader=None):
        self.check_config(config)
        self.input_indices = config['indices']
        self.loader = loader
        self.transform = transform if transform is not None else None
        self.target_loader = target_loader if target_loader is not None else None
        self.log_dataset()

    def __len__(self):
        if isinstance(self.input_indices, (list, dict)):
            return len(self.input_indices)
        elif isinstance(self.input_indices, pd.DataFrame):
            return self.input_indices.shape[0]
            
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


def main():
    # Come from a YAML
    config = {
        'name': 'pixel_0908',
        'data_root': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess\pixel_0908\melspec\img\filenames',
        'indices': None,
        'split': (7, 2, 1),
    }

    img_dataset = ImageDataset(config, data_loading_func=npy_loader)


if __name__ == '__main__':
    main()