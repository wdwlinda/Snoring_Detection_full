
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import json
from typing import Tuple, List, Union

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import csv
import matplotlib.pyplot as plt


class AudioDatasetCOCO(Dataset):
    def __init__(self, config, modes):
        data_roots = config.dataset.index_path
        self.input_data_indices, self.ground_truth_indices = parse_snoring_coco_total(
            data_roots, modes)
        assert len(self.input_data_indices) == len(self.ground_truth_indices), 'Mismatch data and target.'
        self.sr = config.dataset.sample_rate
        self.duration = config.dataset.duration
        self.channel = config.dataset.sound_ch

    def __len__(self):
        return len(self.input_data_indices)

    def __getitem__(self, idx):
        input_path = self.input_data_indices[idx]['path']
        # waveform, sr = torchaudio.load(input_path, normalize=False)
        # XXX: temp
        waveform, sr = torchaudio.load(input_path, normalize=True)
        target = self.ground_truth_indices[idx]['category_id']
        input_data = sequence_legth_adjust(waveform[0], self.sr*self.duration)
        input_data = torch.unsqueeze(input_data, dim=0)
        return {'input': input_data, 'target': target, 'sr': sr}


# TODO: dimension arg for working dimension
def sequence_legth_adjust(
        input_sequence: torch.Tensor, 
        output_length: int,
        tolerate: float = 0.1
    ) -> torch.Tensor:
    """
    Adjust input sequence length. 
    Process 1d audio signal is one of its application

    Args:
        input_sequence (torch.Tensor): Input 1d sequence
        output_length (int): Expected output length

    Returns:
        torch.Tensor: Output 1d sequence
    """
    input_length = input_sequence.shape[0]
    assert abs((input_length-output_length)/input_length) < tolerate, \
        'The dfferen of length are too big.'

    # center padding
    if input_length < output_length:
        pad_lenth = output_length - input_length
        left_pad = pad_lenth // 2
        right_pad = pad_lenth - left_pad
        output_sequence = torch.nn.functional.pad(
            input_sequence, pad=(left_pad, right_pad), mode='constant')
    elif input_length > output_length:
        output_sequence = input_sequence[:output_length]
    else:
        output_sequence = input_sequence
    return output_sequence


def parse_snoring_coco_total(dataset_names: dict, modes: Union[List[str], str] = None) -> Tuple[list, list]:
    total_inputs = []
    total_targets = []
    if isinstance(modes, str):
        modes = [modes]

    if modes is None:
        modes = ['train', 'valid', 'test']
        
    for name, data_root in dataset_names.items():
        for mode in modes:
            coco_path = Path(data_root).joinpath(f'{mode}.json')
            inputs, targets = parse_snoring_coco(coco_path)
            total_inputs.extend(inputs)
            total_targets.extend(targets)
    return total_inputs, total_targets


def parse_snoring_coco(coco_path: str) -> Tuple[list, list]:
    with open(coco_path, newline='') as jsonfile:
        coco_data = json.load(jsonfile)
    return coco_data['waveform'], coco_data['annotations']


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



if __name__ == "__main__":
    pass
        