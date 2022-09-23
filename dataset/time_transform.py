import os
import glob
import array

import torch
import torchaudio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Reverse, Normalize
import numpy as np
import torch_audiomentations
from dataset import dataset_utils


def get_wav_transform():
    augment = torch_audiomentations.Compose([
        torch_audiomentations.AddBackgroundNoise(min_amplitude=50, max_amplitude=100, p=0.5),
        torch_audiomentations.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        torch_audiomentations.PitchShift(min_semitones=-6, max_semitones=6, p=0.5),
        torch_audiomentations.TimeInversion()
        # torch_audiomentations.Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ],
    p=0.75)
    return augment

# TODO: mono for currently
def add_background_noise(waveform, snr_db):
    noise = torch.rand(size=(1, waveform.shape[1]), dtype=torch.float32) - 0.5
    # noise = noise[:, : waveform.shape[1]]

    speech_rms = waveform.norm(p=2)
    noise_rms = noise.norm(p=2)

    snr = 10 ** (snr_db / 20)
    scale = snr * noise_rms / speech_rms
    augmented = (scale * waveform + noise) / 2
    return augmented


# TODO: add rir_raw https://pytorch.org/audio/main/tutorials/audio_data_augmentation_tutorial.html#simulating-room-reverberation
def rir(waveform, rir_raw, sample_rate):
    """Simulating room reverberation"""
    rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
    rir = rir / torch.norm(rir, p=2)
    RIR = torch.flip(rir, [1])

    speech_ = torch.nn.functional.pad(waveform, (RIR.shape[1] - 1, 0))
    augmented = torch.nn.functional.conv1d(speech_[None, ...], RIR[None, ...])[0]
    return augmented


def time_reverse(waveform):
    pass


def augmentation():
    augment = Compose([
        AddGaussianNoise(min_amplitude=50, max_amplitude=100, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-6, max_semitones=6, p=0.5),
        # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        Reverse(),
        # Normalize()
    ], 
    p=0.75)
    return augment


# def augmentation(samples):
#     samples = samples[None]
#     augment = Compose([
#         AddGaussianNoise(min_amplitude=50, max_amplitude=100, p=0.5),
#         TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
#         PitchShift(min_semitones=-6, max_semitones=6, p=0.5),
#         Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
#     ])
    
#     # Augment/transform/perturb the audio data
#     augmented_samples = augment(samples=samples, sample_rate=16000)
#     augmented_samples = augmented_samples[0]
#     return augmented_samples


if __name__ == '__main__':
    test_path1 = r'test_data/1620055140118_4_152.98_154.98_004.wav'
    root = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\test_data'
    files = glob.glob(os.path.join(root, '*.wav'))
    for f in files:
        filename = os.path.split(f)[1]
        sound = dataset_utils.get_pydub_sound(f, 'wav', 16000, channels=1)
        wav1 = np.array(sound.get_array_of_samples(), np.float32)
        wav1_aug = augmentation(wav1)
        wav1_aug = array.array(sound.array_type, wav1_aug)
        new_sound = sound._spawn(wav1_aug)
        new_sound.export(f'test_data/TimeStretch/{filename}.wav', format='wav')
