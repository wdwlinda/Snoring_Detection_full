import time

import numpy as np
import torchaudio
import torch
from timm.data.mixup import Mixup
import matplotlib.pyplot as plt
from torch_audiomentations import (
    Compose, TimeInversion, Gain, AddColoredNoise, 
    PolarityInversion, PitchShift
)

from dataset import input_preprocess
from build.melspec import melspec


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.detach().cpu().numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show()


def plot_spectrogram_raw(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show()

    
class WavtoMelspec_torchaudio():
    def __init__(self, sr, n_class, preprocess_config, device, 
                 is_mixup, is_spec_transform, is_wav_transform,
                 n_fft=2048, hop_length=512, n_mels=128):
        self.device = device
        self.wav_to_melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels,
        )
        self.sr = sr
        self.wav_to_melspec.to(self.device)
        self.power_to_db = torchaudio.transforms.AmplitudeToDB()
        self.power_to_db.to(self.device)
        self.is_mixup = is_mixup
        self.is_wav_transform = is_wav_transform
        self.is_spec_transform = is_spec_transform

        if self.is_mixup:
            self.mixup_fn = get_mixup_fn(n_class)
        
        if self.is_spec_transform:
            self.preprocess_config = preprocess_config
            self.freq_masking = torchaudio.transforms.FrequencyMasking(
                self.preprocess_config['freq_mask_param'])
            self.time_masking = torchaudio.transforms.TimeMasking(
                self.preprocess_config['time_mask_param'])

        if self.is_wav_transform:
            self.wav_transform = get_wav_transform()

    def __call__(self, waveform, target):
        if self.is_wav_transform:
            waveform = self.wav_transform(waveform, self.sr)
            
        if self.is_mixup:
            waveform, target = self.mixup_fn(waveform, torch.argmax(target, 1))

        melspec = self.wav_to_melspec(waveform)
        melspec = self.power_to_db(melspec)

        if self.is_spec_transform:
            # XXX: args for freq_masking, time_masking
            if self.freq_masking is not None:
                melspec = self.freq_masking(melspec)

            if self.time_masking is not None:
                melspec = self.time_masking(melspec)

        # melspec = torch.unsqueeze(melspec, dim=1)
        # xx = input_var.detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(xx[0, 0])
        # plt.show()
        melspec = torch.tile(melspec, (1, 3, 1, 1))
        return melspec, target


def wav_to_spec_cpp(waveform, sr=16000, n_mels=128):
    waveform = waveform.detach().cpu().numpy()
    # XXX: Cpp code melspec is not using args currently insetead define in code.
    melspec_process = melspec.PyMelspec(sr, 12, 25, 10, n_mels, 50, 8000)
    power_to_db = torchaudio.transforms.AmplitudeToDB()

    # XXX: WavformtoMelspec --> WaveformtoMelspec
    total_mel = []
    for sample in waveform:
        mel = melspec_process.WavformtoMelspec(sample)
        mel = np.reshape(np.array(mel), [1, n_mels, -1])
        mel = torch.from_numpy(mel)
        total_mel.append(mel)
    total_mel = torch.cat(total_mel, dim=0)
    input_var = power_to_db(total_mel)
    return input_var


def waveform_transform(input_var):
    # input_var = input_var.detach().cpu().numpy()
    # input_var = wav_transform(input_var, train_dataset.dataset_config.sample_rate)
    # input_var = torch.from_numpy(input_var.copy())
    if np.random.rand() > 0.5:
        # XXX: how to define nosie?
        input_var = input_var + 50*torch.randn_like(input_var)+50
    if np.random.rand() > 0.5:
        input_var = torch.flip(input_var, dims=[1])
    return input_var


def get_mixup_fn(n_class):
    # XXX: modulize mixup
    mixup_args = {
        'mixup_alpha': 1.,
        'cutmix_alpha': 0.,
        'cutmix_minmax': None,
        'prob': 1.0,
        'switch_prob': 0.,
        'mode': 'batch',
        'label_smoothing': 0,
        'num_classes': n_class
    }
    mixup_fn  = Mixup(**mixup_args)
    return mixup_fn


def mixup_and_spec_transform(input_var, target_var=None, device='cuda:0', is_wav_transform=True, mixup=True, 
              is_spec_transform=True, n_class=2, sr=16000, preprocess_config=None):
    if mixup:
        mixup_fn = get_mixup_fn(n_class)

    # wav_transform = augmentation()
    if is_wav_transform:
        input_var = waveform_transform(input_var)

    if mixup and target_var is not None:
        input_var, target_var = mixup_fn(input_var, torch.argmax(target_var, 1))

    # Melspec (trorchaudio)   
    input_var = wav_to_spec_torchaudio(input_var, sr, device)

    if is_spec_transform and preprocess_config is not None:
        input_var = input_preprocess.spectrogram_augmentation(
            input_var, **preprocess_config)
    input_var = torch.unsqueeze(input_var, dim=1)

    # xx = input_var.detach().cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.imshow(xx[0, 0])
    # plt.show()

    input_var = torch.tile(input_var, (1, 3, 1, 1))
    return input_var, target_var


def transform(input_var, target_var=None, device='cuda:0', is_wav_transform=True, mixup=True, 
              is_spec_transform=True, n_class=2, sr=16000, preprocess_config=None):
    if mixup:
        mixup_fn = get_mixup_fn(n_class)

    if is_wav_transform:
        input_var = waveform_transform(input_var)

    if mixup and target_var is not None:
        input_var, target_var = mixup_fn(input_var, torch.argmax(target_var, 1))

    # TODO:
    # Melspec (trorchaudio)   
    # import time
    # t1 = time.time()
    input_var1 = wav_to_spec_torchaudio(input_var, sr, device)
    # t2 = time.time()
    # input_var2 = wav_to_spec_cpp(input_var)
    # t3 = time.time()
    # print(t3-t2, t2-t1)

    if is_spec_transform and preprocess_config is not None:
        input_var = input_preprocess.spectrogram_augmentation(
            input_var, **preprocess_config)
    input_var = torch.unsqueeze(input_var, dim=1)

    # xx = input_var.detach().cpu().numpy()

    # plot_spectrogram_raw(xx[0, 0], title="Spectrogram")    
    # # import matplotlib.pyplot as plt
    # # plt.imshow(xx[0, 0])
    # # plt.show()

    input_var = torch.tile(input_var, (1, 3, 1, 1))
    return input_var, target_var


def get_wav_transform():
    wav_transform = Compose(
        transforms=[
            Gain(
                min_gain_in_db=-15.0,
                max_gain_in_db=5.0,
                p=0.5,
            ),
            PolarityInversion(p=0.5, sample_rate=16000),
            PitchShift(p=0.5, sample_rate=16000),
            AddColoredNoise(p=0.5, sample_rate=16000),
            TimeInversion(p=0.5, sample_rate=16000),
        ]
    )
    return wav_transform