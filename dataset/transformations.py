import matplotlib.pyplot as plt
import librosa.display
from pydub import AudioSegment
import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


def get_audio_features(waveform, sample_rate, transform_methods, transform_config):
    features = {}
    if isinstance(transform_methods, str):
        transform_methods = [transform_methods]
    for method in transform_methods:
        if method == 'fbank':
            spec = fbank(waveform, sample_rate, **transform_config)
        elif method == 'spec':
            spec = spectrogram(waveform, sample_rate, **transform_config)
        elif method == 'mel-spec':
            spec = mel_spec(waveform, sample_rate, **transform_config)
            # spec = np.log2(spec)
        elif method == 'MFCC':
            spec = MFCC(waveform, sample_rate, **transform_config)
        else:
            raise ValueError('Unknown audio transformations')
        # print(features[method].size())

        if transform_config.mean_norm:
            # spec -= (torch.mean(spec, axis=0) + 1e-8)
            spec -= (np.mean(spec, axis=0) + 1e-8)

        features[method] = spec
    return features


def spectrogram(waveform, n_fft, **kwargs):
    return T.Spectrogram(n_fft, **kwargs)(waveform)


# def mel_spec(waveform, sample_rate, **kwargs):
#     melkwargs = {
#       'n_fft': kwargs.get('n_fft'),
#       'n_mels': kwargs.get('n_mels'),
#       'hop_length': kwargs.get('hop_length'),
#       'mel_scale': 'htk',
#     #   'normalized': True
#     }
#     return T.MelSpectrogram(sample_rate, **melkwargs)(torch.from_numpy(waveform)).numpy()


def mel_spec(waveform, sample_rate, **kwargs):
    waveform = waveform[0]
    return librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=256, hop_length=128, n_mels=128, fmax=8000)
    # return librosa.feature.melspectrogram(y=waveform, sr=sample_rate)


def fbank(waveform, sample_rate, **kwargs):
    melkwargs = {
      'num_mel_bins': kwargs.get('n_mels'),
      'htk_compat': True,
      'window_type': 'hanning',
      'use_energy': False,
      'dither': 0.0,
      'frame_shift': 10
    }
    return torchaudio.compliance.kaldi.fbank(torch.from_numpy(waveform), sample_frequency=sample_rate, **melkwargs).numpy()
    


def MFCC(waveform, sample_rate, n_mfcc, **kwargs):
    # melkwargs = {
    #   'n_fft': kwargs.get('n_fft'),
    #   'n_mels': kwargs.get('n_mels'),
    #   'hop_length': kwargs.get('hop_length'),
    #   'mel_scale': 'htk',
    # }
    # return T.MFCC(sample_rate, n_mfcc, melkwargs=melkwargs)(waveform)
    # TODO: check feature extraction method
    waveform = waveform[0]
    return librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)