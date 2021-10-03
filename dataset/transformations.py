import matplotlib.pyplot as plt
import librosa.display
from pydub import AudioSegment
import os
import numpy as np
import torchaudio


def spectrogram(waveform, n_fft, **kwargs):
    return torchaudio.Spectrogram(n_fft, **kwargs)(waveform)


def mel_spec(waveform, n_fft, n_mels, sample_rate, **kwargs):
    return torchaudio.MelSpectrogram(sample_rate, n_fft, n_mels=n_mels, **kwargs)(waveform)


def fbank(waveform, sample_rate):
    return torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sample_rate, use_energy=False, 
        window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)


def MFCC(waveform, sample_rate, n_mfcc, **kwargs):
    return torchaudio.MFCC(sample_rate, n_mfcc, **kwargs)(waveform)