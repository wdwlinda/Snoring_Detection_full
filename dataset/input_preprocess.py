import torchaudio.transforms as T
import torch
import numpy as np
from dataset import transformations


def spectrogram_augmentation(spec, is_freq_masking, is_time_masking, is_time_strech, freq_mask_param, time_mask_param, min_rate, max_rate, **kwargs):
    # SpecAug
    if is_freq_masking:
        freq_masking = T.FrequencyMasking(freq_mask_param)
        spec = freq_masking(spec)
    
    if is_time_masking:
        time_masking = T.TimeMasking(time_mask_param)
        spec = time_masking(spec)

    if is_time_strech:
        strech = T.TimeStretch()
        rate = np.random.uniform(min_rate, max_rate)
        spec = strech(spec, rate)
    return spec


def mix_up(waveform1, waveform2):
    waveform1 = waveform1 - waveform1.mean()
    waveform2 = waveform2 - waveform2.mean()

    if waveform1.shape[1] != waveform2.shape[1]:
        if waveform1.shape[1] > waveform2.shape[1]:
            # padding
            temp_wav = np.zeros((1, waveform1.shape[1]), np.float32)
            temp_wav[0, 0:waveform2.shape[1]] = waveform2
            waveform2 = temp_wav
        else:
            # cutting
            waveform2 = waveform2[0, 0:waveform1.shape[1]]
            
    # sample lambda from uniform distribution
    #mix_lambda = random.random()
    # sample lambda from beta distribtion
    mix_lambda = np.random.beta(10, 10)
    # mix_lambda = np.random.beta(8, 4)

    mix_waveform = mix_lambda * waveform1 + (1-mix_lambda) * waveform2
    waveform = mix_waveform - mix_waveform.mean()
    return waveform, mix_lambda