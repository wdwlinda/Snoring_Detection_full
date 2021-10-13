import torchaudio.transforms as T
import torch
import numpy as np




# TODO: time stratch valueerror
def spectrogram_augmentation(spec, is_freq_masking, is_time_masking, is_time_strech, freq_mask_param, time_mask_param, min_rate, max_rate):
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