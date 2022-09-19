import numpy as np
import torchaudio
import torch
from timm.data.mixup import Mixup

from dataset import input_preprocess





def transform(input_var, target_var=None, device='cuda:0', is_wav_transform=True, mixup=True, is_spec_transform=True,
              n_class=2, sr=16000, preprocess_config=None):
    if mixup:
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

    # wav_transform = augmentation()

    if is_wav_transform:
        # input_var = input_var.detach().cpu().numpy()
        # input_var = wav_transform(input_var, train_dataset.dataset_config.sample_rate)
        # input_var = torch.from_numpy(input_var.copy())
        if np.random.rand() > 0.5:
            # XXX: how to define nosie?
            input_var = input_var + 50*torch.randn_like(input_var)+50
        if np.random.rand() > 0.5:
            input_var = torch.flip(input_var, dims=[1])
    if mixup and target_var is not None:
        input_var, target_var = mixup_fn(input_var, torch.argmax(target_var, 1))

    # Melspec (trorchaudio)   
    n_fft = 2048
    n_mels = 128
    noise = True
    torchaudio_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=512,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=n_mels,
    )
    torchaudio_melspec.to(device)
    power_to_db = torchaudio.transforms.AmplitudeToDB()
    power_to_db.to(device)
    input_var = torchaudio_melspec(input_var)
    input_var = power_to_db(input_var)

    if is_spec_transform and preprocess_config is not None:
        input_var = input_preprocess.spectrogram_augmentation(
            input_var, **preprocess_config)
    input_var = torch.unsqueeze(input_var, dim=1)

    # if noise:
    #     input_var = input_var + torch.rand(input_var.shape[0], input_var.shape[1]) * np.random.rand() / 10
    #     input_var = torch.roll(input_var, np.random.randint(-10, 10), 0)

    # xx = input_var.detach().cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.imshow(xx[0, 0])
    # plt.show()

    input_var = torch.tile(input_var, (1, 3, 1, 1))
    return input_var, target_var