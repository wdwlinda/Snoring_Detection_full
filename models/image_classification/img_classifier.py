from email.policy import strict
import sys
import logging
import os

import torch
import torch.nn as nn
import timm
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

# from models import backbones
from models import layers
# from modules.model import utils
# creat_torchvision_backbone = backbones.creat_torchvision_backbone
# creat_timm_backbone = backbones.creat_timm_backbone
MultiLayerPerceptron = layers.MultiLayerPerceptron
# get_activation = utils.get_activation
# from ....Model_deploy import deploy_torch
# m = deploy_torch.nodule_cls_main
from models.utils import ModelBuilder


class TimmImgClassifierBuilder(ModelBuilder):
    def __init__(self, common_config: dict, model_name: str, new_model_config: dict = None):
        super().__init__(common_config, model_name, new_model_config)

    def interface(self, common_config):
        model_name = common_config.model.name.split('.')[1]
        model_kwargs = {
            'in_channels': common_config.model.in_channels,
            'out_channels': common_config.model.out_channels,
            # 'activation': common_config.model.activation,
            'backbone': model_name,
            'pretrained': common_config.model.pretrained,
            'output_structure': None,
        }
        return model_kwargs
  
    def restore(self, model):
        if self.restore_path is not None:
            state_key = torch.load(self.restore_path, map_location=self.device)
            state_key['encoder'] = {
                '.'.join(layer_name.split('.')[1:]): layer 
                for layer_name, layer in state_key['net'].items()
                if layer_name.startswith('encoder')
            }
            model.encoder.load_state_dict(state_key['encoder'])
            model.encoder = model.encoder.to(self.device)

            state_key['mlp'] = {
                '.'.join(layer_name.split('.')[1:]): layer 
                for layer_name, layer in state_key['net'].items()
                if layer_name.startswith('mlp')
            }
            model.mlp.load_state_dict(state_key['mlp'])
            model.mlp = model.mlp.to(self.device)
        return model


def log_melspec(melspec, top_db=80.0):
    log_melspec = 10.0 * torch.log(torch.clamp(melspec, min=1e-10)) / torch.log(torch.tensor(10.0))
    log_melspec -= 10.0 * torch.log(torch.max(torch.tensor(1e-10), melspec.max())) / torch.log(torch.tensor(10.0))
    if top_db is not None:
        if top_db < 0:
            raise ValueError('top_db must be non-negative')
        log_melspec = torch.max(log_melspec, log_melspec.max() - torch.tensor(top_db))
    return log_melspec


class ImageClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, output_structure=None, activation=None, 
                 backbone='resnet50', pretrained=True, restore_path=None, device='cuda:0',
                 strict=True, replace_gelu=True,
                 *args, **kwargs):
        super(ImageClassifier, self).__init__()
        self.out_channels = out_channels
        self.output_structure = output_structure
        self.strict = strict
        self.encoder = timm.create_model(backbone, pretrained, in_chans=in_channels)
    
        # XXX
        if replace_gelu:
            self.replace_layers(
                self.encoder, 
                timm.models.layers.activations.GELU(), 
                nn.ReLU(),
                sets=True
            )

        self.restore_path = restore_path
        self.device = torch.device('cuda:0')
        
        if in_channels != 3 and pretrained:
            logging.info('Reinitialized first layer')

        encoder_out_node = 1000
        if output_structure:
            output_structure = [encoder_out_node] + output_structure + [out_channels]
            self.mlp = MultiLayerPerceptron(output_structure, 'relu', out_activation=activation)
        else:
            self.mlp = MultiLayerPerceptron([encoder_out_node, out_channels], 'relu', out_activation=activation)
        # self.mlp = self.mlp.to(self.device)
    
        sample_rate = 16000
        hop_size = 320
        window_size = 1024
        mel_bins = 64
        # fmin = 50
        # fmax = 14000
        fmin = 0
        fmax = None

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True, is_log=False)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=12, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        # XXX: device should outsied model, also related to restore
        self.spectrogram_extractor.to(self.device)
        self.logmel_extractor.to(self.device)
        self.spec_augmenter.to(self.device)
        self.bn0.to(self.device)

        # if self.restore_path is not None:
        #     self.restore()

    def forward(self, x):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        # log2 to replace log10 to make onnx works
        x = log_melspec(x, top_db=None)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if not self.training:
            import matplotlib.pyplot as plt
            plt.imshow(x[0, 0].detach().cpu().numpy())
            plt.show()

        if self.training:
            x = self.spec_augmenter(x)

        x = self.encoder(x)
        x = self.mlp(x)
        if self.training:
            x = torch.nn.Softmax()(x)
        return x
    
    def restore(self):
        state_key = torch.load(self.restore_path, map_location=self.device)
        state_key['encoder'] = {
            '.'.join(layer_name.split('.')[1:]): layer 
            for layer_name, layer in state_key['net'].items()
            if layer_name.startswith('encoder')
        }
        self.encoder.load_state_dict(state_key['encoder'])
        self.encoder = self.encoder.to(self.device)

        state_key['mlp'] = {
            '.'.join(layer_name.split('.')[1:]): layer 
            for layer_name, layer in state_key['net'].items()
            if layer_name.startswith('mlp')
        }
        self.mlp.load_state_dict(state_key['mlp'])
        self.mlp = self.mlp.to(self.device)
        
    # XXX:
    def replace_layers(self, model, old, new, sets):
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                self.replace_layers(module, old, new, sets)
            if sets:
                if n == 'act':
                    setattr(model, n, new)
        return model
    