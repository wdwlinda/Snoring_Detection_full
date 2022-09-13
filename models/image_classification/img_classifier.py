from email.policy import strict
import sys
import logging
import os

import torch
import torch.nn as nn
import timm

# from models import backbones
from models import layers
# from modules.model import utils
# creat_torchvision_backbone = backbones.creat_torchvision_backbone
# creat_timm_backbone = backbones.creat_timm_backbone
MultiLayerPerceptron = layers.MultiLayerPerceptron
# get_activation = utils.get_activation


class ImageClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, output_structure=None, activation=None, 
                 backbone='resnet50', pretrained=True, restore_path=None, device='cuda:0',
                 strict=True, replace_gelu=True,
                 *args, **kwargs):
        super(ImageClassifier, self).__init__()
        self.out_channels = out_channels
        self.output_structure = output_structure
        self.strict = strict
        self.encoder = timm.create_model(backbone, pretrained)
    
        if replace_gelu:
            self.replace_layers(
                self.encoder, 
                timm.models.layers.activations.GELU(), 
                nn.ReLU(),
                sets=True
            )

        self.restore_path = restore_path
        self.device = device
        
        if in_channels != 3 and pretrained:
            logging.info('Reinitialized first layer')

        encoder_out_node = 1000
        if output_structure:
            output_structure = [encoder_out_node] + output_structure + [out_channels]
            self.mlp = MultiLayerPerceptron(output_structure, 'relu', out_activation=activation)
        else:
            self.mlp = MultiLayerPerceptron([encoder_out_node, out_channels], 'relu', out_activation=activation)
        # self.mlp = self.mlp.to(self.device)
        
        if self.restore_path is not None:
            self.restore()

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
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        # x = torch.sigmoid(x)
        return x
    
    # XXX:
    def replace_layers(self, model, old, new, sets):
        # for name,child in model._modules:
        #     print(model._modules, name, child)
        for n, module in model.named_children():
            # print(n, module)
            if len(list(module.children())) > 0:
                ## compound module, go inside it
                self.replace_layers(module, old, new, sets)
            if sets:
                if n == 'act':
                    setattr(model, n, new)

                # if module == old:
                # # if isinstance(module, old):
                #     ## simple module
                #     setattr(model, n, new)
        return model
        