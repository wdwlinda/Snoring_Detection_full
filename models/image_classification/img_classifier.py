import logging
import torch.nn as nn
from models import layers
from models import backbones
from models import utils
creat_torchvision_backbone = backbones.creat_torchvision_backbone
creat_timm_backbone = backbones.creat_timm_backbone
MultiLayerPerceptron = layers.MultiLayerPerceptron
get_activation = utils.get_activation


class ImageClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, output_structure=None, activation=None, backbone='resnet50', 
                 pretrained=True, *args, **kwargs):
        super(ImageClassifier, self).__init__()
        self.out_channels = out_channels
        self.output_structure = output_structure
        if 'resnet' in backbone or 'resnext' in backbone:
            self.encoder = creat_torchvision_backbone(in_channels, backbone, pretrained, final_flatten=True)
        elif backbone in ['efficientnet_b0', 'efficientnet_b4']:
            self.encoder = creat_timm_backbone(in_channels, backbone, pretrained, final_flatten=True)
        else:
            raise ValueError(f"Unknown encoder: {backbone}")
        if in_channels != 3 and pretrained:
            logging.info('Reinitialized first layer')

        module = list(self.encoder.children())[0]
        encoder_out_node = list(module.children())[-1].out_features
        if output_structure:
            output_structure = [encoder_out_node] + output_structure + [out_channels]
            self.mlp = MultiLayerPerceptron(output_structure, 'relu', out_activation=activation)
        else:
            self.mlp = MultiLayerPerceptron([encoder_out_node, out_channels], 'relu', out_activation=activation)

        # self.fc1 = nn.Linear(encoder_out_node, out_channels)
        # self.activation_func = get_activation(activation, *args, **kwargs) if activation else None
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        
        # x = self.fc1(x)
        # if self.activation_func:
        #     x = self.activation_func(x)
        return x