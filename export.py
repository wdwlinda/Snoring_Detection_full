from pathlib import Path

import torch
from torch import nn
import numpy as np

from models.snoring_model import create_snoring_model
from utils import configuration


def export_onnx(config):
    restore_path = Path(config.model.restore_path)
    config['model']['restore_path'] = str(restore_path.joinpath(config.model.checkpoint_name))
    save_path = str(restore_path.parent)
    model = create_snoring_model(config)
    deploy(config, model, save_filename=save_path)


class Audio16BitNorm(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.register_buffer('b', b)
        
    def forward(self, x):
        x = x / self.b
        return x


def deploy(config, model, save_filename):
    from deploy.snoring_model_deploy import model_to_onnx

    model = torch.nn.Sequential(
        Audio16BitNorm(b=torch.tensor(32768)),
        model,
        torch.nn.Softmax(1)
    )
    dummy_input = np.float32(np.random.rand(1, 32000))
    restore_path = Path(config.model.restore_path).parent
    run_id = restore_path.name
    save_filename = restore_path.joinpath(f'{config.model.name}_{run_id}.onnx')
    model_to_onnx(dummy_input, model, str(save_filename))


if __name__ == '__main__':
    CONFIG_PATH = r'config/_cnn_train_config.yml'
    config = configuration.load_config(CONFIG_PATH, dict_as_member=True)
    export_onnx(config)