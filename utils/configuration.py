import argparse
import torch
import yaml
from utils import train_utils

logger = train_utils.get_logger('ConfigLoader')


def get_device(device_str=None):
    # Get a device to train on
    if device_str is not None:
        logger.info(f"Device: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    return device


def load_config(config_reference=None, dict_as_member=False):
    if isinstance(config_reference, str):
        parser = argparse.ArgumentParser(description='DL')
        if config_reference is not None:
            parser.add_argument('--config', type=str, help='Path to the YAML config file', default=config_reference)
        else:
            parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
        args = parser.parse_args()
        config = _load_config_yaml(args.config)
    elif isinstance(config_reference, dict):
        config = config_reference

    if dict_as_member:
        config = train_utils.DictAsMember(config)
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))