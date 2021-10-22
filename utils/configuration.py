import argparse
import torch
import yaml
from utils import train_utils

logger = train_utils.get_logger('ConfigLoader')


def get_device(config):
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config


def load_config(config_reference=None):
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

    config = get_device(config)
    config = train_utils.DictAsMember(config)
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))