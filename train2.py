import sys
import os
import torch
import numpy as np
import random
from pprint import pprint
from torch.utils.data import Dataset, DataLoader

import site_path
from dataset.dataloader import AudioDataset
from utils import configuration
CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\_cnn_train_config.yml'

from modules.model import img_classifier
from modules.train import trainer
from modules.utils import train_utils

logger = train_utils.get_logger('train')


def main(config_reference):
    # Configuration
    config = configuration.load_config(config_reference)
    all_checkpoint_path = os.path.join(config.train.project_path, 'checkpoints')
    checkpoint_path = train_utils.create_training_path(all_checkpoint_path)
    config['checkpoint_path'] = checkpoint_path
    pprint(config)
    
    # Set deterministic
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        train_utils.set_deterministic(manual_seed, random, np, torch)
            
    # Dataloader
    train_dataset = AudioDataset(config, mode='train')
    valid_dataset = AudioDataset(config, mode='valid')
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.dataset.batch_size, shuffle=config.dataset.shuffle, pin_memory=config.train.pin_memory, num_workers=config.train.num_workers)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, pin_memory=config.train.pin_memory, num_workers=config.train.num_workers)

    # Logger
    logger.info("Start Training!!")
    logger.info("Training epoch: {} Batch size: {} Shuffling Data: {} Training Samples: {}".
            format(config.train.epoch, config.dataset.batch_size, config.dataset.shuffle, len(train_dataloader.dataset)))
    train_utils.config_logging(os.path.join(checkpoint_path, 'logging.txt'), config, access_mode='w+')

    # Model
    model = img_classifier.ImageClassifier(
        backbone=config.model.name, in_channels=config.model.in_channels,
        out_channels=config.model.out_channels, pretrained=config.model.pretrained, dim=1, output_structure=None)

    # Optimizer
    optimizer = train_utils.create_optimizer_temp(config.optimizer_config, model)

    # Criterion (Loss function)
    def criterion_wrap(outputs, labels):
        criterion = train_utils.create_criterion(config.train.loss)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(outputs, torch.argmax(labels.long(), axis=1))
        else:
            loss = criterion(outputs, labels)
        return loss
    # criterion = train_utils.create_criterion(config.train.loss)

    # Final activation
    activation_func = train_utils.create_activation(config.model.activation)

    # Training
    trainer_instance = trainer.Trainer(config,
                                       model, 
                                       criterion_wrap, 
                                       optimizer, 
                                       train_dataloader, 
                                       valid_dataloader,
                                       logger,
                                       device=config.device,
                                       activation_func=activation_func,
                                       )

    trainer_instance.fit()


if __name__ == '__main__':
    main(CONFIG_PATH)