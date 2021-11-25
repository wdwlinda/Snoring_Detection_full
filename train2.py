import sys
import os
import torch
import numpy as np
import random
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from dataset.dataloader import AudioDataset
from utils import train_utils
from models.image_classification import img_classifier
from utils import configuration
CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\_cnn_train_config.yml'
logger = train_utils.get_logger('train')

sys.path.append("..")
from modules.train import trainer
from modules.utils import train_utils


def main(config_reference):
    # Configuration
    config = configuration.load_config(config_reference)
    checkpoint_path = train_utils.create_training_path(os.path.join(config.train.project_path, 'checkpoints'))
    pprint(config)
    
    # Set deterministic
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        train_utils.set_deterministic(manual_seed)

    # Dataloader
    train_dataset = AudioDataset(config, mode='train')
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.dataset.batch_size, shuffle=config.dataset.shuffle, pin_memory=config.train.pin_memory, num_workers=config.train.num_workers)
    # TODO: config.dataset.preprocess_config.mix_up = None
    config.dataset.preprocess_config.mix_up = None
    valid_dataset = AudioDataset(config, mode='valid')
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, pin_memory=config.train.pin_memory, num_workers=config.train.num_workers)

    # Logger
    # TODO: train_logging
    # TODO: messive
    logger.info("Start Training!!")
    logger.info("Training epoch: {} Batch size: {} Shuffling Data: {} Training Samples: {}".
            format(config.train.epoch, config.dataset.batch_size, config.dataset.shuffle, len(train_dataloader.dataset)))
    
    train_utils._logging(os.path.join(checkpoint_path, 'logging.txt'), config, access_mode='w+')
    experiment = os.path.basename(checkpoint_path)
    config['experiment'] = experiment
    ckpt_dir = os.path.join(config.train.project_path, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)
    train_utils.train_logging(os.path.join(ckpt_dir, 'train_logging.txt'), config)

    # Model
    model = img_classifier.ImageClassifier(
        backbone=config.model.name, in_channels=config.model.in_channels, activation=config.model.activation,
        out_channels=config.model.out_channels, pretrained=config.model.pretrained, dim=1, output_structure=None)

    # Optimizer
    optimizer = train_utils.create_optimizer(config.optimizer_config, model)

    # Criterion
    criterion = train_utils.get_loss(config.train.loss)

    # Training
    trainer_instance = trainer.Trainer(config,
                                       model, 
                                       criterion, 
                                       optimizer, 
                                       train_dataloader, 
                                       valid_dataloader,
                                       logger,
                                       device=config.device,
                                       checkpoint_path=checkpoint_path,
                                       )

    trainer_instance.fit()


if __name__ == '__main__':
    main(CONFIG_PATH)