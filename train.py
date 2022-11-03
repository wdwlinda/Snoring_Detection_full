import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.snoring_model import create_snoring_model
from dataset.dataloader import AudioDatasetCOCO
from dataset.data_transform import WavtoMelspec_torchaudio
from utils import configuration
from utils import trainer
from utils import train_utils

logger = train_utils.get_logger('train')


def run_train(config):
    # Dataloader
    train_dataset = AudioDatasetCOCO(config, modes='train')
    valid_dataset = AudioDatasetCOCO(config, modes='valid')

    drop_last = True if config.TRAIN.MIXUP else False
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.dataset.batch_size, shuffle=config.dataset.shuffle, 
        pin_memory=config.TRAIN.pin_memory, num_workers=config.TRAIN.num_workers, drop_last=drop_last)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, pin_memory=config.TRAIN.pin_memory, 
        num_workers=config.TRAIN.num_workers, drop_last=drop_last)

    # Logger
    logger.info('Start Training!!')
    logger.info((f'Training epoch: {config.TRAIN.EPOCH} '
                 f'Batch size: {config.dataset.batch_size} '
                 f'Shuffling Data: {config.dataset.shuffle} '
                 f' Training Samples: {len(train_dataloader.dataset)}'))
    # train_utils.config_logging(
    #     os.path.join(checkpoint_path, 'logging.txt'), config, access_mode='w+')

    # Model
    model = create_snoring_model(config)

    # Optimizer
    optimizer = train_utils.create_optimizer(config.optimizer_config, model)

    # Criterion (Loss function)
    def criterion_wrap(outputs, labels):
        criterion = train_utils.create_criterion(config.TRAIN.loss)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(outputs, labels.long())
            # loss = criterion(outputs, torch.argmax(labels.long(), axis=1))
        else:
            loss = criterion(outputs.float(), labels.float())
        return loss
    # criterion = train_utils.create_criterion(config.TRAIN.loss)

    # lR scheduler
    lr_scheduler = train_utils.create_lr_scheduler(
        config.optimizer_config.lr_config, optimizer)

    # Final activation
    valid_activation = train_utils.create_activation(config.model.activation)

    # Training
    transform = WavtoMelspec_torchaudio(
        sr=config.dataset.sample_rate,
        n_class=config.model.out_channels,
        preprocess_config=config.dataset.preprocess_config,
        is_mixup=config.TRAIN.MIXUP,
        is_spec_transform=config.dataset.is_data_augmentation,
        is_wav_transform=config.dataset.wav_transform,
        device=config.device
    )    
    valid_transform = WavtoMelspec_torchaudio(
        sr=config.dataset.sample_rate,
        n_class=config.model.out_channels,
        preprocess_config=config.dataset.preprocess_config,
        is_mixup=False,
        is_spec_transform=False,
        is_wav_transform=False,
        device=config.device
    )    
    train_config = {
        'n_class': config.model.out_channels,
        'exp_path': config['CHECKPOINT_PATH'],
        'lr_scheduler': lr_scheduler,
        'train_epoch': config.TRAIN.EPOCH,
        'batch_size': config.dataset.batch_size,
        'valid_activation': valid_activation,
        'checkpoint_saving_steps': config.TRAIN.CHECKPOINT_SAVING_STEPS,
        'history': config.TRAIN.INIT_CHECKPOINT,
        'patience': config.TRAIN.PATIENCE,
        'batch_transform': transform,
        'batch_valid_transform': valid_transform,
    }
    trainer_instance = trainer.Trainer(
        model, 
        criterion_wrap, 
        optimizer, 
        train_dataloader, 
        valid_dataloader,
        logger,
        device=config.device,
        **train_config
    )

    trainer_instance.fit()


