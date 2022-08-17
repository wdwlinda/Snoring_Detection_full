import sys
import os
import random
import copy
from datetime import datetime

from pprint import pprint
import numpy as np
import torch
import mlflow

from torch.utils.data import Dataset, DataLoader
import site_path
from dataset.dataloader import AudioDataset, AudioDatasetfromNumpy
from utils import configuration
from utils import train_utils as local_train_utils
CONFIG_PATH = 'config/_cnn_train_config.yml'

from modules.model.image_calssification import img_classifier
from modules.train import trainer
from modules.utils import train_utils

logger = train_utils.get_logger('train')


def main(config):
    # Set deterministic
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        train_utils.set_deterministic(manual_seed, random, np, torch)
            
    # Dataloader
    train_dataset = AudioDatasetfromNumpy(config, mode='train')
    valid_dataset = AudioDatasetfromNumpy(config, mode='valid')
    # train_dataset = AudioDataset(config, mode='train')
    # valid_dataset = AudioDataset(config, mode='valid')

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.dataset.batch_size, shuffle=config.dataset.shuffle, 
        pin_memory=config.TRAIN.pin_memory, num_workers=config.TRAIN.num_workers)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False, pin_memory=config.TRAIN.pin_memory, 
        num_workers=config.TRAIN.num_workers)

    # Logger
    logger.info('Start Training!!')
    logger.info((f'Training epoch: {config.TRAIN.EPOCH} '
                 f'Batch size: {config.dataset.batch_size} '
                 f'Shuffling Data: {config.dataset.shuffle} '
                 f' Training Samples: {len(train_dataloader.dataset)}'))
    train_utils.config_logging(
        os.path.join(checkpoint_path, 'logging.txt'), config, access_mode='w+')

    # Model
    model = img_classifier.ImageClassifier(
        backbone=config.model.name, in_channels=config.model.in_channels,
        out_channels=config.model.out_channels, pretrained=config.model.pretrained, 
        dim=1, output_structure=None)

    # Optimizer
    optimizer = train_utils.create_optimizer_temp(config.optimizer_config, model)

    # Criterion (Loss function)
    def criterion_wrap(outputs, labels):
        criterion = train_utils.create_criterion(config.TRAIN.loss)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(outputs, torch.argmax(labels.long(), axis=1))
        else:
            loss = criterion(outputs.float(), labels.float())
        return loss
    # criterion = train_utils.create_criterion(config.TRAIN.loss)

    # lR scheduler
    lr_scheduler = train_utils.create_lr_scheduler(config.optimizer_config.lr_config, optimizer)

    # Final activation
    valid_activation = train_utils.create_activation(config.model.activation)

    # Training
    train_config = {
        'n_class': config.model.out_channels,
        'exp_path': config['CHECKPOINT_PATH'],
        'lr_scheduler': lr_scheduler,
        'train_epoch': config.TRAIN.EPOCH,
        'batch_size': config.dataset.batch_size,
        'valid_activation': valid_activation,
        'checkpoint_saving_steps': config.TRAIN.CHECKPOINT_SAVING_STEPS,
        'history': config.TRAIN.INIT_CHECKPOINT,
        'patience': config.TRAIN.PATIENCE
    }
    trainer_instance = trainer.Trainer(
        model, 
        criterion_wrap, 
        optimizer, 
        train_dataloader, 
        valid_dataloader,
        logger,
        device=configuration.get_device(),
        **train_config
    )

    trainer_instance.fit()


if __name__ == '__main__':
    today = datetime.today()
    now = datetime.now()
    # mlflow.set_tracking_uri("file:/.mlruns")
    # mlflow.set_tracking_uri("https://my-tracking-server:5000") <- set to remote server
    
    # Configuration
    config = configuration.load_config(CONFIG_PATH, dict_as_member=False)
    all_checkpoint_path = os.path.join(config['TRAIN']['project_path'], 'checkpoints')
    

    dataset1 = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\Freq2\2_21_2s_my2'
    dataset2 = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\index\Freq2\2_21_2s_my_esc'
    dataset3 = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\2_21_2s_my2'
    config_list = []
    for model_name in ['resnext50_32x4d', 'resnetv2_50', 'seresnext50_32x4d', 'efficientnet_b4']:
    # for model_name in [
    #     'resnetv2_101', 'resnetv2_50', 
    #     'seresnext101_32x4d', 'seresnext50_32x4d',
    #     'resnext101_32x4d', 'resnext50_32x4d',
    #     'efficientnet_b4', 'efficientnet_b7']:
        for is_aug in [True, False]:
            for index_path in [dataset3]:
                for feature in ['mel-spec']:
                    config = copy.deepcopy(config)
                    config['model']['name'] = model_name
                    config['dataset']['index_path'] = index_path
                    config['dataset']['is_data_augmentation'] = is_aug
                    config['dataset']['transform_methods'] = feature
                    checkpoint_path = train_utils.create_training_path(all_checkpoint_path)
                    config['CHECKPOINT_PATH'] = checkpoint_path
                    
                    config_list.append(config)

    for config in config_list:
        # pprint(config)
        try:
            dataset = os.path.split(config['dataset']['index_path'])[1]
            checkpoint_dir = os.path.split(config['CHECKPOINT_PATH'])[1]
            now = datetime.now()
            currentDay = str(now.day)
            currentMonth = str(now.month)
            currentYear = str(now.year)
            exp_name = f"Snoring_Detection_{currentYear}_{currentMonth}_{currentDay}"
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=config['model']['name']):
                mlflow.log_param('dataset', dataset)
                mlflow.log_param('is_data_augmentation', config['dataset']['is_data_augmentation'])
                mlflow.log_param('pretrained', config['model']['pretrained'])
                mlflow.log_param('feature', config['dataset']['transform_methods'])
                mlflow.log_param('checkpoint', checkpoint_dir)
                config = local_train_utils.DictAsMember(config)
                main(config)
        except:
            # TODO: log the error
            print('log error')