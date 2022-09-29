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
import torchaudio

import site_path
CONFIG_PATH = 'config/_cnn_train_config.yml'
from models.image_classification.img_classifier import ImageClassifier
from inference import test
from dataset import transformations
from dataset import input_preprocess
from dataset.time_transform import get_wav_transform, augmentation
from dataset.dataloader import AudioDataset, AudioDatasetfromNumpy
from dataset.get_dataset_name import get_dataset, get_dataset_wav
from dataset.data_transform import WavtoMelspec_torchaudio
from utils import configuration
from utils import trainer
from utils import train_utils

logger = train_utils.get_logger('train')


def run_train(config):
    # Set deterministic
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        train_utils.set_deterministic(manual_seed, random, np, torch)
            
    # Dataloader
    # train_dataset = AudioDatasetfromNumpy(config, mode='train')
    # valid_dataset = AudioDatasetfromNumpy(config, mode='valid')
    # if config.dataset.wav_transform:
    #     t = time_transform.augmentation()
    # else:
    #     t = None
    # train_dataset = SnoringDataset(data_train)
    # valid_dataset = SnoringDataset(data_valid)
    train_dataset = AudioDataset(config, mode='train')
    valid_dataset = AudioDataset(config, mode='valid')

    drop_last = True if config['TRAIN']['MIXUP'] else False
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
    model = ImageClassifier(
        backbone=config.model.name, in_channels=config.model.in_channels,
        out_channels=config.model.out_channels, pretrained=config.model.pretrained, 
        dim=1, output_structure=None)

    # Optimizer
    optimizer = train_utils.create_optimizer_temp(config.optimizer_config, model)

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
    lr_scheduler = train_utils.create_lr_scheduler(config.optimizer_config.lr_config, optimizer)

    # Final activation
    valid_activation = train_utils.create_activation(config.model.activation)

    # Training
    
    transform = WavtoMelspec_torchaudio(
        sr=16000,
        n_class=config.model.out_channels,
        preprocess_config=config.dataset.preprocess_config,
        is_mixup=config.TRAIN.MIXUP,
        is_spec_transform=config.dataset.is_data_augmentation,
        is_wav_transform=config.dataset.wav_transform,
        device=configuration.get_device()
    )    
    valid_transform = WavtoMelspec_torchaudio(
        sr=16000,
        n_class=config.model.out_channels,
        preprocess_config=config.dataset.preprocess_config,
        is_mixup=False,
        is_spec_transform=False,
        is_wav_transform=False,
        device=configuration.get_device()
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
        device=configuration.get_device(),
        **train_config
    )

    trainer_instance.fit()


def main():
    now = datetime.now()
    
    # Configuration
    config = configuration.load_config(CONFIG_PATH, dict_as_member=False)
    all_checkpoint_path = os.path.join(config['TRAIN']['project_path'], 'checkpoints')
    train_utils.clear_empty_dir(all_checkpoint_path)
    
    # train & valid dataset
    dataset_paths = get_dataset()
    dataset_paths = get_dataset_wav()
    # test dataset
    test_dataset = configuration.load_config('dataset/dataset.yml')

    config_list = []
    for model_name in [
        'convnext_tiny_384_in22ft1k', 
    ]:
        for index_path in dataset_paths:
            # test_dataset2 = set(test_dataset['dataset_wav'].items()) ^ set(index_path['train'].items())
            test_dataset = { k : test_dataset['dataset_wav'][k] for k in set(test_dataset['dataset_wav']) - set(index_path['train']) }
            for mixup in [True, False]: 
                for wav_transform in [True, False]:
                    for is_aug in [True, False]:
                    # for feature in ['mel-spec']:
                        config = copy.deepcopy(config)
                        config['model']['name'] = model_name
                        config['dataset']['index_path'] = index_path
                        config['dataset']['is_data_augmentation'] = is_aug
                        config['dataset']['wav_transform'] = wav_transform
                        # config['dataset']['transform_methods'] = feature
                        config['TRAIN']['MIXUP'] = mixup
                        checkpoint_path = train_utils.create_training_path(all_checkpoint_path)
                        config['CHECKPOINT_PATH'] = checkpoint_path
                        
                        config_list.append(config)

    for run_idx, config in enumerate(config_list):
        # try:
            # XXX: 
            # if run_idx == 0: continue
            dataset = '--'.join(list(config['dataset']['index_path']['train'].keys()))
            checkpoint_dir = os.path.split(config['CHECKPOINT_PATH'])[1]
            now = datetime.now()
            currentDay = str(now.day)
            currentMonth = str(now.month)
            currentYear = str(now.year)
            # exp_name = f"Snoring_Detection_new_model_{currentYear}_{currentMonth}_{currentDay}"
            exp_name = f"Snoring_spec_mixup_norm"
            mlflow.set_experiment(exp_name)
            # TODO: add model name as param and change run_name
            with mlflow.start_run(run_name=config['model']['name']):
                mlflow.log_param('dataset', dataset)
                mlflow.log_param('is_data_augmentation', config['dataset']['is_data_augmentation'])
                mlflow.log_param('pretrained', config['model']['pretrained'])
                mlflow.log_param('mixup', config['TRAIN']['MIXUP'])
                mlflow.log_param('wav_transform', config['dataset']['wav_transform'])
                # mlflow.log_param('feature', config['dataset']['transform_methods'])
                mlflow.log_param('checkpoint', checkpoint_dir)
                # config['CHECKPOINT_PATH'] = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_353'
                config['eval'] = {
                    'restore_checkpoint_path': config['CHECKPOINT_PATH'],
                    'checkpoint_name': r'ckpt_best.pth'
                }
                config = train_utils.DictAsMember(config)

                run_train(config)
                total_acc = []
                for test_data_name, test_path in test_dataset.items():
                # for test_data_name, test_path in test_dataset['dataset'].items():
                    # if test_data_name not in ['iphone11_0908', 'iphone11_0908_2', 'pixel_0908', 'pixel_0908_2']: continue
                    # if test_data_name not in ['web_snoring']: continue

                    src_dir = test_path
                    dist_dir = os.path.join(config['CHECKPOINT_PATH'], test_data_name)
                    acc, precision, recall = test(src_dir, dist_dir, config)
                    mlflow.log_metric(f'{test_data_name}_acc', acc)
                    if test_data_name in ['ASUS_snoring_train', 'ASUS_snoring_test', 'ESC50']:
                        mlflow.log_metric(f'{test_data_name}_precision', precision)
                        mlflow.log_metric(f'{test_data_name}_recall', recall)
                    total_acc.append(acc)
                acc_mean = sum(total_acc) / len(total_acc)
                mlflow.log_metric(f'mean_acc', acc_mean)
                    
        # except RuntimeError:
        #     print('RuntimeError')


if __name__ == '__main__':
    main()