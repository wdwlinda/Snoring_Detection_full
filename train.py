from pathlib import Path
import sys
import os
import random
import copy
from datetime import datetime
from typing import Any

from pprint import pprint
import numpy as np
from sklearn.metrics import explained_variance_score
import torch
import mlflow
from torch.utils.data import Dataset, DataLoader
import torchaudio

CONFIG_PATH = 'config/_cnn_train_config.yml'
from models.image_classification.img_classifier import ImageClassifier
from models.snoring_model import create_snoring_model
from inference import run_test
from dataset import transformations
from dataset import input_preprocess
from dataset.time_transform import get_wav_transform, augmentation
from dataset.dataloader import AudioDatasetCOCO
from dataset.get_dataset_name import get_dataset
from dataset.data_transform import WavtoMelspec_torchaudio
from utils import configuration
from utils import trainer
from utils import train_utils

logger = train_utils.get_logger('train')


# TODO: get_device issue

def run_train(config):
    # Set deterministic
    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        train_utils.set_deterministic(manual_seed, random, np, torch)
            
    # Dataloader
    train_dataset = AudioDatasetCOCO(config, modes='train')
    valid_dataset = AudioDatasetCOCO(config, modes='valid')

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
    model = create_snoring_model(config)

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


def assign_exp_value(run_config: dict, keys: str, assign_val: Any) -> dict:
    """AI is creating summary for assign_exp_value

    Args:
        run_config (dict): A config for single experiment running 
                           (usually a nested dict)
        keys (str): The structure key to access parameters in run_config, 
                           e.g., config.model.name
        assign_val (Any): The modified value for assigned parameter, 
                          e.g., config.model.name = 'resnet50'

    Returns:
        (dict): Modified config
    """
    init_config = run_config
    key_list = keys.split('.')
    for key in key_list[:-1]:
        if isinstance(run_config, dict):
            run_config = run_config.get(key, None)
    run_config[key_list[-1]] = assign_val
    return init_config


def get_exp_configs(config: dict, exp_config: dict) -> dict:
    """AI is creating summary for get_exp_configs

    Args:
        config (dict): [description]
        exp_config (dict): [description]

    Returns:
        dict: [description]
    """
    # XXX: Organize code
    def get_configs(root, exp_config, idx):
        params = exp_config[idx]
        new_root = []
        for node in root:
            for p in params:
                new_node = node.copy()
                new_node.append(p)
                new_root.append(new_node)

        idx += 1
        if idx < len(exp_config):
            new_root = get_configs(new_root, exp_config, idx)
        return new_root
    
    root = [[]]
    exp_config_list = list(exp_config.values())  
    params_list = get_configs(root, exp_config_list, 0)
    configs = []
    for params in params_list:
        config_temp = copy.deepcopy(config)
        for idx, param_name in enumerate(exp_config):
            config_temp = assign_exp_value(config_temp, param_name, params[idx])
        configs.append(config_temp)

    return configs
    

def main():
    now = datetime.now()
    
    # Configuration
    config = configuration.load_config(CONFIG_PATH, dict_as_member=False)
    checkpoint_root = os.path.join(config['TRAIN']['project_path'], 'checkpoints')
    train_utils.clear_empty_dir(checkpoint_root)
    
    # train & valid dataset
    dataset_paths = get_dataset()

    # test dataset
    test_dataset = configuration.load_config('dataset/dataset.yml')['data_pre_root']
    # XXX: temporally
    no_test = [
        'Audioset_snoring_strong_0.8',
        'Audioset_snoring_strong_0.6',
        'Audioset_snoring_strong_0.4',
        'Audioset_snoring_strong_0.2',
    ]
    for name in no_test:
        test_dataset.pop(name)

    # TODO: This is only for simple grid search, apply NII hyper-params search later
    # TODO: This should in order to decide which params to be test first?
    exp_config = {
        'dataset.index_path': dataset_paths,
        'model.name': ['pann.ResNet22'],
        # 'model.name': ['pann.ResNet38'],
        # 'model.name': ['timm.resnet34', 'timm.convnext_tiny_384_in22ft1k'],
        # 'model.name': ['pann.ResNet38', 'pann.ResNet54', 'pann.MobileNetV2', 'timm.resnet34', 
        'model.pretrained': [True],
        # 'convnext_tiny_384_in22ft1k'],
        # 'dataset.wav_transform': [True],
        # 'model.dropout': [True, False]
        # 'TRAIN.MIXUP': [True, False],
        # 'dataset.is_data_augmentation': [True, False],
    }
    configs = get_exp_configs(config, exp_config)

    full_test_list = [
        'web_snoring', 'iphone11_0908', 'pixel_0908', 
        '0908_ori', 'iphone11_0908_2', 'pixel_0908_2'
    ]
    for run_idx, config in enumerate(configs):
        dataset = '--'.join(list(config['dataset']['index_path']))
        # dataset = '--'.join(list(config['dataset']['index_path']['train'].keys()))
        checkpoint_path = train_utils.create_training_path(checkpoint_root)
        config['CHECKPOINT_PATH'] = checkpoint_path
        checkpoint_dir = os.path.split(config['CHECKPOINT_PATH'])[1]
        now = datetime.now()
        currentDay = str(now.day)
        currentMonth = str(now.month)
        currentYear = str(now.year)
        # exp_name = f"Snoring_Detection_new_model_{currentYear}_{currentMonth}_{currentDay}"
        # exp_name = f"_Snoring_single_dataset_panns_pretrained_final_3"
        # exp_name = f"_Snoring_single_dataset_panns_data_ratio_final"
        exp_name = f"_Snoring_single_dataset_panns_model_final_2"
        mlflow.set_experiment(exp_name)
        # TODO: add model name as param and change run_name
        
        # XXX: Temp
        # for i in range(1, 17):
        #     restore_path = Path(r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints')
        #     restore_path = restore_path.joinpath(f'run_{700+i}')
        #     config['model']['restore_path'] = restore_path.joinpath('ckpt_best.pth')

        with mlflow.start_run(run_name=config['model']['name']):
            mlflow.log_param('dataset', dataset)
            mlflow.log_param('is_data_augmentation', config['dataset']['is_data_augmentation'])
            mlflow.log_param('pretrained', config['model']['pretrained'])
            mlflow.log_param('mixup', config['TRAIN']['MIXUP'])
            mlflow.log_param('wav_transform', config['dataset']['wav_transform'])
            mlflow.log_param('checkpoint', checkpoint_dir)
            # config['CHECKPOINT_PATH'] = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_742'
            # config['eval'] = {
            #     'restore_checkpoint_path': config['CHECKPOINT_PATH'],
            #     'checkpoint_name': r'ckpt_best.pth'
            # }
            config = train_utils.DictAsMember(config)

            run_train(config)
            total_acc = []

            if config['model']['restore_path'] is not None:
                config['model']['restore_path'] = Path(
                    config['model']['restore_path']).joinpath(config.model.checkpoint_name)
            else:
                config['model']['restore_path'] = Path(
                    config['CHECKPOINT_PATH']).joinpath(config.model.checkpoint_name)

            for test_data_name, test_path in test_dataset.items():
                if test_data_name in full_test_list:
                    split = None
                else:
                    split = 'test'

                print(test_data_name)
                src_dir = test_path
                dist_dir = os.path.join(config['CHECKPOINT_PATH'], test_data_name)

                acc, precision, recall = run_test(src_dir, dist_dir, config, split)

                mlflow.log_metric(f'{test_data_name}_acc', acc)
                precision_and_recall = full_test_list.copy()
                precision_and_recall.extend(['ASUS_snoring', 'ESC50'])
                if test_data_name in precision_and_recall:
                    mlflow.log_metric(f'{test_data_name}_precision', precision)
                    mlflow.log_metric(f'{test_data_name}_recall', recall)
                total_acc.append(acc)
            acc_mean = sum(total_acc) / len(total_acc)
            mlflow.log_metric(f'mean_acc', acc_mean)
                

if __name__ == '__main__':
    main()