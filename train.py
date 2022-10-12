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

import site_path
CONFIG_PATH = 'config/_cnn_train_config.yml'
from models.image_classification.img_classifier import ImageClassifier
from models.snoring_model import create_snoring_model
from inference import test, run_test
from dataset import transformations
from dataset import input_preprocess
from dataset.time_transform import get_wav_transform, augmentation
from dataset.dataloader import AudioDataset, AudioDatasetfromNumpy, AudioDatasetCOCO
from dataset.get_dataset_name import get_dataset, get_dataset_wav, get_dataset_root
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
    # train_dataset = AudioDatasetfromNumpy(config, mode='train')
    # valid_dataset = AudioDatasetfromNumpy(config, mode='valid')
    # if config.dataset.wav_transform:
    #     t = time_transform.augmentation()
    # else:
    #     t = None
    train_dataset = AudioDatasetCOCO(config, modes='train')
    valid_dataset = AudioDatasetCOCO(config, modes='valid')
    # train_dataset = AudioDataset(config, mode='train')
    # valid_dataset = AudioDataset(config, mode='valid')

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
    model = create_snoring_model(config, config.model.name)
    
    
    # model = ImageClassifier(
    #     backbone=config.model.name, in_channels=config.model.in_channels,
    #     out_channels=config.model.out_channels, pretrained=config.model.pretrained, 
    #     dim=1, output_structure=None)
    # XXX: PANNS
    from models.PANNs.pann_model import get_pann_model
    model = get_pann_model(
        config.model.name, 16000, 2, 'cuda:0', pretrained=config.model.pretrained, strict=False,
    )
    # print(model)

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
    exp_config_l = list(exp_config.values())  
    cc = get_configs(root, exp_config_l, 0)
    configs = []
    for params in cc:
        config_temp = {}
        for param_name in exp_config:
            config_temp[param_name] = params
        configs.append(config_temp)

    # cc = get_configs(root, [['a1', 'a2'], ['b1', 'b2', 'b3'], ['c1']], 0)

        # for params in exp_config:
        #     if len(root) == len:

        #     get_configs()


    # for key in exp_config:
    #     c = []
    #     for params in exp_config[key]:
    #         c.append(params)
    #         for k in range(len(root)):
    #             root.append(exp_config[key])

    # configs = []
    # sorted(exp_config, key = lambda key: len(exp_config[key]))
    # for params, values in exp_config.items():
    #     new_config = copy.deepcopy(config)
    #     if len(values) == 1:
    #         val = values[0]
    #         new_config = assign_exp_value(new_config, params, val)
    #     else:
    #         for val in values:
    #             # XXX:
    #             new_config = assign_exp_value(new_config, params, val)
    #             configs.append(new_config)

    # XXX:
    configs = []
    config['model']['name'] = exp_config['model.name'][0]
    config['dataset']['index_path'] = exp_config['dataset.index_path'][0]
    config['dataset']['wav_transform'] = True
    configs.append(config)
    new_config = copy.deepcopy(config)
    new_config['dataset']['wav_transform'] = False
    configs.append(new_config)

    return configs
    

def main():
    now = datetime.now()
    
    # Configuration
    config = configuration.load_config(CONFIG_PATH, dict_as_member=False)
    all_checkpoint_path = os.path.join(config['TRAIN']['project_path'], 'checkpoints')
    train_utils.clear_empty_dir(all_checkpoint_path)
    
    # train & valid dataset
    dataset_paths = get_dataset()
    dataset_paths = get_dataset_wav()
    dataset_paths = get_dataset_root()
    # test dataset
    test_dataset = configuration.load_config('dataset/dataset.yml')

    # TODO: This is only for simple grid search, apply NII hyper-params search later
    exp_config = {
        'dataset.index_path': dataset_paths,
        'model.name': ['ResNet54'],
        # 'model.pretrained': [True, False],
        # 'model.name': ['ResNet38', 'ResNet54', 'MobileNetV2', 'resnet50', 
        # 'convnext_tiny_384_in22ft1k'],
        'dataset.wav_transform': [True, False],
        # 'TRAIN.MIXUP': [True, False],
        # 'dataset.is_data_augmentation': [True, False],
    }
    configs = get_exp_configs(config, exp_config)

    for run_idx, config in enumerate(configs):
        dataset = '--'.join(list(config['dataset']['index_path']))
        # dataset = '--'.join(list(config['dataset']['index_path']['train'].keys()))
        checkpoint_path = train_utils.create_training_path(all_checkpoint_path)
        config['CHECKPOINT_PATH'] = checkpoint_path
        checkpoint_dir = os.path.split(config['CHECKPOINT_PATH'])[1]
        now = datetime.now()
        currentDay = str(now.day)
        currentMonth = str(now.month)
        currentYear = str(now.year)
        # exp_name = f"Snoring_Detection_new_model_{currentYear}_{currentMonth}_{currentDay}"
        exp_name = f"_Snoring_single_dataset_panns_pretrained"
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
            # config['CHECKPOINT_PATH'] = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_402'
            config['eval'] = {
                'restore_checkpoint_path': config['CHECKPOINT_PATH'],
                'checkpoint_name': r'ckpt_best.pth'
            }
            config = train_utils.DictAsMember(config)

            run_train(config)
            total_acc = []
            for test_data_name, test_path in test_dataset['data_pre_root'].items():
            # for test_data_name, test_path in test_dataset['dataset_wav'].items():
            # for test_data_name, test_path in test_dataset.items():
            # for test_data_name, test_path in test_dataset['dataset'].items():
                # if test_data_name not in ['iphone11_0908', 'iphone11_0908_2', 'pixel_0908', 'pixel_0908_2']: continue
                if test_data_name in ['web_snoring', 'iphone11_0908', 'pixel_0908', '0908_ori']:
                    split = None
                else:
                    split = 'test'

                print(test_data_name)
                src_dir = test_path
                dist_dir = os.path.join(config['CHECKPOINT_PATH'], test_data_name)
                acc, precision, recall = run_test(src_dir, dist_dir, config, split)
                mlflow.log_metric(f'{test_data_name}_acc', acc)
                if test_data_name in ['ASUS_snoring', 'ESC50']:
                    mlflow.log_metric(f'{test_data_name}_precision', precision)
                    mlflow.log_metric(f'{test_data_name}_recall', recall)
                total_acc.append(acc)
            acc_mean = sum(total_acc) / len(total_acc)
            mlflow.log_metric(f'mean_acc', acc_mean)
                

if __name__ == '__main__':
    main()