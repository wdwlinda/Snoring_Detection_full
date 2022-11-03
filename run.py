from pathlib import Path
import os
import copy
from datetime import datetime
from typing import Any

import mlflow

from train import run_train
from test import run_test
from dataset.get_dataset_name import get_dataset
from utils import configuration
from utils import train_utils

CONFIG_PATH = 'config/_cnn_train_config.yml'
logger = train_utils.get_logger('train')


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
    # Configuration
    config = configuration.load_config_and_setup(CONFIG_PATH, dict_as_member=False)
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

    full_test_list = [
        'web_snoring', 'iphone11_0908', 'pixel_0908', 
        '0908_ori', 'iphone11_0908_2', 'pixel_0908_2'
    ]
    app_datasets = test_dataset.copy()
    app_datasets.pop('ESC50')
    app_datasets.pop('Audioset_snoring_strong_repeat')
    app_datasets.pop('Kaggle_snoring_pad')
    app_datasets.pop('web_snoring')

    # TODO: This is only for simple grid search, apply NII hyper-params search later
    # TODO: This should in order to decide which params to be test first?
    # TODO: If not using NII then handle by config file instead of hard coding
    exp_config = {
        'dataset.index_path': dataset_paths,
        'model.name': ['pann.MobileNetV2', 'pann.MobileNetV1'],
        # 'model.name': ['pann.ResNet38'],
        # 'model.name': ['timm.resnet34', 'timm.convnext_tiny_384_in22ft1k'],
        # 'model.name': ['pann.ResNet38', 'pann.ResNet54', 'pann.MobileNetV2', 'timm.resnet34', 
        'model.pretrained': [True],
        'model.extra_extractor': ['dw_conv', 'conv', None],
        # 'convnext_tiny_384_in22ft1k'],
        # 'dataset.wav_transform': [True],
        # 'model.dropout': [True, False]
        # 'TRAIN.MIXUP': [True, False],
        # 'dataset.is_data_augmentation': [True, False],
    }
    configs = get_exp_configs(config, exp_config)

    for run_idx, config in enumerate(configs):
        train_datasets = list(config['dataset']['index_path'])
        dataset_tag = '--'.join(train_datasets)
        # dataset = '--'.join(list(config['dataset']['index_path']['train'].keys()))
        checkpoint_path = train_utils.create_training_path(checkpoint_root)
        config['CHECKPOINT_PATH'] = checkpoint_path
        checkpoint_dir = os.path.split(config['CHECKPOINT_PATH'])[1]
        # exp_name = f"Snoring_Detection_new_model_{currentYear}_{currentMonth}_{currentDay}"
        # exp_name = f"_Snoring_single_dataset_panns_pretrained_final_3"
        # exp_name = f"_Snoring_single_dataset_panns_data_ratio_final"
        exp_name = f"_Snoring_Architecture"
        mlflow.set_experiment(exp_name)
        # TODO: add model name as param and change run_name
        
        with mlflow.start_run(run_name=config['model']['name']):
            tags = {'model': config['model']['extra_extractor']}
            mlflow.set_tags(tags)
            mlflow.log_param('dataset', dataset_tag)
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

            # # TODO: Test logging model
            # from models.snoring_model import create_snoring_model
            # model = create_snoring_model(config)
            # mlflow.pytorch.log_model(model, 'model')

            run_train(config)
            total_mAP, total_AUC = [], []
            app_mAP, app_AUC = [], []

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

                mAP, AUC = run_test(src_dir, dist_dir, config, split)
                mlflow.log_metric(f'{test_data_name}_mAP', mAP)
                if test_data_name in train_datasets:
                    mlflow.log_metric(f'{test_data_name}_AUC', AUC)

                total_mAP.append(mAP)
                total_AUC.append(AUC)
                if test_data_name in app_datasets:
                    app_mAP.append(mAP)
                    app_AUC.append(AUC)

            mean_mAP = sum(total_mAP) / len(total_mAP)
            mean_AUC = sum(total_AUC) / len(total_AUC)
            mean_mAP_app = sum(app_mAP) / len(app_mAP)
            mean_AUC_app = sum(app_AUC) / len(app_AUC)
            mlflow.log_metric(f'mean_mAP', mean_mAP)
            mlflow.log_metric(f'mean_AUC', mean_AUC)
            mlflow.log_metric(f'mean_mAP_app', mean_mAP_app)
            mlflow.log_metric(f'mean_AUC_app', mean_AUC_app)


if __name__ == '__main__':
    main()