import os
import time
import torch
import argparse
import yaml
import logging
import sys
import importlib
import torch.optim as optim
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import itertools
import torch.nn as nn    
# def predict(filename, model):



def get_displaying_step(steps, times=5):
    displaying_step = steps//times
    length = 0
    temp = displaying_step
    while (temp):
        temp = temp // 10
        length += 1
    displaying_step = round(displaying_step / 10**(length-1)) * 10**(length-1)
    return displaying_step


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# def minmax_norm(data):
#     data_shape = data.size()
#     data = data.view(data.size(0), -1)
#     # data -= data.min(1, keepdim=True)[0]
#     # data /= data.max(1, keepdim=True)[0]
#     data = (data-data.min(1, keepdim=True)[0]) / (data.max(1, keepdim=True)[0]-data.min(1, keepdim=True)[0])
#     data = data.view(data_shape)
#     return data

def minmax_norm(data):
    data_shape = data.size()
    data = data.view(data.size(0), -1)
    # data -= data.min(1, keepdim=True)[0]
    # data /= data.max(1, keepdim=True)[0]
    mins = data.min(1, keepdim=True)[0]
    maxs = data.max(1, keepdim=True)[0]
    
    data = (data-mins) / (maxs-mins)
    # if torch.sum(torch.isnan(data)) > 0:
    #     print(10)
    data = data.view(data_shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(data[0,0])
    # plt.show()
    return data

def replace_item(obj, key, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = replace_item(v, key, replace_value)
    if key in obj:
        obj[key] = replace_value
    return obj


def get_optimizer(name, net, config):
    if name == 'Adam':
        return optim.Adam(net.parameters(), lr=config.train.learning_rate)

# TODO: understand this code
class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


loggers = {}


def get_loss(name):
    if name == 'CrossEntropy':
        loss_func = nn.CrossEntropyLoss()
    elif name == 'BCE':
        loss_func = nn.BCEWithLogitsLoss()
    else:
        raise ValueError('Unknown Loss name.')
    return loss_func


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied
    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state

    
def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # TODO: Understand why "logger.propagate = False" can prevent duplicate logging inforamtion
        logger.propagate = False
        loggers[name] = logger
        return logger


# def load_config_yaml(config_file):
#     return yaml.safe_load(open(config_file, 'r'))


def create_training_path(train_logdir):
    idx = 0
    path = os.path.join(train_logdir, "run_{:03d}".format(idx))
    while os.path.exists(path):
        # if len(os.listdir(path)) == 0:
        #     os.remove(path)
        idx += 1
        path = os.path.join(train_logdir, "run_{:03d}".format(idx))
    os.makedirs(path)
    return path

# TODO: different indent of dataset config, preprocess config, train config
# TODO: recursively
def config_logging(path, config, access_mode):
    with open(path, access_mode) as fw:
        for dict_key in config:
            dict_value = config[dict_key]
            if isinstance(dict_value , dict):
                for sub_dict_key in dict_value:
                    fw.write(f'{sub_dict_key}: {dict_value[sub_dict_key]}\n')
            else:
                fw.write(f'{dict_key}: {dict_value}\n')

# TODO: create train_loggting.txt automatically
# TODO: check complete after training
# TODO: train config
# TODO: python logging
def train_logging(path, config):
    if not os.path.isfile((path)):
        with open(path, 'w+') as fw:
            fw.write('')
    with open(path, 'r+') as fw:
        if os.stat(path).st_size == 0:
            number = 0
        else:
            for last_line in fw:
                pass
            number = int(last_line.split(' ')[0][1:])
            fw.write('\n')
        local_time = time.ctime(time.time())
        experiment = os.path.basename(config['checkpoint_path'])
        cur_logging = f'#{number+1} {local_time} {experiment}'
        pprint(cur_logging)
        fw.write(cur_logging)


def get_tensorboard_formatter(config):
    if config is None:
        return DefaultTensorboardFormatter()

    class_name = config['name']
    m = importlib.import_module('pytorch3dunet.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**config)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def create_optimizer(optimizer_config, model):
    # TODO: add SGD
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0)
    momentum = optimizer_config.get('momentum', 0)
    betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    optimizer_name = optimizer_config['optimizer']
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, betas=betas, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError('Unknown optimizer name.')
    return optimizer


def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)


def create_sample_plotter(sample_plotter_config):
    if sample_plotter_config is None:
        return None
    class_name = sample_plotter_config['name']
    m = importlib.import_module('pytorch3dunet.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**sample_plotter_config)


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def load_content_from_txt(path, access_mode='r'):
    with open(path, access_mode) as fw:
        content = fw.readlines()
    return content


if __name__ == '__main__':
    PROJECT_PATH = "C:\\Users\\test\\Desktop\\Leon\\Projects\\Breast_Ultrasound\\"
    create_training_path(os.path.join(PROJECT_PATH, 'models'))
