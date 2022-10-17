import importlib
from pathlib import Path
from inspect import getmembers, isfunction, isclass
from typing import Optional

import torch
import torch.nn as nn


def find_model(import_path: str):
    """AI is creating summary for find_model

    Args:
        import_path (str): The import path in relative importing format, 
        e.g., project.module.model

    Returns:
        [class object]: Get the class object by assign name.
    """
    dirs = import_path.split('.')
    module_import_path = '.'.join(dirs[:-1])

    module_path = module_import_path.replace('.', '\\')
    module_path = Path.cwd().parent.joinpath(module_path)
    module_path = module_path.with_suffix('.py')
    model_name = dirs[-1]

    class_obj = find_module(model_name, module_path)
    return class_obj
    

# TODO: Interface
# TODO: use relative importing path Project.snoring_detection.models.PANNs.pann.ResNet38
# TODO: simplify by using dict ouside {pann.ResMet38: Project.snoring_detection.models.PANNs.pann.ResNet38}
# and then different implementation can be separated {pann.ReNet38, timm.ResNet38}
# TODO: inspect path correctness (file exist?)
def find_module(class_name: Path, file_path: Optional[Path] = None):
    """Find class in project

    Args:
        class_name ([type]): [description]
        file_path ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if file_path is not None:
        file_path = Path.cwd().joinpath(file_path)
    # modules = Path.cwd().rglob('*.py')
    modules = Path.cwd().joinpath('models').rglob('*.py')

    match_modules = []
    for f in modules:
        # print(f)
        if f.is_file() and not f.name.endswith('__init__.py'):
            if file_path is not None:
                if f == file_path:
                    match_modules.append(f)
            else:
                match_modules.append(f)

    for module_path in match_modules:
        relpath = module_path.relative_to(Path.cwd())
        # XXX:
        import_format = str(relpath.with_suffix('')).replace('\\', '.')
        try:
            module = importlib.import_module(import_format)
        except ImportError:
            print(f'ImportError in {import_format}')
            module = None

        if module is not None:
            if file_path is not None:
                cls = getattr(module, class_name)
                return cls
            else:
                classes = getmembers(module, isclass)
                for name, cls in classes:
                    if class_name == name:
                        return cls
    return None


class ModelBuilder():
    def __init__(self, common_config: dict, model_name: str, new_model_config: dict = None):
        self.common_config = common_config
        self.model_name = model_name
        # TODO: modify yaml
        self.restore_path = common_config.model.restore_path
        # XXX: params device, statedict_key
        self.device = torch.device('cuda:0')
        self.statedict_key = 'net'
        self.Model = self.get_model(model_name)
        if new_model_config is not None:
            self.new_model_config = new_model_config
        else:
            self.new_model_config = {}
        
    def interface(self):
        raise NotImplementedError("The interface of new model is not implemented")

    def get_model(self, model_name: str):
        return find_model(model_name)

    def build(self):
        new_model_kwargs = self.interface(self.common_config)
        self.new_model_config.update(new_model_kwargs)
        model = self.Model(
            **self.new_model_config
        )
        model = self.restore(model)
        model = model.to(self.device)
        return model

    def restore(self, model):
        # TODO: strict?
        if self.restore_path is not None:
            state_key = torch.load(self.restore_path, map_location=self.device)
            model.load_state_dict(state_key[self.statedict_key])
        return model


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

    
def get_model(model_config):
    def _model_class(class_name):
        modules = ['model']
        for module in modules:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name, None)
            if clazz:
                return clazz

    model_class = _model_class(model_config['name'])
    return model_class(**model_config)


def get_activation(name, *args, **kwargs):
    if name == 'relu':
        return nn.ReLU(inplace=True, *args, **kwargs)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'softmax':
        return nn.Softmax(*args, **kwargs)