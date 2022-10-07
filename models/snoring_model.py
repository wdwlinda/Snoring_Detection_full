import importlib
from pathlib import Path
from inspect import getmembers, isfunction, isclass
from typing import Optional
import os


# TODO: separate same model implementations in different module or library 
# e.g., timm.ResNet38 & custom.ResNet38
# TODO: Interface
# TODO: file_path to model_dir
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


# def get_class(class_ref):
if __name__ == '__main__':
    from datetime import datetime
    print(datetime.now().isoformat(timespec='seconds'))
    # print(find_module('ResNet38', 'models/PANNs/model.py'))
    print(find_module('ResNet38'))
    print(find_module('ImageClassifier'))
    print(datetime.now().isoformat(timespec='seconds'))