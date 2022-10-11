import importlib
from pathlib import Path
from inspect import getmembers, isfunction, isclass
from typing import Optional
import os


def find_model(import_path: str):
    """AI is creating summary for find_model

    Args:
        import_path (str): The import path in relative importing format, e.g., project.module.model

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


# def get_class(class_ref):
if __name__ == '__main__':
    from datetime import datetime
    print(datetime.now().isoformat(timespec='seconds'))
    print(find_model('Snoring_Detection.models.PANNs.model.ResNet38'))
    # print(find_module('ResNet38'))
    # print(find_module('ImageClassifier'))
    print(datetime.now().isoformat(timespec='seconds'))


    