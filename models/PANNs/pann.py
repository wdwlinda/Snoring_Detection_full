import torch
from models.PANNs.pann_model import get_pann_model
from models.utils import ModelBuilder


class PannImgClassifierBuilder(ModelBuilder):
    def __init__(self, 
                 common_config: dict, 
                 model_name: str, 
                 new_model_config: dict = None):
        super().__init__(common_config, model_name, new_model_config)
        self.checkpoint_dir = 'models/PANNs'
        
    def interface(self, common_config):
        model_kwargs = {
            'classes_num': common_config.model.out_channels,
            # 'pretrained': common_config.model.pretrained,
            'dropout': common_config.model.dropout,
            'extra_extractor': common_config.model.extra_extractor
        }
        return model_kwargs

    def restore(self, model):
        model = get_pann_model(
            model_name=self.model_name, 
            model=model,
            pretrained=self.common_config.model.pretrained,
            restore_path=self.common_config.model.restore_path
        )
        return model


if __name__ == '__main__':
    pass