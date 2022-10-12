from models.utils import GetNewModel


class GetPannImgClassifier(GetNewModel):
    def __init__(self, common_config: dict, model_name: str, new_model_config: dict = None):
        super().__init__(common_config, model_name, new_model_config)

    def interface(self, common_config):
        model_kwargs = {
            'class_num': common_config.model.out_channels,
            'class_num': common_config.model.out_channels,
            'class_num': common_config.model.out_channels,
        }
        return model_kwargs


if __name__ == '__main__':
    pass