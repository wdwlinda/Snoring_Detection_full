import importlib

# from models import

# # Get class from name
# def get_pann_model(model_name):
#     m = importlib.import_module('models.PANNs.model')
#     Model = getattr(m, model_name)
#     return Model


# # Get instance from class and arguments
# def interface(config, model_name, new_model_config):
#     Model = get_pann_model(model_name)
#     model = Model(
#         class_num=config.model.out_channels,
#         **new_model_config
#     )
#     return model


class GetNewModel():
    def __init__(self, common_config: dict, model_name: str, new_model_config: dict):
        self.common_config = common_config
        self.Model = self.get_model(model_name)
        self.new_model_config = new_model_config

    def interface(self):
        raise NotImplementedError("The interface of new model is not implemented")

    def get_model(self, model_name: str):
        # XXX:
        m = importlib.import_module('models.PANNs.model')
        Model = getattr(m, model_name)
        return Model

    def __call__(self):
        new_model_kwargs = self.interface(self.common_config)
        self.new_model_config.update(new_model_kwargs)
        model = self.Model(
            **self.new_model_config
        )
        return model


class GetPannModel(GetNewModel):
    def __init__(self):
        super().__init__()

    def interface(self, common_config):
        model_kwargs = {
            'class_num': common_config.model.out_channels
        }
        return model_kwargs


if __name__ == '__main__':
    pann_model = GetPannModel(config, 'pann.ResNet38', new_model_config)