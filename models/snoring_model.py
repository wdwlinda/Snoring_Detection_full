from models.image_classification.img_classifier import TimmImgClassifierBuilder
from models.PANNs.pann import PannImgClassifierBuilder


# TODO: This is bad, the interface should also has this function
def convert_model_name(model_name):
    model_map = {
        'pann.ResNet22': 'Snoring_Detection.models.PANNs.model.ResNet22',
        'pann.ResNet38': 'Snoring_Detection.models.PANNs.model.ResNet38',
        'pann.ResNet54': 'Snoring_Detection.models.PANNs.model.ResNet54',
        'pann.MobileNetV1': 'Snoring_Detection.models.PANNs.model.MobileNetV1',
        'pann.MobileNetV2': 'Snoring_Detection.models.PANNs.model.MobileNetV2',
    }
    if model_name.split('.')[0] == 'timm':
        converted_name = 'Snoring_Detection.models.image_classification.img_classifier.ImageClassifier'
    else:
        converted_name = model_map.get(model_name, model_name)
    return model_name, converted_name
    

def create_snoring_model(common_config: dict):
    # TODO: independent and autoing mapping part
    # TODO: Add timm models
    # TODO: All the if-else statement will be redundent in the future

    # TODO: return the same variable name is confusing
    model_name = common_config.model.name
    model_name, converted_name = convert_model_name(model_name)

    if 'timm' in model_name:
        model_builder = TimmImgClassifierBuilder(common_config, converted_name)
        model = model_builder.build()
    elif 'pann' in model_name:
        hop_size = 320
        window_size = 1024
        mel_bins = 64
        fmin = 0
        fmax = None
        new_model_config = {
            'sample_rate': 16000,
            'window_size': window_size,
            'hop_size': hop_size,
            'mel_bins': mel_bins,
            'fmin': fmin,
            'fmax': fmax,
        }
        model_builder = PannImgClassifierBuilder(
            common_config, converted_name, new_model_config)
        model = model_builder.build()
    return model


if __name__ == '__main__':
    pass


    