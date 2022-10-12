from models.image_classification.img_classifier import GetTimmImgClassifier
from models.PANNs.pann import GetPannImgClassifier


def create_snoring_model(common_config: dict, model_name: str):
    if 'timm' in model_name:
        GetTimmImgClassifier(common_config, model_name)
    elif 'pann' in model_name:
        # hop_size = 512
        # window_size = 2048
        # mel_bins = 128
        hop_size = 320
        window_size = 1024
        mel_bins = 64
        fmin = 50
        fmax = 14000
        new_model_config = {
            'sample_rate': 16000,
            'window_size': window_size,
            'hop_size': hop_size,
            'mel_bins': mel_bins,
            'fmin': fmin,
            'fmax': fmax,
        }
        GetPannImgClassifier(common_config, model_name, new_model_config)


if __name__ == '__main__':
    pass
    # from datetime import datetime
    # print(datetime.now().isoformat(timespec='seconds'))
    # print(find_model('Snoring_Detection.models.PANNs.model.ResNet38'))
    # # print(find_module('ResNet38'))
    # # print(find_module('ImageClassifier'))
    # print(datetime.now().isoformat(timespec='seconds'))


    