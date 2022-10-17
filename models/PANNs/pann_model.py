from pathlib import Path
import importlib

import torch
# https://arxiv.org/abs/1912.10211


def replace_last_layer(model, out_channels):
    model_list = list(model.children())
    fc_last = model_list[-1]
    new_fc_last = torch.nn.Linear(fc_last.in_features, out_channels)
    model = torch.nn.Sequential(*model_list[:-1], new_fc_last)
    return model


def get_local_checkpoint(model_name, checkpoint_dir):
    checkpoints_dir = Path(checkpoint_dir)
    checkpoints = checkpoints_dir.rglob('*.pth')

    match_ckpt = None
    match_times = 0
    for checkpoint in checkpoints:
        ckpt_name = checkpoint.stem
        if ckpt_name.startswith(model_name):
            match_ckpt = checkpoint
            match_times += 1

        assert match_times <= 1, 'Multiple matching result (ambiguous model name)'
    return match_ckpt
    

# TODO: arguments
def get_pann_model(
    model_name, model, device=torch.device('cuda:0'), pretrained=True, strict=False,
    restore_path=None, checkpoint_dir='models/PaNNs'):
    # # hop_size = 512
    # # window_size = 2048
    # # mel_bins = 128
    # # fmin = 0
    # # fmax = None

    # hop_size = 320
    # window_size = 1024
    # mel_bins = 64
    # fmin = 0
    # fmax = None
    # # fmin = 50
    # # fmax = 14000

    # # Model
    # m = importlib.import_module('models.PANNs.model')
    # Model = getattr(m, model_name)
    # model = Model(
    #     sample_rate=sample_rate, window_size=window_size, 
    #     hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
    #     classes_num=classes_num, dropout=dropout
    # )

    if pretrained:
        model_name = model_name.split('.')[-1]
        checkpoint_path = get_local_checkpoint(model_name, checkpoint_dir)
        assert checkpoint_path is not None, 'Missing checkpoint'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model']
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=strict)

    # model = replace_last_layer(model, classes_num)
    if restore_path is not None:
        checkpoint = torch.load(restore_path, map_location=device)
        model.load_state_dict(checkpoint['net'], strict=strict)
        model.to(torch.device(device))
    return model


if __name__ == '__main__':
    model = get_pann_model(
        'ResNet38',
        16000, 
        2,
        'cuda:0'
    )
    print(model)
    print(model)