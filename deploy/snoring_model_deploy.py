import numpy as np

from deploy import onnx_model
from models.image_classification import img_classifier


def onnx_model_deploy(timm_model_name, in_channels, out_channels, checkpoint, save_filename):
    # dummy_input = np.ones((1, 3, 128, 128), np.float32)

    # data = np.load(r'C:\Users\test\Desktop\Leon\Projects\compute-mfcc\1606921286802_1_8.93_10.93_001.npy')
    # data = np.load(r'C:\Users\test\Desktop\Leon\Projects\compute-mfcc\1598482996718_21_106.87_108.87_001.npy')
    f = r'C:\Users\test\Desktop\Leon\Projects\compute-mfcc\data\test\1630681292279_88_10.40_12.40_003.npy'
    f = r'C:\Users\test\Desktop\Leon\Projects\compute-mfcc\data\test\1630779176834_98_30.00_32.00_010.npy'
    # f = r'C:\Users\test\Desktop\Leon\Projects\compute-mfcc\data\test\1630681292279_88_10.40_12.40_003.csv'
    
    data = np.load(f)

    import pandas as pd
    # df = pd.read_csv(f)
    # # df = pd.read_csv(f, header=None)
    # data = df.to_numpy()

    # data = data.T
    data = np.float32(data)[None, None]
    dummy_input = np.tile(data, (1, 3, 1, 1))

    model = img_classifier.ImageClassifier(
        backbone=timm_model_name, 
        in_channels=in_channels,
        out_channels=out_channels, 
        pretrained=False,
        restore_path=checkpoint,
        device='cpu',
    )

    dynamic_axes = {'input' : {0: 'batch', 2 : 'height', 3: 'width'},
                    'output' : {0: 'batch', 2: 'height', 3: 'width'}}

    onnx_model.torch_to_onnx(
        dummy_input, 
        model, 
        save_filename,
        dynamic_axes=dynamic_axes
    )


def main():
    in_channels = 3
    out_channels = 2

    # timm_model_name = 'convnext_tiny_384_in22ft1k'
    # checkpoint = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_050\ckpt_best.pth'
    # save_filename = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_050\snoring.onnx'
    
    # timm_model_name = 'edgenext_small'
    # checkpoint = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_018\ckpt_best.pth'
    # save_filename = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_018\snoring.onnx'
    
    timm_model_name = 'convnext_tiny_384_in22ft1k'
    checkpoint = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_083\ckpt_best.pth'
    save_filename = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_083\snoring.onnx'
    
    onnx_model_deploy(timm_model_name, in_channels, out_channels, checkpoint, save_filename)

    # # inference
    # data = np.load(r'C:\Users\test\Desktop\Leon\Projects\compute-mfcc\1598482996718_21_106.87_108.87_001.npy')
    # data = np.float32(data)[None, None]
    # data = np.tile(data, (1, 3, 1, 1))
    # data = [data]
    # save_filename = r'C:\Users\test\Desktop\Leon\Projects\compute-mfcc\snoring.onnx'
    # ort_outs = onnx_model.ONNX_inference(data, save_filename)
    


if __name__ == '__main__':
    main()

    # from dataset import dataset_utils
    # f = r'C:\Users\test\Desktop\Leon\Projects\compute-mfcc\data\1598482996718_21_106.87_108.87_001.wav'
    # y = dataset_utils.get_pydub_sound(f, 'wav', 16000, channels=1)
    # waveform = np.float32(y.get_array_of_samples())
    # from librosa import load
    # w = load(f, sr=None)
    # print(waveform)

