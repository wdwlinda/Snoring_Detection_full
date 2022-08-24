import numpy as np

from deploy import onnx_model
from models.image_classification import img_classifier


def onnx_model_deploy(timm_model_name, in_channels, out_channels, checkpoint, save_filename):
    dummy_input = np.ones((1, 3, 128, 128), np.float32)
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
    timm_model_name = 'edgenext_small'
    in_channels = 3
    out_channels = 2
    checkpoint = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_018\ckpt_best.pth'
    # checkpoint = None
    save_filename = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_018\snoring.onnx'
    onnx_model_deploy(timm_model_name, in_channels, out_channels, checkpoint, save_filename)


if __name__ == '__main__':
    main()
