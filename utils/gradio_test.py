# import
from __future__ import annotations
import os

import argparse
from typing import Callable, Optional, List, Any
import gradio as gr
from gradio.components import Component

from dataset.dataloader import AudioDatasetCOCO
# from . import Predictor
from utils import configuration
from models.image_classification.img_classifier import ImageClassifier
from dataset.data_transform import WavtoMelspec_torchaudio
from inference import Inferencer
CONFIG_PATH = 'config/_cnn_train_config.yml'



def get_predictor(
    config: str = None, 
    show_info: bool = False, splits: str = None) -> dict:
    data_path = r'C:\Users\test\Desktop\Leon\Datasets\test\web_snoring_pre\web_snoring'
    save_path = r'C:\Users\test\Desktop\Leon\Weekly\0930\web_snoring'

    if not config:
        config = configuration.load_config(CONFIG_PATH, dict_as_member=True)
    name = os.path.split(save_path)[1]
    config['dataset']['index_path'] = {name: data_path}
    test_dataset = AudioDatasetCOCO(config, modes=splits)
    
    net = ImageClassifier(
        backbone=config.model.name, in_channels=config.model.in_channels, activation=config.model.activation,
        out_channels=config.model.out_channels, pretrained=False, dim=1, output_structure=None,
        restore_path=os.path.join(
            config['eval']['restore_checkpoint_path'], config['eval']['checkpoint_name']
        )
    )

    # FIXME: params for sr, device
    test_transform = WavtoMelspec_torchaudio(
        sr=16000,
        n_class=config.model.out_channels,
        preprocess_config=config.dataset.preprocess_config,
        is_mixup=False,
        is_spec_transform=False,
        is_wav_transform=False,
        device=configuration.get_device()
    ) 

    inferencer = Inferencer(
        config, dataset=test_dataset, model=net, save_path=save_path, transform=test_transform)
    return inferencer


# class
class PredictorGUI:
    def __init__(
        self,
        project_parameters: argparse.Namespace,
        loader,
        gradio_inputs: Optional[str | Component | List[str | Component]],
        gradio_outputs: Optional[str | Component | List[str | Component]],
        examples: Optional[List[Any] | List[List[Any]] | str] = None
    ) -> None:
        # self.predictor = Predictor(project_parameters=project_parameters,
        #                            loader=loader)
        self.predictor = get_predictor(project_parameters, splits='test')

        self.gui = gr.Interface(fn=self.inference,
                                inputs=gradio_inputs,
                                outputs=gradio_outputs,
                                examples=examples,
                                cache_examples=True,
                                live=True,
                                interpretation='default')
        self.classes = project_parameters.model.out_channels

    def inference(self, inputs):
        prediction = self.predictor.run(inputs)  #prediction dimension is (1, num_classes)
        prediction = prediction[0].tolist()
        # print(prediction)
        # return {0: 0.2, 1: 0.8}
        result = {cls: proba for cls, proba in zip([0, 1], prediction)}
        return result

    def __call__(self):
        self.gui.launch(inbrowser=True, share=True)


def main():
    # project parameters
    config = configuration.load_config(CONFIG_PATH, dict_as_member=True)
    config['CHECKPOINT_PATH'] = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_386'
    config['eval'] = {
        'restore_checkpoint_path': config['CHECKPOINT_PATH'],
        'checkpoint_name': r'ckpt_best.pth'
    }
    config['model']['name'] = 'convnext_tiny_384_in22ft1k'

    #create predictor gui
    # loader = AudioDatasetCOCO(config, modes='test')
    gradio_inputs = gr.Audio(source='upload', type='filepath', label='input')
    gradio_outputs = gr.Label()
    examples = []
    predictor_gui = PredictorGUI(project_parameters=config,
                                 loader=None,
                                 gradio_inputs=gradio_inputs,
                                 gradio_outputs=gradio_outputs,
                                 examples=examples)

    #launch
    result = predictor_gui()


if __name__ == '__main__':
    main()
    # import numpy as np
    # prediction = np.float32((0.8, 0.2))
    # print({cls: proba for cls, proba in zip([0, 1], prediction)})