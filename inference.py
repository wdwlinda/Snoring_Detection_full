
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.serialization import save
from torch.utils.data import Dataset, DataLoader
import site_path
from modules.model.image_calssification import img_classifier
# from models.image_classification import img_classifier
from dataset.dataloader import AudioDataset, SimpleAudioDataset, SimpleAudioDatasetfromNumpy
from utils import train_utils
from utils import metrics
from utils import configuration
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import csv
import train
from dataset import dataset_utils
ImageClassifier = img_classifier.ImageClassifier

# TODO: solve device problem, check behavoir while GPU using
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\_cnn_valid_config.yml'


class build_inferencer():
    def __init__(self, config, dataset, model, save_path=None, batch_size=1, shuffle=False):
        self.config = config
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size, shuffle)
        self.model = model
        self.device = configuration.get_device()
        self.save_path = save_path
        self.prediction = {}
        self.restore()

    def restore(self):
        checkpoint = os.path.join(
            self.config.eval.restore_checkpoint_path, self.config.eval.checkpoint_name)
        state_key = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state_key['net'])
        self.model = self.model.to(self.device)

    def inference(self):
        with torch.no_grad():
            self.model.eval()

            if len(self.data_loader) == 0:
                raise ValueError('No Data Exist. Please check the data path or data_plit.')

            total_prob = []
            for i, data in enumerate(self.data_loader, 1):
                # if i>10: break
                inputs = data['input']
                # inputs = train_utils.minmax_norm(inputs)
                inputs = inputs.to(self.device)
                output = self.model(inputs)
                prob = torch.sigmoid(output)
                # prob = torch.nn.functional.softmax(output)
                prediction = torch.argmax(prob, dim=1).item()
                # prediction = prediction.cpu().detach().numpy()
                prob = prob.detach().cpu().numpy()
                prob_p = prob[0,1]
                print(f'Sample: {i}', prob_p, prediction, self.dataset.input_data_indices[i-1])
                self.record_prediction(i-1, prob, prediction)
        return self.prediction        
    
    def record_prediction(self, index, prob, pred):
        # if prediction[0] == 1:
        if self.save_path is not None:
            path = self.save_path
        else:
            path = os.path.join(self.config.eval.restore_checkpoint_path, self.config.eval.running_mode, os.path.basename(self.config.dataset.index_path))
        if not os.path.isdir(path):
            os.makedirs(path)
        name = f'{os.path.basename(self.config.dataset.index_path)}_pred.csv'
        # name = f'{os.path.basename(self.dataset.path)}_pred .csv'
        if index == 0:
            with open(os.path.join(path, name), mode='w+', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([''])

        with open(os.path.join(path, name), mode='a+', newline='') as csv_file:
            writer = csv.writer(csv_file)
            sample_name = os.path.basename(self.dataset.input_data_indices[index])[:-4]
            writer.writerow(
                [sample_name, prob, pred, self.dataset.input_data_indices[index]])
        self.prediction[sample_name] = {'prob': prob, 'pred': pred}
        
def pred(data_path, save_path):
    config = configuration.load_config(CONFIG_PATH, dict_as_member=True)
    # test_dataset = AudioDataset(config, mode=config.eval.running_mode, eval_mode=False)
    test_dataset = SimpleAudioDataset(config, data_path)
    net = ImageClassifier(
        backbone=config.model.name, in_channels=config.model.in_channels, activation=config.model.activation,
        out_channels=config.model.out_channels, pretrained=False, dim=1, output_structure=None)

    inferencer = build_inferencer(config, dataset=test_dataset, model=net, save_path=save_path)
    inferencer.inference()


def pred_from_feature(data_path, save_path):
    config = configuration.load_config(CONFIG_PATH, dict_as_member=True)
    # test_dataset = AudioDataset(config, mode=config.eval.running_mode, eval_mode=False)
    test_dataset = SimpleAudioDatasetfromNumpy(config, data_path)
    net = ImageClassifier(
        backbone=config.model.name, in_channels=config.model.in_channels, activation=config.model.activation,
        out_channels=config.model.out_channels, pretrained=False, dim=1, output_structure=None)

    inferencer = build_inferencer(config, dataset=test_dataset, model=net, save_path=save_path)
    prediction = inferencer.inference()
    return prediction


if __name__ == "__main__":
    data_path = rf'C:\Users\test\Downloads\1112\app_test\iOS\clips_2_2_6dB'
    data_path = rf'C:\Users\test\Desktop\Leon\Weekly\1112\3min_test'
    save_path = rf'C:\Users\test\Downloads\1112\app_test\iOS'
    pred(data_path, save_path)
    
    