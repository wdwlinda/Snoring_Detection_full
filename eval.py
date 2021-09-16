
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
# from model import ImageClassifier
from models.image_classification import img_classifier
from dataset.dataloader import AudioDataset
from utils import train_utils
from utils import metrics
from utils import configuration
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
ImageClassifier = img_classifier.ImageClassifier
EVAL_DIR_KEY = ''
# TODO: solve device problem, check behavoir while GPU using
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\_cnn_valid_config.yml'


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# TODO: batch size cannot be like training mode (14 vs 4)
def eval():
    config = configuration.load_config(CONFIG_PATH)
    dataset_config = config['dataset']
    dataset_config['dir_key'] = EVAL_DIR_KEY
    test_dataset = AudioDataset(config, mode=config.eval.running_mode)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model
    net = ImageClassifier(
        backbone=config.model.name, in_channels=config.model.in_channels, activation=config.model.activation,
        out_channels=config.model.out_channels, pretrained=False, dim=1, output_structure=None)
    checkpoint = os.path.join(config.eval.restore_checkpoint_path, config.eval.checkpoint_name)
    state_key = torch.load(checkpoint, map_location=device)
    net.load_state_dict(state_key['net'])
    net = net.to(device)
    with torch.no_grad():
        net.eval()
        total_precision, total_recall, total_dsc, total_iou = [], [], [], []
        evaluator = metrics.SegmentationMetrics(num_class=config.model.out_channels, 
                                                metrics=['precision', 'recall', 'accuracy'])



        if len(test_dataloader) == 0:
            raise ValueError('No Data Exist. Please check the data path or data_plit.')
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(6, 2))
        y_true, y_pred = [], []
        for i, data in enumerate(test_dataloader):
            print('Sample: {}'.format(i+1))
            inputs, labels = data['input'], data['gt']
            inputs, labels = inputs.to(device), labels.to(device)
            output = net(inputs)
            prediction = torch.argmax(output, dim=1)
            labels = labels.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            
            # TODO: Evaluator
            # print(output, F.softmax(output), labels, prediction)
            evals = evaluator(labels, prediction)
            tp, fp, fn = evaluator.tp, evaluator.fp, evaluator.fn
            # TODO: should be dynamic
            # if (tp + fp + fn) != 0:
            #     total_dsc.append(evals['f1'])
            #     total_iou.append(evals['iou'])
            total_precision.append(evals['precision'])
            total_recall.append(evals['recall'])

            # TODO: if EVAL_DIR_KEY = ''
            if not os.path.exists(os.path.join(config.eval.restore_checkpoint_path, 'images', EVAL_DIR_KEY)):
                os.makedirs(os.path.join(config.eval.restore_checkpoint_path, 'images', EVAL_DIR_KEY))

            # TODO: Visualizer
            if config.eval.save_segmentation_result or config.eval.show_segmentation_result:
                fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(6, 2))
                ax1.imshow(inputs.cpu()[0,0].detach().numpy(), 'gray')
                ax2.imshow(labels.cpu()[0,0].detach().numpy(), 'gray')
                ax3.imshow(prediction.cpu()[0,0].detach().numpy(), 'gray')
                if config.eval.save_segmentation_result:
                    image_code = i + 1
                    image_code = test_dataset.input_data[i].split('\\')[-1]
                    fig.savefig(os.path.join(
                        config.eval.restore_checkpoint_path, 'images', EVAL_DIR_KEY, 
                        f'{EVAL_DIR_KEY}_{config.eval.running_mode}_{image_code}.png'))
                    plt.close(fig)
                if config.eval.show_segmentation_result:
                    plt.show()
        
            y_true.append(labels)
            y_pred.append(prediction)
        
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        plot_confusion_matrix(cm, [0,1,2], normalize=True)
        plt.show()

        precision = metrics.precision(evaluator.total_tp, evaluator.total_fp)
        recall = metrics.recall(evaluator.total_tp, evaluator.total_fn)
        specificity = metrics.specificity(evaluator.total_tn, evaluator.total_fp)
        accuracy = metrics.accuracy(np.sum(evaluator.total_tp), np.sum(evaluator.total_fp), np.sum(evaluator.total_fn))
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        mean_specificity = np.mean(specificity)
        # mean_accuracy = np.mean(accuracy)
        print(30*'-')
        print(f'total precision: {mean_precision:.4f}')
        print(f'total recall: {mean_recall:.4f}\n')
        print(f'total specificity: {mean_specificity:.4f}\n')
        print(f'total accuracy: {accuracy:.4f}\n')

        # mean_precision = sum(total_precision)/len(total_precision)
        # mean_recall = sum(total_recall)/len(total_recall)
        # mean_dsc = sum(total_dsc)/len(total_dsc) if len(total_dsc) != 0 else 0
        # mean_iou = sum(total_iou)/len(total_iou) if len(total_iou) != 0 else 0
        # if sum(total_dsc) == 0:
        #     std_dsc = 0
        # else:
        #     std_dsc = [(dsc-mean_dsc)**2 for dsc in total_dsc]
        #     std_dsc = ((sum(std_dsc) / len(std_dsc))**0.5).item()
        # print(f'mean precision: {mean_precision:.4f}')
        # print(f'mean recall: {mean_recall:.4f}')
        # print(f'mean DSC: {mean_dsc:.4f}')
        # print(f'std DSC: {std_dsc:.4f}')
        # print(f'mean IoU: {mean_iou:.4f}')

        # TODO: write to txt or excel

if __name__ == "__main__":
    eval()
    