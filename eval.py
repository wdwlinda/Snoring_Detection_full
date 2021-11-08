
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from models.image_classification import img_classifier
from dataset.dataloader import AudioDataset
from utils import train_utils
from utils import metrics
from utils import configuration
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import csv
import train
ImageClassifier = img_classifier.ImageClassifier

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
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14)

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# TODO: separate checkpoint and evaluation result. (can keep new evaluation while using new dataset)
def eval():
    config = configuration.load_config(CONFIG_PATH)
    dataset_config = config['dataset']
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
        total_precision, total_recall, total_dsc, total_iou = np.array([], dtype=np.float32), np.array([], dtype=np.float32), [], []
        evaluator = metrics.SegmentationMetrics(num_class=config.model.out_channels, 
                                                metrics=['precision', 'recall', 'accuracy'])

        if len(test_dataloader) == 0:
            raise ValueError('No Data Exist. Please check the data path or data_plit.')
        # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(6, 2))
        y_true, y_pred, prediction_record = np.array([], dtype=np.float32), np.array([], dtype=np.float32), []
        prob_p = np.array([], dtype=np.float32)
        prob_n = np.array([], dtype=np.float32)
        pred_filenames = np.array([], dtype=object)
        for i, data in enumerate(test_dataloader):
            print('Sample: {}'.format(i+1))
            inputs, labels = data['input'], data['gt']
            inputs = train_utils.minmax_norm(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            output = net(inputs)
            # prob = torch.nn.functional.softmax(output, dim=1)
            # prediction = torch.argmax(prob, dim=1)
            prob = torch.sigmoid(output)
            prediction = torch.argmax(prob, dim=1)
            labels = torch.argmax(labels, dim=1)
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
            total_precision = np.append(total_precision, evals['precision'])
            total_recall = np.append(total_recall, evals['recall'])

            result_path = os.path.join(config.eval.restore_checkpoint_path, config.eval.running_mode)
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            # TODO: Visualizer
            if config.eval.save_segmentation_result or config.eval.show_segmentation_result:
                fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(6, 2))
                ax1.imshow(inputs.cpu()[0,0].detach().numpy(), 'gray')
                ax2.imshow(labels.cpu()[0,0].detach().numpy(), 'gray')
                ax3.imshow(prediction.cpu()[0,0].detach().numpy(), 'gray')
                if config.eval.save_segmentation_result:
                    image_code = i + 1
                    image_code = test_dataset.input_data[i].split('\\')[-1]
                    fig.savefig(os.path.join(result_path, f'{config.eval.running_mode}_{image_code}.png'))
                    plt.close(fig)
                if config.eval.show_segmentation_result:
                    plt.show()
        
            y_true = np.append(y_true, labels)
            y_pred = np.append(y_pred, prediction)

            prob_p = np.append(prob_p, prob[0][1].item())
            prob_n = np.append(prob_n, prob[0][0].item())
            # TODO: file_name accessing only work if shuffle=False
            prediction_record.append({
                'label': labels[0],
                'pred': prediction[0],
                'prob': prob[0][1].item(),
                'file_name': os.path.basename(test_dataset.input_data_indices[i])})


        for th_prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            y_pred_th = np.array(prob_p)
            y_pred_th[y_pred_th>th_prob] = 1
            y_pred_th[y_pred_th<=th_prob] = 0
            # y_true_th = np.append(y_true[prob_p>th_prob], y_true[prob_p<(1-th_prob)])
            # y_pred_th = np.append(y_pred[prob_p>th_prob], y_pred[prob_p<(1-th_prob)])
            # y_true_th = np.where(prob_p>th_prob, 1, np.where(prob_p<(1-th_prob), 1, 0))
            if len(y_pred_th) > 0:
                # print(y_true.max(), y_pred_th.max(), y_pred.max())
                cm_th = confusion_matrix(y_true, y_pred_th)
                print(f'{th_prob}: acc = {(cm_th[0,0]+cm_th[1,1])/np.sum(cm_th)*100:.2f} %')
            else:
                print(f'{th_prob}: null')
            # plot_confusion_matrix(cm_th, [0,1], normalize=False)
            # plt.savefig(os.path.join(config.eval.restore_checkpoint_path, f'cm_{th_prob}.png'))

        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, [0,1], normalize=False)
        plt.savefig(os.path.join(config.eval.restore_checkpoint_path, config.eval.running_mode, 'cm.png'))
        # plt.show()
        precision = metrics.precision(evaluator.total_tp, evaluator.total_fp)
        recall = metrics.recall(evaluator.total_tp, evaluator.total_fn)
        specificity = metrics.specificity(evaluator.total_tn, evaluator.total_fp)
        accuracy = metrics.accuracy(np.sum(evaluator.total_tp), np.sum(evaluator.total_fp), np.sum(evaluator.total_fn), np.sum(evaluator.total_tn))
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        mean_specificity = np.mean(specificity)
        # mean_accuracy = np.mean(accuracy)

        print(30*'-')
        print(f'total precision: {100*mean_precision:.2f} %', '\n')
        print(f'total recall: {100*mean_recall:.2f} %', '\n')
        print(f'total accuracy: {100*accuracy:.2f} %', '\n')
        print(f'total specificity: {100*mean_specificity:.2f} %', '\n')
        
        pred_file = f'{os.path.split(config.eval.restore_checkpoint_path)[1]}_prediction.csv'
        th_prob = 0.9
        with open(os.path.join(result_path, pred_file), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['#', 'file', 'label', 'prediction', 'positive prob', 'correctness'])
            for i, v in enumerate(prediction_record):
                writer.writerow(
                    [str(i+1), v['file_name'], str(v['label']), str(v['pred']), str(v['prob']), str(int(v['label']==v['pred']))])
                
                
                # if prob_p > th_prob or prob_p < (1-th_prob):
                #     np.append(pred_filenames, prediction_record['file_name'])n

        prob_pp, prob_nn = [], []
        prob_pp2, prob_nn2 = [], []
        th = 0.1
        for i, (p, n) in enumerate(zip(prob_p, prob_n)):
            if abs(p-n) < th:
                prob_pp.append(prob_p[i])
                prob_nn.append(prob_n[i])
            else:
                prob_pp2.append(prob_p[i])
                prob_nn2.append(prob_n[i])

        fig, ax = plt.subplots()
        # ax.scatter(prob_p, prob_n, alpha=0.4)
        ax.scatter(prob_pp, prob_nn, alpha=0.4, color='r')
        ax.scatter(prob_pp2, prob_nn2, alpha=0.4)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('positive')
        ax.set_ylabel('negative')
        ax.set_title('Probability distribution')
        ax.legend([f'diff < {th} ({len(prob_pp)/len(prob_p)*100:.2f} %)', f'diff >= {th} ({len(prob_pp2)/len(prob_p)*100:.2f} %)'])
        fig.savefig(os.path.join(result_path, 'class_prob_dist.png'))
        # with open(os.path.join(config.eval.restore_checkpoint_path, 'error_samples.txt'), 'w+') as fw:
        #     errors.sort(key=len)
        #     for i, v in enumerate(errors):
        #         true = v['true']
        #         pred = v['pred']
        #         value = v['value']
        #         fw.write(f'{i+1}  true: {true} pred: {pred}  value: {value}\n')

        # with open('prediction.txt', 'w+') as fw:
        #     for f in test_dataset.input_data_indices:
        #         fw.write(f'{f}: {prediction}')
        
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
    