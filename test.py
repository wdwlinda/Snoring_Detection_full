
import os
from pathlib import Path
import time

from cv2 import normalize
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.serialization import save
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torchaudio
import onnxruntime

from models.image_classification import img_classifier
from dataset.dataloader import AudioDatasetCOCO
from dataset import dataset_utils
from dataset.data_transform import WavtoMelspec_torchaudio
from utils import train_utils
# from utils import metrics
from utils import configuration
from utils import train_utils as local_train_utils
from models.snoring_model import create_snoring_model
from models.utils import get_activation
from models.import_model import build_tflite


# TODO: solve device problem, check behavoir while GPU using
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
ImageClassifier = img_classifier.ImageClassifier
CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\_cnn_valid_config.yml'


def onnx_inference(inputs, ort_session):
    # compute ONNX Runtime output prediction
    input_names = ort_session.get_inputs()
    assert len(inputs) == len(input_names)
    ort_inputs = {
        input_session.name: input_data for input_session, input_data in zip(input_names, inputs)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs


def ONNX_inference(inputs, onnx_model):
    """AI is creating summary for ONNX_inference

    Args:
        inputs ([type]): [description]
        onnx_model ([type]): [description]

    Returns:
        [type]: [description]
    """
    ort_session = onnxruntime.InferenceSession(onnx_model)
    # compute ONNX Runtime output prediction
    input_names = ort_session.get_inputs()
    assert len(inputs) == len(input_names)
    ort_inputs = {
        input_session.name: input_data for input_session, input_data in zip(input_names, inputs)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs


def pred_data():
    total_data_info = {}

    # preprocess_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp'
    # _0727_data = ['1658889529250_RedmiNote8', '1658889531056_Pixel4XL', '1658889531172_iPhone11']
    # for dataset in _0727_data:
    #     src_dir = os.path.join(preprocess_dir, dataset, '16000', 'img', 'filenames')
    #     dist_dir = os.path.join(preprocess_dir, dataset, '16000', 'pred')
    #     gt_dir = os.path.join(preprocess_dir, dataset, '16000', 'filenames.csv')
    #     total_data_info[dataset] = {'src': src_dir, 'dist': dist_dir, 'gt': gt_dir}

    preprocess_dir = r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\preprocess'
    # _0811_data = ['Samsung_Note10Plus_night']
    _0811_data = ['Mi11_office', 'Redmi_Note8_night', 'Samsung_Note10Plus_night']
    # _0811_data = ['Mi11_night', 'Mi11_office', 'Redmi_Note8_night', 'Samsung_Note10Plus_night']
    for dataset in _0811_data:
        src_dir = os.path.join(preprocess_dir, dataset, 'melspec', 'img', 'filenames')
        dist_dir = os.path.join(preprocess_dir, dataset, 'pred3')
        gt_dir = os.path.join(preprocess_dir, dataset, 'melspec', 'filenames.csv')
        total_data_info[dataset] = {'src': src_dir, 'dist': dist_dir, 'gt': gt_dir}

    # test = {'Test': {
    #     'src': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\temp_test',
    #     'dist': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\temp_test',
    #     'gt': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\2_21_2s_my2\test.csv',
    #     }
    # }
    # total_data_info.update(test)

    # esc50 = {'ESC-50': {
    #     'src': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\esc50\44100\img\file_names',
    #     'dist': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\esc50\44100\pred',
    #     'gt': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\esc50\44100\file_names.csv',
    #     }
    # }
    # total_data_info.update(esc50)

    # asus_snoring = {'ASUS_snoring': {
    #     'src': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\2_21_2s_my2\img\test',
    #     'dist': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\2_21_2s_my2\pred',
    #     'gt': r'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_cpp\2_21_2s_my2\test.csv',
    #     }
    # }
    # total_data_info.update(asus_snoring)

    total_confidence = {}
    data_names = []
    for dataset, data_info in total_data_info.items():
        data_names.append(dataset)
        src_dir = data_info['src']
        dist_dir = data_info['dist']
        prediction = pred_from_feature(src_dir, dist_dir)

        y_true, y_pred, confidence = [], [], []
    
        # df = pd.read_csv(data_info['gt'])
        # for index, sample_gt in df.iterrows():
        #     if prediction.get(sample_gt['input'], None):
        #         true_val = sample_gt['label']
        #         y_true.append(true_val)
        #         y_pred.append(prediction[sample_gt['input']]['pred'])
        #         confidence.append(sample['prob'][0, true_val])

        true_val = 0
        for index, sample in prediction.items():
            y_pred.append(sample['pred'])
            y_true.append(true_val)
            confidence.append(sample['prob'][0, true_val])

        acc = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, zero_division=0)
        with open(os.path.join(data_info['dist'], 'result.txt'), 'w') as fw:
            fw.write(f'Precision {precision:.4f}\n')
            fw.write(f'Recall {recall:.4f}\n')
            fw.write(f'Accuracy {acc:.4f}\n')
        # print(acc)

        plot_confusion_matrix(y_true, y_pred, save_path=data_info['dist'])
        total_confidence[dataset] = confidence

    fig, ax = plt.subplots(1, 1)
    for idx, (dataset, confidence) in enumerate(total_confidence.items(), 1):
        ax.scatter(np.ones_like(confidence, dtype=np.int32)*idx, confidence, s=0.5, alpha=0.5)
    ax.set_xlabel('dataset')
    ax.set_ylabel('probability')
    ax.set_title('Prediction confidence comparision')
    ax.set_xticks(np.arange(1, len(total_confidence)+1), data_names)
    ax.plot([1, len(total_confidence)+1], [0.5, 0.5], 'k--')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(os.path.join(data_info['dist'], 'confidence_comp.png'))


class Inferencer():
    def __init__(self, config, dataset, model, save_path=None, batch_size=1, shuffle=False, transform=None):
        self.config = config
        self.dataset = dataset
        # TODO: Inference will be wrong if batch_size > 1
        self.data_loader = DataLoader(self.dataset, batch_size, shuffle)
        self.model = model
        self.device = configuration.get_device()
        self.save_path = save_path
        self.prediction = {}
        self.transform = transform
        self.activation = get_activation(config.model.activation)

        # # XXX: temp
        # tflite_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_741\pann.ResNet38_run_741.tflite'
        # self.interpreter = build_tflite(tflite_path, [1, 32000])
        
        # import onnxruntime
        # onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_741\pann.ResNet38_run_741.onnx'
        # self.ort_session = onnxruntime.InferenceSession(onnx_path)

        # # Get TF output
        # tf_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_741\pann.ResNet38_run_741'
        # new_model = tf.saved_model.load(tf_path)
        # self.infer = new_model.signatures["serving_default"]
        
    # TODO: from data dir
    def run(self, wav_path):
        inputs, sr = torchaudio.load(wav_path, normalize=True)
        inputs = inputs.to(self.device)
        inputs, _ = self.transform(inputs, None) 

        prob = self.model(inputs)
        prediction = torch.argmax(prob, dim=1).item()
        prob = prob.detach().cpu().numpy()
        return prob
        
    def pred_with_model(self, inputs):
        return self.model(inputs)

    def __call__(self, show_info=False):
        with torch.no_grad():
            self.model.eval()
            if len(self.data_loader) == 0:
                raise ValueError(
                    'No Data Exist. Please check the data path or data_plit.')

            total_prob = []
            for i, data in enumerate(self.data_loader, 1):
                inputs = data['input']
                inputs = inputs.to(self.device)
                target = data.get('target', None)
                target = target.to(self.device)
                inputs, target = self.transform(inputs, target) 

                output = self.pred_with_model(inputs)
                if self.activation is not None:
                    output = self.activation(output)
                prediction = torch.argmax(output, dim=1)

                output = output.detach().cpu().numpy()
                target = target.detach().cpu().numpy()[0]
                prediction = prediction.item()
                sample_name = self.dataset.input_data_indices[i-1]['file_name'][:-4]
                self.prediction[sample_name] = {
                    'prob_0': output[0, 0], 'prob_1': output[0, 1], 
                    'pred': prediction, 'target': target
                }

                if show_info:
                    print(f'Sample: {i}', output, prediction, self.dataset.input_data_indices[i-1])
            self.record(self.prediction)
        return self.prediction        

    def record(self, prediction):
        if self.save_path is not None:
            path = self.save_path
        else:
            path = os.path.join(
                self.config.eval.restore_checkpoint_path, 
                self.config.eval.running_mode, 
                os.path.basename(self.config.dataset.index_path))
        if not os.path.isdir(path):
            os.makedirs(path)
        # XXX: pred.csv -> kaggle_snorin_pred.csv
        name = f'pred.csv'

        for index, (sample_name, pred_info) in enumerate(prediction.items()):
            if index == 0:
                with open(os.path.join(path, name), mode='w+', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['sample'] + list(prediction[sample_name].keys()))

            with open(os.path.join(path, name), mode='a+', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    sample_name, pred_info['prob_0'], pred_info['prob_1'], 
                    pred_info['pred'], pred_info['target']
                ])


def plot_confusion_matrix(y_true, y_pred, save_path=''):
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(save_path, 'cm.png'))



def tflite_inference(inputs, interpreter):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    inputs = tf.constant(inputs)
    interpreter.set_tensor(input_details[0]['index'], inputs)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def pred_tflite(config, dataset_mapping, tflite_path):
    total_acc = {}
    config = local_train_utils.DictAsMember(config)
    inf_time = {}

    # tflite model
    interpreter = build_tflite(tflite_path, [1, 32000])
    # interpreter = build_tflite(tflite_path, [1, 3, 128, 59])
    
    # Get ONNX output
    onnx_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_741\pann.ResNet38_run_741.onnx'
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # Get TF output
    tf_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_741\pann.ResNet38_run_741'
    new_model = tf.saved_model.load(tf_path)
    infer = new_model.signatures["serving_default"]

    for test_data_name, data_path in dataset_mapping['data_pre_root'].items():
        # test_dataset = SimpleAudioDatasetfromNumpy_csv(config, data_path)
        data_name = Path(data_path).name
        config['dataset']['index_path'] = {data_name: data_path}
        test_dataset = AudioDatasetCOCO(config, modes='test')
        test_dataloader = DataLoader(test_dataset, 1, False)

        

        p = 0
        data_infer_t = 0
        # small_size = 10
        y_true, y_pred, confidence = [], [], []
        for i, data in enumerate(test_dataloader):
            # if i > small_size: break
            input_data, target = data['input'], data['target']
            input_data = input_data.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            input_data = input_data[:,0]
            start_t = time.time()
            # XXX: temp
            # TFlite
            prediction = tflite_inference(input_data, interpreter)
            
            # ONNX
            output_node = onnx_inference([input_data], ort_session)
            onnx_output = output_node[0]

            # Tensorflow
            output = infer(tf.constant(input_data))
            tf_output = output['output'].numpy()


            # ===
            end_t = time.time()
            infer_t = end_t - start_t

            print(i, test_dataset.input_data_indices[i], prediction[0,1], infer_t)
            data_infer_t += infer_t
            y_true.append(target[0])
            if prediction[0,1] > 0.5:
                # p += 1
                y_pred.append(1)
            else:
                y_pred.append(0)

        acc = metrics.accuracy_score(y_true, y_pred)
        # inf_time[test_data_name] = data_inf_t/small_size
        # inf_time[test_data_name] = data_inf_t/len(test_dataloader.dataset)
        print('--')
        print(inf_time, acc, y_true, y_pred)
        
    return prediction


def run_test(src_dir, dist_dir, config, splits):
    test_dataset_name = Path(dist_dir).stem
    prediction = single_test(src_dir, dist_dir, config, splits=splits)
    y_true, y_pred, confidence = [], [], []
    for sample_name in prediction:
        y_true.append(prediction[sample_name]['target'])
        y_pred.append(prediction[sample_name]['pred'])
    
    all_zeros = [
        'redmi',
        'pixel',
        'iphone',
        'Mi11_night',
        'Mi11_office',
        'Redmi_Note8_night',
        'Samsung_Note10Plus_night'
    ]

    if test_dataset_name in all_zeros:
        mAP = AUC = metrics.accuracy_score(y_true, y_pred)
    else:
        mAP = metrics.average_precision_score(y_true, y_pred, average=None)
        AUC = metrics.roc_auc_score(y_true, y_pred, average=None)

    plot_confusion_matrix(y_true, y_pred, save_path=dist_dir)
    return mAP, AUC


def single_test(
    data_path: str, save_path: str, config: str = None, show_info: bool = False, 
    splits: str = None) -> dict:
    if not config:
        config = configuration.load_config_and_setup(CONFIG_PATH, dict_as_member=True)
    name = os.path.split(save_path)[1]
    config['dataset']['index_path'] = {name: data_path}
    
    test_dataset = AudioDatasetCOCO(config, modes=splits)

    test_transform = WavtoMelspec_torchaudio(
        sr=config.dataset.sample_rate,
        n_class=config.model.out_channels,
        preprocess_config=config.dataset.preprocess_config,
        is_mixup=False,
        is_spec_transform=False,
        is_wav_transform=False,
        device=config.device
    ) 
    
    model = create_snoring_model(config)

    inferencer = Inferencer(
        config, dataset=test_dataset, model=model, save_path=save_path, 
        transform=test_transform
    )
    prediction = inferencer(show_info)
    return prediction
    


if __name__ == "__main__":
    CONFIG_PATH = 'config/_cnn_train_config.yml'
    checkpoint = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_741'
    # checkpoint = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_723'
    # checkpoint = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_741'

    # checkpoint = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\pann_run_461'
    # checkpoint = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_082'
    # model_name = 'convnext_tiny_384_in22ft1k'
    # model_name = 'edgenext_small'
    model_name = 'pann.Resnet38'
    # model_name = 'pann.MobileNetV2'

    test_dataset = configuration.load_config('dataset/dataset.yml')
    # test_dataset['dataset'].pop('ASUS_snoring_train')
    # test_dataset['dataset'].pop('ESC50')

    config = configuration.load_config(CONFIG_PATH, dict_as_member=False)
    config['CHECKPOINT_PATH'] = checkpoint
    config['model']['name'] = model_name
    # inference_test(config, test_dataset)


    # tflite_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\pann_run_461\pann_run_461.tflite'
    tflite_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_741\snoring_model_1021.tflite'
    # tflite_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_723\pann.MobileNetV2_run_723.tflite'
    # tflite_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_722\pann.ResNet54_run_722.tflite'
    # tflite_path = r'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\checkpoints\run_082\snoring_relu_trained.tflite'
    pred_tflite(config, test_dataset, tflite_path)


    
    