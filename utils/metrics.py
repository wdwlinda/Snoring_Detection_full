import importlib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


# TODO: check correctness and multi-class solution
# def dsc(prediction, label):
#     tp = ((prediction.data == 1) & (label.data == 1)).sum()
#     fn = ((prediction.data == 0) & (label.data == 1)).sum()
#     fp = ((prediction.data == 1) & (label.data == 0)).sum()
#     denominator = 2*tp + fp +fn
#     if denominator == 0:
#         return 1
#     else:
#         return 2*tp / denominator

def check_zero_division(func):
    def warp():
        
        func()
    return warp
    
def sensitivity(prediction, label):
    pass

def specificity(prediction, label):
    pass

# TODO: w/ label and w/o label
# TODO: multi-classes example
# TODO: decorate with result printing, zero-division judgeing
def precision(tp, fp):
    denominator = tp + fp
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return tp / denominator


def recall(tp, fn):
    denominator = tp + fn
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return tp / denominator


# TODO: correct
def accuracy(tp, fp, fn):
    denominator = tp + fp
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return tp / denominator


def f1(tp, fp, fn):
    denominator = (2 * tp + fp + fn)
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return 2 * tp / denominator


def iou(tp, fp, fn):
    denominator = (tp + fp + fn)
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return tp / denominator


def specificity(tn, fp):
    denominator = tn + fp
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return tn / denominator


# def precision(tp, fp):
#     return (tp) / (tp + fp) if tp > 0 else 0

# def recall(tp, fn):
#     return (tp) / (tp + fn) if tp > 0 else 0

# def accuracy(tp, fp, fn):
#     return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


# def f1(tp, fp, fn):
#     return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


# def iou(tp, fp, fn):
#     return tp / (tp + fp + fn) if tp > 0 else 0


# TODO: property for all avaiable metrics
# TODO: input output type --> numpy or tensor
# TODO: should implement in @staticmethod
class SegmentationMetrics():
    def __init__(self, num_class, metrics=None):
        # TODO: parameter check
        self.num_class = num_class
        self.total_tp = None
        self.total_fp = None
        self.total_fn = None
        self.total_tn = None
        # TODO: check value and class (Dose sklearn func do this part?)
        # TODO: check shape
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = ['precision', 'recall', 'accuracy', 'f1', 'iou']
        
    def __call__(self, label, pred):
        self.label = label
        self.pred = pred
        self.tp, self.fp, self.fn, self.tn = self.confusion_matrix()
        self.total_tp = self.tp if self.total_tp is None else self.total_tp + self.tp
        self.total_fp = self.fp if self.total_fp is None else self.total_fp + self.fp
        self.total_fn = self.fn if self.total_fn is None else self.total_fn + self.fn
        self.total_tn = self.tn if self.total_tn is None else self.total_tn + self.tn
        # self.total_tp += self.tp
        # self.total_fp += self.fp
        # self.total_fn += self.fn
        # self.total_tn += self.tn
        eval_result = {}
        for m in self.metrics:
            if m == 'precision':
                eval_result[m] = precision(self.tp, self.fp)
            elif m == 'recall':
                eval_result[m] = recall(self.tp, self.fn)
            elif m == 'accuracy':
                eval_result[m] = accuracy(self.tp, self.fp, self.fn)
            elif m == 'f1':
                eval_result[m] = f1(self.tp, self.fp, self.fn)
            elif m == 'iou':
                eval_result[m] = iou(self.tp, self.fp, self.fn)
        return eval_result

    def confusion_matrix(self):
        num_class = self.num_class if self.num_class > 1 else self.num_class + 1
        # print(self.label.max(), self.label.min())
        # print(self.pred.max(), self.pred.min())
        # import matplotlib.pyplot as plt
        # plt.imshow(self.pred)
        # plt.show()
        # plt.imshow(self.label)
        # plt.show()
        self.label = np.reshape(self.label, [-1])
        self.pred = np.reshape(self.pred, [-1])
        # print(self.label.shape, self.pred.shape)
        # self.pred = np.int32(self.pred)
        cm = confusion_matrix(self.label, self.pred, labels=np.arange(0, num_class))
        tp = np.diagonal(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        # tp = ((self.pred.data == 1) & (self.label.data == 1)).sum()
        # tn = ((self.pred.data == 0) & (self.label.data == 0)).sum()
        # fn = ((self.pred.data == 0) & (self.label.data == 1)).sum()
        # fp = ((self.pred.data == 1) & (self.label.data == 0)).sum()
        return (tp, fp, fn, tn)
    

# def torch_confusion_matrix
def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('utils.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)


# TODO: get Confusion matrix
# TODO: flexible for single sample test
# TODO: flexibility for tensorflow
class BaseEvaluator():
    def __init__(self, loader, net, metrics_name, data_keys=['input', 'gt'], *args, **kwargs):
        self.loader = loader
        self.net = net
        self.metrics_name = metrics_name
        self.data_keys = data_keys

    def get_evaluation(self):
        input_key, gt_key = self.data_keys
        metrics_func = self.get_metrics()
        for _, data in enumerate(self.loader):
            inputs, labels = data[input_key], data[gt_key]
            outputs = self.net(inputs)
            metrics_func(labels, outputs)
            evaluation = self.aggregation()
        return evaluation

    def get_metrics(self):
        def metrics(label, pred):
            self.total_cm = 0
            for m in self.metrics_name:
                if m in ('precsion', 'recall'):
                    cm = confusion_matrix(label, pred)
                else:
                    raise ValueError('Undefined metrics name.')
                self.total_cm += cm

        return metrics

    def aggregation(self):
        pass 

    def check_format(self):
        pass

    def check_shape(self):
        pass



    # def get_metrics(self):
    #     def metrics(label, pred):
    #         for m in self.metrics:
    #             if m in ['precsion', 'recall', 'f1', 'accuracy']:
    #                 cm = get_confusion_matrix(label, pred)
    #             elif m in ['mean_DSC', 'mean_IoU']
    #     return metrics


# suedo code
# loader = Dataloader()
# net = ModelBuilder()
# base_evaluator = BaseEvaluator(loader, net, metrics=['precision', 'recall'])
# evaluation = base_evaluator.get_evaluation()

