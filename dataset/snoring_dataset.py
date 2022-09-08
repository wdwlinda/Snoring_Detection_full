import numpy as np
import cv2

from dataset_builder import ImageDataset


def img_loader(ref):
    return cv2.imread(ref)


def npy_loader(ref):
    return np.load(ref)


class MelSpecDataset(ImageDataset):
    def __init__(self, loader, ):
        pass
