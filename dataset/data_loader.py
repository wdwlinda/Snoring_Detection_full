import numpy as np
import cv2


"""All the dataloader. The dataloader load the data (image, array, Dataframe) from reference."""
# TODO: The higher class embeding can be considered.


def img_loader(ref):
    return cv2.imread(ref)


def npy_loader(ref):
    return np.load(ref)