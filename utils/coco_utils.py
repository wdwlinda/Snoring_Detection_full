import json
import itertools

import numpy as np


class coco_structure_converter():
    def __init__(self, cat_ids, height, width):
        self.images = []
        self.annotations = []
        self.cat_ids = cat_ids
        self.cats =[{'name': name, 'id': id} for name, id in self.cat_ids.items()]
        self.height = height
        self.width = width
        self.idx = 0

    def sample(self, img_path, mask, image_id, category):
        if np.sum(mask):
            image = {'id': image_id, 'width':self.width, 'height':self.height, 'file_name': f'{img_path}'}
            self.images.append(image)

            ys, xs = np.where(mask)
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            enc =binary_mask_to_rle(mask)
            seg = {
                'segmentation': enc, 
                'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)],
                'area': int(np.sum(mask)),
                'image_id': image_id, 
                'category_id': self.cat_ids[category], 
                'iscrowd': 0, 
                'id':self.idx
            }
            self.idx += 1
            self.annotations.append(seg)


    def create_coco_structure(self):
        return {'categories':self.cats, 'images': self.images, 'annotations': self.annotations}



# From https://newbedev.com/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    x = binary_mask.ravel(order='F')
    y = itertools.groupby(x)
    z = enumerate(y)
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def merge_coco_structure(coco_group):
    output_coco = coco_group[0]
    for idx in range(1, len(coco_group)):
        # TODO: correct categories (if two different cat combine)
        # output_coco['categories'].update(coco_group[idx]['categories'])
        output_coco['images'].extend(coco_group[idx]['images'])
        output_coco['annotations'].extend(coco_group[idx]['annotations'])
    return output_coco