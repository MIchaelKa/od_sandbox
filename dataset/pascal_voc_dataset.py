import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision.transforms import functional as TF
from PIL import Image

IMG_WIDTH = 384
IMG_HEIGHT = IMG_WIDTH
MODEL_SCALE = 8 # How to calculate this?

# from transforms import get_transform_to_show

class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split='TRAIN', transforms=None, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """

        assert split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult
        self.split = split.upper()
        self.transforms = transforms

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = np.array(objects['boxes'])  # (n_objects, 4)
        labels = np.array(objects['labels'])  # (n_objects)
        difficulties = np.array(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            mask = difficulties != 1
            boxes = boxes[mask]
            labels = labels[mask]
            difficulties = difficulties[mask]

        orig_width, orig_height = image.size

        image = TF.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = TF.to_tensor(image).float()

        center_mask = np.zeros((IMG_WIDTH // MODEL_SCALE, IMG_HEIGHT // MODEL_SCALE), dtype='float32')
        regr_bbox = np.zeros((IMG_WIDTH // MODEL_SCALE, IMG_HEIGHT // MODEL_SCALE, 4), dtype='float32')

        for box in boxes:
            xmin, ymin, xmax, ymax = box

            x = xmin + (xmax - xmin) // 2
            y = ymin + (ymax - ymin) // 2

            x = x * (IMG_WIDTH / orig_width)
            x = x / MODEL_SCALE
            x = np.round(x).astype('int')

            y = y * (IMG_HEIGHT / orig_height)
            y = y / MODEL_SCALE
            y = np.round(y).astype('int')

            center_mask[y, x] = 1
            regr_bbox[y, x] = [xmin * (IMG_WIDTH / orig_width),
                               ymin * (IMG_HEIGHT / orig_height),
                               xmax * (IMG_WIDTH / orig_width),
                               ymax * (IMG_HEIGHT / orig_height)]

        regr_bbox = np.transpose(regr_bbox, (2, 0, 1))

        return image, center_mask, regr_bbox