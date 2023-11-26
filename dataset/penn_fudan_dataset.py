import os
import random
import numpy as np
import math

from torch.utils.data import Dataset

from torchvision.transforms import functional as TF
from PIL import Image

from transforms import norm_imagenet

class PennFudanDataset(Dataset):
    def __init__(self, root, train, stride=8, size=384, format='xyxy'):
        self.root = root
        self.train = train

        assert(format in ['xyxy', 'cxcywh'])
        self.format = format
        self.stride = stride
        self.size = size


        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

        self.g_kernel = np.array([
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]]
        ) * 4
        # print(self.g_kernel)
        
        self.g_kernel_w = self.g_kernel.shape[0] // 2

        self.gt_size = (self.size // self.stride, self.size // self.stride)

    def __len__(self):
        return len(self.imgs)

    def transform(self, image, mask):

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image) # [:,:,::-1] # [C,H,W]
            mask = TF.hflip(mask)
        
        return image, mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.train:
            img, mask = self.transform(img, mask)

        img = TF.resize(img, (self.size, self.size))
        # can't resize mask due to incorrect result (see colab notebook)
        # mask = TF.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

        img = TF.to_tensor(img).float()
        # img = TF.normalize(img, norm_imagenet['mean'], norm_imagenet['std'])
        
        # get masks
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:] # remove background id
        masks = (mask == obj_ids[:, None, None])

        # get bboxs from masks
        bboxs = []
        num_objs = len(obj_ids)
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            bboxs.append([xmin, ymin, xmax, ymax])

        # mask.shape == original shape
        if self.format == 'xyxy':
            center_mask, regr_bbox = self.get_gt_xyxy(bboxs, mask.shape)
        elif self.format == 'cxcywh':
            center_mask, regr_bbox = self.get_gt_cxcywh(bboxs, mask.shape)
        else:
            raise Exception('Unknown format')
    
        regr_bbox = np.transpose(regr_bbox, (2, 0, 1)) # (4, W, H)

        return img, center_mask, regr_bbox
    
    def get_gt_xyxy(self, bboxs, orig_shape):
        orig_height, orig_width = orig_shape

        center_mask = np.zeros(self.gt_size, dtype='float32')
        regr_bbox = np.zeros((*self.gt_size, 4), dtype='float32')

        for bbox in bboxs:
            xmin, ymin, xmax, ymax = bbox

            x = xmin + (xmax - xmin) / 2
            y = ymin + (ymax - ymin) / 2

            # TODO: is it correct to use math.floor?
            # math.floor fixes oob error for small feature maps
            x = x * (self.size / orig_width)
            x = round(x / self.stride)
            
            y = y * (self.size / orig_height)
            y = round(y / self.stride)

            center_mask[y, x] = 1
            # center_mask[y-self.g_kernel_w:y+self.g_kernel_w+1,
            #             x-self.g_kernel_w:x+self.g_kernel_w+1] = self.g_kernel
            
            regr_bbox[y, x] = [xmin / orig_width,
                               ymin / orig_height,
                               xmax / orig_width,
                               ymax / orig_height]
            
        return center_mask, regr_bbox
    
    def get_gt_cxcywh(self, bboxs, orig_shape):

        orig_height, orig_width = orig_shape

        center_mask = np.zeros(self.gt_size, dtype='float32')
        regr_bbox = np.zeros((*self.gt_size, 4), dtype='float32')

        for bbox in bboxs:
            xmin, ymin, xmax, ymax = bbox

            # original coordinates
            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w / 2
            cy = ymin + h / 2

            # fractional form
            # same numbers on original image and resized image
            f_cx = cx / orig_width
            f_cy = cy / orig_height

            # coordinates in feature map
            cm_x = round(f_cx * self.size / self.stride) 
            cm_y = round(f_cy * self.size / self.stride)

            center_mask[cm_y, cm_x] = 1

            # coordinates in resized image
            # center_mask -> orig_image
            # these coordinates are close to the center of the receptive field
            orig_x = cm_x * self.stride + self.stride // 2
            orig_y = cm_y * self.stride + self.stride // 2

            # fractional form
            orig_x = orig_x / self.size
            orig_y = orig_y / self.size

            regr_bbox[cm_y, cm_x] = [
                f_cx - orig_x,
                f_cy - orig_y,
                w / orig_width,
                h / orig_height
            ]

        return center_mask, regr_bbox
