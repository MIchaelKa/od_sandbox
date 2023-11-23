import os
import random
import numpy as np

from torch.utils.data import Dataset

from torchvision.transforms import functional as TF
from PIL import Image

IMG_WIDTH = 384
IMG_HEIGHT = IMG_WIDTH
MODEL_SCALE = 8 # How to calculate this?

class PennFudanDataset(Dataset):
    def __init__(self, root, train):
        self.root = root
        self.train = train
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

        img = TF.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # can't resize mask due to incorrect result (see colab notebook)
        # mask = TF.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

        img = TF.to_tensor(img).float()
        
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:] # remove background id

        orig_height, orig_width = mask.shape

        masks = (mask == obj_ids[:, None, None])

        center_mask = np.zeros((IMG_WIDTH // MODEL_SCALE, IMG_HEIGHT // MODEL_SCALE), dtype='float32')
        regr_bbox = np.zeros((IMG_WIDTH // MODEL_SCALE, IMG_HEIGHT // MODEL_SCALE, 4), dtype='float32')

        # get center coordinates for each mask
        num_objs = len(obj_ids)
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            x = xmin + (xmax - xmin) // 2
            y = ymin + (ymax - ymin) // 2

            x = x * (IMG_WIDTH / orig_width)
            x = x / MODEL_SCALE
            x = np.round(x).astype('int')

            y = y * (IMG_HEIGHT / orig_height)
            y = y / MODEL_SCALE
            y = np.round(y).astype('int')

            center_mask[y, x] = 1
            # center_mask[y-self.g_kernel_w:y+self.g_kernel_w+1,
            #             x-self.g_kernel_w:x+self.g_kernel_w+1] = self.g_kernel
            
            regr_bbox[y, x] = [xmin / orig_width,
                               ymin / orig_height,
                               xmax / orig_width,
                               ymax / orig_height]

        regr_bbox = np.transpose(regr_bbox, (2, 0, 1))

        return img, center_mask, regr_bbox