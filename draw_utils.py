
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

import torchvision.transforms as T

from common.logger import logger

def show_images(images, titles, size=5):
    
    count = len(images)
    plt.figure(figsize=(count * size, size))
    
    for i in range(count):    
        plt.subplot(1, count, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i])
        # plt.grid(False)
        plt.axis('off')

        # if i == (count - 1):
        #     plt.colorbar()


def visualize_dataset(dataset, count=5, size=6):
    plt.figure(figsize=(count * size, size * 2))
    
    for i, (img, mask, bboxs) in enumerate(dataset):
        if i == count:
            break

        # draw bbox
        color = (255, 0, 0)
        img_arr = np.array(T.ToPILImage()(img))
        xs, ys = np.nonzero(mask)
        bboxs = np.transpose(bboxs, (1, 2, 0))
        center_bboxs = np.int32(bboxs[xs, ys])
        for bbox in center_bboxs:
            cv2.rectangle(img_arr, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, 2)

        # image    
        plt.subplot(2, count, i + 1)
        plt.imshow(img_arr)
        plt.axis('off')
        # mask
        plt.subplot(2, count, count + i + 1)
        plt.imshow(mask)
        # plt.imshow(bboxs[:,:,2])
        plt.axis('off')

def make_prediction(model, dataset, index, threshold = 0.5):
    test_img_t, test_mask_t, test_bboxs_t = dataset[index]

    model.eval()
    pred = model(test_img_t)

    pred_mask = pred[:,0]
    # TODO: how to get good distribution for sigmoid?
    pred_mask = torch.sigmoid(pred[:,0])
    
    bboxs = pred[:,1:]

    pred_img = pred_mask.squeeze().detach().cpu().numpy()
    # pred_img = test_mask_t

    logger.info(f'threshold: {threshold}')
    logger.info(f'min: {np.min(pred_img)}, max: {np.max(pred_img)}')
    pred_img = np.where(pred_img > threshold, pred_img, 0)

    img_arr = np.array(T.ToPILImage()(test_img_t))

    # draw bbox
    color = (255, 0, 0)
    xs, ys = np.nonzero(test_mask_t) # pred_img, test_mask_t
    bboxs = bboxs.squeeze().detach().cpu().numpy()
    bboxs = np.transpose(bboxs, (1, 2, 0)) * 384
    center_bboxs = np.int32(bboxs[xs, ys])
    for bbox in center_bboxs:
        # print(bbox)
        cv2.rectangle(img_arr, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, 2)
    
    # visualize
    show_images([img_arr, test_mask_t, pred_img],
                ["Original", "GT Mask", "Prediction"])