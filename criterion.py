
import torch
from common.logger import logger

import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss

def criterion_1(prediction, mask, bbox):

    # 1. Binary mask loss
    pred_mask = torch.sigmoid(prediction[:,0])
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)

    # TODO: sum vs. mean
    mask_loss = -mask_loss.mean(0).mean()
    # mask_loss = -mask_loss.sum(0).sum()
    # mask_loss = -mask_loss.mean(0).sum() # original
    
    logger.debug(f'mask_loss: {mask_loss}')

    # 2. L1 loss for bbox coords
    pred_bbox = prediction[:,1:]
    regr_loss = (torch.abs(pred_bbox - bbox).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    logger.debug(f'regr_loss: {regr_loss}')

    loss = mask_loss + regr_loss

    return loss

def criterion_1_1(prediction, mask, bbox):

    # 1. Binary mask loss
    pred_mask = torch.sigmoid(prediction[:,0])
    alpha = 1000
    mask_loss = alpha * mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    logger.info(f'mask_loss: {mask_loss}')

    # 2. L1 loss for bbox coords
    pred_bbox = prediction[:,1:]
    regr_loss = (torch.abs(pred_bbox - bbox).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    logger.info(f'regr_loss: {regr_loss}')

    loss = mask_loss + alpha * regr_loss

    return loss

def criterion_1_5(prediction, mask, bbox):
    # TODO: is this loss work with g_kernel?

    # 1. Binary mask loss
    pred_mask = torch.sigmoid(prediction[:,0])
    alpha = 0.995
    mask_loss = alpha * mask * torch.log(pred_mask + 1e-12) + (1 - alpha) * (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    logger.info(f'mask_loss: {mask_loss}')

    # 2. L1 loss for bbox coords
    pred_bbox = prediction[:,1:]

    # TODO: sum(1).sum(1) vs. mean() only
    regr_loss = (torch.abs(pred_bbox - bbox).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    logger.info(f'regr_loss: {regr_loss}')

    beta = 0.5
    loss = (1 - beta) * mask_loss + beta * regr_loss

    return loss

def criterion_2(prediction, mask, bbox):

    # mask_loss = F.binary_cross_entropy_with_logits(prediction[:,0], mask, reduction='mean')
    mask_loss = sigmoid_focal_loss(prediction[:,0], mask, alpha=0.95, gamma=2, reduction='mean')
    logger.debug(f'mask_loss: {mask_loss}')

    pred_bbox = prediction[:,1:]
    regr_loss = F.l1_loss(pred_bbox * mask.unsqueeze(1), bbox, reduction='mean')
    logger.debug(f'regr_loss: {regr_loss}')

    loss = mask_loss + regr_loss

    return loss

# TODO: add center-ness loss