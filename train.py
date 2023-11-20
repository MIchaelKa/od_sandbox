
from centernet import create_model
from dataset.penn_fudan.penn_fudan_dataset import PennFudanDataset

import torch
from torch.utils.data import Subset, DataLoader

import torch.nn.functional as F

from common.logger import logger
import logging

from common.utils import get_device, seed_everything

def get_dataset():
    ROOT_DIR = 'dataset/penn_fudan/PennFudanPed'

    dataset_train = PennFudanDataset(ROOT_DIR, train=True)
    dataset_test = PennFudanDataset(ROOT_DIR, train=False)

    test_split_index = 50

    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = Subset(dataset_train, indices[:-test_split_index])
    dataset_test = Subset(dataset_test, indices[-test_split_index:])

    logger.info(f'Train size: {len(dataset_train)} / Test size: {len(dataset_test)}')
    logger.info(f'first idxs: {indices[:5]}')

    return dataset_train, dataset_test

def criterion(prediction, mask, bbox):

    # 1. Binary mask loss
    pred_mask = torch.sigmoid(prediction[:,0])
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).mean()
    # mask_loss = -mask_loss.sum(0).sum()
    # mask_loss = -mask_loss.mean(0).sum() # original
    logger.debug(f'mask_loss: {mask_loss}')

    # 2. L1 loss for bbox coords
    pred_bbox = prediction[:,1:]
    regr_loss = (torch.abs(pred_bbox - bbox).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean()
    logger.debug(f'regr_loss: {regr_loss}')

    loss = mask_loss + regr_loss

    return loss

def criterion_2(prediction, mask, bbox):

    mask_loss = F.binary_cross_entropy_with_logits(prediction[:,0], mask, reduction='mean')
    logger.debug(f'mask_loss: {mask_loss}')

    pred_bbox = prediction[:,1:]
    regr_loss = F.l1_loss(pred_bbox * mask.unsqueeze(1), bbox, reduction='mean')
    logger.debug(f'regr_loss: {regr_loss}')

    loss = mask_loss + regr_loss

    return loss

def train(model, data_loader_train, model_save_name):
    num_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        logger.info('epoch: {}'.format(epoch))
        train_epoch(model, data_loader_train, optimizer)

    torch.save(model.state_dict(), model_save_name)
    logger.info(f'Model saved to {model_save_name}')

def train_epoch(model, data_loader, optimizer):

    for iter_num, data_tensor in enumerate(data_loader):
        image, mask, bbox = data_tensor

        output = model(image)

        logger.info(f'min: {torch.min(output[:,0]).item()}, max: {torch.max(output[:,0]).item()}')
        loss = criterion_2(output, mask, bbox)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_item = loss.item()
        logger.info('iter: {:>4d}, loss = {:.5f}'.format(iter_num, loss_item))

def test_loss(model, data_loader):

    logger.debug('Testing loss...')
    data_tensor = next(iter(data_loader))
    image, mask, bbox = data_tensor # [B,C,H,W], [B,H,W], [B,4,H,W]
    logger.debug(f'image: {image.shape}, mask: {mask.shape}, bbox: {bbox.shape}')

    loss = criterion(model(image), mask, bbox)
    logger.debug(f'loss: {loss}')

    loss_2 = criterion_2(model(image), mask, bbox)
    logger.debug(f'loss_2: {loss_2}')

    
def main():

    seed_everything(1024)

    # DEBUG INFO WARNING ERROR CRITICAL
    logger.setLevel(logging.INFO)

    dataset_train, dataset_test = get_dataset()

    data_loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=0)
    data_loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=0)

    model = create_model()

    # test_loss(model, data_loader_train)

    model_name = 'centernet_v1'
    model_save_name = f'{model_name}.pth'
    train(model, data_loader_train, model_save_name)

if __name__ == "__main__":
    main()