
from centernet import create_model
from dataset.penn_fudan.penn_fudan_dataset import PennFudanDataset

import torch
from torch.utils.data import Subset, DataLoader


from common.logger import logger
import logging

from common.utils import get_device, seed_everything

def train():

    seed_everything(1024)

    # DEBUG INFO WARNING ERROR CRITICAL
    logger.setLevel(logging.INFO)

    ROOT_DIR = 'dataset/penn_fudan/PennFudanPed'

    dataset_train = PennFudanDataset(ROOT_DIR, train=True)
    dataset_test = PennFudanDataset(ROOT_DIR, train=False)

    test_split_index = 50

    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = Subset(dataset_train, indices[:-test_split_index])
    dataset_test = Subset(dataset_test, indices[-test_split_index:])

    print(f'Train size: {len(dataset_train)} / Test size: {len(dataset_test)}')

    data_loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=0)
    data_loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=0)

    model = create_model()

    # test_loss(model, data_loader_train)

    num_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        logger.info('epoch: {}'.format(epoch))
        train_epoch(model, data_loader_train, optimizer)

    model_name = 'centernet_v0'
    model_save_name = f'{model_name}.pth'
    torch.save(model.state_dict(), model_save_name)
        


def train_epoch(model, data_loader, optimizer):

    for iter_num, data_tensor in enumerate(data_loader):
        image, mask, bbox = data_tensor

        output = model(image)

        loss = criterion(output, mask, bbox)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_item = loss.item()
        logger.info('iter: {:>4d}, loss = {:.5f}'.format(iter_num, loss_item))


def test_loss(model, data_loader):

    print('Testing loss...')
    data_tensor = next(iter(data_loader))
    image, mask, bbox = data_tensor # [B,C,H,W], [B,H,W], [B,4,H,W]
    print(image.shape, mask.shape, bbox.shape)

    loss = criterion(model(image), mask, bbox)
    print(loss)
    

# TODO: Use losses from pytorch
def criterion(prediction, mask, bbox):
    # 1. Binary mask loss
    pred_mask = torch.sigmoid(prediction[:,0])
    alpha = 10000
    mask_loss = alpha * mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()

    # 2. L1 loss for bbox coords
    pred_bbox = prediction[:,1:]
    regr_loss = (torch.abs(pred_bbox - bbox).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean()

    beta = 10000
    loss = mask_loss + beta * regr_loss

    return loss




    

if __name__ == "__main__":
    train()