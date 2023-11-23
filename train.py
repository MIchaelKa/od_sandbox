
from centernet import create_model
from dataset.penn_fudan_dataset import PennFudanDataset

import torch
from torch.utils.data import Subset, DataLoader

from common.logger import logger
import logging

from common.utils import get_device, seed_everything
from criterion import *

def get_dataset():
    ROOT_DIR = 'data/PennFudanPed'

    dataset_train = PennFudanDataset(ROOT_DIR, train=True)
    dataset_test = PennFudanDataset(ROOT_DIR, train=False)

    test_split_index = 50

    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = Subset(dataset_train, indices[:-test_split_index])
    dataset_test = Subset(dataset_test, indices[-test_split_index:])

    logger.info(f'Train size: {len(dataset_train)} / Test size: {len(dataset_test)}')
    logger.info(f'first idxs: {indices[:5]}')

    return dataset_train, dataset_test

def train(model, data_loader_train, model_save_name):
    num_epochs = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        logger.info('epoch: {}'.format(epoch))
        model.train()
        train_epoch(model, data_loader_train, optimizer)

    torch.save(model.state_dict(), model_save_name)
    logger.info(f'Model saved to {model_save_name}')

def train_epoch(model, data_loader, optimizer):

    for iter_num, data_tensor in enumerate(data_loader):
        image, mask, bbox = data_tensor

        output = model(image)

        logger.info(f'min: {torch.min(output[:,0]).item()}, max: {torch.max(output[:,0]).item()}')
        loss = criterion_1_5(output, mask, bbox)

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

    loss = criterion_1(model(image), mask, bbox)
    logger.debug(f'loss: {loss}')

    loss_2 = criterion_2(model(image), mask, bbox)
    logger.debug(f'loss_2: {loss_2}')

    
def main():

    seed_everything(1024)

    # DEBUG INFO WARNING ERROR CRITICAL
    logger.setLevel(logging.INFO)

    dataset_train, dataset_test = get_dataset()

    dataset_train = Subset(dataset_train, range(8))

    data_loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=0)
    data_loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=0)

    model = create_model()

    # test_loss(model, data_loader_train)

    model_name = 'centernet_v1'
    model_save_name = f'{model_name}.pth'
    train(model, data_loader_train, model_save_name)

if __name__ == "__main__":
    main()