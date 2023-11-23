
from centernet import create_model

from dataset.penn_fudan_dataset import PennFudanDataset
from dataset.pascal_voc_dataset import PascalVOCDataset

import torch
from torch.utils.data import Subset, DataLoader

from common.logger import logger
import logging

from common.logger import create_tb_writer

from common.utils import get_device, seed_everything
from criterion import *

from trainer import Trainer

def get_dataset():
    return  get_dataset_voc()

def get_dataset_pf():
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

def get_dataset_voc():

    dataset_train = PascalVOCDataset('./data/VOCdevkit/', 'TRAIN', transforms=None)
    dataset_test = PascalVOCDataset('./data/VOCdevkit/', 'TRAIN', transforms=None)

    test_split_index = int(len(dataset_train) * 0.2)

    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = Subset(dataset_train, indices[:-test_split_index])
    dataset_test = Subset(dataset_test, indices[-test_split_index:]) 

    logger.info(f'Train size: {len(dataset_train)} / Test size: {len(dataset_test)}')
    logger.info(f'first idxs: {indices[:5]}')

    return dataset_train, dataset_test

    
def main():

    seed_everything(1024)

    experiment_name = 'base'
    tb_writer = create_tb_writer(experiment_name)

    # DEBUG INFO WARNING ERROR CRITICAL
    logger.setLevel(logging.INFO)

    device = get_device()

    dataset_train, dataset_test = get_dataset()

    # dataset_train = Subset(dataset_train, range(4))

    data_loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)
    data_loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=0)

    model = create_model().to(device)
    # test_loss(model, data_loader_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = Criterion()
    num_epochs = 3

    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        tb_writer=tb_writer
    )
    trainer.fit(data_loader_train, data_loader_test, num_epochs)

if __name__ == "__main__":
    main()