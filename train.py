
import torch
from torch.utils.data import Subset, DataLoader

from centernet import create_model

from dataset.penn_fudan_dataset import PennFudanDataset
from dataset.pascal_voc_dataset import PascalVOCDataset

from common.logger import logger
import logging

from common.logger import create_tb_writer

from common.utils import get_device, seed_everything
from criterion import *

from trainer import Trainer

def get_dataset(dataset_name, stride):
    if dataset_name == 'voc':
        return  get_dataset_voc(stride)
    elif dataset_name == 'penn_fud':
        return get_dataset_pf(stride)
    else:
        logger.error(f'No datasets with name : {dataset_name}')

def get_dataset_pf(stride):
    ROOT_DIR = 'data/PennFudanPed'

    dataset_train = PennFudanDataset(ROOT_DIR, train=True, stride=stride)
    dataset_test = PennFudanDataset(ROOT_DIR, train=False, stride=stride)

    test_split_index = 50

    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = Subset(dataset_train, indices[:-test_split_index])
    dataset_test = Subset(dataset_test, indices[-test_split_index:])

    logger.info(f'Train size: {len(dataset_train)} / Test size: {len(dataset_test)}')
    logger.info(f'first idxs: {indices[:5]}')

    return dataset_train, dataset_test

def get_dataset_voc(stride):

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

    experiment_name = '3_1024_mesh_fm1'
    num_epochs = 50
    dataset_name = 'penn_fud' # voc, penn_fud
    feature_map = 1
    add_mesh = True

    strides = { i: 2**(i+3) for i in range(4) } 
    stride = strides[feature_map]

    tb_writer = create_tb_writer(experiment_name, dataset_name)

    # DEBUG INFO WARNING ERROR CRITICAL
    logger.setLevel(logging.INFO)

    device = get_device()

    dataset_train, dataset_test = get_dataset(dataset_name, stride)

    # dataset_train = Subset(dataset_train, range(4))

    data_loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)
    data_loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=0)

    model = create_model(feature_map, add_mesh).to(device)

    criterion = Criterion()
    # test_loss(model, data_loader_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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