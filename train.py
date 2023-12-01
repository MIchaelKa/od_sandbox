
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

def get_dataset(dataset_name, stride, format):
    if dataset_name == 'voc':
        return  get_dataset_voc(stride)
    elif dataset_name == 'penn_fud':
        return get_dataset_pf(stride, format)
    else:
        logger.error(f'No datasets with name : {dataset_name}')

def get_dataset_pf(stride, format):
    ROOT_DIR = 'data/PennFudanPed'

    dataset_train = PennFudanDataset(ROOT_DIR, train=True, stride=stride, format=format)
    dataset_test = PennFudanDataset(ROOT_DIR, train=False, stride=stride, format=format)

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

    #
    # config
    #

    # v2 = 3_1024_mesh_fm1
    experiment_version = 'v2'
    experiment_name = 'cxcywh_g'
    num_epochs = 50
    feature_map = 1
    add_mesh = True

    dataset_format = 'cxcywh'
    assert(dataset_format in ['xyxy', 'cxcywh'])

    dataset_name = 'penn_fud' # voc, penn_fud
    dataset_size = 384
    
    strides = { i: 2**(i+3) for i in range(4) } 
    stride = strides[feature_map]

    dataset_info = dict(
        name=dataset_name,
        format=dataset_format,
        size=dataset_size,
        stride=stride,
    )

    #
    # init
    #

    tb_writer = create_tb_writer(
        experiment_name,
        experiment_version,
        dataset_name
    )

    # DEBUG INFO WARNING ERROR CRITICAL
    logger.setLevel(logging.INFO)

    device = get_device()

    dataset_train, dataset_test = get_dataset(dataset_name, stride, dataset_format)

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
        tb_writer=tb_writer,
        dataset_info=dataset_info,
    )
    trainer.fit(data_loader_train, data_loader_test, num_epochs)


def debug():

    seed_everything(1024)

    dataset_format = 'cxcywh' # ['xyxy', 'cxcywh']
    dataset_name = 'penn_fud' # voc, penn_fud
    stride = 8

    dataset_train, dataset_test = get_dataset(dataset_name, stride, dataset_format)

    index = 0
    image, mask, bboxs = dataset_test[index]

    print(image.shape)
    # print(bboxs.shape)

    import torchvision
    from torchvision.models.detection.fcos import FCOS_ResNet50_FPN_Weights
    
    model = torchvision.models.detection.fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
    # model.eval()

    bboxs = torch.tensor([
        [5,10,20,30],
        [50,50,100,120],
        [200,200,220,250]
    ])
    labels = torch.tensor([1,2,3]).long()
    targets = dict(
        boxes=bboxs,
        labels=labels,
    )

    image = image.unsqueeze(0)
    outputs = model(image, [targets]) 

    # print(outputs)

if __name__ == "__main__":
    # main()
    debug()