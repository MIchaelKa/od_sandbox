import torch

from common.logger import logger

class Trainer():
    def __init__(self, model, device, criterion, optimizer, tb_writer):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.tb_writer = tb_writer

        model_name = 'centernet'
        version = 2
        self.model_save_name = f'./ckpts/{model_name}_v{version}.pth'

        self.print_every = 10

        self.save_checkpoint = True

    def fit(self, train_loader, val_loader, num_epochs):

        logger.info('start training...')

        for epoch in range(num_epochs):
            logger.info('train epoch: {}'.format(epoch))
            self.train_epoch(epoch, train_loader)

        if self.save_checkpoint:
            torch.save(self.model.state_dict(), self.model_save_name)
            logger.info(f'model saved to {self.model_save_name}')

        self.writer.close()
     

    def train_epoch(self, epoch, data_loader):

        self.model.train()

        for iter_num, data_tensor in enumerate(data_loader):
            image, mask, bbox = data_tensor
            image = image.to(self.device)
            mask = mask.to(self.device)
            bbox = bbox.to(self.device)

            # should mask have integer type?
            # logger.info(f'tensor types: {image.dtype}, {mask.dtype}, {bbox.dtype}')

            output = self.model(image)

            logger.debug(f'min: {torch.min(output[:,0]).item()}, max: {torch.max(output[:,0]).item()}')
            loss = self.criterion(output, mask, bbox)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_item = loss.item()

            if self.print_every > 0 and iter_num % self.print_every == 0:
                logger.info('iter: {:>4d}, loss = {:.5f}'.format(iter_num, loss_item))
