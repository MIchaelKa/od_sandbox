import torch

import time
from common.logger import logger, format_time
from common.visualizer import Visualizer

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

        self.print_every = 1

        self.save_checkpoint = True

        self.visualizer = Visualizer(
            model=model,
            device=device,
            tb_writer=tb_writer,
        )

    def fit(self, train_loader, val_loader, num_epochs):

        t0 = time.time()
        logger.info('start training...')

        vis_last_epoch = 3

        # self.visualizer.vis_preds(0, val_loader, split='val')
        # return

        for epoch in range(num_epochs):

            t1 = time.time()
            logger.info('train epoch: {}'.format(epoch))

            self.train_epoch(epoch, train_loader)

            epoch_time = time.time() - t1
            self.tb_writer.add_scalar('train/epoch_time', epoch_time, epoch)
            logger.info('finish epoch: {}, epoch_time: {}'.format(epoch, format_time(epoch_time)))

            if epoch > num_epochs - vis_last_epoch - 1:
                t2 = time.time()
                self.visualizer.vis_preds(epoch, train_loader, split='train')
                self.visualizer.vis_preds(epoch, val_loader, split='val')
                logger.info('vis epoch: {}, vis_time: {}'.format(epoch, format_time(time.time() - t2)))

        if self.save_checkpoint:
            torch.save(self.model.state_dict(), self.model_save_name)
            logger.info(f'model saved to {self.model_save_name}')

        logger.info(f'finish training, time: {format_time(time.time() - t0)}')

        self.tb_writer.close()
            

    def train_epoch(self, epoch, data_loader):

        self.model.train()

        iter_start = epoch * len(data_loader)

        for iter_num, data_tensor in enumerate(data_loader):
            image, mask, bbox = data_tensor

            image = image.to(self.device)
            mask = mask.to(self.device)
            bbox = bbox.to(self.device)

            global_iter = iter_start+iter_num

            # TODO: visualize image, mask, bbox

            # should mask have integer type?
            # logger.info(f'tensor types: {image.dtype}, {mask.dtype}, {bbox.dtype}')
            
            # logger.info(f'image mean: {torch.mean(image)}')

            output = self.model(image)

            mask_max = torch.max(output[:,0]).item()
            mask_min = torch.min(output[:,0]).item()

            self.tb_writer.add_scalar('train/mask_max', mask_max, global_iter)

            loss_dict = self.criterion(output, mask, bbox)

            loss = loss_dict['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_item = loss.item()
            mask_loss = loss_dict['mask_loss'].item()
            box_loss = loss_dict['box_loss'].item()

            self.tb_writer.add_scalar('train/loss', loss_item, global_iter)
            self.tb_writer.add_scalar('train/mask_loss', mask_loss, global_iter)
            self.tb_writer.add_scalar('train/box_loss', box_loss, global_iter)

            if self.print_every > 0 and iter_num % self.print_every == 0:
                logger.info('iter: {:>4d}, loss = {:.5f}'.format(iter_num, loss_item))
                logger.debug('iter: {:>4d}, mask_loss = {:.5f}, box_loss = {:.5f}'
                             .format(iter_num, mask_loss, box_loss))
                
                logger.debug('iter: {:>4d}, mask: min = {:.5f}, max = {:.5f}'.format(iter_num, mask_min, mask_max))


