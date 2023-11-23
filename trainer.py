import torch

import numpy as np
import torchvision

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

        self.make_predictions(0, val_loader)

        # for epoch in range(num_epochs):
        #     logger.info('train epoch: {}'.format(epoch))
        #     self.train_epoch(epoch, train_loader)

        if self.save_checkpoint:
            torch.save(self.model.state_dict(), self.model_save_name)
            logger.info(f'model saved to {self.model_save_name}')

        self.tb_writer.close()

    def make_predictions(self, epoch, data_loader):

        # get one batch
        data_tensor = next(iter(data_loader))
        image, gt_mask, gt_bbox = data_tensor

        image = image.to(self.device)
        gt_mask = gt_mask.to(self.device)
        gt_bbox = gt_bbox.to(self.device)

        self.model.eval()

        batch_pred = self.model(image)
        batch_pred = batch_pred.detach().cpu()

        # pred_mask = batch_pred[:,0,:,:]
        # pred_mask = pred_mask.unsqueeze(1)
        # print(pred_mask.shape)
        # grid = torchvision.utils.make_grid(pred_mask)
        # self.tb_writer.add_image('masks', grid, 0)
        # return

        for i, pred in enumerate(batch_pred):

            print(pred.shape)

            pred_mask = pred[0]
            pred_mask = torch.sigmoid(pred_mask)
            # pred_mask = pred_mask.unsqueeze(0)
            pred_mask = pred_mask.numpy()
            threshold = 0.5
            pred_mask = np.int32(np.where(pred_mask > threshold, pred_mask, 0))
            
            print(pred_mask)

            pred_bbox = pred[1:]
            pred_bbox = pred_bbox.numpy()

            pred_bbox = np.transpose(pred_bbox, (1, 2, 0)) * 384

            print(pred_bbox.shape)

            img = image[0].detach().cpu().numpy()
            img = np.int32(np.transpose(img, (1, 2, 0)) * 255)
            print(img.shape)
            

            # self.tb_writer.add_image('image', img, i)
            self.tb_writer.add_image('mask', pred_mask, i, dataformats='HW')
            # self.tb_writer.add_image('gt_mask', pred_mask, i)
     

    def train_epoch(self, epoch, data_loader):

        self.model.train()

        iter_start = epoch * len(data_loader)

        for iter_num, data_tensor in enumerate(data_loader):
            image, mask, bbox = data_tensor
            image = image.to(self.device)
            mask = mask.to(self.device)
            bbox = bbox.to(self.device)

            global_iter = iter_start+iter_num

            # should mask have integer type?
            # logger.info(f'tensor types: {image.dtype}, {mask.dtype}, {bbox.dtype}')

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


