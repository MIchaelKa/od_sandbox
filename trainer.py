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

        # self.make_predictions(0, val_loader)

        for epoch in range(num_epochs):
            logger.info('train epoch: {}'.format(epoch))
            self.train_epoch(epoch, train_loader)
            self.make_predictions(epoch, val_loader)

        if self.save_checkpoint:
            torch.save(self.model.state_dict(), self.model_save_name)
            logger.info(f'model saved to {self.model_save_name}')

        self.tb_writer.close()

    def make_predictions(self, epoch, data_loader):

        # get one batch
        data_tensor = next(iter(data_loader))
        images, gt_masks, gt_bboxs = data_tensor

        images = images.to(self.device) # (B, 3, W, H)
        gt_masks = gt_masks.to(self.device) # (B, W, H)
        gt_bboxs = gt_bboxs.to(self.device) # (B, 4, W, H)

        self.model.eval()
        batch_pred = self.model(images)

        batch_pred = batch_pred.detach().cpu() # (B, 5, W, H)
        images = images.detach().cpu()
        gt_masks = gt_masks.detach().cpu()
        gt_bboxs = gt_bboxs.detach().cpu()

        # pred_masks = batch_pred[:2,0,:,:].unsqueeze(1)
        # grid_masks = torchvision.utils.make_grid(pred_masks)
        # self.tb_writer.add_image('masks', grid_masks, 0)

        # self.tb_writer.add_images('images', images, 0)

        # pred.shape = (5, W, H)
        for i, pred in enumerate(batch_pred):
          
            pred_mask = pred[0] # (W, H)
            pred_mask = torch.sigmoid(pred_mask)
            # pred_mask = pred_mask.numpy()
            threshold = 0.5
            pred_mask = np.where(pred_mask > threshold, pred_mask, 0)
           
            pred_bbox = pred[1:] # (4, W, H)
            pred_bbox = np.transpose(pred_bbox, (1, 2, 0)) * 384 # (W, H, 4)
            xs, ys = np.nonzero(gt_masks[i].numpy()) # gt_mask, pred_mask
            pred_bbox = pred_bbox[xs, ys] # (N, 4)
            
            gt_bbox = gt_bboxs[i]
            gt_bbox = np.transpose(gt_bbox, (1, 2, 0)) * 384 # (W, H, 4)
            xs, ys = np.nonzero(gt_masks[i].numpy())
            gt_bbox = gt_bbox[xs, ys] # (N, 4)

            # self.tb_writer.add_image('image', image[i], i)
            self.tb_writer.add_image_with_boxes(f'pred_image/{i}', images[i], pred_bbox, epoch)
            self.tb_writer.add_image(f'pred_mask/{i}', pred_mask, epoch, dataformats='HW')

            if epoch == 0:
                self.tb_writer.add_image_with_boxes(f'gt_image/{i}', images[i], gt_bbox, epoch)
                self.tb_writer.add_image(f'gt_mask/{i}', gt_masks[i], epoch, dataformats='HW')
            

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


