import torch

import numpy as np
import torchvision

from transforms import unnormalize

import cv2

color_gt = (255, 0, 0)
color_pred = (0, 255, 0)

class Visualizer:

    def __init__(self, model, device, tb_writer):
        self.model = model
        self.device = device
        self.tb_writer = tb_writer

    def vis_preds(self, epoch, data_loader, split='val', num_vis=4):

        # get one batch
        data_tensor = next(iter(data_loader))
        images, gt_masks, gt_bboxs = data_tensor

        images = images.to(self.device) # (B, 3, W, H)
        gt_masks = gt_masks.to(self.device) # (B, W, H)
        gt_bboxs = gt_bboxs.to(self.device) # (B, 4, W, H)

        self.model.eval()
        batch_pred = self.model(images)

        batch_pred = batch_pred.detach().cpu()[:num_vis] # (B, 5, W, H)

        images = unnormalize(images)

        images = images.detach().cpu().numpy()
        gt_masks = gt_masks.detach().cpu().numpy()
        gt_bboxs = gt_bboxs.detach().cpu().numpy()

        # pred.shape = (5, W, H)
        for i, pred in enumerate(batch_pred):
          
            pred_mask = pred[0] # (W, H)
            pred_mask = torch.sigmoid(pred_mask)

            # pred_mask = pred_mask.numpy()
            threshold = 0.5
            pred_mask = np.where(pred_mask > threshold, pred_mask, 0)
           
            pred_bbox = pred[1:] # (4, W, H)

            pred_bbox = self.extract_bboxs(pred_bbox, gt_masks[i]) # pred_mask
            gt_bbox = self.extract_bboxs(gt_bboxs[i], gt_masks[i])

            image = images[i].transpose(1, 2, 0) # (W, H, 3)

            gt_img = self.draw_bboxs(image.copy(), gt_bbox, color_gt)
            pred_img = self.draw_bboxs(image.copy(), pred_bbox, color_pred)

            gt_mask_img = self.get_mask_to_draw(gt_masks[i], image.shape[:2])
            pred_mask_img = self.get_mask_to_draw(pred_mask, image.shape[:2])

            image_to_draw = np.stack([gt_img, gt_mask_img, pred_img, pred_mask_img], axis=0)

            self.tb_writer.add_images(f'{split}_image/{i}', image_to_draw, epoch, dataformats='NHWC')

    def extract_bboxs(self, bboxs, mask):
        bboxs = np.transpose(bboxs, (1, 2, 0)) * 384 # (W, H, 4)
        xs, ys = np.nonzero(mask)
        bboxs = np.int32(bboxs[xs, ys]) # (N, 4)
        return bboxs

    def draw_bboxs(self, image, bboxs, color):
        image = (image * 255).astype(np.uint8)
        for bbox in bboxs:
            cv2.rectangle(image, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color, 2)
        return image

    def get_mask_to_draw(self, mask, shape):
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, shape, interpolation=cv2.INTER_NEAREST)
        mask = np.tile(mask[:,:,np.newaxis], (1, 1, 3))
        return mask

    # TODO: old implementation
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