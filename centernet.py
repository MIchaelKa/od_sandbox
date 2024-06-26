import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    
class Head(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # in_channels = [in_channels, 258, 258]
        # in_channels = [in_channels, 512, 512, 512]
        channels = [in_channels, 512, 1024]

        conv = []
        for i in range(len(channels)-1):
            conv.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1))
            conv.append(nn.BatchNorm2d(channels[i+1]))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # Same as using one nn.Conv2d(channels[-1], 5, kernel_size=1, padding=0)?
        # Not the same since we want F.relu only for head_reg output
        self.head_reg = nn.Conv2d(channels[-1], 4, kernel_size=1, padding=0)
        self.head_cls = nn.Conv2d(channels[-1], 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        cls = self.head_cls(x)
        reg = F.relu(self.head_reg(x))
        x = torch.cat([cls, reg], dim=1)
        return x

class CenterNet(nn.Module):

    def __init__(self, backbone, feature_map=0, add_mesh=False):
        super().__init__()
        self.backbone = backbone
        self.feature_map = feature_map
        self.add_mesh = add_mesh

        out_channels = self.backbone.out_channels
        if add_mesh:
            out_channels += 2

        self.head = Head(out_channels)

    def forward(self, x):
        features = self.backbone(x)
        features = list(features.values())
        features = features[self.feature_map]

        if self.add_mesh:
            b, c, h, w = features.shape
            mesh = get_mesh(b, h, w).to(features.device)
            features = torch.cat([features, mesh], 1)
        
        out = self.head(features)
        return out

import numpy as np

def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x), torch.tensor(mg_y)], 1)
    return mesh
   
def create_model(feature_map, add_mesh):
    backbone = resnet_fpn_backbone(
        'resnet18', # resnet18, resnet50
        pretrained=True,
        trainable_layers=5,
        returned_layers=[2,3]
    )
    net = CenterNet(backbone, feature_map, add_mesh)
    return net

if __name__ == "__main__":
 
    net = create_model()
    x = torch.randn(1, 3, 512, 512)
    y = net(x)

    print(y.shape)
    
    # center_logits = y[:, 0, :, :]
    center_logits = y[0,0]
    print(center_logits.shape)