import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class SharedHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 5, kernel_size=1, padding=0),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.head(x)
        return x
    
class DecoupledHead(nn.Module):
    def __init__(self):
        super().__init__()

        num_convs = 3
        in_channels = 256
        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            # conv.append(nn.BatchNorm2d(256))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.head_reg = nn.Conv2d(256, 4, kernel_size=1, padding=0)
        self.head_cls = nn.Conv2d(256, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        cls = self.head_cls(x)
        reg = F.relu(self.head_reg(x))
        x = torch.cat([cls, reg], dim=1)
        return x

class CenterNet(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = DecoupledHead() # SharedHead, DecoupledHead

    def forward(self, x):
        features = self.backbone(x)
        features = list(features.values())
        features = features[0]

        # b, c, h, w = features.shape
        # mesh = get_mesh(b, h, w)
        # features = torch.cat([features, mesh], 1)
        
        out = self.head(features)
        return out

import numpy as np

def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x), torch.tensor(mg_y)], 1)
    return mesh
   
def create_model():
    backbone = resnet_fpn_backbone(
        'resnet18', # resnet18, resnet50
        pretrained=True,
        trainable_layers=5,
        returned_layers=[2,3]
    )
    net = CenterNet(backbone)
    return net

if __name__ == "__main__":
 
    net = create_model()
    x = torch.randn(1, 3, 512, 512)
    y = net(x)

    print(y.shape)
    
    # center_logits = y[:, 0, :, :]
    center_logits = y[0,0]
    print(center_logits.shape)