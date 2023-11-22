import torch
import torch.nn as nn

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class SharedHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
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
        
        self.head_cls = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.head_reg = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        cls = self.head_cls(x)
        reg = self.head_reg(x)
        x = torch.cat([cls, reg], dim=1)
        return x

class CenterNet(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = SharedHead() # SharedHead, DecoupledHead

    def forward(self, x):
        features = self.backbone(x)
        features = list(features.values())
        # print(features[2].shape)
        out = self.head(features[0])
        return out
    
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