import torch
import torch.nn as nn

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class Head(nn.Module):

    def __init__(self):
        super().__init__()

        # Shared head for center-ness and box regression
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 5, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.head(x)
        return x

class CenterNet(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = Head()

    def forward(self, x):
        features = self.backbone(x)
        features = list(features.values())

        # print(features[0].shape)

        out = self.head(features[0])
        return out
    
def create_model():
    backbone = resnet_fpn_backbone(
        'resnet18', # resnet18, resnet50
        pretrained=True,
        trainable_layers=2,
        returned_layers=[2,3,4]
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