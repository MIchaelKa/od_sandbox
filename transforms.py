import torch

norm_imagenet = {
    'mean': [0.485, 0.456, 0.406], 
    'std': [0.229, 0.224, 0.225]
}

def unnormalize(img, norm=norm_imagenet):
    img = img * torch.tensor(norm['std']).view(3, 1, 1)
    img = img + torch.tensor(norm['mean']).view(3, 1, 1)
    return img