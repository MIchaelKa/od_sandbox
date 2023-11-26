import torch

norm_imagenet = {
    'mean': [0.485, 0.456, 0.406], 
    'std': [0.229, 0.224, 0.225]
}

def unnormalize(img, norm=norm_imagenet):
    img = img * torch.tensor(norm['std']).to(img.device).view(3, 1, 1)
    img = img + torch.tensor(norm['mean']).to(img.device).view(3, 1, 1)
    return img