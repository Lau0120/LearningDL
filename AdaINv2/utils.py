import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms


def AdaIN(cont_feature_map, styl_feature_map):
    cont_mean = cont_feature_map.mean(dim=(2, 3), keepdim=True)
    cont_stds = cont_feature_map.std(dim=(2, 3), keepdim=True)
    styl_mean = styl_feature_map.mean(dim=(2, 3), keepdim=True)
    styl_stds = styl_feature_map.std(dim=(2, 3), keepdim=True)
    return (styl_stds * (cont_feature_map - cont_mean) / (cont_stds + 1e-5)) + styl_mean


def load_image(filename, size=None, scale=None, transform=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img if transform is None else transform(img)


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def inv_normz(img):
    stds = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(img.device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(img.device)
    return torch.clamp(img * stds + mean, 0, 1)
