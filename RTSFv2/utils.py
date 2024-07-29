import torch

from PIL import Image


def calc_gram_matrix(x):
    b, c, h, w = x.shape
    features = x.view(b, c, h * w)
    features_t = features.transpose(1, 2)
    return features.bmm(features_t) / (c * h * w)


def normalize_batch(b):
    mean = b.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    stds = b.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    b = b.div_(255.0)
    return (b - mean) / stds


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
