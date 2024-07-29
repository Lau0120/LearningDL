import torch
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms

from utils import load_image, save_image, AdaIN, inv_normz
from net import Decoder, Encoder


content_image = load_image(
    "../Images/Content/dancing.jpg",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1080, 1080)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]),
)
content_image = content_image.unsqueeze(0).to(torch.device("cuda"))


style_image = load_image(
    "../Images/Style/vg.jpg",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1080, 1080)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]),
)
style_image = style_image.unsqueeze(0).to(torch.device("cuda"))


encoder = Encoder().to(torch.device("cuda")).eval()
decoder = Decoder()
decoder.load_state_dict(torch.load("./ckpt_nets/adain_transfer_e{}b{}.pth".format(5, 10000)))
decoder.to(torch.device("cuda")).eval()


fc = encoder(content_image)
fs = encoder(style_image)
t = AdaIN(fc[-1], fs[-1])
gt = inv_normz(decoder(t).squeeze().cpu().detach())
save_image("./outputs/dancing_vg.jpg", gt.mul(255))
