import torch
from torchvision import transforms

from utils import load_image, save_image
from config import hp
from net import StyleTransferNet


# ! load & configure content image
content_image = load_image(
    "../Images/Content/tanjore.jpg",
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
)
content_image = content_image.unsqueeze(0).to(hp.device)

# ! load style transfer net & set parameters
transfer_net = StyleTransferNet()
transfer_net.load_state_dict(torch.load("./ckpt_nets/transfer_oil_net_e{}b{}.pth".format(2, 7000)))
transfer_net.to(hp.device).eval()

# ! stylize & save output
output = transfer_net(content_image).cpu()
save_image("outputs/taj_oil.jpg", output[0].detach())
