import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from train import training_loop
from net import StyleTransferNet, Vgg16
from config import hp
from utils import load_image, calc_gram_matrix, normalize_batch


# ! configure loader
cocos = ImageFolder(
    transform=transforms.Compose([
        transforms.Resize(hp.image_size),
        transforms.CenterCrop(hp.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ]),
    root=hp.dataset,
)
cocos_loader = DataLoader(cocos, hp.batch_size, shuffle=True)


# ! calculate gram matrix of style
loss_calculator = Vgg16(requires_grad=False).to(hp.device)
styl_img = load_image(
    hp.style_image,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ]),
    size=hp.image_size,
)
styl_img = styl_img.repeat(hp.batch_size, 1, 1, 1).to(hp.device)
features_style = loss_calculator(normalize_batch(styl_img))
gram_style = [calc_gram_matrix(y) for y in features_style]


# ! start training
rtstn = StyleTransferNet().to(hp.device)
training_loop(
    n_epochs=hp.n_epochs,
    loader=cocos_loader,
    transfer_net=rtstn,
    optimizer=optim.Adam(rtstn.parameters(), lr=hp.learning_rate),
    loss_net=loss_calculator,
    cw=hp.cont_weight,
    sw=hp.styl_weight,
    gram_styl=gram_style,
)
