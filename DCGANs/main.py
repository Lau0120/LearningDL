import torch.nn as nn
import torch.optim as optim
import pickle

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from training import training_loop
from generator import generator
from discriminator import discriminator


celeba = ImageFolder(
    root="G:\Project\Deep_Learning\Datasets\CelebA\celeba",
    transform=transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        )
    ])
)

celeba_loader = DataLoader(celeba, 128, shuffle=True)

imagesL = []
lossesG = []
lossesD = []
training_loop(
    epochs=5,
    discriminator=discriminator,
    lossesD=lossesD,
    optimizerD=optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
    generator=generator,
    lossesG=lossesG,
    optimizerG=optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
    loss_fn=nn.BCELoss(),
    imagesL=imagesL,
    loader=celeba_loader,
)

with open("./pickles/images_L.pickle", "wb") as f:
    pickle.dump(imagesL, f)
with open("./pickles/losses_G.pickle", "wb") as f:
    pickle.dump(lossesG, f)
with open("./pickles/losses_D.pickle", "wb") as f:
    pickle.dump(lossesD, f)
