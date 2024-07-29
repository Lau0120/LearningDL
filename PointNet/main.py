import pickle

from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import ModelNet
from model import ClaPointNet, ClaPointNet2SSG
from train import training_loop


model_n = ModelNet("../../Datasets/ModelNet/ModelNet40_PLY")
model_trn_loader = DataLoader(model_n, batch_size=32, shuffle=True)


net = ClaPointNet2SSG(k=40)
net.train().to("cuda")
losses = training_loop(
    n_iterations=10000,
    network=net,
    loader=model_trn_loader,
    loss_fn=nn.CrossEntropyLoss(),
    optim=optim.Adam(net.parameters(), lr=0.001)
)


with open("losses.pkl", 'wb') as file:
    pickle.dump(losses, file)
