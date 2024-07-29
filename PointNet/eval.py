import torch
from model import ClaPointNet, ClaPointNet2SSG
from dataset import ModelNet

from torch.utils.data import DataLoader

model_n = ModelNet("../../Datasets/ModelNet/ModelNet40_PLY", is_train=False)
model_val_loader = DataLoader(model_n, batch_size=32, shuffle=True)

net = ClaPointNet2SSG(k=40)
net.load_state_dict(torch.load("./ckpts/pointnet2_ssg_v{}.pth".format(10000)))
net.eval().cuda()

correct = 0
total = 0
with torch.no_grad():
    for pts, lbs in model_val_loader:
        pts = pts.to("cuda", dtype=torch.float)
        lbs = lbs.to("cuda")
        output = net(pts)
        _, predicted = torch.max(output, dim=1)
        total = total + lbs.shape[0]
        correct = correct + (predicted == lbs).sum()
print("Accuracy: {:.4f}".format(correct / total))
