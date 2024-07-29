import torch.nn.functional as F
import torch

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


config = {
    "path": "../Datasets/CIFAR10",
    "weight_decay": 0.0001,
    "momentum": 0.9,
    "learning_rate": 0.01,
    "batch_size": 128,
    "iterations_count": 32000,
}


trn_set = datasets.CIFAR10(
    root=config.get("path"),
    train=True,
    transform=transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32)),
        transforms.ToTensor(),
    ]),
)


val_set = datasets.CIFAR10(
    root=config.get("path"),
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)


class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()

        # input
        self.input = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.ipool = nn.MaxPool2d(kernel_size=2, stride=2)

        # block1
        self.block1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.block1_btnm1 = nn.BatchNorm2d(num_features=16)
        self.block1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.block1_btnm2 = nn.BatchNorm2d(num_features=16)
        self.block1_conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.block1_btnm3 = nn.BatchNorm2d(num_features=16)
        self.block1_conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.block1_btnm4 = nn.BatchNorm2d(num_features=16)
        self.block1_conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.block1_btnm5 = nn.BatchNorm2d(num_features=16)
        self.block1_conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.block1_btnm6 = nn.BatchNorm2d(num_features=16)
        self.block1_pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block2
        self.block2_conv0 = nn.Conv2d(16, 32, kernel_size=1)
        self.block2_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.block2_btnm1 = nn.BatchNorm2d(num_features=32)
        self.block2_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.block2_btnm2 = nn.BatchNorm2d(num_features=32)
        self.block2_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.block2_btnm3 = nn.BatchNorm2d(num_features=32)
        self.block2_conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.block2_btnm4 = nn.BatchNorm2d(num_features=32)
        self.block2_conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.block2_btnm5 = nn.BatchNorm2d(num_features=32)
        self.block2_conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.block2_btnm6 = nn.BatchNorm2d(num_features=32)
        self.block2_pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        # block3
        self.block3_conv0 = nn.Conv2d(32, 64, kernel_size=1)
        self.block3_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.block3_btnm1 = nn.BatchNorm2d(num_features=64)
        self.block3_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block3_btnm2 = nn.BatchNorm2d(num_features=64)
        self.block3_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block3_btnm3 = nn.BatchNorm2d(num_features=64)
        self.block3_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block3_btnm4 = nn.BatchNorm2d(num_features=64)
        self.block3_conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block3_btnm5 = nn.BatchNorm2d(num_features=64)
        self.block3_conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block3_btnm6 = nn.BatchNorm2d(num_features=64)
        self.block3_pool7 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 10-way fully-connected layer
        self.output = nn.Linear(64 * 2 * 2, 10)

    def forward(self, x):
        # input
        out = self.ipool(F.relu(self.input(x)))

        # block1
        out = F.relu(self.block1_btnm2(self.block1_conv2(F.relu(self.block1_btnm1(self.block1_conv1(out)))) + out))
        out = F.relu(self.block1_btnm4(self.block1_conv4(F.relu(self.block1_btnm3(self.block1_conv3(out)))) + out))
        out = F.relu(self.block1_btnm6(self.block1_conv6(F.relu(self.block1_btnm5(self.block1_conv5(out)))) + out))
        out = self.block1_pool7(out)

        # block2
        out = F.relu(self.block2_btnm2(self.block2_conv2(F.relu(self.block2_btnm1(self.block2_conv1(out)))) + self.block2_conv0(out)))
        out = F.relu(self.block2_btnm4(self.block2_conv4(F.relu(self.block2_btnm3(self.block2_conv3(out)))) + out))
        out = F.relu(self.block2_btnm6(self.block2_conv6(F.relu(self.block2_btnm5(self.block2_conv5(out)))) + out))
        out = self.block2_pool7(out)

        # block3
        out = F.relu(self.block3_btnm2(self.block3_conv2(F.relu(self.block3_btnm1(self.block3_conv1(out)))) + self.block3_conv0(out)))
        out = F.relu(self.block3_btnm4(self.block3_conv4(F.relu(self.block3_btnm3(self.block3_conv3(out)))) + out))
        out = F.relu(self.block3_btnm6(self.block3_conv6(F.relu(self.block3_btnm5(self.block3_conv5(out)))) + out))
        out = self.block3_pool7(out)

        # output
        return self.output(out.view(out.shape[0], -1))


resnet = ResNet20()


def training_loop(n_iterations, optimizer, model, loss_fn, loader):
    for i in range(n_iterations):
        imgs, labels = next(iter(loader))
        imgs, labels = imgs.to(device=dev), labels.to(device=dev)

        # forward
        output = model(imgs)
        loss = loss_fn(output, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 2000 == 0:
            print("Iterations Count {},\t Training Loss {:.2f}".format(i, loss))

def validation(model, trn_loader, val_loader):
    for name, loader in [("trn", trn_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device=dev), labels.to(device=dev)
                output = model(imgs)
                _, predicted = torch.max(output, dim=1)
                total += labels.shape[0]
                correct += (predicted == labels).sum()

        print("Accuracy {}: {:.2f}".format(name, correct / total))

resnet.to(dev)
training_loop(
    n_iterations=config.get("iterations_count"),
    optimizer=optim.SGD(
        params=resnet.parameters(),
        weight_decay=config.get("weight_decay"),
        momentum=config.get("momentum"),
        lr=config.get("learning_rate"),
    ),
    model=resnet,
    loss_fn=nn.CrossEntropyLoss(),
    loader=DataLoader(
        dataset=trn_set,
        batch_size=config.get("batch_size"),
        shuffle=True,
    ),
)

torch.save(resnet.state_dict(), "res20_params_64k.pth")

validation(
    model=resnet,
    trn_loader=DataLoader(
        dataset=trn_set,
        batch_size=config.get("batch_size"),
        shuffle=True,
    ),
    val_loader=DataLoader(
        dataset=val_set,
        batch_size=config.get("batch_size"),
        shuffle=True,
    ),
)
