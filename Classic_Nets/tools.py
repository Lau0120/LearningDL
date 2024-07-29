import torch
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


conf = {
    "TRN_SET_PATH": "../Datasets/ImageNet2012/train",
    "VAL_SET_PATH": "../Datasets/ImageNet2012/val"
}


def init_weights(model, mean, std):
    if isinstance(model, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(model.weight, mean, std)


def init_bias(model, val):
    if isinstance(model, (nn.Conv2d, nn.Linear)):
        nn.init.constant_(model.bias, val)


class ImageNet(Dataset):
    def __init__(self, transform, is_train=True):
        super().__init__()
        if is_train:
            self.set = ImageFolder(conf["TRN_SET_PATH"], transform)
        else:
            self.set = ImageFolder(conf["VAL_SET_PATH"], transform)

    def __getitem__(self, index):
        return self.set[index]

    def __len__(self):
        return len(self.set)


def training_loop(n_epochs, optimizer, model, loss_fn, loader, dev):
    # train mode
    model.train()

    for epoch in range(1, n_epochs + 1):
        trn_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device=dev), labels.to(device=dev)

            # forward
            output = model(imgs)
            loss = loss_fn(output, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trn_loss += loss.item()

        print("Epoch {}, Training Loss {}".format(epoch, trn_loss / len(loader)))


def evaluate(model, loader, dev):
    # eval mode
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device=dev), labels.to(device=dev)

            output = model(imgs)
            # top1
            _, predicted_top1 = torch.max(output, dim=1)
            correct_top1 += (predicted_top1 == labels).sum()
            # top5
            _, predicted_top5 = torch.topk(output, k=5, dim=1)
            correct_top5 += (predicted_top5 == labels).sum()
            # total
            total += labels.shape[0]

    print("top-1 error rate: {:.2f}".format(correct_top1 / total))
    print("top-5 error rate: {:.2f}".format(correct_top5 / total))


def validation(model, trn_loader, val_loader, dev):
    # eval mode
    model.eval()

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
