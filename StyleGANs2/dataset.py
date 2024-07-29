from torch.utils.data import Dataset
from torchvision import datasets


class CeleaHQ(Dataset):
    def __init__(self, path, transform):
        self.data = datasets.ImageFolder(path, transform)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class FFHQ(Dataset):
    def __init__(self, path, transform):
        self.data = datasets.ImageFolder(path, transform)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
