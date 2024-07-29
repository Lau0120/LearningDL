import os
import numpy as np
import torch

from torch.utils.data import Dataset

from utils import load_point_cloud


class PointCloudBase(Dataset):
    def __init__(self, path, is_train=True):
        super(PointCloudBase, self).__init__()
        self.fps = []
        self.lbs = []
        class_index = 0
        for dir in os.listdir(path):
            if os.path.isfile(os.path.join(path, dir)):
                continue
            target_path = os.path.join(path, dir, "train" if is_train else "test")
            files = os.listdir(target_path)
            self.lbs = self.lbs + [class_index] * len(files)
            for file in files:
                self.fps.append(os.path.join(target_path, file))
            class_index = class_index + 1
 
    def __getitem__(self, index):
        _, pts = load_point_cloud(self.fps[index])
        pts = torch.from_numpy(pts)
        return pts.transpose(0, 1), torch.tensor(self.lbs[index])

    def __len__(self):
        return len(self.fps)


class ModelNet(PointCloudBase):
    def __init__(self, path, is_train=True):
        super(ModelNet, self).__init__(path, is_train)


class ShapeNet(PointCloudBase):
    def __init__(self, path, is_train=True, n_points=2500):
        super(ShapeNet, self).__init__(path, is_train)
        self.n_points = n_points

    def __getitem__(self, index):
        _, pts = load_point_cloud(self.fps[index])
        indices = np.random.choice(len(pts[0]), self.n_points, replace=True)
        pts = torch.from_numpy(pts[indices])
        return pts, torch.tensor(self.lbs[index])
