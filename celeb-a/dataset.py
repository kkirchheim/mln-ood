from torchvision.datasets import VisionDataset
from os.path import join
from os import listdir
import pandas as pd
from PIL import Image
import torch
import numpy as np


class CelebA(VisionDataset):
    def __init__(self, root, transforms=None, target_transform=None):
        super().__init__(root)
        self.d = join(root, "img_align_celeba", "img_align_celeba")
        # files = listdir(self.d)
        self.transforms = transforms
        self.target_transform = target_transform

        self.data = pd.read_csv(join(root, "list_attr_celeba.csv"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        v = self.data.iloc[item].values
        img = Image.open(join(self.d, v[0]))

        if self.transforms:
            img = self.transforms(img)

        y = torch.tensor(v[1:].astype(np.int32))
        y[y == -1] = 0

        if self.target_transform:
            y = self.target_transform(y)

        return img, y
