from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import torch
import numpy as np
import logging

log = logging.getLogger(__name__)


class CUB200(Dataset):
    """ """

    def __init__(
        self, root="./", transform=None, target_transform=None, attributes=None
    ):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv("cub-labels.csv")
        self.root_dir = root
        self.transform = transform
        self.target_transform = target_transform
        self.images = self.df.path

        self.atts = attributes

        df_values = self.df[self.atts].values.astype(np.float32)

        self.values = torch.tensor(df_values).long()

        for n, att in enumerate(attributes):
            log.info(f"{att}: {n} -> {len(self.values[:,n].unique())}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)

        attributes = self.values[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            attributes = self.target_transform(attributes)

        return image, attributes
