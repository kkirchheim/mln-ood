"""
GTSRB dataset as downloaded from kaggle, with an additional labels.txt
"""

from torch.utils.data import Dataset
from os.path import join
import pandas as pd
from PIL import Image
import numpy as np

attribute_index_map = {"class": 0, "color": 1, "shape": 2}

category_value_map = {
    "color": {"red": 0, "blue": 1, "yellow": 2, "white": 3},
    "class": {
        s.replace(" ", "_"): i
        for i, s in enumerate(
            [
                "speed limit 20",
                "speed limit 30",
                "speed limit 50",
                "speed limit 60",
                "speed limit 70",
                "speed limit 80",
                "restriction ends 80",
                "speed limit 100",
                "speed limit 120",
                "no overtaking",
                "no overtaking trucks",
                "priority at next intersection",
                "priority road",
                "give way",
                "stop",
                "no traffic both ways",
                "no trucks",
                "no entry",
                "danger",
                "bend left",
                "bend right",
                "bend",
                "uneven road",
                "slippery road",
                "road narrows",
                "construction",
                "traffic signal",
                "pedestrian crossing",
                "school crossing",
                "cycles crossing",
                "snow",
                "animals",
                "restriction ends",
                "go right",
                "go left",
                "go straight",
                "go right or straight",
                "go left or straight",
                "keep right",
                "keep left",
                "roundabout",
                "restriction ends overtaking",
                "restriction ends overtaking trucks",
            ]
        )
    },
    "shape": {
        s.replace(" ", "_"): i
        for i, s in enumerate(
            [
                "triangle",
                "circle",
                "square",
                "octagon",
                "inverse triangle",
            ]
        )
    },
}


class GTSRB(Dataset):
    """ """

    shape_to_name = {
        0: "triangle",
        1: "circle",
        2: "square",
        3: "octagon",
        4: "inverse triangle",
    }

    color_to_name = {
        0: "red",
        1: "blue",
        2: "yellow",
        3: "white",
    }

    class_to_name = {
        k: v
        for k, v in enumerate(
            [
                "speed limit 20",
                "speed limit 30",
                "speed limit 50",
                "speed limit 60",
                "speed limit 70",
                "speed limit 80",
                "restriction ends 80",
                "speed limit 100",
                "speed limit 120",
                "no overtaking",
                "no overtaking trucks",
                "priority at next intersection",
                "priority road",
                "give way",
                "stop",
                "no traffic both ways",
                "no trucks",
                "no entry",
                "danger",
                "bend left",
                "bend right",
                "bend",
                "uneven road",
                "slippery road",
                "road narrows",
                "construction",
                "traffic signal",
                "pedestrian crossing",
                "school crossing",
                "cycles crossing",
                "snow",
                "animals",
                "restriction ends",
                "go right",
                "go left",
                "go straight",
                "go right or straight",
                "go left or straight",
                "keep right",
                "keep left",
                "roundabout",
                "restriction ends overtaking",
                "restriction ends overtaking trucks",
            ]
        )
    }

    def __init__(
        self, root, train=True, transforms=None, target_transform=None, transform=None
    ):
        self.root = join(root, "GTSRB")
        self.meta_csv = pd.read_csv(join(self.root, "Meta.csv"))
        self.class_to_color = {}
        self.class_to_shape = {}
        for idx, (clazz, shape, color) in self.meta_csv[
            ["ClassId", "ShapeId", "ColorId"]
        ].iterrows():
            self.class_to_color[clazz] = color
            self.class_to_shape[clazz] = shape

        self.data = pd.read_csv(join(self.root, f"{'Train' if train else 'Test'}.csv"))
        self.paths = list(self.data["Path"])
        self.labels = list(self.data["ClassId"])
        self.transform = transform
        self.transforms = transforms
        self.target_transform = target_transform

        with open(join(self.root, "labels.txt"), "r") as f:
            label_lines = [l.strip().replace("'", "") for l in f.readlines()]

        self.class_to_name = {n: name for n, name in enumerate(label_lines, 0)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        color = self.class_to_color[label]
        shape = self.class_to_shape[label]
        path = join(self.root, self.paths[index])

        y = np.array([label, color, shape])
        x = Image.open(path).convert("RGB")

        if self.transforms:
            x = self.transforms(x)

        if self.target_transform:
            y = self.target_transform(y)

        if self.transform:
            x, y = self.transform(x, y)

        return x, y
