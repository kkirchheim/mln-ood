from torch.utils.data import Dataset
from os.path import join
from os import listdir
from PIL import Image
import numpy as np
import json


class RIVAL10(Dataset):
    attributes = [
        "long_snout",
        "wings",
        "wheels",
        "text",
        "horns",
        "floppy_ears",
        "ears",
        "colored_eyes",
        "tail",
        "mane",
        "beak",
        "hairy",
        "metallic",
        "rectangular",
        "wet",
        "long",
        "tall",
        "patterned",
        "label",
    ]

    attribute_index_map = {c: n for n, c in enumerate(attributes)}

    category_value_map = {
        **{a: [0, 1] for a in attributes[:-1]},
        "label": {
            "truck": 0,
            "car": 1,
            "plane": 2,
            "ship": 3,
            "cat": 4,
            "dog": 5,
            "equine": 6,
            "deer": 7,
            "frog": 8,
            "bird": 9,
        },
        #     ["truck",
        #     "car",
        #     "plane",
        #     "ship",
        #     "cat",
        #     "dog",
        #     "equine",
        #     "deer",
        #     "frog",
        #     "bird"])
        # ],
    }

    def __init__(
        self, root, train=True, transforms=None, target_transform=None, transform=None
    ):
        if train:
            self.root = join(root, "RIVAL10", "train", "ordinary")
        else:
            self.root = join(root, "RIVAL10", "test", "ordinary")

        # self.pickles = [s for s in listdir(self.root) if s.endswith(".pkl")]
        # self.pickles.sort()

        self.numpys = [s for s in listdir(self.root) if s.endswith(".npy")]
        self.numpys.sort()

        self.images = [
            s for s in listdir(self.root) if s.endswith(".JPEG") and "mask" not in s
        ]
        self.images.sort()

        self.wnids = [s.split("_")[0] for s in self.numpys]

        self.wnid_map = json.load(
            open(join(root, "RIVAL10", "meta", "wnid_to_class.json"), "r")
        )
        self.label_mapping = json.load(
            open(join(root, "RIVAL10", "meta", "label_mappings.json"), "r")
        )

        self.wnid_to_label_map = {}
        for wnid, name in self.wnid_map.items():
            if name in self.label_mapping:
                self.wnid_to_label_map[wnid] = self.label_mapping[name]

        self.classes = [self.wnid_to_label_map[i][1] for i in self.wnids]

        self.transform = transform
        self.transforms = transforms
        self.target_transform = target_transform

    def __len__(self):
        return len(self.numpys)

    def __getitem__(self, index):
        # pkl = self.pickles[index]
        img = self.images[index]
        npy = self.numpys[index]
        claz = self.classes[index]

        x = Image.open(join(self.root, img)).convert("RGB")
        npy = np.load(open(join(self.root, npy), "rb"))[
            0
        ]  # for some reason, the 2nd seems to be empty

        atts = npy.tolist()
        atts.append(claz)
        y = np.array(atts)

        if self.transforms:
            x = self.transforms(x)

        if self.target_transform:
            y = self.target_transform(y)

        if self.transform:
            x, y = self.transform(x, y)

        return x, y
