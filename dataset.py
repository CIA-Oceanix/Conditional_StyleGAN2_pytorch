import os
import glob

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

from PIL import Image

from config import EXTS


def cycle(iterable):
    """
    Transform an iterable into a generator.
    """
    while True:
        for i in iterable:
            yield i


class Dataset(data.Dataset):
    """
    The Dataset object is used to read files from a given folder and generate both the labels and the tensor.
    """

    def __init__(self, folder, image_size):
        """
        Initialize the Dataset.

        :param folder: the path to the folder containing either pictures or subfolder with pictures.
        :type folder: str
        :param image_size: the size of the tensor to output.
        :type image: int
        """
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        self.labels = [subfolder for subfolder in os.listdir(folder) if os.path.isdir(os.path.join(folder, subfolder))]
        if not self.labels:
            self.labels = '.'
        self.label_number = len(self.labels)

        self.path_keys = [[p for ext in EXTS for p in glob.glob(os.path.join(folder, label, f'*.{ext}'))]
                          for i, label in enumerate(self.labels)]
        self.length = sum([len(path_keys) for path_keys in self.path_keys])
        assert self.length, f"Didn't find any picture inside {folder}"

        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label_index = index % self.label_number  # we select one label after another
        path_keys = self.path_keys[label_index]
        index = (index // self.label_number) % (len(path_keys))
        # print('dataset.Dataset.__getitem__ l63:',
        #       f"The index of the picture is {index} for label {self.labels[label_index]}")

        with Image.open(path_keys[index]) as image_file:
            img = self.transform(image_file)

        label = torch.from_numpy(np.eye(self.label_number)[label_index]).cuda().float()
        return img, label
