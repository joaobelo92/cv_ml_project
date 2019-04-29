import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from PIL import Image
import random


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class SpatialDataset(Dataset):
    """Spatial Dataset"""

    def __init__(self, root_dir, cvs_file, transform=None, t=5):
        self.classes = pd.read_csv(root_dir + cvs_file)
        self.root_dir = root_dir
        self.transform = transform
        self.t = t

    def __len__(self):
        return len(self.classes)

    # Initially we want to be able to train each stream separately, so we don't synchronize
    # the spatial dataloader with the temporal. This should be considered as future work.
    def __getitem__(self, index):
        # Get t random images from the video
        frames = []
        item = self.classes.iloc[index]
        diff = item[2] // self.t
        for i in range(self.t):
            frames.append(random.randint(i * diff, (i+1) * diff))
        data = self.__load_images(item[0], frames)
        return data, item[1]

    def __load_images(self, filename, frames):
        data = []
        for f in frames:
            path = os.path.join(self.root_dir, 'jpegs_256', filename,
                                'frame{:06d}.jpg'.format(f))
            image = pil_loader(path)
            if self.transform:
                image = self.transform(image)
            data.append(image)
        return data

# dataset = SpatialDataset('/Users/joaobelo/Datasets/ucf-101/', 'trainlist01.csv')
# i, l = dataset.__getitem__(5)
# for x in i:
#     x.show(x)
