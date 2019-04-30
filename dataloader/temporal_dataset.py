import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


class TemporalDataset(Dataset):
    """Temporal dataset"""

    def __init__(self, root_dir, cvs_file, transform=None, temporal_chunks=5,
                 time_between_frames=1, frames_temporal_flow=10):
        assert frames_temporal_flow % 2 == 0
        self.classes = pd.read_csv(root_dir + cvs_file)
        self.root_dir = root_dir
        self.transform = transform
        self.temporal_chunks = temporal_chunks
        self.time_between_frames = time_between_frames
        self.frames_temporal_flow = frames_temporal_flow

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, index):
        frames = []
        item = self.classes.iloc[index]
        offset = self.time_between_frames * self.frames_temporal_flow // 2
        assert offset * 2 + self.temporal_chunks <= item['frames']

        # Index difference between image we will pick to get the optical flow
        diff = (item['frames'] - (self.time_between_frames * self.frames_temporal_flow)) // self.temporal_chunks
        for i in range(self.temporal_chunks):
            # It's necessary to add 1 because images start with label 1
            frames.append(np.random.randint(offset + i * diff + 1, offset + (i + 1) * diff + 1))
        data = self.__load_optical_images(item['file'], frames)
        return data, item['label']-1

    def __load_optical_images(self, filename, frames):
        offset = self.frames_temporal_flow // 2
        # data = torch.
        for f in frames:
            r = list(range(f - offset * self.time_between_frames, f, self.time_between_frames))
            r += list(range(f + 1, f + offset * self.time_between_frames + 1, self.time_between_frames))
            for image_number in r:
                print(f, image_number)
                image = pil_loader(os.path.join(self.root_dir, 'tvl1_flow', 'u', filename,
                                                'frame{:06d}.jpg'.format(image_number)))
                if self.transform:
                    image = self.transform(image)
                data = image if i is 0 else torch.cat((data, image))
                print(data.size())
        return data


dataset = TemporalDataset('/home/joao/Datasets/ucf101/', 'trainlist01.csv', transform=transforms.ToTensor())
i, l = dataset.__getitem__(5)
print(i.size())
for x in i[0:10]:
    i = x.numpy()

    image = Image.fromarray(i, mode='L')
    image.show()

