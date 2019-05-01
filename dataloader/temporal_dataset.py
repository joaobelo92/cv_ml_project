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
        img = img.convert('RGB')
        return img


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
        self.image_width = 224
        self.image_height = 224

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
        data = torch.empty(self.frames_temporal_flow * 2 * self.temporal_chunks, self.image_width,
                           self.image_height)
        offset = self.frames_temporal_flow // 2
        i = 0
        for f in frames:
            r = list(range(f - offset * self.time_between_frames, f, self.time_between_frames))
            r += list(range(f + 1, f + offset * self.time_between_frames + 1, self.time_between_frames))
            for image_number in r:
                image_u = pil_loader(os.path.join(self.root_dir, 'tvl1_flow', 'u', filename,
                                                  'frame{:06d}.jpg'.format(image_number)))
                image_v = pil_loader(os.path.join(self.root_dir, 'tvl1_flow', 'v', filename,
                                                  'frame{:06d}.jpg'.format(image_number)))
                if self.transform:
                    image_u = self.transform(image_u)
                    image_v = self.transform(image_v)
                data[i, :, :] = image_u[0, :, :]
                i += 1
                data[i, :, :] = image_v[0, :, :]
                i += 1
        return data


# t = transforms.Compose([
#     transforms.CenterCrop(224),
#     transforms.ToTensor()
# ])
# dataset = TemporalDataset('/Users/joaobelo/Datasets/ucf101/', 'trainlist01.csv', transform=t)
# i, l = dataset.__getitem__(5)
# print(i.size())
# for x in i[0:4]:
#     print(x)
#     i = (x.numpy() * 255).astype(np.uint8)
#
#     image = Image.fromarray(i, mode='L')
#     image.show()



