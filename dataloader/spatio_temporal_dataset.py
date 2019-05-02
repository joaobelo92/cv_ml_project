import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return img


class SpatioTemporalDataset(Dataset):
    def __init__(self, root_dir, cvs_file, temporal_chunks=5,
                 time_between_frames=1, frames_temporal_flow=10):
        assert frames_temporal_flow % 2 == 0
        self.classes = pd.read_csv(root_dir + cvs_file)
        self.root_dir = root_dir
        self.temporal_chunks = temporal_chunks
        self.time_between_frames = time_between_frames
        self.frames_temporal_flow = frames_temporal_flow
        self.image_dimensions = (224, 224)

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, index):
        frames = []
        item = self.classes.iloc[index]
        offset = self.time_between_frames * self.frames_temporal_flow // 2
        assert offset * 2 + self.temporal_chunks <= item['frames'] - 1

        # Index difference between image we will pick to get the optical flow
        diff = (item['frames'] - 1 - (self.time_between_frames * self.frames_temporal_flow)) // self.temporal_chunks
        for i in range(self.temporal_chunks):
            # It's necessary to add 1 because images start with label 1
            frames.append(np.random.randint(offset + i * diff + 1, offset + (i + 1) * diff + 1))

        # this is where we decide the same transformation to all the pictures
        # load one image to know its dimensions
        sample = pil_loader(os.path.join(self.root_dir, 'jpegs_256', item['file'],
                            'frame000001.jpg'))
        crop_params = transforms.RandomCrop.get_params(sample, output_size=self.image_dimensions)
        horizontal_flip = np.random.random() > 0.5

        temporal_data = self.__load_temporal_images(item['file'], frames, crop_params, horizontal_flip)
        spatial_data = self.__load_spatial_images(item['file'], frames, crop_params, horizontal_flip)
        return spatial_data, temporal_data, item['label']-1

    @staticmethod
    def __transform_images(image, crop_params, horizontal_flip, normalize_image=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = TF.crop(image, *crop_params)
        if horizontal_flip:
            image = TF.hflip(image)
        image = transforms.ToTensor()(image)
        if normalize_image:
            image = normalize(image)
        return image

    def __load_temporal_images(self, filename, frames, crop_params, horizontal_flip):
        data = torch.empty(self.frames_temporal_flow * 2 * self.temporal_chunks, *self.image_dimensions)
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
                image_u = self.__transform_images(image_u, crop_params, horizontal_flip)
                image_v = self.__transform_images(image_v, crop_params, horizontal_flip)
                data[i, :, :] = image_u[0, :, :]
                i += 1
                data[i, :, :] = image_v[0, :, :]
                i += 1
        return data

    def __load_spatial_images(self, filename, frames, crop_params, horizontal_flip):
        for i, f in enumerate(frames):
            path = os.path.join(self.root_dir, 'jpegs_256', filename,
                                'frame{:06d}.jpg'.format(f))
            image = pil_loader(path)
            image = self.__transform_images(image, crop_params, horizontal_flip, True)
            data = image if i is 0 else torch.cat((data, image))
        return data


# dataset = SpatioTemporalDataset('/home/joao/Datasets/ucf101/', 'trainlist01.csv')
# i, f, l = dataset.__getitem__(5)
# print(i.size(), f.size())
# print(i[0:3].numpy())
# image = (i[0:3].numpy().transpose(1, 2, 0))
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# image = std * image + mean
# image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
# image = Image.fromarray(image, mode='RGB')
# image.show()
#
# for x in f[0:10]:
#     print(x)
#     i = (x.numpy() * 255).astype(np.uint8)
#
#     image = Image.fromarray(i, mode='L')
#     image.show()