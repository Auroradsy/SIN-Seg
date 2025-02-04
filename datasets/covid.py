import os
import numpy as np
from PIL import Image
import pickle

import torch
from torch.utils.data import Dataset


class covid_original(Dataset):
    def __init__(self, transform, data_dict, state='train'):
        self.data_dict = data_dict
        self.input_paths = self.data_dict[state]['input']
        self.label_paths = self.data_dict[state]['label']
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        if self.input_paths[idx].split('/')[-1] != self.label_paths[idx].split('/')[-1]:
            raise ValueError('input and label should have the same name!')

        input_path = self.input_paths[idx]
        label_path = self.label_paths[idx]
        subject_name = input_path.split('/')[-1].split('.')[0]

        image = np.load(input_path).astype(np.float32)  # shoule be (H, W, C) or (H, W)
        mask = np.load(label_path).astype(np.float32)  # should be (H, W)
        # if len(image.shape) == 2:
        #     image = np.expand_dims(image, axis=-1)
        #     image = np.repeat(image, 3, axis=-1)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # mask.unsqueeze_(0)

        return [image, mask, subject_name]
    