import os
import numpy as np
from PIL import Image
import pickle

import torch
from torch.utils.data import Dataset


class brats_original(Dataset):
    def __init__(self, transform=None, data_dict=None, state='t2'):
        # self.data_split_root = r'/data2/siyuan/BraTS2021/train_processed/brats_ori.pkl'
        # # open the pickle file
        # with open(self.data_split_root, 'rb') as f:
        #     self.data_dict = pickle.load(f)
        self.input_paths = data_dict[state]
        self.label_paths = data_dict['seg']
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_path = self.input_paths[idx]
        label_path = self.label_paths[idx]

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
        
        if image.shape[-2:] != mask.shape[-2:]:
            print(f'image shape: {image.shape}, mask shape: {mask.shape}')
            raise ValueError('image and mask should have the same size!')

        return [image, mask, self.input_paths[idx].split('/')[-1].split('.')[0]]
        # return [image, mask]


class brats_frequency(Dataset):
    def __init__(self, path_imgs, path_labels, list_imgs, list_labels, transform1=None, transform2=None):
        self.path_imgs = path_imgs
        self.path_labels = path_labels
        self.imgs = list_imgs
        self.imgs.sort(key=lambda x: int(x[7:10]))  # for cell segmentation
        # self.imgs.sort(key=lambda x: int(x[5:11]))  
        self.labels = list_labels
        self.labels.sort(key=lambda x: int(x[7:10]))
        # self.labels.sort(key=lambda x: int(x[5:11]))
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_imgs, self.imgs[idx])
        label_path = os.path.join(self.path_labels, self.labels[idx])

        image = np.load(img_path).astype(np.float32)
        image = torch.from_numpy(image)
        mask = np.load(label_path)
        mask = (255 * mask).astype(np.uint8)
        mask = Image.fromarray(mask)

        # if self.transform1 is not None:
        #     image = self.transform1(image)
        if self.transform2 is not None:
            mask = self.transform2(mask)

        return [image, mask, self.imgs[idx].split('.')[0]]


class brats_ycbcr(Dataset):
    def __init__(self, path_imgs, path_labels, list_imgs, list_labels, transform1=None, transform2=None):
        self.path_imgs = path_imgs
        self.path_labels = path_labels
        self.imgs = list_imgs
        self.imgs.sort(key=lambda x: int(x[7:10]))
        self.labels = list_labels
        self.labels.sort(key=lambda x: int(x[7:10]))
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_imgs, self.imgs[idx])
        label_path = os.path.join(self.path_labels, self.labels[idx])

        image = Image.open(img_path)
        mask = np.load(label_path)
        mask = (255 * mask).astype(np.uint8)
        mask = Image.fromarray(mask)

        if self.transform1 is not None:
            image = self.transform1(image)
        if self.transform2 is not None:
            mask = self.transform2(mask)

        return [image, mask, self.imgs[idx].split('.')[0]]

