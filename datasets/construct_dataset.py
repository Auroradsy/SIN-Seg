from .cell import cell_original, cell_frequency, cell_ycbcr
from .msd import msd_original, msd_frequency, msd_ycbcr
from .chaos import chaos_original, chaos_frequency, chaos_ycbcr
from .brats import brats_original
from .covid import covid_original
from utils import brats_data_split, split_covid_dict

import torch
from torch.utils.data import DataLoader

import albumentations as albu
from albumentations.augmentations import transforms
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose, OneOf

from omegaconf import OmegaConf
from PIL import Image

def dateloader_construct(
        cfg: OmegaConf
):
    """
    Create dataset loaders for training, validation and testing.
    :return: train_loader, val_loader, test_loader
        if no specific test dataset, test_loader = val_loader
    """
    dataset = {
        "brats": {
            "original": brats_original,
            # "brats_frequency": brats_frequency,
            # "brats_ycbcr": brats_ycbcr,
        },
        "cell": {
            "original": cell_original,
            "frequency": cell_frequency,
            "ycbcr": cell_ycbcr,
        },
        "msd": {
            "original": msd_original,
            "frequency": msd_frequency,
            "ycbcr": msd_ycbcr,
        },
        "chaos": {
            "original": chaos_original,
            "frequency": chaos_frequency,
            "ycbcr": chaos_ycbcr,
        },
        "covid20": {
            "original": covid_original,
            # "frequency": covid_frequency,
        },
        "covid100": {
            "original": covid_original,
            # "frequency": covid_frequency,
        },
        "covid_mos_med": {
            "original": covid_original,
            # "frequency": covid_frequency,
        },
    }[cfg.dataset.name][cfg.dataset.type]
    print(f'Using dataset: {cfg.dataset.type} {cfg.dataset.name}')

    interpolation = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
    }[cfg.dataset.interpolation]

    # set the transforms
    if cfg.dataset.transform == 'None':
        train_transforms = Compose([
            albu.Resize(cfg.dataset.size, cfg.dataset.size, interpolation=interpolation),
            ToTensorV2(),
        ])
    if cfg.dataset.name == 'brats':
        train_data_split, val_data_split = brats_data_split()
        train_dataset = dataset(train_transforms, train_data_split)
        print(f'train_dataset_test: {train_dataset.__getitem__(0)}')
        val_dataset = dataset(train_transforms, val_data_split)
    elif cfg.dataset.name == 'cell':
        train_dataset = dataset(train_transforms, 'train')
        print(f'train_dataset_test: {train_dataset.__getitem__(0)}')
        val_dataset = dataset(train_transforms, 'val')
    elif 'covid' in cfg.dataset.name:
        splited_data_dict = split_covid_dict(cfg.dataset.name)
        train_dataset = dataset(train_transforms, splited_data_dict, state='train')
        print(f'train_dataset_test: {train_dataset.__getitem__(0)}')
        val_dataset = dataset(train_transforms, splited_data_dict, 'val')

    if cfg.dataset.name == 'cell':
        test_dataset = dataset(train_transforms, 'test')
    else:
        test_dataset = val_dataset

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    return dataloaders