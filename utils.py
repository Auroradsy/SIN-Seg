from omegaconf import OmegaConf
import pandas as pd
import pickle

import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import wandb

import torch
import sys
# sys.path.append(r'/home/siyuan/projects/Segmentation/SIN-Seg/datasets')
# import msd as md
# from ..datasets.msd import *


def flatten_configdict(
    cfg: OmegaConf,
    sep: str = ".",
):
    cfgdict = OmegaConf.to_container(cfg)
    cfgdict = pd.json_normalize(cfgdict, sep=sep)

    return cfgdict.to_dict(orient="records")[0]

def split_covid_dict(data_inst, split_ratio=0.8):
    splited_data_dict = {'train':
                            {'input': [],
                             'label': []},
                        'val':
                            {'input': [],
                             'label': []}
                        }
    data_split_root = '/data_new2/siyuan/covid19/processed_data/' + data_inst + '/' + data_inst + '_paths.pkl'
    
    with open(data_split_root, 'rb') as f:
        data_dict = pickle.load(f)
    data_lentgh = len(data_dict['input'])
    train_length = int(data_lentgh * split_ratio)
    train_index = np.random.choice(data_lentgh, train_length, replace=False)
    val_index = list(set(range(data_lentgh)) - set(train_index))
    for key, value in data_dict.items():
        splited_data_dict['train'][key] = [value[i] for i in train_index]
        splited_data_dict['val'][key] = [value[i] for i in val_index]

    return splited_data_dict

def save_visual(
    cfg: OmegaConf,
    model: torch.nn.Module,
    best_epoch: int,
    data_loader: torch.utils.data.DataLoader,
):
    model.load_state_dict(torch.load(os.path.join(cfg.exp_root, f"{cfg.model.name}_{cfg.dataset.name}_best_{best_epoch}.pth")))
    model.eval()
    pbar = tqdm(data_loader)
    for i, (inputs, labels, name) in enumerate(pbar):
        device = torch.device("cuda:{}".format(cfg.device.index[0]))
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(inputs)
        preds = preds.transpose(1, 3)
        preds = preds.transpose(1, 2)
        labels = labels.unsqueeze(-1)
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        if 'covid' in cfg.dataset.name:
            preds = preds[:, :, :, 0]
            labels = labels[:, :, :, 0]
        for j in range(preds.shape[0]):
            if cfg.dataset.name == 'brats':
                exp_file = cfg.exp_root.split('/')[-2:]
                exp_file = exp_file[0] + '/' + exp_file[1]
                new_root = os.path.join('/data2/siyuan/BraTS2021/exps', exp_file)
                pred_root = os.path.join(new_root, 'pred')
                mask_root = os.path.join(new_root, 'mask')
                pred_path = os.path.join(pred_root, f'{name[j][0]}_{name[j][1]}_pred.npy')
                mask_path = os.path.join(mask_root, f'{name[j][0]}_{name[j][1]}_mask.npy')
            elif cfg.dataset.name == 'cell':
                pred_root = os.path.join(cfg.exp_root, 'pred')
                mask_root = os.path.join(cfg.exp_root, 'mask')
                pred_path = os.path.join(pred_root, f'{name[j][0]}_{name[j][1]}_pred.npy')
                mask_path = os.path.join(mask_root, f'{name[j][0]}_{name[j][1]}_mask.npy')
            elif 'covid' in cfg.dataset.name:
                pred_root = os.path.join(cfg.exp_root, 'pred')
                mask_root = os.path.join(cfg.exp_root, 'mask')
                pred_path = os.path.join(pred_root, f'{name[j]}_pred.npy')
                mask_path = os.path.join(mask_root, f'{name[j]}_mask.npy')
            os.makedirs(pred_root, exist_ok=True)
            os.makedirs(mask_root, exist_ok=True)
            np.save(pred_path, preds[j])
            np.save(mask_path, labels[j])
            
            compare_img = np.concatenate([preds[j], labels[j]], axis=0)
            images = wandb.Image(
                compare_img,
                caption=f"Top: preds vs Bottom: labels",
            )
            wandb.log({f"preds vs labels_{name[j]}": images})

def brats_data_split():
    data_split_root = r'/data2/siyuan/BraTS2021/train_processed/brats_paths.pkl'
    data_sub_root = r'/data2/siyuan/BraTS2021/train_processed/brats_sub_name.pkl'
    # open the pickle file
    with open(data_split_root, 'rb') as f:
        data_dict = pickle.load(f)
    with open(data_sub_root, 'rb') as f:
        data_sub_dict = pickle.load(f)

    # use the first 20 subjects
    data_sub_dict['subject_name'] = data_sub_dict['subject_name'][:35]
    # randomly split the subjects into train and val
    train_sub = (np.random.choice(data_sub_dict['subject_name'], int(0.8*len(data_sub_dict['subject_name'])), replace=False)).tolist()
    train_sub.sort(key=lambda x: int(x[-4:])) 
    val_sub = list(set(data_sub_dict['subject_name']) - set(train_sub))
    val_sub.sort(key=lambda x: int(x[-4:]))

    # every element in the dict is splitted into train and val
    train_data_split = {}
    val_data_split = {}
    for key, value in data_dict.items():
        train_data_split[key] = []
        val_data_split[key] = []
        for sub_value in value:
            # if value in train_sub then append to train_data_split
            # else append to val_data_split
            cur_name = sub_value.split('/')[-2].split('_')[:2]
            cur_name = cur_name[0] + '_' + cur_name[1] 
            if cur_name in train_sub:
                train_data_split[key].append(sub_value)
            elif cur_name in val_sub:
                val_data_split[key].append(sub_value)
    return train_data_split, val_data_split

def dice_loss(pred, trg, smooth=1):
    pred = pred.contiguous()
    target = trg.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

if __name__ == "__main__":
    print(1)
    _s, _d = brats_data_split()