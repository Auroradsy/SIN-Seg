import os
from tqdm import tqdm
import numpy as np
import skimage.io as sio
import pickle
import warnings
import cv2

import hydra
from omegaconf import OmegaConf

def find_min_max(input_volume):
    min_val = 100000
    max_val = -100000
    for i in range(input_volume.shape[-1]):
        input_slice = input_volume[:, :, i]
        min_val = min(min_val, input_slice.min())
        max_val = max(max_val, input_slice.max())

    return min_val, max_val

def get_covid_list(root_path, data_inst, pkl_state = False):
    """
        Get the list of the normalized .npy formate covid19 dataset
        :param root_path: the root of the covid19 dataset
        :param data_inst: the collected institution of the covid19 dataset
        :return: the list of the covid19 dataset
    """
    img_root = os.path.join(root_path, 'imgs')
    img_names = os.listdir(img_root)
    if data_inst == 'covid20':
        label_root = os.path.join(root_path, 'masks')
        label_names = os.listdir(label_root)
        img_names.sort(key=lambda x: (x.split('_')[0], int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0])))
        label_names.sort(key=lambda x: (x.split('_')[0], int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0])))
        covid_img_list = [os.path.join(img_root, img_name) for img_name in img_names]
        covid_label_list = [os.path.join(label_root, label_name) for label_name in label_names]

        if pkl_state:
            pkl_name = data_inst + '_paths.pkl'
            with open(os.path.join(root_path, pkl_name), 'wb') as f:
                pickle.dump({'input': covid_img_list, 'label': covid_label_list}, f)
        return covid_img_list, covid_label_list
    
    if data_inst == 'covid_mos_med':
        label_root = os.path.join(root_path, 'masks')
        label_names = os.listdir(label_root)
        img_names.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0])))
        label_names.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0])))
        covid_img_list = [os.path.join(img_root, img_name) for img_name in img_names]
        covid_label_list = [os.path.join(label_root, label_name) for label_name in label_names]

        if pkl_state:
            pkl_name = data_inst + '_paths.pkl'
            with open(os.path.join(root_path, pkl_name), 'wb') as f:
                pickle.dump({'input': covid_img_list, 'label': covid_label_list}, f)
        return covid_img_list, covid_label_list
    
    elif data_inst == 'covid100':
        label_root = os.path.join(root_path, 'masks')
        label_names = os.listdir(label_root)
        label3_root = os.path.join(root_path, 'masks3')
        label3_names = os.listdir(label3_root)
        img_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        label_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        label3_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        covid_img_list = [os.path.join(img_root, img_name) for img_name in img_names]
        covid_label1_list = [os.path.join(label_root, label1_name) for label1_name in label_names]
        covid_label3_list = [os.path.join(label3_root, label3_name) for label3_name in label3_names]

        if pkl_state:
            pkl_name = data_inst + '_paths.pkl'
            with open(os.path.join(root_path, pkl_name), 'wb') as f:
                pickle.dump({'input': covid_img_list, 'label': covid_label1_list, 'label3': covid_label3_list}, f)
        return covid_img_list, covid_label1_list, covid_label3_list



@hydra.main(config_path="../cfg", config_name="config.yaml")
def main(
    cfg: OmegaConf):
    if 'covid' in cfg.dataset.name:
        covid_root = os.path.join(r'/data_new2/siyuan/covid19/processed_data/', cfg.dataset.name)
        if cfg.dataset.name == 'covid100':
            covid100_input_list, covid100_label1_list, covid100_label3_list = get_covid_list(covid_root, cfg.dataset.name, True)
        else:
            covid_input_list, covid_label_list = get_covid_list(covid_root, cfg.dataset.name, True)

if __name__ == '__main__':
    main()
