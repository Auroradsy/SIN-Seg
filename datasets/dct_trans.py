import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

def covid_spectrum_patch(src_root, patch_size=8):
    """
    Used to transform the covid19 dataset into dct domain
    :param src_root: the root of the processed 2-d data in [0,1]
    :param patch_size: the size of the patch, needs to be the element of weight and height
    """
    dct_root = os.path.join(src_root, 'dct_patch_{}'.format(patch_size))
    os.makedirs(dct_root, exist_ok=True)
    ori_img_root = os.path.join(src_root, 'imgs')
    ori_img_list = os.listdir(ori_img_root)
    for ori_img_name in tqdm(ori_img_list):
        new_name = 'dct_' + ori_img_name
        trg_path = os.path.join(dct_root, new_name)

        ori_img_path = os.path.join(ori_img_root, ori_img_name)
        ori_img = np.load(ori_img_path).astype(np.float32)
        dct_list = []
        dct = np.zeros_like(ori_img)
        dct_matrix = np.zeros(shape=(patch_size * patch_size, int(ori_img.shape[0] / patch_size), int(ori_img.shape[1] / patch_size)))
        for i in range(0, ori_img.shape[0], patch_size):
            for j in range(0, ori_img.shape[1], patch_size):
                dct[i:i + patch_size, j:j + patch_size] = cv2.dct(ori_img[i:i + patch_size, j:j + patch_size])
                dct_matrix[:, i // patch_size, j // patch_size] = dct[i:i + patch_size, j:j + patch_size].flatten()
        dct_list.append(dct)
        dct_image = dct_matrix

        # normalize the dct image to [0,1] for frequnecy by frequency
        for i in range(dct_image.shape[0]):
            dct_image[i] = (dct_image[i] - np.min(dct_image[i])) / (np.max(dct_image[i]) - np.min(dct_image[i]))

        np.save(trg_path, dct_image)
    return None

def brats_spectrum(trg_root):
    dct_root = os.path.join(trg_root, 'dct')

    data_split_root = r'/data2/siyuan/BraTS2021/train_processed/brats_paths.pkl'
    data_sub_root = r'/data2/siyuan/BraTS2021/train_processed/brats_sub_name.pkl'
    # open the pickle file
    with open(data_split_root, 'rb') as f:
        data_dict = pickle.load(f)
    with open(data_sub_root, 'rb') as f:
        data_sub_dict = pickle.load(f)

    dct_mat = {'t1': [],
               't1ce': [],
               't2': [],
               'flair': [],
               }
    for keys, values in data_dict.items():
        if keys == 'seg':
            continue
        pbar = tqdm(values)
        for i, sub_value in enumerate(pbar):
            ori_name = sub_value.split('/')[-2:]
            trg_path = os.path.join(dct_root, keys, ori_name[0])
            os.makedirs(trg_path, exist_ok=True)

            ori_img = np.load(sub_value)
            ori_img = cv2.resize(ori_img, (128, 128), interpolation=cv2.INTER_NEAREST)
            ori_img = np.float32(ori_img)
            
            dct_list = []
            dct = np.zeros_like(ori_img)
            dct_matrix = np.zeros(shape=(8 * 8, int(ori_img.shape[0] / 8), int(ori_img.shape[1] / 8)))
            for i in range(0, ori_img.shape[0], 8):
                for j in range(0, ori_img.shape[1], 8):
                    dct[i:i + 8, j:j + 8] = cv2.dct(ori_img[i:i + 8, j:j + 8])
                    dct_matrix[:, i // 8, j // 8] = dct[i:i + 8, j:j + 8].flatten()
            dct_list.append(dct)
            dct_image = dct_matrix
            dct_image = cv2.normalize(dct_image, None, 0, 1, norm_type=cv2.NORM_MINMAX)

            new_path = os.path.join(trg_path, 'dct_' + ori_name[1])
            dct_mat[keys].append(new_path)
            np.save(new_path, dct_image)

    with open(os.path.join(trg_root, 'brats_dct_paths.pkl'), 'wb') as f:
        pickle.dump(dct_mat, f)

if __name__ == '__main__':
    # trg_root = r'/data2/siyuan/BraTS2021/train_processed'
    # brats_spectrum(trg_root)
    covid_20_root = r'/data2/siyuan/covid19/processed_data/covid20'
    covid_100_root = r'/data2/siyuan/covid19/processed_data/covid100'
    covid_mosmed_root = r'/data2/siyuan/covid19/processed_data/covid_mos_med'
    covid_spectrum_patch(covid_20_root, patch_size=8)
    covid_spectrum_patch(covid_100_root, patch_size=8)
    covid_spectrum_patch(covid_mosmed_root, patch_size=8)
