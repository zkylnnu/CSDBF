import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import random

def load_dataset(Dataset):

    if Dataset== "RV":
        data_b = sio.loadmat('datasets/river/river_before.mat')
        data_a = sio.loadmat('datasets/river/river_after.mat')
        data_before = data_b['river_before']
        data_after = data_a['river_after']
        gt_mat = sio.loadmat('datasets/river/river_gt.mat')
        gt = gt_mat['river_gt']
        dataset_name = "river"

    height, width, bands = data_before.shape
    if height < 500 :
        gt = 2 - gt

    print(dataset_name)
    print(height, width, bands)
    data_concat = np.concatenate((data_before, data_after), axis=-1)
    data_before = np.reshape(data_before, [height * width, bands])
    data_after = np.reshape(data_after, [height * width, bands])
    data_concat = np.reshape(data_concat, [height * width, 2 * bands])
    minMax = preprocessing.StandardScaler()
    data_before = minMax.fit_transform(data_before)
    data_after = minMax.fit_transform(data_after)
    data_concat = minMax.fit_transform(data_concat)
    data_before = np.reshape(data_before, [height, width, bands])
    data_after = np.reshape(data_after, [height, width, bands])
    data_concat = np.reshape(data_concat, [height, width, 2 * bands])

    return data_before, data_after, data_concat, gt, dataset_name

def sampling(sampling_mode, train_rate, gt):

    train_rand_idx = []
    gt_1d = np.reshape(gt, [-1])

    if sampling_mode == 'random':

        idx = np.where(gt_1d < 3)[-1]
        samplesCount = len(idx)
        rand_list = [i for i in range(samplesCount)]
        rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_rate).astype('int32'))
        rand_real_idx_per_class = idx[rand_idx]
        train_rand_idx.append(rand_real_idx_per_class)

        train_rand_idx = np.array(train_rand_idx)
        train_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_index.append(a[j])
        train_index = np.array(train_index)

        train_index = set(train_index)
        all_index = [i for i in range(len(gt_1d))]
        all_index = set(all_index)

        background_idx = np.where(gt_1d == 0)[-1]
        background_idx = set(background_idx)
        test_index = all_index - train_index - background_idx

        val_count = int(0.01 * (len(test_index) + len(train_index)))
        val_index = random.sample(test_index, val_count)
        val_index = set(val_index)
        test_index = test_index - val_index

        test_index = list(test_index)
        train_index = list(train_index)
        val_index = list(val_index)

    return train_index, val_index, test_index

def one_hot(gt_mask, height, width):
    gt_one_hot = []
    for i in range(gt_mask.shape[0]):
        for j in range(gt_mask.shape[1]):
            temp = np.zeros(2, dtype=np.float32)
            if gt_mask[i, j] != 0:
                temp[int(gt_mask[i, j]) - 1] = 1
            gt_one_hot.append(temp)
    gt_one_hot = np.reshape(gt_one_hot, [height, width, 2])

    return gt_one_hot

def make_mask(gt_mask, height, width):

    label_mask = np.zeros([height * width, 2])
    temp_ones = np.ones([2])
    gt_mask_1d = np.reshape(gt_mask, [height * width])
    for i in range(height * width):
        if gt_mask_1d[i] != 0:
            label_mask[i] = temp_ones
    label_mask = np.reshape(label_mask, [height * width, 2])
    return label_mask

def get_mask_onehot(gt, index):
    height, width = gt.shape
    gt_1d = np.reshape(gt, [-1])
    gt_mask = np.zeros_like(gt_1d)
    for i in range(len(index)):
        gt_mask[index[i]] = gt_1d[index[i]]
        pass
    gt_mask = np.reshape(gt_mask, [height, width])
    sampling_gt = gt_mask
    gt_onehot = one_hot(gt_mask, height, width)
    gt_mask = make_mask(gt_mask, height, width)

    return gt_onehot, gt_mask, sampling_gt
