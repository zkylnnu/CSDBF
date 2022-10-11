import numpy as np
import time
import torch
from superpixel_seg import seg_module, get_temporal_adj
import model
import train
import test
from data_pre import load_dataset, sampling, get_mask_onehot
from generate_pic import generate_png

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global Dataset

Dataset = "RV"  # RV, FM, USA, YC, SANTA, BAY

sampling_mode = "random"
scale = 30
ITER, lr, num_epochs, sampling_rate = 4, 1e-4, 1000, 0.0336

data_before, data_after, data_concat, gt, dataset_name = load_dataset(Dataset)

superpixel_num = data_before.shape[0] * data_before.shape[1] / scale
height, width, bands = data_before.shape
OA_record = []
kappa_record = []
alltime_record = []


for index_iter in range(ITER):
    datapre_time_before = time.time()
    train_index, val_index, test_index = sampling(sampling_mode, sampling_rate, gt)
    print("-----load data successfully-----")
    train_onehot, train_mask, train_gt = get_mask_onehot(gt, train_index)
    val_onehot, val_mask, val_gt = get_mask_onehot(gt, val_index)
    test_onehot, test_mask, test_gt = get_mask_onehot(gt, test_index)
    datapre_time_after = time.time()
    data_pre_time = datapre_time_after - datapre_time_before

    SS_time_before = time.time()
    #Q1, S1, A1, seg1, supernum1 = seg_module(data_before, train_gt, superpixel_num)
    # Q2, S2, A2, seg2, supernum2 = seg_module(data_after, train_gt, superpixel_num)
    Q3, S3, A3, seg3, supernum3 = seg_module(data_concat, train_gt, superpixel_num)
    # Reduce the supernum1 and supernum2 there when the sampling rate is high
    Q1, S1, A1, seg1, supernum1 = seg_module(data_before, train_gt, 0.2* superpixel_num)
    Q2, S2, A2, seg2, supernum2 = seg_module(data_after, train_gt, 0.2* superpixel_num)
    A12, A21 = get_temporal_adj(S1, S2)
    A13, A31 = get_temporal_adj(S1, S3)
    A23, A32 = get_temporal_adj(S2, S3)
    SS_time_after = time.time()
    SS_time = SS_time_after - SS_time_before
    print("SS time: {}".format(SS_time))
    print("-----superpixel segmentation ok-----")

    transfer_time_before = time.time()
    Q1 = torch.from_numpy(Q1).to(device)
    Q2 = torch.from_numpy(Q2).to(device)
    Q3 = torch.from_numpy(Q3).to(device)
    A1 = torch.from_numpy(A1).to(device)
    A2 = torch.from_numpy(A2).to(device)
    A3 = torch.from_numpy(A3).to(device)
    A12 = torch.from_numpy(A12).to(device)
    A13 = torch.from_numpy(A13).to(device)
    A23 = torch.from_numpy(A23).to(device)
    A21 = torch.from_numpy(A21).to(device)
    A31 = torch.from_numpy(A31).to(device)
    A32 = torch.from_numpy(A32).to(device)
    train_mask = torch.from_numpy(train_mask.astype(np.float32)).to(device)
    val_mask = torch.from_numpy(val_mask.astype(np.float32)).to(device)
    test_mask = torch.from_numpy(test_mask.astype(np.float32)).to(device)
    train_onehot = torch.from_numpy(train_onehot.astype(np.float32)).to(device)
    val_onehot = torch.from_numpy(val_onehot.astype(np.float32)).to(device)
    test_onehot = torch.from_numpy(test_onehot.astype(np.float32)).to(device)
    train_gt = torch.from_numpy(train_gt.astype(np.float32)).to(device)
    val_gt = torch.from_numpy(val_gt.astype(np.float32)).to(device)
    test_gt = torch.from_numpy(test_gt.astype(np.float32)).to(device)
    net_input_before = np.array(data_before, np.float32)
    net_input_after = np.array(data_after, np.float32)
    net_input_concat = np.array(data_concat, np.float32)
    net_input_before = torch.from_numpy(net_input_before.astype(np.float32)).to(device)
    net_input_after = torch.from_numpy(net_input_after.astype(np.float32)).to(device)
    net_input_concat = torch.from_numpy(net_input_concat.astype(np.float32)).to(device)
    net = model.CSDBF(height, width, bands, Q1, A1, Q2, A2, Q3, A3, A12, A13, A23, A21, A31, A32,
                          supernum1, supernum2, supernum3)
    net.to(device)
    transfer_time_after = time.time()
    transfer_time = transfer_time_after - transfer_time_before

    train_before = time.time()
    train.train(net, lr, num_epochs, net_input_before, net_input_after, net_input_concat, train_gt, train_onehot, train_mask, val_gt, val_onehot, val_mask)
    train_after = time.time()
    train_time = train_after - train_before
    print("train time: {}".format(train_time))
    test_before = time.time()
    pred, test_OA, test_kappa = test.test(net, net_input_before, net_input_after, net_input_concat, test_gt, test_onehot, test_mask)
    test_after = time.time()
    test_time = test_after - test_before
    print("test time: {}".format(test_time))

    classification_map = torch.argmax(pred, 1).reshape([height, width]).cpu() + 1
    generate_png(classification_map, "results\\" + dataset_name + str(test_OA))
    OA_record.append(test_OA)
    kappa_record.append(test_kappa)
    alltime = data_pre_time + SS_time + train_time + test_time
    alltime_record.append(alltime)

OA = np.array(OA_record)
kappa = np.array(kappa_record)
alltime = np.array(alltime)
print('OA=', np.mean(OA), '+-', np.std(OA))
print('kappa=', np.mean(kappa), '+-', np.std(kappa))
print('alltime=', np.mean(alltime))

f = open('results\\' + dataset_name + '_results.txt', 'a+')
str_results ='\nOA='+ str(np.mean(OA))+ '+-'+ str(np.std(OA)) \
+'\nkappa='+ str(np.mean(kappa))+ '+-'+ str(np.std(kappa)) \
+'\nalltime='+ str(np.mean(alltime))+ '+-'+ str(np.std(alltime))
f.write(str_results)
f.close()











