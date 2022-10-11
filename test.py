import torch
import numpy as np
from sklearn import metrics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cal_kappa(network_output, test_gt, test_onehot, zeros, height, width):

    with torch.no_grad():
        test_gt_cpu = test_gt.cpu().detach().numpy()
        Test_GT = np.reshape(test_gt_cpu, [height, width])
        # OA
        available_label_idx = (test_gt != 0).float()
        available_label_count = available_label_idx.sum()
        correct_prediction = torch.where(
            torch.argmax(network_output, 1) == torch.argmax(test_onehot, 1), available_label_idx,
            zeros).sum()
        OA = correct_prediction.cpu() / available_label_count
        OA = OA.cpu().numpy()
        output_data = network_output.cpu().numpy()
        # kappa
        test_pre_label_list = []
        test_real_label_list = []
        output_data = np.reshape(output_data, [height * width, 2])
        idx = np.argmax(output_data, axis=-1)
        idx = np.reshape(idx, [height, width])
        for ii in range(height):
            for jj in range(width):
                if Test_GT[ii][jj] != 0:
                    test_pre_label_list.append(idx[ii][jj] + 1)
                    test_real_label_list.append(Test_GT[ii][jj])
        test_pre_label_list = np.array(test_pre_label_list)
        test_real_label_list = np.array(test_real_label_list)
        kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                              test_real_label_list.astype(np.int16))
        test_kpp = kappa
        print("test OA=", OA, 'kpp=', test_kpp)
        return OA, kappa

def test(net, net_input_before, net_input_after, net_input_concat, test_gt, test_onehot, test_mask):
    torch.cuda.empty_cache()
    height = test_onehot.size()[0]
    width = test_onehot.size()[1]
    zeros = torch.zeros([height * width]).to(device).float()
    test_onehot = test_onehot.reshape([-1, 2])
    test_gt = test_gt.reshape([-1])
    with torch.no_grad():
        net.load_state_dict(torch.load("model\\best_model.pt"))
        net.eval()
        output = net(net_input_before, net_input_after, net_input_concat)
        test_OA, test_kappa = cal_kappa(output, test_gt, test_onehot, zeros, height, width)
        print("test OA={}\t test kappa={}".format(test_OA, test_kappa))

    return output, test_OA, test_kappa