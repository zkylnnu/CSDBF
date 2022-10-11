import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor,
                 reallabel_mask: torch.Tensor):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict + 1e-10))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy

def cal_OA(network_output, train_samples_gt, train_samples_gt_onehot, zeros):

    with torch.no_grad():
        available_label_idx = (train_samples_gt != 0).float()
        available_label_count = available_label_idx.sum()
        correct_prediction = torch.where(
            torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1), available_label_idx,
            zeros).sum()
        OA = correct_prediction.cpu() / available_label_count
        return OA

def train(net, lr, num_epochs, net_input_before, net_input_after, net_input_concat, train_gt, train_onehot, train_mask, val_gt, val_onehot, val_mask):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_loss = 1000000
    height = train_onehot.size()[0]
    width = train_onehot.size()[1]
    zeros = torch.zeros([height * width]).to(device).float()
    net.train()
    train_onehot = train_onehot.reshape([-1,2])
    val_onehot = val_onehot.reshape([-1, 2])
    train_gt = train_gt.reshape([-1])
    val_gt = val_gt.reshape([-1])
    for i in range(num_epochs + 1):
        optimizer.zero_grad()
        output = net(net_input_before, net_input_after, net_input_concat)
        loss = compute_loss(output, train_onehot, train_mask)
        loss.backward(retain_graph=False)
        optimizer.step()
        if i % 10 == 0:
            with torch.no_grad():
                net.eval()
                output = net(net_input_before, net_input_after, net_input_concat)
                trainloss = compute_loss(output, train_onehot, train_mask)
                trainOA = cal_OA(output, train_gt, train_onehot, zeros)
                valloss = compute_loss(output, val_onehot, val_mask)
                valOA = cal_OA(output, val_gt, val_onehot, zeros)
                print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA,
                                                                                       valloss, valOA))
                if valloss < best_loss:
                    best_loss = valloss
                    torch.save(net.state_dict(), "model\\best_model.pt")
                    print('saving model')
            torch.cuda.empty_cache()
            net.train()
    print("\n-----training ok. start testing-----\n")
