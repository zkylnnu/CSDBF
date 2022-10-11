import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device =torch.device("cpu")

class GraphAttentionLayer(nn.Module):

    def __init__(self, A1, A2, A3, A12, A13, A23, A21, A31, A32, nodes1, nodes2, nodes3, in_features, out_features,
                 dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A12 = A12
        self.A13 = A13
        self.A23 = A23
        self.A21 = A21
        self.A31 = A31
        self.A32 = A32
        self.nodes1 = nodes1
        self.nodes2 = nodes2
        self.nodes3 = nodes3
        self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * self.out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        Wh = torch.mm(h, self.W)  # (nodes, hid)

        e = self._prepare_attentional_mechanism_input(Wh)  # (nodes, nodes)
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.zeros_like(e)
        attention[0:self.nodes1, 0:self.nodes1] = torch.where(self.A1 > 0, e[0:self.nodes1, 0:self.nodes1],
                                                              zero_vec[0:self.nodes1, 0:self.nodes1])
        attention[self.nodes1:self.nodes1 + self.nodes2, 0:self.nodes1] = torch.where(self.A21 > 0, e[
                                                                                                    self.nodes1:self.nodes1 + self.nodes2,
                                                                                                    0:self.nodes1],
                                                                                      zero_vec[
                                                                                      self.nodes1:self.nodes1 + self.nodes2,
                                                                                      0:self.nodes1])
        attention[0:self.nodes1, self.nodes1:self.nodes1 + self.nodes2] = torch.where(self.A12 > 0, e[0:self.nodes1,
                                                                                                    self.nodes1:self.nodes1 + self.nodes2],
                                                                                      zero_vec[0:self.nodes1,
                                                                                      self.nodes1:self.nodes1 + self.nodes2])
        attention[self.nodes1:self.nodes1 + self.nodes2, self.nodes1:self.nodes1 + self.nodes2] = torch.where(
            self.A2 > 0, e[self.nodes1:self.nodes1 + self.nodes2, self.nodes1:self.nodes1 + self.nodes2],
            zero_vec[self.nodes1:self.nodes1 + self.nodes2, self.nodes1:self.nodes1 + self.nodes2])
        attention[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3, 0:self.nodes1] = torch.where(
            self.A31 > 0, e[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3, 0:self.nodes1],
            zero_vec[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3, 0:self.nodes1])
        attention[0:self.nodes1, self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3] = torch.where(
            self.A13 > 0, e[0:self.nodes1, self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3],
            zero_vec[0:self.nodes1, self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3])
        attention[self.nodes1:self.nodes1 + self.nodes2,
        self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3] = torch.where(self.A23 > 0, e[
                                                                                                       self.nodes1:self.nodes1 + self.nodes2,
                                                                                                       self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3],
                                                                                         zero_vec[
                                                                                         self.nodes1:self.nodes1 + self.nodes2,
                                                                                         self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3])
        attention[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
        self.nodes1:self.nodes1 + self.nodes2] = torch.where(self.A32 > 0, e[
                                                                           self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
                                                                           self.nodes1:self.nodes1 + self.nodes2],
                                                             zero_vec[
                                                             self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
                                                             self.nodes1:self.nodes1 + self.nodes2])
        attention[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
        self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3] = torch.where(self.A3 > 0, e[
                                                                                                      self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
                                                                                                      self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3],
                                                                                         zero_vec[
                                                                                         self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
                                                                                         self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3])
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        mid = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(mid)
        else:
            return mid

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, A1, A2, A3, A12, A13, A23, A21, A31, A32, nodes1, nodes2, nodes3, nfeat, nhid, nclass, dropout,
                 nheads, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A12 = A12
        self.A13 = A13
        self.A23 = A23
        self.A21 = A21
        self.A31 = A31
        self.A32 = A32
        self.nodes1 = nodes1
        self.nodes2 = nodes2
        self.nodes3 = nodes3
        self.attentions3 = [
            GraphAttentionLayer(self.A1, self.A2, self.A3, self.A12, self.A13, self.A23, self.A21, self.A31, self.A32,
                                 self.nodes1, self.nodes2, self.nodes3, nfeat, nhid, dropout=dropout, alpha=alpha,
                                 concat=True) for _ in
            range(nheads)]
        for i, attention in enumerate(self.attentions3):
            self.add_module('attention3_{}'.format(i), attention)

    def forward(self, x):
        x1 = F.dropout(x, self.dropout, training=self.training)

        x2 = torch.cat([att1(x1) for att1 in self.attentions3], dim=1)

        x3 = x2[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3, :]
        x4 = x2[0:self.nodes1, :]
        x5 = x2[self.nodes1:self.nodes1 + self.nodes2, :]

        x3 = F.dropout(x3, self.dropout, training=self.training)
        x4 = F.dropout(x4, self.dropout, training=self.training)
        x5 = F.dropout(x5, self.dropout, training=self.training)
        return F.log_softmax(x3, dim=1), F.log_softmax(x4, dim=1), F.log_softmax(x5, dim=1)

class SSConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class CSDBF(nn.Module):
    def __init__(self, height: int, width: int, changel: int, Q1: torch.Tensor, A1: torch.Tensor,
                 Q2: torch.Tensor, A2: torch.Tensor, Q3: torch.Tensor, A3: torch.Tensor, A12: torch.Tensor,
                 A13: torch.Tensor, A23: torch.Tensor, A21: torch.Tensor, A31: torch.Tensor, A32: torch.Tensor,
                 num1: int, num2: int, num3: int):
        super(CSDBF, self).__init__()
        self.channel = changel
        self.height = height
        self.width = width
        self.Q1 = Q1
        self.Q2 = Q2
        self.Q3 = Q3
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A12 = A12
        self.A13 = A13
        self.A23 = A23
        self.A21 = A21
        self.A31 = A31
        self.A32 = A32
        self.nodes1 = num1
        self.nodes2 = num2
        self.nodes3 = num3
        self.norm_col_Q1 = Q1 / (torch.sum(Q1, 0, keepdim=True))
        self.norm_col_Q2 = Q2 / (torch.sum(Q2, 0, keepdim=True))
        self.norm_col_Q3 = Q3 / (torch.sum(Q3, 0, keepdim=True))
        layers_count = 2

        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        self.CNN_denoise1 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise1.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(2 * self.channel))
                self.CNN_denoise1.add_module('CNN_denoise_Conv' + str(i),
                                             nn.Conv2d(2 * self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise1.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise1.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise1.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise1.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        self.CNN_Branch = nn.Sequential()
        for i in range(1):
            if i < 0:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 128, kernel_size=5))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))

        self.myGAT = GAT(self.A1, self.A2, self.A3, self.A12, self.A13, self.A23, self.A21, self.A31, self.A32,
                         self.nodes1, self.nodes2, self.nodes3, nfeat=128, nhid=64, nclass=64, dropout=0.3, nheads=3,
                         alpha=0.2)
        self.Softmax_linear = nn.Sequential(nn.Linear(640, 2))

    def forward(self, x: torch.Tensor, y: torch.Tensor, abs: torch.Tensor):

        (h, w, c) = x.shape
        noise_x = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]),
                                                   0))
        noise_x = torch.squeeze(noise_x, 0).permute([1, 2, 0])
        clean_x = noise_x
        noise_y = self.CNN_denoise(torch.unsqueeze(y.permute([2, 0, 1]),
                                                   0))
        noise_y = torch.squeeze(noise_y, 0).permute([1, 2, 0])
        clean_y = noise_y

        noise_abs = self.CNN_denoise1(torch.unsqueeze(abs.permute([2, 0, 1]), 0))
        noise_abs = torch.squeeze(noise_abs, 0).permute([1, 2, 0])
        clean_abs = noise_abs

        clean_x_flatten = clean_x.reshape([h * w, -1])
        clean_y_flatten = clean_y.reshape([h * w, -1])
        clean_abs_flatten = clean_abs.reshape([h * w, -1])

        superpixels_flatten_x = torch.mm(self.norm_col_Q1.t(),
                                         clean_x_flatten)
        superpixels_flatten_y = torch.mm(self.norm_col_Q2.t(),
                                         clean_y_flatten)
        superpixels_flatten_abs = torch.mm(self.norm_col_Q3.t(), clean_abs_flatten)

        CNN_result_abs = self.CNN_Branch(torch.unsqueeze(clean_abs.permute([2, 0, 1]), 0))
        CNN_result_abs = torch.squeeze(CNN_result_abs, 0).permute([1, 2, 0]).reshape([h * w, -1])


        GAT_input1 = superpixels_flatten_x
        GAT_input2 = superpixels_flatten_y
        GAT_input3 = superpixels_flatten_abs
        GAT_input_concat = torch.cat([GAT_input1, GAT_input2, GAT_input3], dim=0)

        GAT_result1, GAT_result2, GAT_result3 = self.myGAT(GAT_input_concat)
        GAT_result1 = torch.matmul(self.Q3, GAT_result1)
        GAT_result2 = torch.matmul(self.Q1, GAT_result2)
        GAT_result3 = torch.matmul(self.Q2, GAT_result3)
        GAT_result = torch.cat([GAT_result1, GAT_result2, GAT_result3], dim=1)
        Y = torch.cat([CNN_result_abs,GAT_result],dim=-1)
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y
