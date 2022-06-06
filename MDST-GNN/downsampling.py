import math
from torch import nn
import torch
import numpy as np

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))

class downsampling_layer(nn.Module):
    def __init__(self, in_channel, splitting=True, kernel=5, dropout=0.5, groups=1, hidden_size=1):
        super(downsampling_layer, self).__init__()
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1
            pad_r = self.dilation * (self.kernel_size) // 2 + 1

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size

        modules_P += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),

            nn.Conv2d(in_channel * prev_size, int(in_channel * size_hidden),
                      kernel_size=(1, self.kernel_size), dilation=(1, self.dilation), groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(int(in_channel * size_hidden), in_channel, kernel_size=(1, 3), groups=self.groups),
            nn.Tanh()
        ]

        modules_U += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(in_channel * prev_size, int(in_channel * size_hidden),
                      kernel_size=(1, self.kernel_size), dilation=(1, self.dilation), groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(int(in_channel * size_hidden), in_channel, kernel_size=(1, 3), groups=self.groups),
            nn.Tanh()
        ]

        # [batch,channel,num_nodes,seq_length//2]
        modules_phi += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(in_channel * prev_size, int(in_channel * size_hidden),
                      kernel_size=(1, self.kernel_size), dilation=(1, self.dilation), groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(int(in_channel * size_hidden), in_channel, kernel_size=(1, 3), groups=self.groups),
            nn.Tanh()
        ]

        # [batch,channel,num_nodes,seq_length//2]
        modules_psi += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(in_channel * prev_size, int(in_channel * size_hidden),
                      kernel_size=(1, self.kernel_size), dilation=(1, self.dilation), groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(int(in_channel * size_hidden), in_channel, kernel_size=(1, 3), groups=self.groups),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        d = x_odd.mul(torch.exp(self.phi(x_even)))
        c = x_even.mul(torch.exp(self.psi(x_odd)))
        x_even_update = c - self.U(d)
        x_odd_update = d - self.P(c)

        return (x_even_update, x_odd_update)

class downsampling_tree(nn.Module):
    def __init__(self, in_channel,current_layer, kernel_size, dropout, groups, hidden_size):
        super().__init__()
        self.current_layer = current_layer

        self.workingblock = downsampling_layer(in_channel=in_channel, splitting=True,
                                kernel=kernel_size, dropout=dropout, groups=groups, hidden_size=hidden_size)

        if current_layer != 0:
            self.SCINet_Tree_odd = downsampling_tree(in_channel, current_layer - 1, kernel_size, dropout, groups, hidden_size)
            self.SCINet_Tree_even = downsampling_tree(in_channel, current_layer - 1, kernel_size, dropout, groups, hidden_size)

    # concat subsequence
    def zip_up_the_pants(self, even, odd):
        even = even.permute(3, 0, 1, 2)
        odd = odd.permute(3, 0, 1, 2)  # L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len:
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_, 0).permute(1, 2, 3, 0)  # B, L, D

    def forward(self, x):
        x_even_update, x_odd_update = self.workingblock(x)
        if self.current_layer == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))

class Downsampling(nn.Module):
    def __init__(self, output_len, input_len, in_channel,input_dim=9, hid_size=1,
                 num_layer=3, groups=1, kernel=5, dropout=0.5):
        super(Downsampling, self).__init__()

        self.in_channel = in_channel
        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_layer = num_layer
        self.groups = groups
        self.kernel_size = kernel
        self.dropout = dropout

        self.blocks1 = downsampling_tree(
            in_channel=in_channel,
            current_layer=num_layer - 1,
            kernel_size=self.kernel_size,
            dropout=dropout,
            groups=groups,
            hidden_size=self.hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.projection1 = nn.Linear(self.input_len,self.output_len,bias=False)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01,inplace=True)

    def forward(self, x):
        # ensure sequence can be average split
        assert self.input_len % (np.power(2, self.num_layer)) == 0
        res1 = x
        x = self.blocks1(x)
        x = self.projection1(x)
        x += res1[:,:,:,-x.size(3):]

        return self.leaky_relu(x)

