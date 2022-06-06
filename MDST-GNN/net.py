from MDSTGNN.layers import *
from MDSTGNN.downsampling import *

class mdstnet(nn.Module):
    def __init__(self, gl_true, num_layer, gl_depth, num_nodes, device, batch_size,dropout=0.3,ds_dropout=0.5,
                 number_neighbors=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=16,
                 skip_channels=32, end_channels=64, seq_length=12, in_dim=2, out_dim=12, layers=3, prop_beta=0.05,
                 tanh_alpha=3, out_len = 1):
        super(mdstnet, self).__init__()
        self.gl_true = gl_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.ds_dropout = ds_dropout
        self.batch_size = batch_size
        self.layers = layers
        self.obs_len = []
        self.temp_scinet_in_len = []

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        if gl_true:
            self.gl = graph_learning(num_nodes, number_neighbors, node_dim, device, alpha=tanh_alpha)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        if self.seq_length > self.receptive_field:
            self.obs_len.append(self.seq_length)
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
        else:
            self.obs_len.append(self.receptive_field)
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1

            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))

                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length - rf_size_j + 1)))
                    self.obs_len.append(self.seq_length - rf_size_j + 1)

                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, self.receptive_field - rf_size_j + 1)))
                    self.obs_len.append(self.receptive_field - rf_size_j + 1)

                if self.gl_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gl_depth, dropout, prop_beta))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gl_depth, dropout, prop_beta))
                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1)))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1)))

                new_dilation *= dilation_exponential
                self.temp_scinet_in_len.append(self.obs_len[j - 1] if self.obs_len[j - 1] % (2**num_layer) == 0
                                               else self.obs_len[j - 1] - (self.obs_len[j - 1] % (2**num_layer)))

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.downsampling = Downsampling(output_len=out_len, input_len=self.temp_scinet_in_len[-1],
                                         in_channel=conv_channels, input_dim=self.num_nodes, hid_size=1,
                                         num_layer=num_layer, groups=1, kernel=5, dropout=self.ds_dropout).cuda()

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)

        self.drop = nn.Dropout(self.dropout)
        self.drop2 = nn.Dropout(self.dropout)
        self.pad = nn.ReplicationPad2d((0,self.receptive_field - self.seq_length,0,0))
        self.conv = nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=(1, 1), bias=False)
        self.linear = nn.Linear(self.obs_len[-1], self.temp_scinet_in_len[-1], bias=False)

    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input = self.pad(input)

        adj = None
        if self.gl_true:
            adj = self.gl(idx)

        x = self.start_conv(input)
        skip = self.skip0(self.drop(input))

        for i in range(self.layers):
            residual = x

            # temporal convolution
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = self.drop2(x)
            s = self.skip_convs[i](x)
            skip = s + skip

            # graph convolution
            if self.gl_true:
                x = self.gconv1[i](x, adj) + self.gconv2[i](x, adj.transpose(1, 0))

            # residual connection
            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, idx)

        x1 = self.linear(x)
        x_downsampling = self.downsampling(x1)
        skip = self.conv(x_downsampling) + skip
        x = torch.relu(skip)
        x = torch.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
