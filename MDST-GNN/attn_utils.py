import torch.nn as nn
import torch
from math import log
import math
from functools import reduce

class SELayer(nn.Module):
    def __init__(self,channel,reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)

class SELayer_nodes(nn.Module):
    def __init__(self,channel,reduction=4):
        super(SELayer_nodes, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel),
            nn.Sigmoid()
        )
    def forward(self,x1):
        x = x1.permute(0,2,1,3)
        b,nodes,_,_ = x.size()
        y = self.avg_pool(x).view(b,nodes)
        y = self.fc(y).view(b,nodes,1,1)
        return (x*y.expand_as(x)).permute(0,2,1,3)

class SELayer_nodes2(nn.Module):
    def __init__(self,channel,channel2,reduction=4):
        super(SELayer_nodes2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channel2, channel2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel2 // reduction, channel2),
            nn.Sigmoid()
        )
    def forward(self,x1):
        x = x1.permute(0,2,3,1)
        b,nodes,_,_ = x.size()
        y = self.avg_pool(x).view(b,nodes)
        y = self.fc(y).view(b,nodes,1,1).permute(0,2,1,3)

        x2 = x1.permute(0,3,2,1)
        b, seq, _, _ = x2.size()
        y2 = self.avg_pool(x2).view(b, seq)
        y2 = self.fc2(y2).view(b, seq, 1, 1).permute(0,3,2,1)

        return (x1*y.expand_as(x1)*y2.expand_as(x1))

class SELayer_seq(nn.Module):
    def __init__(self,channel,reduction=4):
        super(SELayer_seq, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel),
            nn.Sigmoid()
        )
    def forward(self,x1):
        x = x1.permute(0, 3, 2, 1)
        b,seq,_,_ = x.size()
        y = self.avg_pool(x).view(b,seq)
        y = self.fc(y).view(b,seq,1,1)
        return (x*y.expand_as(x)).permute(0,3,2,1)

#SK-Net
class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,G = 8,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        '''
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   # 计算向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=G,bias=False),
                                           nn.GroupNorm(out_channels//r,out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.GroupNorm(out_channels//r,d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z降维
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加
        return V


#channel and spartial attn
class CBAM(nn.Module):
    def __init__(self,channel,kernel_size,reduction =4):
        super(CBAM, self).__init__()
        # self.cha = ChannelAttn(channel,reduction=reduction)
        self.spa = Spartial_Attn(channel,kernel_size=kernel_size)

    def forward(self,x):
        # out = self.cha(x)
        return self.spa(x)

class Spartial_Attn(nn.Module):
    def __init__(self,channel,kernel_size=7):
        super(Spartial_Attn, self).__init__()
        # assert kernel_size % 2 == 1,'kernel_size={}'.format(kernel_size)
        # if kernel_size%2 == 0:
        #     kernel_size += 1
        # padding = (kernel_size-1)//2
        # self.atn = AttentionLayer(ProbAttention(output_attention=True), kernel_size, 4)
        self.layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, 1)),
        )
        # self.en_conv = nn.Conv2d(1, channel, kernel_size=(1, 1))
        # self.layer = nn.Sequential(
        #     nn.Conv2d(2,8,kernel_size=(1,kernel_size)),
        #     nn.ConvTranspose2d(in_channels=8,out_channels=1,kernel_size=(1,kernel_size)),
        #     nn.Sigmoid()
        # )
    def forward(self,x):
        avg_mask = torch.mean(x,dim=1,keepdim=True)
        # print(f'avg:{avg_mask.shape}')
        max_mask,_ = torch.max(x,dim=1,keepdim=True)
        # print(f'max_mask:{max_mask.shape},{_.shape}')
        mask = torch.cat([avg_mask,max_mask],dim=1)
        # print(f'mask:{mask.shape}')
        mask = self.layer(mask)
        # mask = mask.squeeze()
        # x,atn = self.atn(mask,mask,mask,None)
        # x= x.unsqueeze(dim=1)

        # print(f'mask2:{mask.shape}')
        # print(f'mask:{mask.shape}')
        # print(f'shape:{mask.expand_as(x).shape}')
        return self.en_conv(x)

class ChannelAttn(nn.Module):
    def __init__(self,channel,reduction=4):
        super(ChannelAttn, self).__init__()
        mid_channel = channel//reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels=channel,out_channels=mid_channel,kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channel,out_channels=channel,kernel_size=(1,1))
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return x*self.sigmoid(avgout+maxout)

#-----ANN
class APNB(nn.Module):
    def __init__(self, channel):
        super(APNB, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

#------Triplet
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.channel_pool = ChannelPool()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1)
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(self.channel_pool(x))
        return out * self.sigmod(out)


class TripletAttention(nn.Module):
    def __init__(self, spatial=True):
        super(TripletAttention, self).__init__()
        self.spatial = spatial
        self.height_gate = SpatialGate()
        self.width_gate = SpatialGate()
        if self.spatial:
            self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.height_gate(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.width_gate(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()

        if self.spatial:
            x_out3 = self.spatial_gate(x)
            return (1/3) * (x_out1 + x_out2 + x_out3)
        else:
            return (1/2) * (x_out1 + x_out2)


#GCB
class GlobalContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(GlobalContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


import matplotlib.pyplot as plt
#--------SEVariants
class cSE_Module(nn.Module):
    def __init__(self, channel,ratio = 2):
        super(cSE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
                nn.Linear(in_features=channel, out_features=channel),
                nn.Sigmoid()
            )
    def forward(self, x):

        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class sSE_Module(nn.Module):
    def __init__(self, channel):
        super(sSE_Module, self).__init__()
        self.spatial_excitation = nn.Sequential(
                nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1,stride=1,padding=0),
                nn.Sigmoid()
            )
    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)


class scSE_Module(nn.Module):
    def __init__(self, channel,ratio = 4):
        super(scSE_Module, self).__init__()
        self.cSE = cSE_Module(channel,ratio)
        self.sSE = sSE_Module(channel)

    def forward(self, x):
        # print(x.size())
        # l1, = plt.plot(x[0,0,0,:].data.cpu().numpy())
        res1 = self.cSE(x)
        res2 = self.sSE(x)
        # l8, = plt.plot(res1[0,0,0,:].data.cpu().numpy())
        # l2, = plt.plot(res2[0, 0, 0, :].data.cpu().numpy())
        # plt.legend(handles=[l1, l8,l2], labels=['x', 'res1','res2'], loc='best')
        # plt.show()
        return res1+res2

#ECA module
class ECA_Module(nn.Module):
    def __init__(self, channel,gamma=2, b=1):
        super(ECA_Module, self).__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(log(channel, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,y):
        b, c, _, _ = x.size()
        y = self.avg_pool(y)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        y = y.transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class DSCLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DSCLayer, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=(3,3),padding=1,groups=in_ch)
        # self.selayer = SELayer(in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=(1,1),groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        # out = self.selayer(out)
        out = self.point_conv(out)
        return out

class Shrinkage(nn.Module):
    def __init__(self, gap_size, channel):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        # self.conv = nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
        )
        self.norm = nn.BatchNorm1d(channel)
        self.fc1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(channel, 1),
            nn.Sigmoid(),
        )

    def forward(self,x,y):
        x_raw = x
        x = y
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)
        average = x
        # print(f'x:{x.shape}')
        x = self.fc(x)
        if x.size(0) == 1:
            x = self.fc1(x)
        else:
            x = self.fc1(self.norm(x))
        x = torch.mul(average, x)
        x = x.unsqueeze(2).unsqueeze(2)
        # 软阈值化
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        # x = self.conv(x)
        return x

class LSA(nn.Module):
    def __init__(self,batch_size,num_nodes, attn_dim, kernel_size=31, filters=16):
        super().__init__()
        self.conv = nn.Conv1d(2, filters, padding=(kernel_size - 1) // 2, kernel_size=kernel_size, bias=False)
        self.L = nn.Linear(filters, attn_dim, bias=True)
        self.W = nn.Linear(attn_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.cumulative = None
        self.attention = None

        self.cumulative = torch.zeros(batch_size, num_nodes)
        self.attention = torch.zeros(batch_size, num_nodes)

    # def init_attention(self, encoder_seq_proj):
    #     device = next(self.parameters()).device  # use same device as parameters
    #     b, t, c = encoder_seq_proj.size()
    #     self.cumulative = torch.zeros(b, t, device=device)
    #     self.attention = torch.zeros(b, t, device=device)

    def forward(self, encoder_seq_proj, query):
        processed_query = self.W(query).unsqueeze(1)

        location = torch.cat([self.cumulative.unsqueeze(1), self.attention.unsqueeze(1)], dim=1)
        processed_loc = self.L(self.conv(location).transpose(1, 2))

        u = self.v(torch.tanh(processed_query + encoder_seq_proj + processed_loc))
        u = u.squeeze(-1)

        # Smooth Attention
        scores = torch.sigmoid(u) / torch.sigmoid(u).sum(dim=1, keepdim=True)
        # scores = F.softmax(u, dim=1)
        self.attention = scores
        self.cumulative += self.attention

        return scores.unsqueeze(-1).transpose(1, 2)


class TPALSTM(nn.Module):
    # TPALSTM(num_nodes, num_nodes, num_nodes, seq_length, 1)
    def __init__(self, input_size, output_horizon, hidden_size, obs_len, batch_size, num_layers, conv_channels):
        super(TPALSTM, self).__init__()
        self.st_conv = nn.Conv2d(conv_channels,1,kernel_size=(1,1))
        self.end_conv = nn.Conv2d(1, conv_channels, kernel_size=(1, 1))
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, \
                            bias=True, batch_first=True)  # output (batch_size, obs_len, hidden_size)
        self.hidden_size = hidden_size
        self.filter_num = 32
        self.filter_size = 1
        self.batch_size = batch_size
        self.output_horizon = output_horizon
        self.seq_length = obs_len
        self.attention = TemporalPatternAttention(self.filter_size, \
                                                  self.filter_num, obs_len, hidden_size)
        self.den = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.num_layers = num_layers
        self.linear = nn.Linear(hidden_size * self.num_layers, output_horizon)

    def forward(self, x):
        x = self.st_conv(x)
        x = torch.relu(x)
        x = x.transpose(3,2).squeeze()
        batch_size, obs_len, num_nodes = x.size()
        xconcat = torch.relu(self.hidden(x))

        res = torch.zeros(batch_size, self.num_layers, self.hidden_size).cuda()
        res_ht = torch.zeros(batch_size, self.seq_length, self.hidden_size).cuda()

        ct = torch.zeros(1, batch_size, num_nodes).cuda()
        ht = torch.zeros(1, batch_size, num_nodes).cuda()
        attn_out = torch.zeros(self.batch_size, 1, num_nodes).cuda()
        H = torch.zeros(batch_size, obs_len, num_nodes).cuda()

        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1)
            for i in range(self.num_layers):

                if xt.shape[0] < self.batch_size:
                    xt = torch.cat([xt, xt[-(self.batch_size - batch_size):, :, :]], dim=0)
                xt = self.den(torch.cat([xt, attn_out], dim=2))
                # lstm
                out, (ht, ct) = self.lstm(xt, (ht, ct))

                htt = ht.permute(1, 0, 2)
                ctt = ct.permute(1, 0, 2)
                htt, ctt = htt[:, -1, :], ctt[:, -1, :]
                htt2 = torch.squeeze(torch.cat([ctt, htt], dim=1))

                # reshape hidden states H
                # print(H.shape)
                H = H.view(-1, 1, obs_len, self.hidden_size)

                new_ht, new_H = self.attention(H, htt2)
                output = self.den(torch.cat([torch.squeeze(out), new_ht], dim=1))
                new_H = torch.cat([new_H, torch.unsqueeze(output, dim=1)], dim=1)
                new_H = new_H.view(-1, obs_len ,self.hidden_size)
                H = new_H
                attn_out = torch.unsqueeze(new_ht, dim=1)
                H = self.relu(H)

                res[:, i, :] = new_ht[:batch_size, :]
                xt = torch.unsqueeze(output, dim=1)

            res_ht[:, t, :] = res[:, -1, :]
        # final_res = torch.flatten(res, start_dim=1)
        # ypred = self.linear(final_res)

        res_ht = res_ht.unsqueeze(1).transpose(3, 2)
        return self.end_conv(torch.relu(res_ht))

class TemporalPatternAttention(nn.Module):

    def __init__(self, filter_size, filter_num, attn_len, attn_size):
        super(TemporalPatternAttention, self).__init__()
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.feat_size = attn_size - self.filter_size + 1
        self.conv = nn.Conv2d(1, filter_num, (attn_len, filter_size))
        self.linear1 = nn.Linear(attn_size * 2, filter_num)
        self.linear2 = nn.Linear(attn_size * 2 + self.filter_num, attn_size)
        self.relu = nn.ReLU()

    def forward(self, H, ht):
        batch_size, channels, seq_len, attn_size = H.size()
        new_ht = ht.view(-1, 1, attn_size * 2)
        w = self.linear1(new_ht)  # batch_size, 1, filter_num
        conv_vecs = self.conv(H)

        conv_vecs = torch.transpose(torch.squeeze(conv_vecs), 1, 2)
        conv_vecs = conv_vecs.view(-1, self.feat_size, self.filter_num)
        conv_vecs = torch.relu(conv_vecs)

        # score function
        s = torch.mul(conv_vecs, w).sum(dim=2)
        alpha = torch.sigmoid(s)
        new_alpha = alpha.view(-1, self.feat_size, 1)
        v = torch.mul(new_alpha, conv_vecs).sum(dim=1).view(-1, self.filter_num)
        concat = torch.cat([torch.squeeze(ht), v], dim=1)
        new_ht = self.linear2(torch.relu(concat))
        new_ht2 = torch.squeeze(H[:, :, 1:, :])
        return new_ht, new_ht2

class Spatial_attn_conv(nn.Module):

    def __init__(self, ch_in,ch_out,num_nodes,kernel_size):
        super(Spatial_attn_conv, self).__init__()
        # self.fc = nn.Sequential(
        #     # nn.AdaptiveAvgPool2d((321,1)),
        #     # nn.Conv2d(ch_in, ch_out, kernel_size=(321, 1)),
        #     nn.Conv2d(ch_in,ch_out,kernel_size=(num_nodes,1)),
        # )
        self.conv = nn.Conv2d(ch_in,ch_out,kernel_size=(1,kernel_size))

    def forward(self,x):
        # x1 = self.fc(x)
        # print(x.shape)
        x = self.conv(x)*x
        return x

class Spatial_attn_para(nn.Module):

    def __init__(self, channel,batch_size,num_nodes,kernel_size):
        super(Spatial_attn_para, self).__init__()

        # self.conv = nn.Conv2d(ch_in,ch_out,kernel_size=(1,kernel_size))
        self.attn = nn.Parameter(torch.ones(batch_size, channel, num_nodes, 1).cuda())

    def forward(self,x):
        # x = torch.sigmoid(self.conv(x))*x
        # print(x.shape,self.attn.shape)
        if self.attn.size(0) > x.size(0):
            out = torch.mul(self.attn[:x.size(0), ...], x)
        else:
            out = torch.mul(self.attn, x)
        x = out
        return x

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=4, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # print(len(mapper_x),len(mapper_y))
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        # print(y.shape,x.shape)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        # assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class Spatial_attn(nn.Module):
    def __init__(self,batch_size,seq_len,num_nodes):
        super(Spatial_attn, self).__init__()
        self.global_seq = seq_len
        self.global_num = num_nodes
        # global_v = Variable(torch.zeros(global_attention_vec_size)) # v_l
        self.global_v = nn.Parameter(torch.FloatTensor(self.global_seq)).to('cuda:0')

        # global_attn = Variable(torch.zeros(batch_size, global_attn_length))
        self.global_attn = nn.Parameter(torch.FloatTensor(batch_size, self.global_num)).to('cuda:0')

        nn.init.normal_(self.global_v)
        nn.init.xavier_uniform_(self.global_attn)

    def spatial_attention(self,global_inputs, attention_states):
        global_attention_states = attention_states

        self.batch_size = attention_states.size(0)
        if self.batch_size<self.global_attn.size(0):
            self.global_attn = self.global_attn[:self.batch_size,:]

        # A trick: to calculate U_l * x^{i,k} by a 1-by-1 convolution
        global_hidden = global_attention_states.contiguous().view(-1, self.global_seq, self.global_num,
                                                                  1)
        # Size of query vectors for attention.
        # global_attention_vec_size = self.global_seq
        # global_k = nn.Conv2d(self.global_seq, global_attention_vec_size, (1, global_n_input), (1, 1))
        # global_hidden_features = global_k(global_hidden.float())

        outputs = []
        i = 0
        global_inputs = global_inputs.permute(2,0,1)
        # i is the index of the which time step
        for global_inp in global_inputs:
            # print(global_inp.device,global_inputs.device,global_attn.device)
            global_x = self.global_attn * global_inp
            # cell_output, state = cell(global_x)
            self.global_attn = self.global_attention([global_x],global_hidden)
            # Attention output projection
            outputs.append(global_x)
            i += 1

        return outputs

    def global_attention(self,query,hidden):
        # linear map
        y = Linear(query, self.global_seq, True)
        y = y.view(-1, 1, 1, self.global_seq)
        # Attention mask is a softmax of v_g^{\top} * tanh(...)
        # print(f'global----:{global_v.device,global_v.size(),global_hidden_features.device,global_hidden_features.size(),y.device,y.size()}')
        s = torch.sum(self.global_v * torch.tanh(hidden + y), dim=[1, 3])
        a = torch.softmax(s, dim=1)
        return a

def Linear(args, output_size, bias, bias_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias(default is all zeros).
      kernel_initializer: starting value to initialize the weight.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.data.size(1) for a in args]
    for shape in shapes:
        total_arg_size += shape

    # Now the computation.
    weights = nn.Parameter(torch.FloatTensor(total_arg_size, output_size)).to('cuda:0')
    nn.init.xavier_uniform_(weights)
    # weights = Variable(torch.zeros(total_arg_size, output_size))
    if len(args) == 1:
        res = torch.matmul(args[0], weights)
    else:
        res = torch.matmul(torch.cat(args, 1), weights)
    if not bias:
        return res

    if bias_initializer is None:
        biases = nn.Parameter(torch.zeros(output_size)).to('cuda:0')

    return torch.add(res, biases)

class eca_layer(nn.Module):
    def __init__(self,channel,k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=k_size,padding=(k_size-1)//2,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        b,c,h,w = x.size()
        x = x.permute(0,2,3,1)
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return (x*y.expand_as(x)).permute(0,3,1,2)
