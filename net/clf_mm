import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class norm_block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, padding=1):
        super(norm_block, self).__init__()
        ops = []
        ops.append(nn.Conv3d(in_planes, out_planes, kernel, stride, padding))
        ops.append(nn.BatchNorm3d(out_planes))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class channel_attention_3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention_3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class spatial_attention_3D(nn.Module):
    def __init__(self):
        super(spatial_attention_3D, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, 3, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return self.sigmoid(x)


class res_conv_blk(nn.Module):
    def __init__(self, n_stages, planes, kernel=3, stride=1, padding=1):
        super(res_conv_blk, self).__init__()
        ops = []
        for i in range(n_stages):
            ops.append(nn.Conv3d(planes, planes, kernel, stride, padding))
            ops.append(nn.BatchNorm3d(planes))
            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x) + x
        x = self.relu(x)
        return x


class downsample_conv_blk(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, kernel=2, stride=2, padding=0):
        super(downsample_conv_blk, self).__init__()
        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel, stride, padding))
        ops.append(nn.BatchNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class downsample_pool_blk(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, pool_k=2):
        super(downsample_pool_blk, self).__init__()
        ops = []
        ## nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        ops.append(nn.MaxPool3d(pool_k))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, 1))
        ops.append(nn.BatchNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


# benign and malignant
"""
16 64 256 1024
"""
class be_ma_net(nn.Module):
    def __init__(self, in_planes, use_pool=False):
        super(be_ma_net, self).__init__()
        if use_pool:
            dn_blk = downsample_pool_blk
        else:
            dn_blk = downsample_conv_blk

        base=16
        ratio=4

        self.d1_conv_first = norm_block(in_planes, 16,kernel = 1, stride = 1, padding=0)
        self.d1_rescv1 = res_conv_blk(2, base)
        self.d1_clat1 = channel_attention_3D(base)
        self.d1_spat1 = spatial_attention_3D()
        self.d1_dncv1 = dn_blk(base, base*ratio)
        base*=ratio

        self.d1_rescv2 = res_conv_blk(2, base)
        self.d1_clat2 = channel_attention_3D(base)
        self.d1_spat2 = spatial_attention_3D()
        self.d1_dncv2 = dn_blk(base,base*ratio)
        base *= ratio

        self.d1_rescv3 = res_conv_blk(2, base)
        self.d1_clat3 = channel_attention_3D(base)
        self.d1_spat3 = spatial_attention_3D()
        self.d1_dncv3 = dn_blk(base, base*ratio)

        base*=ratio
        self.d1_rescv4 = res_conv_blk(2, base)
        self.d1_clat4 = channel_attention_3D(base)
        self.d1_spat4 = spatial_attention_3D()

        #print(base)
        self.avg=nn.AdaptiveAvgPool3d((1))

        self.fc_d1 = nn.Sequential(
            nn.Linear(base, 1024),
            nn.ReLU(True),
            # nn.Dropout(),

            nn.Linear(1024, 512),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(512, 2),
        )

    def forward(self, d1):

        d1_x = self.d1_conv_first(d1)
        d1_x = self.d1_rescv1(d1_x)
        d1_cl1 = self.d1_clat1(d1_x)
        d1_x = d1_x * d1_cl1
        d1_sp1 = self.d1_spat1(d1_x)
        d1_x = d1_x * d1_sp1
        d1_x = self.d1_dncv1(d1_x)

        d1_x = self.d1_rescv2(d1_x)
        d1_cl2 = self.d1_clat2(d1_x)
        d1_x = d1_x * d1_cl2
        d1_sp2 = self.d1_spat2(d1_x)
        d1_x = d1_x * d1_sp2
        d1_x = self.d1_dncv2(d1_x)

        d1_x = self.d1_rescv3(d1_x)
        d1_cl3 = self.d1_clat3(d1_x)
        d1_x = d1_x * d1_cl3
        d1_sp3 = self.d1_spat3(d1_x)
        d1_x = d1_x * d1_sp3
        d1_x = self.d1_dncv3(d1_x)

        d1_x = self.d1_rescv4(d1_x)
        d1_cl4 = self.d1_clat4(d1_x)
        d1_x = d1_x * d1_cl4
        d1_sp4 = self.d1_spat4(d1_x)
        d1_x = d1_x * d1_sp4

        d1_x=self.avg(d1_x)
        d1_x=d1_x.view(d1_x.size(0),-1)
        #print(d1_x.shape)

        out = self.fc_d1(d1_x)
        return out


def test():
    bs = 1
    ml_feature = torch.rand((bs, 8))
    img1 = torch.rand((bs, 1, 64, 64, 64))
    img2 = torch.rand((bs, 1, 64, 64, 64))
    img3 = torch.rand((bs, 1, 64, 64, 64))
    img4 = torch.rand((bs, 1, 64, 64, 64))

    net = be_ma_net(1)
    out = net(img1)
    print(out.shape)


#test()
