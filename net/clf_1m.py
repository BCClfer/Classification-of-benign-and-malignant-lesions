import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable

class norm_block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3,stride=1,padding=1):
        super(norm_block, self).__init__()
        ops=[]
        ops.append(nn.Conv3d(in_planes, out_planes, kernel,stride,padding))
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



class spatial_attention(nn.Module):
    def __init__(self):
        super(spatial_attention, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, 3,1,1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return self.sigmoid(x)


class res_conv_blk(nn.Module):
    def __init__(self, n_stages,planes,kernel=3,stride=1,padding=1):
        super(res_conv_blk, self).__init__()
        ops = []
        for i in range(n_stages):
            ops.append(nn.Conv3d(planes, planes, kernel, stride,padding))
            ops.append(nn.BatchNorm3d(planes))
            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)+x
        x = self.relu(x)
        return x

class downsample_conv_blk(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, kernel=2,stride=2,padding=0):
        super(downsample_conv_blk, self).__init__()
        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out,kernel, stride,padding))
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
        ops.append(nn.Conv3d(n_filters_in, n_filters_out,1))
        ops.append(nn.BatchNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class be_ma_net(nn.Module):
    def __init__(self,in_planes):
        super(be_ma_net, self).__init__()
        
        self.conv_first=norm_block(in_planes,8)
        #self.cl_attention=channel_attention_3D(8)

        self.res_conv1=res_conv_blk(2,8)
        self.cl_at1=channel_attention_3D(8) 
        self.sp_at1=spatial_attention()
        self.down_conv1=downsample_conv_blk(8,16)
        
        self.res_conv2=res_conv_blk(2,16)
        self.down_conv2=downsample_conv_blk(16,32)
        self.cl_at2=channel_attention_3D(16)
        self.sp_at2=spatial_attention()

        self.res_conv3=res_conv_blk(2,32)
        self.down_conv3=downsample_conv_blk(32,64)
        self.cl_at3=channel_attention_3D(32)
        self.sp_at3=spatial_attention()
 
        
        self.res_conv4=res_conv_blk(2,64)
        self.down_conv4=downsample_conv_blk(64,128)
        self.cl_at4=channel_attention_3D(64)
        self.sp_at4=spatial_attention()
        
        self.res_conv5 = res_conv_blk(2, 128)
        self.cl_at5=channel_attention_3D(128)
        self.sp_at5=spatial_attention()
        
        self.fc1=nn.Sequential(
            nn.Linear(4096*2,2048),
            nn.ReLU(True),
            #nn.Dropout(0.2),
            
            nn.Linear(2048,512),
            nn.ReLU(True),
            #nn.Dropout(0.3),
            nn.Linear(512,256),

            nn.ReLU(True),
            nn.Linear(256,2),
        )


        
        
    def forward(self,x):
       
        x=self.conv_first(x)
        
        
        x=self.res_conv1(x)
        cl_1=self.cl_at1(x)
        x=cl_1*x
        sp_1=self.sp_at1(x)
        x=sp_1*x
        x=self.down_conv1(x)
        
        x = self.res_conv2(x)
        cl_2=self.cl_at2(x)
        x=cl_2*x
        sp_2=self.sp_at2(x)
        x=sp_2*x
        x = self.down_conv2(x)
        
        x = self.res_conv3(x)
        cl_3=self.cl_at3(x)
        x=cl_3*x
        sp_3=self.sp_at3(x)
        x=sp_3*x
        x = self.down_conv3(x)
        
        x=self.res_conv4(x)
        cl_4=self.cl_at4(x)
        x=cl_4*x
        sp_4=self.sp_at4(x)
        x=sp_4*x
        x=self.down_conv4(x)
        
        x=self.res_conv5(x)
        cl_5=self.cl_at5(x)
        x=cl_5*x
        sp_5=self.sp_at5(x)
        x=sp_5*x
        

        dl_fe = x.view(x.size(0), -1)

        out=self.fc1(dl_fe)
        return out
