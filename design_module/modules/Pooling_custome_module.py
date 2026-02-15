import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.shiftwise import SWConv2d
import math

class MSSW2d(nn.Module):
    """Multi-scale convolution using shift-wise conolution"""
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.cv2 = SWConv2d(in_channels=int(c1), out_channels=int(c2), big_kernel=5, small_kernel=3, stride=1, N_rep=3)
        self.cv3 = SWConv2d(in_channels=int(c1), out_channels=int(c2), big_kernel=7, small_kernel=3, stride=1, N_rep=3)
        self.cv4 = SWConv2d(in_channels=int(c1), out_channels=int(c2), big_kernel=9, small_kernel=3, stride=1, N_rep=3)
    def forward(self,x):
        x2 = self.cv2(x)
        x3 = self.cv3(x)
        x4 = self.cv4(x)
        out = x2 + x3 + x4
        return out

class CAE(nn.Module):
    def __init__(self, cin: int):
        super(CAE, self).__init__()
        self.cv1 = MSSW2d(c1=int(cin), c2=int(cin))
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) 
        self.cv2 = nn.Conv2d(int(cin), int(cin), kernel_size=1, stride=1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.avg(x)+self.max(x)
        x3 = x2*x1

        B, C, H, W = x3.shape
        gmp_vec = F.max_pool2d(x3, kernel_size=(H, W))  # (B, C, 1, 1)
        x4 = self.act(self.cv2(gmp_vec))
        
        B, C, H, W = x4.shape
        activ = torch.softmax(x4.view(B, C, -1), dim=1).view(B, C, H, W)
        activ_exp = activ.expand_as(x)
        
        out = x*activ_exp
        return out

class SAE(nn.Module):
    def __init__(self, cin: int):
        super(SAE, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.max = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.cv1 = MSSW2d(c1=2, c2=2)
        self.cv2 = nn.Conv2d(2, 1, kernel_size=1, stride=1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        max3= self.max(x)
        A_max, _ = torch.max(max3, dim=1, keepdim=True)
        avg3= self.avg(x)
        A_avg = torch.mean(avg3, dim=1, keepdim=True)
        x1= torch.cat((A_max, A_avg), dim=1)
        x2= self.cv1(x1)
        x3= x2+x1
        x4= self.cv1(x3)
        x5= x4+x3+x1
        x6= self.act(self.cv2(x5))
        B, C, H, W = x6.shape
        activ = torch.softmax(x6.view(B, C, -1), dim=2).view(B, C, H, W)
        activ_exp = activ.expand_as(x)
        out = x*activ_exp  
        return out    

class CSAE(nn.Module):
    def __init__(self, cin: int, cout: int):
        super(CSAE, self).__init__()
        self.cae = CAE(cin = int(cin))
        self.sae = SAE(cin = int(2*cin))
        self.conv2 = Conv(c1=int(4*cin), c2=int(cout), k=1, s=1)
    def forward(self, x):
        x1 = self.cae(x)
        dns1 = torch.cat((x, x1), dim=1)
        x2 = self.sae(dns1)
        dns2 = torch.cat((dns1, x2), dim=1)
        out = self.conv2(dns2)
        return out  

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: [N, C, H, W]
        n = x.shape[2] * x.shape[3] - 1

        # (x - mean)^2
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # variance per channel
        v = d.sum(dim=[2, 3], keepdim=True) / n

        # energy function
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5

        # attention
        return x * torch.sigmoid(e_inv)

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0

        self.softmax = nn.Softmax(-1)
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, 1)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)

        # === Thay AdaptiveAvgPool2d ===
        x_h = group_x.mean(dim=3, keepdim=True)              # (None,1)
        x_w = group_x.mean(dim=2, keepdim=True).permute(0,1,3,2)  # (1,None)

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0,1,3,2).sigmoid())
        x2 = self.conv3x3(group_x)

        # === Thay AdaptiveAvgPool2d((1,1)) ===
        x11 = self.softmax(x1.mean(dim=(2,3), keepdim=True)
                              .reshape(b*self.groups, -1, 1)
                              .permute(0,2,1))
        x12 = x2.reshape(b*self.groups, c//self.groups, -1)

        x21 = self.softmax(x2.mean(dim=(2,3), keepdim=True)
                              .reshape(b*self.groups, -1, 1)
                              .permute(0,2,1))
        x22 = x1.reshape(b*self.groups, c//self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)) \
                    .reshape(b*self.groups, 1, h, w)

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class SEBlock(nn.Module):
    """
    https://github.com/AILab-CVC/UniRepLKNet/blob/main/unireplknet.py#L120C1-L141C62
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CA, self).__init__()
        #self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        #self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True)   # (N, C, H, 1)
        x_w = torch.mean(x, dim=2, keepdim=True)   # (N, C, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)               # (N, C, W, 1)

        #x_h = self.pool_h(x)
        #x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

x = torch.randn(1, 32, 64, 64)
ca = CA(32, 32)
y = ca(x)

print(x.shape, y.shape)
