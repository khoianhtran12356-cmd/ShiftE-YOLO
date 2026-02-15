import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.sob_lap_linear import LapDws, SobDws
from ultralytics.nn.modules.shiftwise import SWConv

class SOLA_Bottleneck(nn.Module):
    """using Ghost convolution and Depth wise convolution for Densenet module"""
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.cv1= Conv(c1=int(c1), c2= int(0.5*c1), k=1, s=1)

        self.cv2 = Conv(c1=int(0.5*c1), c2= int(0.5*c1), k=3, s=1, p=1) 
        self.lap = LapDws(c1= int(0.5*c1), c2= int(0.5*c1), groups=int(0.5*c1))
        self.cv3 = nn.Conv2d(int(c1), int(0.5*c1), kernel_size=3, stride=1, padding=1)
        
        self.cv4 = Conv(c1=int(c1), c2= int(c1), k=3, s=1, p=1)
        self.sob = SobDws( c1= int(c1), c2= int(c1), groups=int(c1) )
        self.cv5 = nn.Conv2d( int(2*c1), int(c1), kernel_size=3, stride=1, padding=1 )
        
        self.cv6 = nn.Conv2d(int(2*c1), int(c2), kernel_size=1, stride=1)
    def forward(self, x):
        x1 = self.cv1(x)
        
        x2 = self.cv2(x1)
        x3 = self.lap(x2)
        gs1= torch.cat((x3, x2), dim=1)
        x4 = self.cv3(gs1)
        dense1= torch.cat((x4, x1), dim=1)
        
        x5 = self.cv4(dense1)
        x6 = self.sob(x5)
        gs2= torch.cat((x6, dense1), dim=1)
        x7 = self.cv5(gs2)
        dense2= torch.cat((x7, dense1), dim=1)
        
        x8 = self.cv6(dense2)
        return x8

class MSSW(nn.Module):
    """Multi-scale large kernel convolution using shift-wise conolution"""
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.cv1= Conv(c1=int(c1), c2= int(0.5*c1), k=1, s=1)
        self.cv2 = SWConv(in_channels=int(0.5*c1), out_channels=int(0.5*c1), big_kernel=5, small_kernel=3, stride=1, ghost_ratio=0.23, N_rep=3,)
        self.cv3 = SWConv(in_channels=int(0.5*c1), out_channels=int(0.5*c1), big_kernel=7, small_kernel=3, stride=1, ghost_ratio=0.23, N_rep=3,)
        self.cv4 = SWConv(in_channels=int(0.5*c1), out_channels=int(0.5*c1), big_kernel=9, small_kernel=3, stride=1, ghost_ratio=0.23, N_rep=3,)
        self.cv5 = nn.Conv2d(int(0.5*c1), int(c2), kernel_size=1, stride=1)
    def forward(self,x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        x3 = self.cv3(x1)
        x4 = self.cv4(x1)
        out = x2 + x3 + x4
        x5 = self.cv5(out)
        return x5

class SW_Bottleneck(nn.Module):
    """using Shift-wise convolution for Densenet module"""
    def __init__(self, c1: int, c2: int):
        super(SW_Bottleneck, self).__init__()
        self.cv1= Conv(c1=int(c1), c2=int(c1), k=3, s=1)
        self.dns1 = MSSW(c1=int(c1), c2=int(c1))
        self.dns2 = MSSW(c1=int(2*c1), c2=int(2*c1))
        self.cv2 = Conv(c1=int(4*c1), c2=int(c2), k=3, s=1)
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.dns1(x1)
        dn1= torch.cat((x1, x2), dim=1)
        x3 = self.dns2(dn1)
        dn2= torch.cat((x3, dn1), dim=1)
        x4 = self.cv2(dn2)
        return x4

if __name__ == "__main__":
    x = torch.randn(1, 16, 64, 64)
    SWConv = SW_Bottleneck(16, 16)
    y = SWConv(x)
    print(y.shape)
