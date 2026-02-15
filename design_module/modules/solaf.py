import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
import torch.nn.functional as F
from ultralytics.nn.modules.sob_lap_linear import SobDws, LapDws

class SOLAF(nn.Module):
    """using Ghost convolution and Depth wise convolution for Densenet module"""
    def __init__(self, cin: int, cout: int):
        super(SOLAF, self).__init__()
        
        self.cv1 = Conv(c1=int(cin), c2= int(cin), k=3, s=1, p=1) 
        self.lap = LapDws(c1= int(cin), c2= int(cin), groups= int(cin))
        self.cv2 = nn.Conv2d(int(2*cin), int(cin), kernel_size=3, stride=1, padding=1)

        self.cv3 = Conv(int(2*cin), int(2*cin), k=3, s=1, p=1)
        self.sob = SobDws(c1= int(2*cin), c2= int(2*cin), groups= int(2*cin))
        self.cv4 = nn.Conv2d(int(4*cin), int(2*cin), kernel_size=3, stride=1, padding=1)
        
        self.cv5 = Conv(c1=int(4*cin),c2=int(cout), k=3, s=1, p=1)
    
    def forward(self, x):

        x1 = self.cv1(x)
        x2 = self.lap(x1)
        gs1= torch.cat((x1, x2), dim=1)
        x3 = self.cv2(gs1)
        dense1= torch.cat((x3, x), dim=1)

        x4 = self.cv3(dense1)
        x5 = self.sob(x4)
        gs2= torch.cat((x5, x4), dim=1)
        x6 = self.cv4(gs2)
        dense2= torch.cat((dense1, x6), dim=1)
        
        x7 = self.cv5(dense2)
        return x7

if __name__ == "__main__":
    x= torch.randn(1, 8, 16, 16)
    sola= SOLAF(cin=8,cout=8)
    y= sola(x)
    print(y.shape)