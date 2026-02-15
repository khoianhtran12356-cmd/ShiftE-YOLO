import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

# ----- Multi angle Sobel Conv -----
class SobConvAng(nn.Module):
    def __init__(self, c1: int, c2: int,  angle: int, groups: int=1):
        super().__init__()
        assert c2 % groups == 0
        self.groups = groups
        self.angle= angle
        if angle == 0:
            kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        elif angle == 45:
            kernel = [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
        elif angle == 90:
            kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        elif angle == 135:
            kernel = [[2, 1, 0], [1, 0, -1], [0, -1, -2]]
        else:
            raise ValueError("Only angles 0, 45, 90, 135 are supported.")

        kernel = torch.tensor(kernel, dtype=torch.float32).view(1, 1, 3, 3)
        kernel = kernel.repeat(c2, c1 // groups, 1, 1)
        self.register_buffer("sob_kernel", kernel)

        self.pad = nn.ReflectionPad2d(1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = F.conv2d(self.pad(x), self.sob_kernel, stride=1, groups=self.groups)
        x2 = self.act(self.bn(x1))
        return x2


# ----- Sobel Conv Block-----
class SobConv(nn.Module):
    def __init__(self, c1: int, c2: int, groups: int=1): #, c2: int
        super().__init__()
        self.groups = groups

        self.sobel_0 = SobConvAng(c1, c2,  0, groups= self.groups)
        self.sobel_90 = SobConvAng(c1, c2, 90, groups= self.groups)

    def forward(self, x):
        cv1 = torch.sqrt(self.sobel_0(x)**2+ self.sobel_90(x)**2 + 1e-6)
        return cv1 

# ----- Laplacian conv với Group/ Deptwise -----
class LapConv(nn.Module):
    def __init__(self, c1: int, c2: int, groups: int=1):
        super().__init__()
        assert c2 % groups == 0
        self.groups = groups

        kernel = [[0,  1,  0], 
                  [1, -4,  1], 
                  [0,  1,  0]]
        kernel = torch.tensor(kernel, dtype=torch.float32).view(1, 1, 3, 3)
        kernel = kernel.repeat(c2, c1 // groups, 1, 1)
        self.register_buffer("lap_kernel", kernel)

        self.pad = nn.ReflectionPad2d(1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        cv1= F.conv2d(self.pad(x), self.lap_kernel, stride=1, groups=self.groups)
        cv2= self.act(self.bn(cv1))
        return cv2 