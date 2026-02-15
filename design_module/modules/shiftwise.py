import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PureAddShiftMP(nn.Module):
    """
    Pure-PyTorch simulation of AddShift_mp_module.
    - big_kernel, small_kernel: as in original
    - c_out: number of output "rep" channels (repN)
    - c_in: number of input channels (out_n in original = repN * nk)
    - group_in: number of shuffle groups (N_rep in original)
    Forward signature: forward(x, b, hout, wout) -> returns (x1, x2, x3)
    """
    def __init__(self, big_kernel:int, small_kernel:int, c_out:int, c_in:int, group_in:int):
        super().__init__()
        self.big_kernel = big_kernel
        self.small_kernel = small_kernel
        self.c_out = c_out
        self.c_in = c_in
        self.group_in = group_in

        # nk = how many small kernels fit into big kernel (rep factor)
        self.nk = math.ceil(big_kernel / small_kernel)
        # padding (approx as in original shift)
        padding, real_pad = self._compute_padding((small_kernel, big_kernel))
        # extra_pad used for conv padding construction
        self.extra_pad = padding - (small_kernel // 2)
        self.padding = self.extra_pad  # we'll use this as conv padding

        # Prepare three different convolution kernels (group_out = 3)
        # We create learn-free kernels (one-hot) to select shifted positions.
        # Note: we create them as buffers (non-trainable) but you can make them Parameters if you want learnable.
        self.register_buffer('_kernel_group_0', self._make_one_hot_kernels(group_idx=0))
        self.register_buffer('_kernel_group_1', self._make_one_hot_kernels(group_idx=1))
        self.register_buffer('_kernel_group_2', self._make_one_hot_kernels(group_idx=2))

    def _compute_padding(self, kernels):
        mink, maxk = min(kernels), max(kernels)
        nk = math.ceil(maxk / mink)
        padding = mink - 1
        mid = maxk // 2
        real_pad = []
        for i in range(nk):
            extra_pad = mid - i * mink - padding
            real_pad.append(extra_pad)
        return padding, real_pad

    def _make_one_hot_kernels(self, group_idx:int):
        """
        Construct a kernel tensor of shape (C_out, C_in, nk, nk) with one-hot spatial positions.
        We distribute the c_in channels across output positions:
           - For out channel i, its corresponding input channels are [i*nk .. i*nk + nk-1]
           - For each relative k in 0..nk-1 we set a single '1' at position (kh, kw) inside nk x nk.
        group_idx just permutes the mapping to simulate different shuffles for each group.
        """
        nk = self.nk
        C_out = self.c_out
        C_in = self.c_in
        # kernel spatial size (we use nk x nk)
        kH = nk
        kW = nk

        # create zeros
        weight = torch.zeros((C_out, C_in, kH, kW), dtype=torch.float32)

        # a simple deterministic mapping to positions: map k -> (k // nk_row, k % nk_row)
        # but since we have nk*nk >= nk usually, we'll lay k along first row then next.
        # Also we permute k by group_idx to simulate different shuffles.
        base_positions = []
        for k in range(nk):
            row = k // nk if nk > 0 else 0
            col = k % nk
            # small adjustment so positions spread inside the nk x nk grid
            row = k // int(math.ceil(math.sqrt(nk)))
            col = k % int(math.ceil(math.sqrt(nk)))
            base_positions.append((row, col))

        # build weight
        # For out channel i, link it to C_in indices from i*nk to i*nk+nk-1 (if available)
        for i in range(C_out):
            for k in range(nk):
                in_idx = i * nk + k
                if in_idx >= C_in:
                    # if out_n doesn't align exactly, wrap around (safe fallback)
                    in_idx = in_idx % C_in
                # apply small permutation based on group_idx to vary positions between groups
                perm_k = (k + group_idx) % nk
                # clamp position inside kH/kW
                row, col = base_positions[perm_k]
                row = row % kH
                col = col % kW
                weight[i, in_idx, row, col] = 1.0

        return weight  # shape (C_out, C_in, kH, kW)

    def forward(self, x, b, hout, wout):
        """
        x: (B, C_in, H_in, W_in)
        b: B (redundant but kept for compat)
        hout, wout: expected output H/W
        returns three tensors (x1, x2, x3) each shape (B, C_out, hout, wout)
        """
        device = x.device
        # ensure kernel tensors on same device
        k0 = self._kernel_group_0.to(device)
        k1 = self._kernel_group_1.to(device)
        k2 = self._kernel_group_2.to(device)

        # Use conv2d with padding=self.padding. Because kernels are sparse one-hot,
        # this effectively selects shifted positions from the input.
        # Note: groups=1 to allow cross-channel mapping as in original mapping
        # If needed, can use groups=self.c_out to make depthwise-like mapping.
        out0 = F.conv2d(x, k0, bias=None, stride=1, padding=self.padding)
        out1 = F.conv2d(x, k1, bias=None, stride=1, padding=self.padding)
        out2 = F.conv2d(x, k2, bias=None, stride=1, padding=self.padding)

        # If output size doesn't match (due to padding differences), crop/resize to (hout,wout)
        # Crop center region to hout x wout
        #def crop_to(out, H, W):
        #    _, _, Ht, Wt = out.shape
        #    if Ht == H and Wt == W:
        #        return out
            # center crop
        #    start_h = max((Ht - H) // 2, 0)
        #    start_w = max((Wt - W) // 2, 0)
        #    return out[:, :, start_h:start_h+H, start_w:start_w+W]
        def crop_to(out, H, W):
            _, _, Ht, Wt = out.shape

            if Ht > H or Wt > W:
                start_h = (Ht - H)//2
                start_w = (Wt - W)//2
                out = out[:, :, start_h:start_h+H, start_w:start_w+W]

            if out.shape[2] != H or out.shape[3] != W:
                out = F.interpolate(out, size=(H,W), mode="nearest")
            return out
        out0 = crop_to(out0, hout, wout)
        out1 = crop_to(out1, hout, wout)
        out2 = crop_to(out2, hout, wout)

        return out0, out1, out2


# Convenience wrapper matching original API name
class AddShift_mp_module(nn.Module):
    def __init__(self, big_kernel, small_kernel, c_out, c_in, group_in):
        super().__init__()
        self._impl = PureAddShiftMP(big_kernel, small_kernel, c_out, c_in, group_in)

    def forward(self, x, b, hout, wout):
        return self._impl(x, b, hout, wout)

# ------------------- ShiftWiseConv2d -------------------
class SWConv2d(nn.Module):
    '''
    Using shift is equivalent to using a large convolution kernel.
    '''
    def __init__(self,
                in_channels,
                out_channels,
                big_kernel,
                small_kernel=3,
                stride=1,
                N_rep=4,
                ):
        super(SWConv2d, self).__init__()
        #===============khởi tạo chia ghost và rep =================
        self.kernels = (small_kernel, big_kernel)
        self.stride = stride
        # add same padding for vertical and horizon axis. should delete it accordingly
        padding, real_pad = self.shift(self.kernels)
        self.pad = padding, real_pad
        self.nk = math.ceil(big_kernel / small_kernel)
        #================sinh kênh ảo bằng depth wise==================
        out_n = in_channels * self.nk #kênh ảo để shift
        self.LoRA1 = nn.Conv2d(in_channels, out_n, kernel_size=small_kernel, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.LoRA2 = nn.Conv2d(in_channels, out_n, kernel_size=small_kernel, stride=stride, padding=padding, groups=in_channels, bias=False)
        #================================================================
        #==================Shift ========================================
        self.loras=AddShift_mp_module(big_kernel, small_kernel, in_channels, out_n, N_rep) #cần đầu ra chạy GPU
        # => sổ ra repN kênh X1 X2 X3
        #=====================Batchnorm ==================================
        self.bn_lora1 = nn.BatchNorm2d(in_channels)
        self.bn_lora2 = nn.BatchNorm2d(in_channels)
        self.bn_small = nn.BatchNorm2d(in_channels)
        #=======================================================
        self.cv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1,bias=False)
    def forward(self, inputs):
        ori_b, ori_c, ori_h, ori_w = inputs.shape
        LoRA1 = self.LoRA1(inputs)
        LoRA2 = self.LoRA2(inputs)
        LoRA = LoRA1 + LoRA2
        x1, x2, x3 = self.loras(LoRA, ori_b, ori_h, ori_w)
        lora1_x = self.bn_lora1(x1)
        lora2_x = self.bn_lora2(x2)
        small_x = self.bn_small(x3)
        x = lora1_x + lora2_x + small_x + inputs
        x=self.cv(x)
        return x
    
    def shift(self, kernels):
        '''
        We assume the conv does not change the feature map size, so padding = bigger_kernel_size//2. Otherwise,
        you may configure padding as you wish, and change the padding of small_conv accordingly.
        '''
        mink, maxk = min(kernels), max(kernels)
        nk = math.ceil(maxk / mink) 
        # 2. padding
        padding = mink -1  
        # 3. pads for each idx
        mid=maxk // 2
        real_pad=[]
        for i in range(nk): 
            extra_pad=mid-i*mink - padding 
            real_pad.append(extra_pad)
        return padding, real_pad

# ------------------- ShiftWiseConv -------------------
class SWConv(nn.Module):
    '''
    Using shift is equivalent to using a large convolution kernel.
    '''
    def __init__(self,
                in_channels,
                out_channels,
                big_kernel,
                small_kernel=3,
                stride=1,
                ghost_ratio=0.23,
                N_rep=4,
                ):
        super(SWConv, self).__init__()

        #===============khởi tạo chia ghost và rep =================
        self.kernels = (small_kernel, big_kernel)
        self.stride = stride
        
        # add same padding for vertical and horizon axis. should delete it accordingly
        padding, real_pad = self.shift(self.kernels)
        self.pad = padding, real_pad
        self.nk = math.ceil(big_kernel / small_kernel)
        
        # only part of input will using shift-wise chia tỷ lệ ghost
        repN=int(in_channels*(1-ghost_ratio))//2*2
        ghostN= in_channels - repN
        
        """trộn kênh"""
        np.random.seed(123)
        ghost = np.random.choice(in_channels,ghostN,replace=False).tolist()
        ghost.sort()
        rep=list(set(range(in_channels))-set(ghost))
        rep.sort()
        assert len(rep)==repN,f'len(rep):{len(rep)}==repN:{repN}'
        
        self.ghost=torch.IntTensor(ghost) #ngõ ra ghost xử lý cần CUDA
        self.rep=torch.IntTensor(rep) #ngõ ra rep xử lý cần CUDA
        #==============================================================
        
        #================sinh kênh ảo bằng depth wise==================
        out_n = repN * self.nk #kênh ảo để shift
        self.LoRA1 = nn.Conv2d(repN, out_n, kernel_size=small_kernel, stride=stride, padding=padding, groups=repN, bias=False)
        self.LoRA2 = nn.Conv2d(repN, out_n, kernel_size=small_kernel, stride=stride, padding=padding, groups=repN, bias=False)
        #================================================================

        #==================Shift ========================================
        self.loras=AddShift_mp_module(big_kernel, small_kernel, repN, out_n, N_rep) #cần đầu ra chạy GPU
        # => sổ ra repN kênh X1 X2 X3

        #=====================Batchnorm ==================================
        self.bn_lora1 = nn.BatchNorm2d(repN)
        self.bn_lora2 = nn.BatchNorm2d(repN)
        self.bn_small = nn.BatchNorm2d(repN)
        #=======================================================

        self.cv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1,bias=False)
    def forward(self, inputs):
        device = inputs.device
        self.ghost = self.ghost.to(device)
        self.rep   = self.rep.to(device)
        
        ori_b, ori_c, ori_h, ori_w = inputs.shape
        ghost_inputs = torch.index_select(inputs, dim=1, index=self.ghost)
        rep_inputs   = torch.index_select(inputs, dim=1, index=self.rep)
        
        LoRA1 = self.LoRA1(rep_inputs)
        LoRA2 = self.LoRA2(rep_inputs)
        LoRA = LoRA1 + LoRA2

        x1, x2, x3 = self.loras(LoRA, ori_b, ori_h, ori_w)

        lora1_x = self.bn_lora1(x1)
        lora2_x = self.bn_lora2(x2)
        small_x = self.bn_small(x3)

        x = lora1_x + lora2_x + small_x + rep_inputs

        x=self.cv(torch.cat([x,ghost_inputs],dim=1))
        return x
    
    def shift(self, kernels):
        '''
        We assume the conv does not change the feature map size, so padding = bigger_kernel_size//2. Otherwise,
        you may configure padding as you wish, and change the padding of small_conv accordingly.
        '''
        mink, maxk = min(kernels), max(kernels)
        nk = math.ceil(maxk / mink) 
        # 2. padding
        padding = mink -1  
        # 3. pads for each idx
        mid=maxk // 2
        real_pad=[]
        for i in range(nk): 
            extra_pad=mid-i*mink - padding 
            real_pad.append(extra_pad)
        return padding, real_pad

if __name__ =="__main__":
    x = torch.randn(1, 16, 32, 32)
    cae = SWConv(16, 16, 5, 3, 1, 0.23, 4)
    y = cae(x)
    print(y.shape)