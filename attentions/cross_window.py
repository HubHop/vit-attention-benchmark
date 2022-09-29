"""
    Paper: CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped
    Link: https://arxiv.org/abs/2107.00652
"""

import torch
import torch.nn as nn
import numpy as np

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x

    def flops(self, H, W):
        flops = 0
        N = H * W
        # q, k shape

        #  v, lepe = self.get_lepe(v, self.get_v)
        flops += conv_flops(H, W, self.get_v.in_channels, self.get_v.out_channels, kernel_size=self.get_v.kernel_size, stride=self.get_v.stride, padding=self.get_v.padding, groups=self.dim)

        num_windows = (H // self.H_sp) * (W // self.W_sp)
        num_tokes_per_windpow = self.H_sp * self.W_sp

        # Q@K and attn@v
        flops += num_windows * num_tokes_per_windpow * self.dim * num_tokes_per_windpow * 2

        return flops

class CrossWindowAttention(nn.Module):
    def __init__(self, dim, patch_resolution=14, branch_num=2, split_size=7, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.branch_num = branch_num
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attns = nn.ModuleList([
                LePEAttention(
                    dim//2, resolution=patch_resolution, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
                for i in range(branch_num)])
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        x1 = self.attns[0](qkv[:, :, :, :C // 2])
        x2 = self.attns[1](qkv[:, :, :, C // 2:])
        x = torch.cat([x1, x2], dim=2)
        x = self.proj(x)
        return x

    def flops(self, H, W):
        flops = 0
        N = H * W
        # qkv
        flops += N * self.dim * self.dim * 3
        # attention 1
        flops += self.attns[0].flops(H, W)
        flops += self.attns[1].flops(H, W)

        # projection
        flops += N * self.dim * self.dim

        return flops

if __name__ == '__main__':
    dim = 768
    num_heads = 12
    H = W = 14
    B = 64

    # special for CrossWindowAttention
    split_size = 7
    branch_num = 2

    model = CrossWindowAttention(dim, H,  branch_num=branch_num, split_size=split_size, num_heads=num_heads)

    from utils import measure_flops_params, measure_throughput_cpu, measure_throughput_gpu
    x = torch.randn(1, H*W, dim)
    measure_flops_params(model, x)
    measure_throughput_cpu(model)
    measure_throughput_gpu(model)
