"""
Paper: Visual Attention Network
Link: https://arxiv.org/abs/2202.09741
"""

import torch
import torch.nn as nn

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class VanAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dim = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x, H=14, W=14):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x.reshape(B, C, N).permute(0, 2, 1)

if __name__ == '__main__':
    dim = 768
    num_heads = 12
    H = W = 14
    B = 64

    model = VanAttention(d_model=dim)

    from utils import measure_flops_params, measure_throughput_cpu, measure_throughput_gpu

    x = torch.randn(1, H * W, dim)
    measure_flops_params(model, x)
    measure_throughput_cpu(model)
    measure_throughput_gpu(model)