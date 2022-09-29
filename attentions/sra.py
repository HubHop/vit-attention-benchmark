"""
    Paper: Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
    Link: https://arxiv.org/abs/2102.12122
"""

import torch
import torch.nn as nn
from utils import conv_flops

class SRAttention(nn.Module):
    """
    Spatial Reduction Attention

    Paper: Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
    Link: https://arxiv.org/abs/2102.12122
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H=14, W=14):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def flops(self, N):
        flops = 0
        # q
        flops += N * self.dim * self.dim
        H = int(N ** 0.5)
        kv_len = (H // self.sr_ratio) ** 2
        # spatial reduction
        flops += conv_flops(self.sr_ratio, self.dim, self.dim, self.sr_ratio, 0, H)
        # norm
        flops += self.dim * kv_len

        # kv
        flops += kv_len * self.dim * self.dim * 2
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * kv_len
        #  x = (attn @ v)
        flops += self.num_heads * N * kv_len * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

if __name__ == '__main__':
    dim = 768
    num_heads = 12
    H = W = 14
    B = 128

    # special
    sr_ratio = 2

    model = SRAttention(dim=dim, num_heads=num_heads, qkv_bias=True, sr_ratio=sr_ratio)

    from utils import measure_flops_params, measure_throughput_cpu, measure_throughput_gpu

    x = torch.randn(1, H * W, dim)
    measure_flops_params(model, x)
    measure_throughput_cpu(model)
    measure_throughput_gpu(model)