"""
    Paper: HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions
    Link: https://arxiv.org/abs/2207.14284
"""

import torch
import torch.nn as nn

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dim = dim
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        # print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, H=14, W=14):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x.reshape(B, C, N).permute(0, 2, 1)



if __name__ == '__main__':
    dim = 768
    num_heads = 12
    H = W = 14
    B = 64

    # special for HorNet at 1/16 scale
    order = 4
    s = 1.0/3.0
    gflayer = None
    model = gnconv(dim=dim, order=order, gflayer=gflayer, s=s)

    from utils import measure_flops_params, measure_throughput_cpu, measure_throughput_gpu

    x = torch.randn(1, H * W, dim)
    measure_flops_params(model, x)
    measure_throughput_cpu(model)
    measure_throughput_gpu(model)