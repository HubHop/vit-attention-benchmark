import torch
import time
from fvcore.nn import FlopCountAnalysis



def measure_flops_params(model, x):
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops = FlopCountAnalysis(model,  x)
    converted = flops.total() / 1e6
    print(f'Number of Params: {round(n_parameters/ 1e6, 2)} M')
    print(f'FLOPs = {round(converted, 2)} M')


def measure_throughput_gpu(model):
    H = W = 14
    B = 64
    x = torch.randn(B, H*W, model.dim).cuda()

    model = model.cuda()
    print(f"throughput averaged with 30 times")
    torch.cuda.synchronize()
    tic1 = time.time()
    for i in range(30):
        model(x)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {B} throughput on GPU {int(30 * B / (tic2 - tic1))}")


def measure_throughput_cpu(model):
    H = W = 14
    B = 64
    x = torch.randn(B, H*W, model.dim)
    print(f"throughput averaged with 30 times")
    tic1 = time.time()
    for i in range(30):
        model(x)
    tic2 = time.time()
    print(f"batch_size {B} throughput on CPU {int(30 * B / (tic2 - tic1))}")



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def get_relative_position_index(q_windows, k_windows):
    """
    Args:
        q_windows: tuple (query_window_height, query_window_width)
        k_windows: tuple (key_window_height, key_window_width)
    Returns:
        relative_position_index: query_window_height*query_window_width, key_window_height*key_window_width
    """
    # get pair-wise relative position index for each token inside the window
    coords_h_q = torch.arange(q_windows[0])
    coords_w_q = torch.arange(q_windows[1])
    coords_q = torch.stack(torch.meshgrid([coords_h_q, coords_w_q]))  # 2, Wh_q, Ww_q

    coords_h_k = torch.arange(k_windows[0])
    coords_w_k = torch.arange(k_windows[1])
    coords_k = torch.stack(torch.meshgrid([coords_h_k, coords_w_k]))  # 2, Wh, Ww

    coords_flatten_q = torch.flatten(coords_q, 1)  # 2, Wh_q*Ww_q
    coords_flatten_k = torch.flatten(coords_k, 1)  # 2, Wh_k*Ww_k

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  # 2, Wh_q*Ww_q, Wh_k*Ww_k
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh_q*Ww_q, Wh_k*Ww_k, 2
    relative_coords[:, :, 0] += k_windows[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += k_windows[1] - 1
    relative_coords[:, :, 0] *= (q_windows[1] + k_windows[1]) - 1
    relative_position_index = relative_coords.sum(-1)  #  Wh_q*Ww_q, Wh_k*Ww_k
    return relative_position_index

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_partition_noreshape(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows


def conv_flops(H, W, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    batch_size = 1
    output_dims = [H, W]

    kernel_dims = list(kernel_size)
    in_channels = in_channels
    out_channels = out_channels
    groups = groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
                              in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)