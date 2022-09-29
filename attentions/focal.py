"""
    Paper: Focal Self-attention for Local-Global Interactions in Vision Transformers
    Link: https://arxiv.org/abs/2107.00641
"""

import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_
import math
from utils import window_reverse, window_partition, window_partition_noreshape, get_relative_position_index


class FocalWindowAttention(nn.Module):
    r""" Focal Attention

    Args:
        dim (int): Number of input channels.
        expand_size (int): The expand size at focal level 1.
        window_size (tuple[int]): The height and width of the window.
        focal_window (int): Focal region size.
        focal_level (int): Focal attention level.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pool_method (str): window pooling method. Default: none
    """

    def __init__(self, dim, expand_size=3, window_size=7, focal_window=3, focal_level=2, num_heads=12,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pool_method="none", shift_size=0):

        super().__init__()


        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        # from block-wise setting
        self.shift_size = shift_size
        self.window_size_glo = self.window_size
        self.pool_layers = nn.ModuleList()
        if self.pool_method != "none":
            for k in range(self.focal_level - 1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                if self.pool_method == "fc":
                    self.pool_layers.append(nn.Linear(window_size_glo * window_size_glo, 1))
                    self.pool_layers[-1].weight.data.fill_(1. / (window_size_glo * window_size_glo))
                    self.pool_layers[-1].bias.data.fill_(0)
                elif self.pool_method == "conv":
                    self.pool_layers.append(
                        nn.Conv2d(dim, dim, kernel_size=window_size_glo, stride=window_size_glo, groups=dim))

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        # define a parameter table of relative position bias for each window
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        if self.expand_size > 0 and focal_level > 0:
            # define a parameter table of position bias between window and its fine-grained surroundings
            self.window_size_of_key = self.window_size * self.window_size if self.expand_size == 0 else \
                (4 * self.window_size * self.window_size - 4 * (self.window_size - self.expand_size) * (
                            self.window_size - self.expand_size))
            self.relative_position_bias_table_to_neighbors = nn.Parameter(
                torch.zeros(1, num_heads, self.window_size * self.window_size,
                            self.window_size_of_key))  # Wh*Ww, nH, nSurrounding
            trunc_normal_(self.relative_position_bias_table_to_neighbors, std=.02)

            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size, self.window_size);
            mask_tl[:-self.expand_size, :-self.expand_size] = 0
            mask_tr = torch.ones(self.window_size, self.window_size);
            mask_tr[:-self.expand_size, self.expand_size:] = 0
            mask_bl = torch.ones(self.window_size, self.window_size);
            mask_bl[self.expand_size:, :-self.expand_size] = 0
            mask_br = torch.ones(self.window_size, self.window_size);
            mask_br[self.expand_size:, self.expand_size:] = 0
            mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", mask_rolled.nonzero().view(-1))

        if pool_method != "none" and focal_level > 1:
            self.relative_position_bias_table_to_windows = nn.ParameterList()
            self.unfolds = nn.ModuleList()

            # build relative position bias between local patch and pooled windows
            for k in range(focal_level - 1):
                stride = 2 ** k
                kernel_size = 2 * (self.focal_window // 2) + 2 ** k + (2 ** k - 1)
                # define unfolding operations
                self.unfolds += [nn.Unfold(
                    kernel_size=(kernel_size, kernel_size),
                    stride=stride, padding=kernel_size // 2)
                ]

                # define relative position bias table
                relative_position_bias_table_to_windows = nn.Parameter(
                    torch.zeros(
                        self.num_heads,
                        (self.window_size + self.focal_window + 2 ** k - 2) * (
                                    self.window_size + self.focal_window + 2 ** k - 2),
                    )
                )
                trunc_normal_(relative_position_bias_table_to_windows, std=.02)
                self.relative_position_bias_table_to_windows.append(relative_position_bias_table_to_windows)

                # define relative position bias index
                relative_position_index_k = get_relative_position_index((self.window_size, self.window_size),
                                                                        to_2tuple(self.focal_window + 2 ** k - 1))
                self.register_buffer("relative_position_index_{}".format(k), relative_position_index_k)

                # define unfolding index for focal_level > 0
                if k > 0:
                    mask = torch.zeros(kernel_size, kernel_size);
                    mask[(2 ** k) - 1:, (2 ** k) - 1:] = 1
                    self.register_buffer("valid_ind_unfold_{}".format(k), mask.flatten(0).nonzero().view(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def to_windows(self, x, H, W):
        # H, W = self.input_resolution
        B, L, C = x.shape
        if min(H, W) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.expand_size = 0
            self.shift_size = 0
            self.window_size = min(H, W)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        B, H, W, C = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows_all = [shifted_x]
        x_window_masks_all = [self.attn_mask]

        if self.focal_level > 1 and self.pool_method != "none":
            # if we add coarser granularity and the pool method is not none
            for k in range(self.focal_level - 1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                pooled_h = math.ceil(H / self.window_size) * (2 ** k)
                pooled_w = math.ceil(W / self.window_size) * (2 ** k)
                H_pool = pooled_h * window_size_glo
                W_pool = pooled_w * window_size_glo

                x_level_k = shifted_x
                # trim or pad shifted_x depending on the required size
                if H > H_pool:
                    trim_t = (H - H_pool) // 2
                    trim_b = H - H_pool - trim_t
                    x_level_k = x_level_k[:, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = (H_pool - H) // 2
                    pad_b = H_pool - H - pad_t
                    x_level_k = F.pad(x_level_k, (0, 0, 0, 0, pad_t, pad_b))

                if W > W_pool:
                    trim_l = (W - W_pool) // 2
                    trim_r = W - W_pool - trim_l
                    x_level_k = x_level_k[:, :, trim_l:-trim_r]
                elif W < W_pool:
                    pad_l = (W_pool - W) // 2
                    pad_r = W_pool - W - pad_l
                    x_level_k = F.pad(x_level_k, (0, 0, pad_l, pad_r))

                x_windows_noreshape = window_partition_noreshape(x_level_k.contiguous(),
                                                                 window_size_glo)  # B, nw, nw, window_size, window_size, C
                nWh, nWw = x_windows_noreshape.shape[1:3]
                if self.pool_method == "mean":
                    x_windows_pooled = x_windows_noreshape.mean([3, 4])  # B, nWh, nWw, C
                elif self.pool_method == "max":
                    x_windows_pooled = x_windows_noreshape.max(-2)[0].max(-2)[0].view(B, nWh, nWw,
                                                                                      C)  # B, nWh, nWw, C
                elif self.pool_method == "fc":
                    x_windows_noreshape = x_windows_noreshape.view(B, nWh, nWw, window_size_glo * window_size_glo,
                                                                   C).transpose(3, 4)  # B, nWh, nWw, C, wsize**2
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).flatten(
                        -2)  # B, nWh, nWw, C
                elif self.pool_method == "conv":
                    x_windows_noreshape = x_windows_noreshape.view(-1, window_size_glo, window_size_glo, C).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2).contiguous()  # B * nw * nw, C, wsize, wsize
                    x_windows_pooled = self.pool_layers[k](x_windows_noreshape).view(B, nWh, nWw,
                                                                                     C)  # B, nWh, nWw, C

                x_windows_all += [x_windows_pooled]
                x_window_masks_all += [None]
        return x_windows_all, x_window_masks_all


    def forward(self, x, H=14, W=14):
        """
        Args:
            x_all (list[Tensors]): input features at different granularity
            mask_all (list[Tensors/None]): masks for input features at different granularity
        """
        x_all, mask_all = self.to_windows(x, H, W)
        x = x_all[0]  #

        B, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, nH, nW, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nH, nW, C

        # partition q map
        (q_windows, k_windows, v_windows) = map(
            lambda t: window_partition(t, self.window_size).view(
                -1, self.window_size * self.window_size, self.num_heads, C // self.num_heads
            ).transpose(1, 2),
            (q, k, v)
        )

        if self.expand_size > 0 and self.focal_level > 0:
            (k_tl, v_tl) = map(
                lambda t: torch.roll(t, shifts=(-self.expand_size, -self.expand_size), dims=(1, 2)), (k, v)
            )
            (k_tr, v_tr) = map(
                lambda t: torch.roll(t, shifts=(-self.expand_size, self.expand_size), dims=(1, 2)), (k, v)
            )
            (k_bl, v_bl) = map(
                lambda t: torch.roll(t, shifts=(self.expand_size, -self.expand_size), dims=(1, 2)), (k, v)
            )
            (k_br, v_br) = map(
                lambda t: torch.roll(t, shifts=(self.expand_size, self.expand_size), dims=(1, 2)), (k, v)
            )

            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(-1, self.window_size * self.window_size,
                                                                        self.num_heads, C // self.num_heads),
                (k_tl, k_tr, k_bl, k_br)
            )
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(-1, self.window_size * self.window_size,
                                                                        self.num_heads, C // self.num_heads),
                (v_tl, v_tr, v_bl, v_br)
            )
            k_rolled = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), 1).transpose(1, 2)
            v_rolled = torch.cat((v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), 1).transpose(1, 2)

            # mask out tokens in current window
            k_rolled = k_rolled[:, :, self.valid_ind_rolled]
            v_rolled = v_rolled[:, :, self.valid_ind_rolled]
            k_rolled = torch.cat((k_windows, k_rolled), 2)
            v_rolled = torch.cat((v_windows, v_rolled), 2)
        else:
            k_rolled = k_windows;
            v_rolled = v_windows;

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            for k in range(self.focal_level - 1):
                stride = 2 ** k
                x_window_pooled = x_all[k + 1]  # B, nWh, nWw, C
                nWh, nWw = x_window_pooled.shape[1:3]

                # generate mask for pooled windows
                mask = x_window_pooled.new(nWh, nWw).fill_(1)
                unfolded_mask = self.unfolds[k](mask.unsqueeze(0).unsqueeze(1)).view(
                    1, 1, self.unfolds[k].kernel_size[0], self.unfolds[k].kernel_size[1], -1).permute(0, 4, 2, 3,
                                                                                                      1).contiguous(). \
                    view(nWh * nWw // stride // stride, -1, 1)

                if k > 0:
                    valid_ind_unfold_k = getattr(self, "valid_ind_unfold_{}".format(k))
                    unfolded_mask = unfolded_mask[:, valid_ind_unfold_k]

                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                x_window_masks = x_window_masks.masked_fill(x_window_masks == 0, float(-100.0)).masked_fill(
                    x_window_masks > 0, float(0.0))
                mask_all[k + 1] = x_window_masks

                # generate k and v for pooled windows
                qkv_pooled = self.qkv(x_window_pooled).reshape(B, nWh, nWw, 3, C).permute(3, 0, 4, 1, 2).contiguous()
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]  # B, C, nWh, nWw

                (k_pooled_k, v_pooled_k) = map(
                    lambda t: self.unfolds[k](t).view(
                        B, C, self.unfolds[k].kernel_size[0], self.unfolds[k].kernel_size[1], -1).permute(0, 4, 2, 3,
                                                                                                          1).contiguous(). \
                        view(-1, self.unfolds[k].kernel_size[0] * self.unfolds[k].kernel_size[1], self.num_heads,
                             C // self.num_heads).transpose(1, 2),
                    (k_pooled_k, v_pooled_k)  # (B x (nH*nW)) x nHeads x (unfold_wsize x unfold_wsize) x head_dim
                )

                if k > 0:
                    (k_pooled_k, v_pooled_k) = map(
                        lambda t: t[:, :, valid_ind_unfold_k], (k_pooled_k, v_pooled_k)
                    )

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k]
            k_all = torch.cat([k_rolled] + k_pooled, 2)
            v_all = torch.cat([v_rolled] + v_pooled, 2)
        else:
            k_all = k_rolled
            v_all = v_rolled

        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        attn = (q_windows @ k_all.transpose(-2,
                                            -1))  # B*nW, nHead, window_size*window_size, focal_window_size*focal_window_size

        window_area = self.window_size * self.window_size
        window_area_rolled = k_rolled.shape[2]

        # add relative position bias for tokens inside window
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn[:, :, :window_area, :window_area] = attn[:, :, :window_area,
                                                 :window_area] + relative_position_bias.unsqueeze(0)

        # add relative position bias for patches inside a window
        if self.expand_size > 0 and self.focal_level > 0:
            attn[:, :, :window_area, window_area:window_area_rolled] = attn[:, :, :window_area,
                                                                       window_area:window_area_rolled] + self.relative_position_bias_table_to_neighbors

        if self.pool_method != "none" and self.focal_level > 1:
            # # add relative position bias for different windows in an image
            offset = window_area_rolled
            for k in range(self.focal_level - 1):
                # add relative position bias
                relative_position_index_k = getattr(self, 'relative_position_index_{}'.format(k))
                relative_position_bias_to_windows = self.relative_position_bias_table_to_windows[k][:,
                                                    relative_position_index_k.view(-1)].view(
                    -1, self.window_size * self.window_size, (self.focal_window + 2 ** k - 1) ** 2,
                )  # nH, NWh*NWw,focal_region*focal_region
                attn[:, :, :window_area, offset:(offset + (self.focal_window + 2 ** k - 1) ** 2)] = \
                    attn[:, :, :window_area, offset:(offset + (
                                self.focal_window + 2 ** k - 1) ** 2)] + relative_position_bias_to_windows.unsqueeze(0)
                # add attentional mask
                if mask_all[k + 1] is not None:
                    attn[:, :, :window_area, offset:(offset + (self.focal_window + 2 ** k - 1) ** 2)] = \
                        attn[:, :, :window_area, offset:(offset + (self.focal_window + 2 ** k - 1) ** 2)] + \
                        mask_all[k + 1][:, :, None, None, :].repeat(attn.shape[0] // mask_all[k + 1].shape[1], 1, 1, 1,
                                                                    1).view(-1, 1, 1, mask_all[k + 1].shape[-1])

                offset += (self.focal_window + 2 ** k - 1) ** 2

        if mask_all[0] is not None:
            nW = mask_all[0].shape[0]
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads, window_area, N)
            attn[:, :, :, :, :window_area] = attn[:, :, :, :, :window_area] + mask_all[0][None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, window_area, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x[:, :self.window_size ** 2]

        # merge windows
        x = x.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(x, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x[:, :H, :W].contiguous().view(B, -1, C)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N, window_size, unfold_size):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        if self.pool_method != "none" and self.focal_level > 1:
            flops += self.num_heads * N * (self.dim // self.num_heads) * (unfold_size * unfold_size)
        if self.expand_size > 0 and self.focal_level > 0:
            flops += self.num_heads * N * (self.dim // self.num_heads) * (
                        (window_size + 2 * self.expand_size) ** 2 - window_size ** 2)

            #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        if self.pool_method != "none" and self.focal_level > 1:
            flops += self.num_heads * N * (self.dim // self.num_heads) * (unfold_size * unfold_size)
        if self.expand_size > 0 and self.focal_level > 0:
            flops += self.num_heads * N * (self.dim // self.num_heads) * (
                        (window_size + 2 * self.expand_size) ** 2 - window_size ** 2)

            # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


if __name__ == '__main__':
    dim = 768
    num_heads = 12
    H = W = 14
    B = 64

    # setting for 1/16 feature map
    expand_size = 3
    focal_level = 2
    window_size = 7
    focal_window = 3
    pool_method = 'fc'

    model = FocalWindowAttention(
            dim, expand_size=expand_size, window_size=window_size,
            focal_window=focal_window, focal_level=focal_level, num_heads=num_heads, pool_method=pool_method)

    from utils import measure_flops_params, measure_throughput_cpu, measure_throughput_gpu

    x = torch.randn(1, H * W, dim)
    measure_flops_params(model, x)
    measure_throughput_cpu(model)
    measure_throughput_gpu(model)


