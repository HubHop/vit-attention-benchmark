"""
    Paper: QuadTree Attention for Vision Transformers
    Link: https://arxiv.org/abs/2201.02767
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

import sys
sys.path.append("./QuadTreeAttention")
from QuadtreeAttention.modules.quadtree_attention import QTAttA, QTAttB
from einops.einops import rearrange

class QTAttAPytorch(nn.Module):
    def __init__(self, nhead, dim, scale, use_dropout=False, attention_dropout=0.1, topk=1):
        super().__init__()
        self.use_dropout = use_dropout
        self.topk = topk
        self.nhead = nhead
        self.dim = dim

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Quadtree attention
        Args:
            query: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        bs = queries[0].shape[0]
        # Compute the unnormalized attention and apply the masks
        message = 0
        for i, (query, key, value) in enumerate(zip(reversed(queries), reversed(keys), reversed(values))):
            bs, c, h, w = key.shape
            key = rearrange(key, "b c h w -> b (h w) c").view(bs, -1, self.nhead, self.dim)  # [N, S, H, D]
            value = rearrange(value, "b c h w -> b (h w) c").view(bs, -1, self.nhead, self.dim)  # [N, S, H, D]
            value_old = value
            if i == 0:
                query = rearrange(query, "b c h w -> b (h w) c").view(bs, -1, self.nhead, self.dim)
                QK = torch.einsum("nlhd,nshd->nlsh", query, key)
            else:
                query = query.view(bs, c, h // 2, 2, w // 2, 2)
                query = rearrange(query, "b c h t1 w t2-> b (h w) (t1 t2) c ").view(bs, -1, 4, self.nhead, self.dim)
                topk_pos = topk_pos * 2
                key_gather = []
                value_gather = []
                idx_gather = []

                for x in [0, 1]:
                    for y in [0, 1]:
                        idx = (topk_pos[0] + x) * w + topk_pos[1] + y  # convert to index

                        idx_gather.append(idx)
                        idx = idx.view(bs, -1, self.nhead, 1).repeat(1, 1, 1, self.dim)  # [N, L, K, H, D]

                        k = torch.gather(key, index=idx, dim=1).view(bs, -1, self.topk, self.nhead, self.dim)

                        v = torch.gather(value, index=idx, dim=1).view(bs, -1, self.topk, self.nhead, self.dim)
                        key_gather.append(k)
                        value_gather.append(v)
                idx = torch.stack(idx_gather, dim=3)  # [N, L, K, 4, H, D]

                # query: [b, N, 4, H, D]
                # key: [b, 4N, H, D]
                # idx: [b, N, 4K, H, D]
                # QK: [b, N, 4, K, 4, H]

                key_gather = torch.stack(key_gather, dim=3)  # [N, L, K, 4, H, D]
                value = torch.stack(value_gather, dim=3)  # [N, L, K, 4, H, D]

                QK = torch.einsum("nlwhd,nlkfhd->nlwkfh", query, key_gather)

            softmax_temp = 1.0 / self.dim ** 0.5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=-2)  # [N, L//scale**i, K, 4, H]

            if i != 0:
                # A: [b, N, 4, K, 4, H]
                # topk_score: [b, N, 1, K, 1, H]
                # return: A, [b, N, 4, K*4, H]
                A = (A * topk_score.unsqueeze(-2).unsqueeze(2)).reshape(bs, -1, 4, self.topk * 4, self.nhead)
                # value: [b, N, K, 4, H, D]
                value = value.reshape(bs, -1, self.topk * 4, self.nhead, self.dim)  # [N, L, K*4, H, D]

            topk_score, topk_idx = torch.topk(A, dim=-2, k=self.topk, largest=True)  # [N, L, 4, K, H]

            mask = torch.ones_like(A)
            if i != len(keys) - 1:
                mask = mask.scatter(dim=-2, index=topk_idx, src=torch.zeros_like(topk_idx).float())
            if i == 0:
                message += torch.einsum("nlsh,nshd->nlhd", A * mask, value)
                # [b, N, H, D]
            else:
                # A: [b, N, 4, 4K, H]
                # value: [b, N, 4K, H, D]
                # message: [b, N, 4, H, D]
                new_message = torch.einsum("nlwkh,nlkhd->nlwhd", A * mask, value)

                idx = idx.view(bs, -1, 1, self.topk * 4, self.nhead).repeat(1, 1, 4, 1, 1)  # [N, L,4, K*4, H]
                # A: [b, N, 4, K*4, H]
                # value_old: [b, 4N, H, D]
                # idx: [b, N, 4, 4K, H]

                message = message.unsqueeze(2) + new_message
                message = message.view(bs, h // 2, w // 2, 2, 2, self.nhead, self.dim)
                message = rearrange(message, "b h w t1 t2 nh d -> b (h t1 w t2) nh d")  # reshape

                topk_idx = torch.gather(idx, index=topk_idx, dim=-2)

                topk_idx = topk_idx.view(bs, h // 2, w // 2, 2, 2, self.topk, self.nhead)

                topk_idx = rearrange(topk_idx, "b h w t1 t2 k nh -> b (h t1 w t2) k nh")  # reshape back

                topk_score = topk_score.view(bs, h // 2, w // 2, 2, 2, self.topk, self.nhead)
                topk_score = rearrange(topk_score, "b h w t1 t2 k nh -> b (h t1 w t2) k nh")  # reshape back

            topk_pos = torch.stack([topk_idx // w, topk_idx % w])  # convert to coordinate

        return message
class QTAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1, attn_type=None, cpu=False
    ):
        self.attn_type = attn_type
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)

        if attn_type == "B":
            if cpu:
                self.py_att = QTAttAPytorch(num_heads, dim // num_heads, scale=sr_ratio, topk=8)
            else:
                self.py_att = QTAttB(num_heads, dim // num_heads, scale=sr_ratio, topks=[8, 8, 8, 8], lepe=True)
            self.value_branchs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.GroupNorm(1, dim),
                        nn.GELU(),
                        nn.Conv2d(dim, dim, kernel_size=2, stride=2),
                        nn.GroupNorm(1, dim),
                        nn.GELU(),
                        nn.Conv2d(dim, dim, kernel_size=1, stride=1),
                    )
                    for i in range(sr_ratio - 1)
                ]
            )
        else:
            self.py_att = QTAttA(num_heads, dim // num_heads, topks=[8, 8, 8, 8])

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            trunc_normal_(m.weight, std=0.02)
            m.init = True
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H=14, W=14):

        B, N, C = x.shape
        # H = W = int(N ** 0.5)
        y = x
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        target = x
        keys = []
        values = []
        queries = []

        q = self.q_proj(x)
        k = self.k_proj(target)
        v = self.v_proj(target)

        for i in range(self.sr_ratio):
            keys.append(k.float())
            values.append(v.float())
            queries.append(q.float())
            if i != self.sr_ratio - 1:
                k = F.avg_pool2d(k, kernel_size=2, stride=2)
                q = F.avg_pool2d(q, kernel_size=2, stride=2)
                if self.attn_type == "B":
                    v = self.value_branchs[i](v)
                else:
                    v = F.avg_pool2d(v, kernel_size=2, stride=2)

        msg = self.py_att(queries, keys, values).contiguous().view(B, -1, C)
        x = self.proj(msg)
        x = self.proj_drop(x)

        return x

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

if __name__ == '__main__':
    dim = 768
    num_heads = 12
    H = W = 14
    B = 64

    # special for QTAttention
    sr_ratio = 2
    attn_type = "B"
    from utils import measure_flops_params, measure_throughput_cpu, measure_throughput_gpu
    cpu = True

    if cpu:
        model = QTAttention(dim, num_heads=num_heads, qkv_bias=True, qk_scale=None, sr_ratio=sr_ratio,
                            attn_type=attn_type)
        x = torch.randn(1, H * W, dim)
        measure_flops_params(model, x)
        measure_throughput_cpu(model)
    else:
        model = QTAttention(dim, num_heads=num_heads, qkv_bias=True, qk_scale=None, sr_ratio=sr_ratio,
                            attn_type=attn_type, cpu=cpu)
        x = torch.randn(1, H * W, dim)
        measure_flops_params(model, x)
        measure_throughput_gpu(model)