# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn
from ..builder import BACKBONES
import torch
import math
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import ssd_selective_scan

def init_conv(conv):
    """Initialize convolution layers."""
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


@BACKBONES.register_module()
class SECONDMamba(BaseModule):
    def __init__(self):
        super(SECONDMamba, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                SECONDMambaBlock(use_conv=False,d_model=256,H=180,W=180),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.SyncBatchNorm(256, eps=0.001, momentum=0.01),
                nn.ReLU()),
            nn.Sequential(
                SECONDMambaBlock(use_conv=True,d_model=256,H=180,W=180),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.SyncBatchNorm(256, eps=0.001, momentum=0.01, ),
                nn.ReLU()),
            nn.Sequential(
                nn.ConvTranspose2d(256,256,2,2,bias=False),
                nn.SyncBatchNorm(256, eps=0.001, momentum=0.01),
                nn.ReLU())])
        self.norm = nn.SyncBatchNorm(256, eps=0.001, momentum=0.01)

    def forward(self, x):
        outs = []
        x1 = self.block[0](x)
        x2 = self.block[1](x1)
        x3 = self.block[2](x2)
        x4 = self.norm(x1+x3)
        outs.append(x1)
        outs.append(x4)
        return tuple(outs)


class SECONDMambaBlock(nn.Module):
    def __init__(self,
                 channel_reduct=False,
                 channel_increase=False,
                 use_conv=False,
                 d_model=256,
                 d_state=4,
                 headdim=64,
                 A_init_range=(1, 16),
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init_floor=1e-4,
                 dt_limit=(0.0, float("inf")),
                 bias=False,
                 chunk_size=256,
                 device=None,
                 dtype=None,
                 H=180,
                 W=180):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = self.d_inner = self.d_ssm = d_model
        self.d_state = d_state
        self.headdim = headdim
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.d_in_proj = d_in_proj
        self.in_proj_H = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.in_proj_V = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.act = nn.SiLU()
        dt_H = torch.clamp(torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)), min=dt_init_floor)
        inv_dt_H = dt_H + torch.log(-torch.expm1(-dt_H))
        dt_V = torch.clamp(torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)), min=dt_init_floor)
        inv_dt_V = dt_V + torch.log(-torch.expm1(-dt_V))
        self.dt_bias_H = nn.Parameter(inv_dt_H)
        self.dt_bias_V = nn.Parameter(inv_dt_V)
        self.dt_bias_H._no_weight_decay = True
        self.dt_bias_V._no_weight_decay = True
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A_H = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_V = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log_H = torch.log(A_H).to(dtype=dtype)
        A_log_V = torch.log(A_V).to(dtype=dtype)

        self.A_log_H = nn.Parameter(A_log_H)
        self.A_log_V = nn.Parameter(A_log_V)
        self.A_log_H._no_weight_decay = True
        self.A_log_V._no_weight_decay = True
        self.D_H = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_V = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_H._no_weight_decay = True
        self.D_V._no_weight_decay = True
        assert not (channel_reduct and channel_increase), "channel_reduct and channel_increase cannot both be True"
        if channel_reduct:
            self.out_proj = nn.Linear(self.d_model*2, self.d_model//2, bias=bias, **factory_kwargs)
        elif channel_increase:
            self.out_proj = nn.Linear(self.d_model*2, self.d_model*2, bias=bias, **factory_kwargs)
        else:
            self.out_proj = nn.Linear(self.d_model*2, self.d_model, bias=bias, **factory_kwargs)
        self.out_norm = nn.LayerNorm(self.d_model, eps=1e-6, **factory_kwargs)
        self.use_conv=use_conv
        self.channel_reduct=channel_reduct
        self.channel_increase=channel_increase
        if self.use_conv:
            self.conv1d_H = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False)
            self.conv1d_V = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False)
            init_conv(self.conv1d_H)
            init_conv(self.conv1d_V)
            mask_low_resolution = torch.ones(1, 1, H//2, W//2, device=device if device is not None else 'cpu')
            morton_H_indices_low, morton_V_indices_low = self.morton_code_extraction(mask_low_resolution)
            inverse_H_indices_low = torch.empty_like(morton_H_indices_low)
            inverse_H_indices_low[morton_H_indices_low] = torch.arange(morton_H_indices_low.size(0), device=morton_H_indices_low.device)
            inverse_V_indices_low = torch.empty_like(morton_V_indices_low)
            inverse_V_indices_low[morton_V_indices_low] = torch.arange(morton_V_indices_low.size(0), device=morton_V_indices_low.device)
            self.morton_H_indices_low = morton_H_indices_low
            self.morton_V_indices_low = morton_V_indices_low
            self.inverse_H_indices_low = inverse_H_indices_low
            self.inverse_V_indices_low = inverse_V_indices_low
        # >>> 新增部分: 计算并保存morton和inverse索引 >>>
        self.H = H
        self.W = W
        mask = torch.ones(1, 1, H, W, device=device if device is not None else 'cpu')
        morton_H_indices, morton_V_indices = self.morton_code_extraction(mask)
        inverse_H_indices = torch.empty_like(morton_H_indices)
        inverse_H_indices[morton_H_indices] = torch.arange(morton_H_indices.size(0), device=morton_H_indices.device)
        inverse_V_indices = torch.empty_like(morton_V_indices)
        inverse_V_indices[morton_V_indices] = torch.arange(morton_V_indices.size(0), device=morton_V_indices.device)
        self.morton_H_indices = morton_H_indices
        self.morton_V_indices = morton_V_indices
        self.inverse_H_indices = inverse_H_indices
        self.inverse_V_indices = inverse_V_indices


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_conv(m)

    def forward(self, x):
        B, C, H, W = x.shape
        # 直接使用self保存的索引
        morton_H_indices = self.morton_H_indices.to(x.device)
        morton_V_indices = self.morton_V_indices.to(x.device)
        # morton_H_indices_low = self.morton_H_indices_low.to(x.device)
        # morton_V_indices_low = self.morton_V_indices_low.to(x.device)
        inverse_H_indices = self.inverse_H_indices.to(x.device)
        inverse_V_indices = self.inverse_V_indices.to(x.device)
        if self.use_conv:
            inverse_H_indices_low = self.inverse_H_indices_low.to(x.device)
            inverse_V_indices_low = self.inverse_V_indices_low.to(x.device)
        x_flat = x.view(B, C, -1)
        x_morton_H = x_flat[:, :, morton_H_indices].permute(0,2,1)
        x_morton_V = x_flat[:, :, morton_V_indices].permute(0,2,1)
        zxbcdt_H = self.in_proj_H(x_morton_H)
        zxbcdt_V = self.in_proj_V(x_morton_V)
        A_H = -torch.exp(self.A_log_H.float())
        A_V = -torch.exp(self.A_log_V.float())

        dim = self.d_ssm

        z_H, xBC_H, dt_H = torch.split(zxbcdt_H, [dim, dim + 2 * self.d_state, self.nheads], dim=-1)
        xBC_H = self.act(xBC_H)
        x_H, B_H, C_H = torch.split(xBC_H, [dim, self.d_state, self.d_state], dim=-1)
        x_H = rearrange(x_H, "b l (h p) -> b l h p", h=self.nheads)
        B_H = rearrange(B_H, "b l (g n) -> b l g n", g=1)
        C_H = rearrange(C_H, "b l (g n) -> b l g n", g=1)
        z_H = rearrange(z_H, "b l (h p) -> b l h p", h=self.nheads)
        out_H = ssd_selective_scan(x_H, dt_H.to(x_H.dtype), A_H, B_H, C_H, D=self.D_H.float(), z=z_H, dt_bias=self.dt_bias_H, dt_softplus=True, dt_limit=self.dt_limit)
        out_H = rearrange(out_H, "b s h p -> b s (h p)")

        z_V, xBC_V, dt_V = torch.split(zxbcdt_V, [dim, dim + 2 * self.d_state, self.nheads], dim=-1)
        xBC_V = self.act(xBC_V)
        x_V, B_V, C_V = torch.split(xBC_V, [dim, self.d_state, self.d_state], dim=-1)
        x_V = rearrange(x_V, "b l (h p) -> b l h p", h=self.nheads)
        B_V = rearrange(B_V, "b l (g n) -> b l g n", g=1)
        C_V = rearrange(C_V, "b l (g n) -> b l g n", g=1)
        z_V = rearrange(z_V, "b l (h p) -> b l h p", h=self.nheads)
        out_V = ssd_selective_scan(x_V, dt_V.to(x_V.dtype), A_V, B_V, C_V, D=self.D_V.float(), z=z_V, dt_bias=self.dt_bias_V, dt_softplus=True, dt_limit=self.dt_limit)
        out_V = rearrange(out_V, "b s h p -> b s (h p)")

        if self.use_conv:
            out_H = self.conv1d_H(out_H.permute(0, 2, 1)).permute(0, 2, 1)
            out_V = self.conv1d_V(out_V.permute(0, 2, 1)).permute(0, 2, 1)
            out_H_inverse = out_H[:,inverse_H_indices_low,:]
            out_V_inverse = out_V[:,inverse_V_indices_low,:]
        else:
            out_H_inverse = out_H[:,inverse_H_indices,:]
            out_V_inverse = out_V[:,inverse_V_indices,:]
        out = self.out_proj(torch.cat([out_H_inverse, out_V_inverse], dim=-1).contiguous())
        out = self.out_norm(out).permute(0,2,1)
        if self.channel_reduct:
            out_C = C//2
        elif self.channel_increase:
            out_C = C*2
        else:
            out_C = C
        if self.use_conv:
            out = out.view(B,out_C,H//2,W//2)
        else:
            out = out.view(B,out_C,H,W)
        return out
    
    def morton_code_extraction(self, mask: torch.Tensor) -> tuple:
        device = mask.device
        h, w = mask[0][0].shape
        row_indices, col_indices = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')    # row_indices[2,3]的值为2，表示该位置的行索引为2，col_indices[2,3]的值为3，表示该位置的列索引为3
        row_indices = row_indices.flatten()
        col_indices = col_indices.flatten()
        valid_indices = mask[0][0].flatten() != 0
        row_indices = row_indices[valid_indices]
        col_indices = col_indices[valid_indices]
        morton_codes_1 = self.interleave_bits(col_indices, row_indices)
        morton_codes_2 = self.interleave_bits_x_last(col_indices, row_indices)
        sorted_indices_1 = torch.argsort(morton_codes_1)
        sorted_indices_2 = torch.argsort(morton_codes_2)
        linear_indices_1 = row_indices[sorted_indices_1] * w + col_indices[sorted_indices_1]
        linear_indices_2 = row_indices[sorted_indices_2] * w + col_indices[sorted_indices_2]
        return linear_indices_1, linear_indices_2

    def interleave_bits(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        y = (y | (y << 8)) & 0x00FF00FF
        y = (y | (y << 4)) & 0x0F0F0F0F
        y = (y | (y << 2)) & 0x33333333
        y = (y | (y << 1)) & 0x55555555
        z = (x << 1) | y
        return z

    def interleave_bits_x_last(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        y = (y | (y << 8)) & 0x00FF00FF
        y = (y | (y << 4)) & 0x0F0F0F0F
        y = (y | (y << 2)) & 0x33333333
        y = (y | (y << 1)) & 0x55555555
        z = (y << 1) | x
        return z
    
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
# input = torch.tensor([[[[1.0,2.0,0.0,0.0],[5.0,6.0,0.0,0.0],[9.0,10.0,11.0,12.0],[13.0,14.0,15.0,16.0]]]]).repeat(1, 256, 1, 1).cuda()
# input = torch.randn(2,256,180,180).cuda()
# model=SECONDMamba().cuda()
# out = model(input)
# print('end')