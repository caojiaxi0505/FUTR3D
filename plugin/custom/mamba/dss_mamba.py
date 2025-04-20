import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class DSSMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        expand,
        # >>> morton重排以及conv1d路径 >>>
        morton_rerrange=False,   # 传入selective_scan前是否对目标查询按照morton码排序
        conv_path=False,         # 是否添加两条conv1d路径
        # <<< morton重排以及conv1d路径 <<<
        # >>> 一般情况不应修改的参数 >>>
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        device=None,
        dtype=None,
        # <<< 一般情况不应修改的参数 <<<
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.morton_rearrange = morton_rerrange
        self.conv_path = conv_path
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        if self.conv_path:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 8, bias=bias, **factory_kwargs)   # xz*2，双向*2，conv1d*2
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 4, bias=bias, **factory_kwargs)
        # >>> 非conv_path配置 >>>
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        dt = torch.exp(torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.act_f = nn.SiLU()
        self.act_b = nn.SiLU()
        self.x_proj_f = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj_f = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_f.weight, dt_init_std)
            nn.init.constant_(self.dt_proj_b.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_f.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_proj_b.weight, -dt_init_std, dt_init_std)
        with torch.no_grad():
            self.dt_proj_f.bias.copy_(inv_dt)
            self.dt_proj_b.bias.copy_(inv_dt)
        self.dt_proj_f.bias._no_reinit = True
        self.dt_proj_b.bias._no_reinit = True
        A_f = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
        A_b = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
        A_log_f = torch.log(A_f)
        A_log_b = torch.log(A_b)
        self.A_log_f = nn.Parameter(A_log_f)
        self.A_log_b = nn.Parameter(A_log_b)
        self.A_log_f._no_weight_decay = True
        self.A_log_b._no_weight_decay = True
        self.D_f = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_f._no_weight_decay = True
        self.D_b._no_weight_decay = True
        # <<< 非conv_path配置 <<<
        # >>> conv_path单独配置
        if self.conv_path:
            self.conv_f_down = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=4, groups=self.d_inner, bias=False)
            self.conv_b_down = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=4, groups=self.d_inner, bias=False)
            self.act_f_down = nn.SiLU()
            self.act_b_down = nn.SiLU()
            self.x_proj_f_down = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            self.x_proj_b_down = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            self.dt_proj_f_down = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.dt_proj_b_down = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            if dt_init == "constant":
                nn.init.constant_(self.dt_proj_f_down.weight, dt_init_std)
                nn.init.constant_(self.dt_proj_b_down.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_proj_f_down.weight, -dt_init_std, dt_init_std)
                nn.init.uniform_(self.dt_proj_b_down.weight, -dt_init_std, dt_init_std)
            with torch.no_grad():
                self.dt_proj_f_down.bias.copy_(inv_dt)
                self.dt_proj_b_down.bias.copy_(inv_dt)
            self.dt_proj_f_down.bias._no_reinit = True
            self.dt_proj_b_down.bias._no_reinit = True
            A_f_down = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
            A_b_down = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
            A_log_f_down = torch.log(A_f_down)
            A_log_b_down = torch.log(A_b_down)
            self.A_log_f_down = nn.Parameter(A_log_f_down)
            self.A_log_b_down = nn.Parameter(A_log_b_down)
            self.A_log_f_down._no_weight_decay = True
            self.A_log_b_down._no_weight_decay = True
            self.D_f_down = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_b_down = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_f_down._no_weight_decay = True
            self.D_b_down._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs) if not self.conv_path else nn.Linear(self.d_inner * 4, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, reference_points=None, grid_size=None, spatial_scale=None):
        # >>> 重拍morton >>>
        if not self.morton_rearrange:
            assert reference_points is None, "reference_points should be None when z_rerrange is False"
            assert grid_size is None, "grid_size should be None when z_rerrange is False"
            assert spatial_scale is None, "spatial_scale should be None when z_rerrange is False"
        batch, seqlen, dim = hidden_states.shape
        if self.morton_rearrange:
            grid_coords = (reference_points * grid_size).floor().long()
            grid_coords = grid_coords.clamp(0, grid_size-1)
            morton_codes = morton_encode_2d_tensor(grid_coords)
            sort_idx = torch.argsort(morton_codes, dim=1)
            unsort_idx = torch.argsort(sort_idx, dim=1)
            hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=sort_idx.unsqueeze(-1).expand(-1, -1, dim)
            )
        # >>> 重拍morton >>>
        xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz_b.dtype), "d -> d 1")
        if not self.morton_rearrange:
            xz_f, xz_b = torch.chunk(xz, 2, dim=1)
            xz_b = xz_b.flip([-1])
            A_f = -torch.exp(self.A_log_f.float())
            A_b = -torch.exp(self.A_log_b.float())
            L_f = xz_f.shape[-1]
            L_b = xz_b.shape[-1]
            delta_rank_f = self.dt_proj_f.weight.shape[1]
            delta_rank_b = self.dt_proj_b.weight.shape[1]
            d_state_f = A_f.shape[-1] * (1 if not A_f.is_complex() else 2)
            d_state_b = A_b.shape[-1] * (1 if not A_b.is_complex() else 2)
            x_f, z_f = xz_f.chunk(2, dim=1)
            x_b, z_b = xz_b.chunk(2, dim=1)
            x_f = self.act_f(x_f)
            x_b = self.act_b(x_b)
            x_dbl_f = F.linear(rearrange(x_f, "b d l -> (b l) d"), self.x_proj_f.weight)
            x_dbl_b = F.linear(rearrange(x_b, "b d l -> (b l) d"), self.x_proj_b.weight)
            delta_f = self.dt_proj_f.weight @ x_dbl_f[:, :delta_rank_f].t()
            delta_b = self.dt_proj_b.weight @ x_dbl_b[:, :delta_rank_b].t()
            delta_f = rearrange(delta_f, "d (b l) -> b d l", l=L_f)
            delta_b = rearrange(delta_b, "d (b l) -> b d l", l=L_b)
            B_f = x_dbl_f[:, delta_rank_f : delta_rank_f + d_state_f]
            B_b = x_dbl_b[:, delta_rank_b : delta_rank_b + d_state_b]
            C_f = x_dbl_f[:, -d_state_f:]
            C_b = x_dbl_b[:, -d_state_b:]
            if not A_f.is_complex():
                B_f = rearrange(B_f, "(b l) dstate -> b dstate l", l=L_f).contiguous()
            else:
                B_f = rearrange(
                    B_f, "(b l) (dstate two) -> b dstate (l two)", l=L_f, two=2
                ).contiguous()
            if not A_b.is_complex():
                B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=L_b).contiguous()
            else:
                B_b = rearrange(
                    B_b, "(b l) (dstate two) -> b dstate (l two)", l=L_b, two=2
                ).contiguous()
            if not A_f.is_complex():
                C_f = rearrange(C_f, "(b l) dstate -> b dstate l", l=L_f).contiguous()
            else:
                C_f = rearrange(
                    C_f, "(b l) (dstate two) -> b dstate (l two)", l=L_f, two=2
                ).contiguous()
            if not A_b.is_complex():
                C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=L_b).contiguous()
            else:
                C_b = rearrange(
                    C_b, "(b l) (dstate two) -> b dstate (l two)", l=L_b, two=2
                ).contiguous()
            y_f = selective_scan_fn(
                x_f,
                delta_f,
                A_f,
                B_f,
                C_f,
                self.D_f.float(),
                z=z_f,
                delta_bias=self.dt_proj_f.bias.float(),
                delta_softplus=True,
            )
            y_b = selective_scan_fn(
                x_b,    # 翻转过
                delta_b,    # 翻转过
                A_b,
                B_b,    # 翻转过
                C_b,    # 翻转过
                self.D_b.float(),
                z=z_b,  # 翻转过
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            ).flip([-1])
            y_f = rearrange(y_f, "b d l -> b l d")
            y_b = rearrange(y_b, "b d l -> b l d")
            y = torch.cat([y_f, y_b], dim=-1)
            out = self.out_proj(y)
            return out

        else:
            if self.conv_path:
                xz_f, xz_f_down, xz_b, xz_b_down = torch.chunk(xz, 4, dim=1)
                xz_b = xz_b.flip([-1])
                xz_b_down = xz_b_down.flip([-1])
                A_f = -torch.exp(self.A_log_f.float())
                A_b = -torch.exp(self.A_log_b.float())
                A_f_down = -torch.exp(self.A_log_f_down.float())
                A_b_down = -torch.exp(self.A_log_b_down.float())
                L_f = xz_f.shape[-1]
                L_b = xz_b.shape[-1]
                L_f_down = xz_f_down.shape[-1]
                L_b_down = xz_b_down.shape[-1]
                delta_rank_f = self.dt_proj_f.weight.shape[1]
                delta_rank_b = self.dt_proj_b.weight.shape[1]
                delta_rank_f_down = self.dt_proj_f_down.weight.shape[1]
                delta_rank_b_down = self.dt_proj_b_down.weight.shape[1]
                d_state_f = A_f.shape[-1] * (1 if not A_f.is_complex() else 2)
                d_state_b = A_b.shape[-1] * (1 if not A_b.is_complex() else 2)
                x_f, z_f = xz_f.chunk(2, dim=1)
                x_b, z_b = xz_b.chunk(2, dim=1)
                x_f = self.act_f(x_f)
                x_b = self.act_b(x_b)
                x_f_down, z_f_down = xz_f_down.chunk(2, dim=1)
                x_b_down, z_b_down = xz_b_down.chunk(2, dim=1)
                x_f_down = self.act_f_down(F.conv1d(x_f_down, self.conv_f_down.weight, padding='same', groups=self.d_inner))
                x_b_down = self.act_b_down(F.conv1d(x_b_down, self.conv_b_down.weight, padding='same', groups=self.d_inner))
                x_dbl_f = F.linear(rearrange(x_f, "b d l -> (b l) d"), self.x_proj_f.weight)
                x_dbl_b = F.linear(rearrange(x_b, "b d l -> (b l) d"), self.x_proj_b.weight)
                delta_f = self.dt_proj_f.weight @ x_dbl_f[:, :delta_rank_f].t()
                delta_b = self.dt_proj_b.weight @ x_dbl_b[:, :delta_rank_b].t()
                delta_f = rearrange(delta_f, "d (b l) -> b d l", l=L_f)
                delta_b = rearrange(delta_b, "d (b l) -> b d l", l=L_b)
                B_f = x_dbl_f[:, delta_rank_f : delta_rank_f + d_state_f]
                B_b = x_dbl_b[:, delta_rank_b : delta_rank_b + d_state_b]
                C_f = x_dbl_f[:, -d_state_f:]
                C_b = x_dbl_b[:, -d_state_b:]
                x_dbl_f_down = F.linear(rearrange(x_f_down, "b d l -> (b l) d"), self.x_proj_f_down.weight)
                x_dbl_b_down = F.linear(rearrange(x_b_down, "b d l -> (b l) d"), self.x_proj_b_down.weight)
                delta_f_down = self.dt_proj_f_down.weight @ x_dbl_f_down[:, :delta_rank_f_down].t()
                delta_b_down = self.dt_proj_b_down.weight @ x_dbl_b_down[:, :delta_rank_b_down].t()
                delta_f_down = rearrange(delta_f_down, "d (b l) -> b d l", l=L_f_down)
                delta_b_down = rearrange(delta_b_down, "d (b l) -> b d l", l=L_b_down)
                B_f_down = x_dbl_f_down[:, delta_rank_f_down : delta_rank_f_down + d_state_f]
                B_b_down = x_dbl_b_down[:, delta_rank_b_down : delta_rank_b_down + d_state_b]
                C_f_down = x_dbl_f_down[:, -d_state_f:]
                C_b_down = x_dbl_b_down[:, -d_state_b:]
                if not A_f.is_complex():
                    B_f = rearrange(B_f, "(b l) dstate -> b dstate l", l=L_f).contiguous()
                else:
                    B_f = rearrange(
                        B_f, "(b l) (dstate two) -> b dstate (l two)", l=L_f, two=2
                    ).contiguous()
                if not A_b.is_complex():
                    B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=L_b).contiguous()
                else:
                    B_b = rearrange(
                        B_b, "(b l) (dstate two) -> b dstate (l two)", l=L_b, two=2
                    ).contiguous()
                if not A_f.is_complex():
                    C_f = rearrange(C_f, "(b l) dstate -> b dstate l", l=L_f).contiguous()
                else:
                    C_f = rearrange(
                        C_f, "(b l) (dstate two) -> b dstate (l two)", l=L_f, two=2
                    ).contiguous()
                if not A_b.is_complex():
                    C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=L_b).contiguous()
                else:
                    C_b = rearrange(
                        C_b, "(b l) (dstate two) -> b dstate (l two)", l=L_b, two=2
                    ).contiguous()
                if not A_f_down.is_complex():
                    B_f_down = rearrange(B_f_down, "(b l) dstate -> b dstate l", l=L_f_down).contiguous()
                else:
                    B_f_down = rearrange(
                        B_f_down, "(b l) (dstate two) -> b dstate (l two)", l=L_f_down, two=2
                    ).contiguous()
                if not A_b_down.is_complex():
                    B_b_down = rearrange(B_b_down, "(b l) dstate -> b dstate l", l=L_b_down).contiguous()
                else:
                    B_b_down = rearrange(
                        B_b_down, "(b l) (dstate two) -> b dstate (l two)", l=L_b_down, two=2
                    ).contiguous()
                if not A_f_down.is_complex():
                    C_f_down = rearrange(C_f_down, "(b l) dstate -> b dstate l", l=L_f_down).contiguous()
                else:
                    C_f_down = rearrange(
                        C_f_down, "(b l) (dstate two) -> b dstate (l two)", l=L_f_down, two=2
                    ).contiguous()
                if not A_b_down.is_complex():
                    C_b_down = rearrange(C_b_down, "(b l) dstate -> b dstate l", l=L_b_down).contiguous()
                else:
                    C_b_down = rearrange(
                        C_b_down, "(b l) (dstate two) -> b dstate (l two)", l=L_b_down, two=2
                    ).contiguous()
                y_f = selective_scan_fn(
                    x_f,
                    delta_f,
                    A_f,
                    B_f,
                    C_f,
                    self.D_f.float(),
                    z=z_f,
                    delta_bias=self.dt_proj_f.bias.float(),
                    delta_softplus=True,
                )
                y_b = selective_scan_fn(
                    x_b,    # 翻转过
                    delta_b,    # 翻转过
                    A_b,
                    B_b,    # 翻转过
                    C_b,    # 翻转过
                    self.D_b.float(),
                    z=z_b,  # 翻转过
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                ).flip([-1])
                y_f_down = selective_scan_fn(
                    x_f_down,
                    delta_f_down,
                    A_f_down,
                    B_f_down,
                    C_f_down,
                    self.D_f_down.float(),
                    z=z_f_down,
                    delta_bias=self.dt_proj_f_down.bias.float(),
                    delta_softplus=True,
                )
                y_b_down = selective_scan_fn(
                    x_b_down,    # 翻转过
                    delta_b_down,    # 翻转过
                    A_b_down,
                    B_b_down,    # 翻转过
                    C_b_down,    # 翻转过
                    self.D_b_down.float(),
                    z=z_b_down,  # 翻转过
                    delta_bias=self.dt_proj_b_down.bias.float(),
                    delta_softplus=True,
                ).flip([-1])
                y_f = rearrange(y_f, "b d l -> b l d")
                y_b = rearrange(y_b, "b d l -> b l d")
                y_f_down = rearrange(y_f_down, "b d l -> b l d")
                y_b_down = rearrange(y_b_down, "b d l -> b l d")
                y = torch.cat([y_f, y_b, y_f_down, y_b_down], dim=-1)
                out = self.out_proj(y)
                out = torch.gather(
                    out,
                    dim=1,
                    index=unsort_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
                )
                return out

def morton_encode_2d(x, y, max_value):
    def part1by1_64(n):
        n = (n | (n << 16)) & 0x0000FFFF0000FFFF
        n = (n | (n << 8)) & 0x00FF00FF00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F0F0F0F0F
        n = (n | (n << 2)) & 0x3333333333333333
        n = (n | (n << 1)) & 0x5555555555555555
        return n
    def next_power_of_2(n):
        return 1 << (n - 1).bit_length() if n != 0 else 1
    power_of_2 = next_power_of_2(max_value)
    x_mapped = x * (power_of_2 - 1) // (max_value -1)
    y_mapped = y * (power_of_2 - 1) // (max_value -1)
    x_enc = part1by1_64(x_mapped)
    y_enc = part1by1_64(y_mapped) << 1
    return x_enc | y_enc

def morton_encode_2d_tensor(coords: torch.Tensor) -> torch.Tensor:
    x = coords[..., 0]
    y = coords[..., 1]
    x = x.long()
    y = y.long()
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555
    y = (y | (y << 16)) & 0x0000FFFF0000FFFF
    y = (y | (y << 8)) & 0x00FF00FF00FF00FF
    y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F
    y = (y | (y << 2)) & 0x3333333333333333
    y = (y | (y << 1)) & 0x5555555555555555
    return x | (y << 1)

def dssmamba_g():
    return DSSMamba(d_model=256, d_state=1024, expand=2)

def dssmamba_h():
    return DSSMamba(d_model=256, d_state=512, expand=2)

def dssmamba_l():
    return DSSMamba(d_model=256, d_state=256, expand=2)

def dssmamba_m():
    return DSSMamba(d_model=256, d_state=256, expand=1)

def dssmamba_s():
    return DSSMamba(d_model=256, d_state=128, expand=1)

def dssmamba_s_morton_conv():
    return DSSMamba(d_model=256, d_state=128, expand=1, morton_rerrange=True, conv_path=True)

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
