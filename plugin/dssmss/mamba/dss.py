import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from mamba_ssm import selective_scan_fn
from mmcv.cnn import build_norm_layer
from timm.layers import DropPath
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from typing import Dict


class DSSMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        expand,
        # >>> morton重排以及conv1d路径 >>>
        morton_rearrange=False,     # 传入selective_scan前是否对目标查询按照morton码排序
        conv_path=False,            # 是否添加两条conv1d路径
        xy=False,                   # 是否使用横纵交错的morton，仅在morton_rearrange=True时使用
        kernel_size=4,              # 仅在conv_path=True时使用
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
        assert not (xy and conv_path), "xy和conv_path不能同时存在"
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.morton_rearrange = morton_rearrange
        self.conv_path = conv_path
        self.xy = xy
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        if self.conv_path:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 8, bias=bias, **factory_kwargs)   # xz*2，fb*2，conv1d*2
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 4, bias=bias, **factory_kwargs)   # xz*2，fb*2
        if self.xy:
            self.in_proj_xy = nn.Linear(self.d_model, self.d_inner * 4, bias=bias, **factory_kwargs)   # xz*2，fb*2
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
        # >>> conv_path单独配置 >>>
        if self.conv_path:
            self.conv_f_down = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=kernel_size, groups=self.d_inner, bias=False)
            self.conv_b_down = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=kernel_size, groups=self.d_inner, bias=False)
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
        if self.xy:
            self.act_f_xy = nn.SiLU()
            self.act_b_xy = nn.SiLU()
            self.x_proj_f_xy = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            self.x_proj_b_xy = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            self.dt_proj_f_xy = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.dt_proj_b_xy = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            if dt_init == "constant":
                nn.init.constant_(self.dt_proj_f_xy.weight, dt_init_std)
                nn.init.constant_(self.dt_proj_b_xy.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_proj_f_xy.weight, -dt_init_std, dt_init_std)
                nn.init.uniform_(self.dt_proj_b_xy.weight, -dt_init_std, dt_init_std)
            with torch.no_grad():
                self.dt_proj_f_xy.bias.copy_(inv_dt)
                self.dt_proj_b_xy.bias.copy_(inv_dt)
            self.dt_proj_f_xy.bias._no_reinit = True
            self.dt_proj_b_xy.bias._no_reinit = True
            A_f_xy = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
            A_b_xy = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=self.d_inner).contiguous()
            A_log_f_xy = torch.log(A_f_xy)
            A_log_b_xy = torch.log(A_b_xy)
            self.A_log_f_xy = nn.Parameter(A_log_f_xy)
            self.A_log_b_xy = nn.Parameter(A_log_b_xy)
            self.A_log_f_xy._no_weight_decay = True
            self.A_log_b_xy._no_weight_decay = True
            self.D_f_xy = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_b_xy = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_f_xy._no_weight_decay = True
            self.D_b_xy._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs) if not (self.conv_path or self.xy) else nn.Linear(self.d_inner * 4, self.d_model, bias=bias, **factory_kwargs)
        self.global_proj = nn.Linear(self.d_inner * 4, self.d_inner * 4)

    def forward(self, hidden_states, reference_points=None, grid_size=None):
        # 希望reference_points乘以grid_size后与BEV范围一致
        # >>> 重排morton >>>
        if not self.morton_rearrange:
            assert reference_points is None, "reference_points should be None when z_rerrange is False"
            assert grid_size is None, "grid_size should be None when z_rerrange is False"
        batch, seqlen, dim = hidden_states.shape
        assert seqlen >= 100, "检查形状正确"
        if self.morton_rearrange:
            grid_coords = (reference_points * grid_size).floor().long()
            grid_coords = grid_coords.clamp(0, grid_size-1)
            assert grid_coords.max() >= 90, "检查形状正确"
            morton_codes = morton_encode_2d_tensor(grid_coords)
            sort_idx = torch.argsort(morton_codes, dim=1)
            unsort_idx = torch.argsort(sort_idx, dim=1)
            hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=sort_idx.unsqueeze(-1).expand(-1, -1, dim)
            )
        if self.xy:
            grid_coords_xy = (reference_points * grid_size).floor().long()
            grid_coords_xy = grid_coords_xy.clamp(0, grid_size-1)
            assert grid_coords_xy.max() >= 90, "检查形状正确"
            morton_codes_xy = morton_encode_2d_tensor_xy(grid_coords_xy)
            sort_idx_xy = torch.argsort(morton_codes_xy, dim=1)
            unsort_idx_xy = torch.argsort(sort_idx_xy, dim=1)
            hidden_states_xy = torch.gather(
                hidden_states,
                dim=1,
                index=sort_idx_xy.unsqueeze(-1).expand(-1, -1, dim)
            )
        # >>> 重排morton >>>
        xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz_b.dtype), "d -> d 1")
        if self.xy:
            xz_xy = rearrange(self.in_proj_xy.weight @ rearrange(hidden_states_xy, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
            if self.in_proj_xy.bias is not None:
                xz_xy = xz_xy + rearrange(self.in_proj_xy.bias.to(dtype=xz_xy.dtype), "d -> d 1")
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
            if self.morton_rearrange:
                out = torch.gather(
                    out,
                    dim=1,
                    index=unsort_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
                )
            return out
        elif self.xy:
            # xy模式仅支持非conv模型
            # x的f和b
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
            # y的f和b
            xz_f_xy, xz_b_xy = torch.chunk(xz_xy, 2, dim=1)
            xz_b_xy = xz_b_xy.flip([-1])
            A_f_xy = -torch.exp(self.A_log_f_xy.float())
            A_b_xy = -torch.exp(self.A_log_b_xy.float())
            L_f_xy = xz_f_xy.shape[-1]
            L_b_xy = xz_b_xy.shape[-1]
            delta_rank_f_xy = self.dt_proj_f_xy.weight.shape[1]
            delta_rank_b_xy = self.dt_proj_b_xy.weight.shape[1]
            d_state_f_xy = A_f_xy.shape[-1] * (1 if not A_f_xy.is_complex() else 2)
            d_state_b_xy = A_b_xy.shape[-1] * (1 if not A_b_xy.is_complex() else 2)
            x_f_xy, z_f_xy = xz_f_xy.chunk(2, dim=1)
            x_b_xy, z_b_xy = xz_b_xy.chunk(2, dim=1)
            x_f_xy = self.act_f_xy(x_f_xy)
            x_b_xy = self.act_b_xy(x_b_xy)
            x_dbl_f_xy = F.linear(rearrange(x_f_xy, "b d l -> (b l) d"), self.x_proj_f_xy.weight)
            x_dbl_b_xy = F.linear(rearrange(x_b_xy, "b d l -> (b l) d"), self.x_proj_b_xy.weight)
            delta_f_xy = self.dt_proj_f_xy.weight @ x_dbl_f_xy[:, :delta_rank_f_xy].t()
            delta_b_xy = self.dt_proj_b_xy.weight @ x_dbl_b_xy[:, :delta_rank_b_xy].t()
            delta_f_xy = rearrange(delta_f_xy, "d (b l) -> b d l", l=L_f_xy)
            delta_b_xy = rearrange(delta_b_xy, "d (b l) -> b d l", l=L_b_xy)
            B_f_xy = x_dbl_f_xy[:, delta_rank_f_xy : delta_rank_f_xy + d_state_f_xy]
            B_b_xy = x_dbl_b_xy[:, delta_rank_b_xy : delta_rank_b_xy + d_state_b_xy]
            C_f_xy = x_dbl_f_xy[:, -d_state_f_xy:]
            C_b_xy = x_dbl_b_xy[:, -d_state_b_xy:]
            if not A_f_xy.is_complex():
                B_f_xy = rearrange(B_f_xy, "(b l) dstate -> b dstate l", l=L_f_xy).contiguous()
            else:
                B_f_xy = rearrange(
                    B_f_xy, "(b l) (dstate two) -> b dstate (l two)", l=L_f_xy, two=2
                ).contiguous()
            if not A_b_xy.is_complex():
                B_b_xy = rearrange(B_b_xy, "(b l) dstate -> b dstate l", l=L_b_xy).contiguous()
            else:
                B_b_xy = rearrange(
                    B_b_xy, "(b l) (dstate two) -> b dstate (l two)", l=L_b_xy, two=2
                ).contiguous()
            if not A_f_xy.is_complex():
                C_f_xy = rearrange(C_f_xy, "(b l) dstate -> b dstate l", l=L_f_xy).contiguous()
            else:
                C_f_xy = rearrange(
                    C_f_xy, "(b l) (dstate two) -> b dstate (l two)", l=L_f_xy, two=2
                ).contiguous()
            if not A_b_xy.is_complex():
                C_b_xy = rearrange(C_b_xy, "(b l) dstate -> b dstate l", l=L_b_xy).contiguous()
            else:
                C_b_xy = rearrange(
                    C_b_xy, "(b l) (dstate two) -> b dstate (l two)", l=L_b_xy, two=2
                ).contiguous()
            y_f_xy = selective_scan_fn(
                x_f_xy,
                delta_f_xy,
                A_f_xy,
                B_f_xy,
                C_f_xy,
                self.D_f_xy.float(),
                z=z_f_xy,
                delta_bias=self.dt_proj_f_xy.bias.float(),
                delta_softplus=True,
            )
            y_b_xy = selective_scan_fn(
                x_b_xy,
                delta_b_xy,
                A_b_xy,
                B_b_xy,
                C_b_xy,
                self.D_b_xy.float(),
                z=z_b_xy,
                delta_bias=self.dt_proj_b_xy.bias.float(),
                delta_softplus=True,
            ).flip([-1])
            y_f_xy = rearrange(y_f_xy, "b d l -> b l d")
            y_b_xy = rearrange(y_b_xy, "b d l -> b l d")
            out_1 = torch.gather(y_f, dim=1,index=unsort_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
            out_2 = torch.gather(y_b, dim=1,index=unsort_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
            out_3 = torch.gather(y_f_xy, dim=1,index=unsort_idx_xy.unsqueeze(-1).expand(-1, -1, self.d_model))
            out_4 = torch.gather(y_b_xy, dim=1,index=unsort_idx_xy.unsqueeze(-1).expand(-1, -1, self.d_model))
            y = torch.cat([out_1, out_2, out_3, out_4], dim=-1)
            global_y = torch.mean(y, dim=1)
            global_y = self.global_proj(global_y)
            global_y = global_y.unsqueeze(1).expand(-1, y.shape[1], -1)
            y = y + global_y


            out = self.out_proj(y)
            return out
        else:
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
            if self.morton_rearrange:
                out = torch.gather(
                    out,
                    dim=1,
                    index=unsort_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
                )
            return out

MAMBA_BUILDERS = {
    "DSSMamba_Gigantic": partial(DSSMamba, d_model=256, d_state=128, expand=1),
    "DSSMamba_Huge": partial(DSSMamba, d_model=256, d_state=64, expand=1),
    "DSSMamba_Large": partial(DSSMamba, d_model=256, d_state=32, expand=1),
    "DSSMamba_Middle": partial(DSSMamba, d_model=256, d_state=16, expand=1),
    "DSSMamba_Small": partial(DSSMamba, d_model=256, d_state=8, expand=1),
    "DSSMamba_Tiny": partial(DSSMamba, d_model=256, d_state=4, expand=1),
    "DSSMamba_Nano": partial(DSSMamba, d_model=256, d_state=2, expand=1),
    "DSSMamba_Pico": partial(DSSMamba, d_model=256, d_state=1, expand=1),
}

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    

class MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN['leaky_relu']

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DSS(nn.Module):
    def __init__(
        self,
        drop_prob: float = None,
        mamba_version: str = None,
        num_layers: int = None,
        # use_rope: bool = False,
        morton_rearrange: bool = False,
        conv_path: bool = False,
        xy=False,
        d_model: int = 256,
        batch_first: bool = False,
        # deepseek_format: bool=False,
    ):
        super(DSS, self).__init__()
        assert drop_prob is not None, "drop_prob must be provided"
        assert mamba_version is not None, "mamba_version must be provided"
        assert num_layers is not None, "num_layers must be provided"
        drop_path_rate = torch.linspace(0, drop_prob, num_layers).tolist()
        # self.use_rope = use_rope
        # if deepseek_format:
        #     self.deepseek_format = True
        #     self.input_layernorm = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        #     self.post_mamba_layernorm = nn.ModuleList([RMSNorm(d_model) for _ in range(num_layers)])
        #     self.mlp = nn.ModuleList([MLP(None, d_model, 4*d_model) for _ in range(num_layers)])
        # else:
        #     self.deepseek_format = False
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_before": nn.LayerNorm(d_model),   # 2025-05-01添加
                "mamba": MAMBA_BUILDERS[mamba_version](
                    morton_rearrange=morton_rearrange,
                    conv_path=conv_path,
                    xy=xy),
                "dropout": DropPath(drop_rate) if drop_rate > 0 else nn.Identity(),
                "norm": nn.LayerNorm(d_model) if i < num_layers - 1 else nn.Identity()}) for i, drop_rate in enumerate(drop_path_rate)])
        self.batch_first = batch_first


    def forward(self, query, query_pos, reference_points=None, grid_size=180):
        # query形状：[bs, num_q, c]
        # query_pos形状: [bs, num_q, c]
        # reference_points形状： [bs, num_q, 2]
        # 1.位置编码
        if self.batch_first:
            x = query + query_pos
        else:
            x = query.permute(1,0,2) + query_pos.permute(1,0,2)
        # if not self.use_rope:
        #     if self.batch_first:
        #         x = query + query_pos
        #     else:
        #         x = query.permute(1,0,2) + query_pos.permute(1,0,2)
        # else:
        #     if not self.batch_first:
        #         query = query.permute(1,0,2)
        #     actual_points = reference_points * torch.tensor([grid_size-1, grid_size-1], device=reference_points.device)
        #     angles = get_rotary_angles(actual_points, query.shape[-1], 10000)
        #     x = apply_rope(query, angles)
        # 独特的前向传播方式
        # if self.deepseek_format:
        #     for i, layer in enumerate(self.layers):
        #         residual = x
        #         hidden_states = self.input_layernorm[i](x)
        #         hidden_states = layer['mamba'](hidden_states, reference_points, grid_size)
        #         hidden_states = residual + hidden_states
        #         residual = hidden_states
        #         hidden_states = self.post_mamba_layernorm[i](hidden_states)
        #         hidden_states = self.mlp[i](hidden_states)
        #         hidden_states = residual + hidden_states
        #         if not self.batch_first:
        #             hidden_states = hidden_states.permute(1,0,2)
        #         return hidden_states
        # 2. 前向传播
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer["norm_before"](x)
            if layer["mamba"].morton_rearrange:
                x = layer["mamba"](x, reference_points, grid_size)
            else:
                x = layer["mamba"](x)
            x = layer["dropout"](x)
            x = residual + x
            x = layer["norm"](x)
        if not self.batch_first:
            x = x.permute(1,0,2)
        return x

def get_rotary_angles(coords, dim, base=10000):
    assert dim % 4 == 0, "维度需为4的倍数以支持x/y分块编码"
    theta = 1.0 / (base ** (torch.arange(0, dim//4, dtype=torch.float32, device=coords.device) / (dim//4)))
    x_angles = coords[..., 0].unsqueeze(-1) * theta
    y_angles = coords[..., 1].unsqueeze(-1) * theta
    angles = torch.cat([x_angles, y_angles], dim=-1)
    return angles

def apply_rope(vec, angles):
    d = vec.shape[-1]
    assert d % 4 == 0, "向量维度需为4的倍数"
    vec_parts = vec.view(*vec.shape[:-1], 4, d//4)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin_x, sin_y = torch.chunk(sin, 2, dim=-1)  # 各 (bs, num_q, d//4)
    cos_x, cos_y = torch.chunk(cos, 2, dim=-1)  # 各 (bs, num_q, d//4)
    rotated = torch.cat([
        vec_parts[..., 0, :] * cos_x + vec_parts[..., 1, :] * sin_x,
        -vec_parts[..., 0, :] * sin_x + vec_parts[..., 1, :] * cos_x,
        vec_parts[..., 2, :] * cos_y + vec_parts[..., 3, :] * sin_y,
        -vec_parts[..., 2, :] * sin_y + vec_parts[..., 3, :] * cos_y
    ], dim=-1)
    return rotated

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

def morton_encode_2d_tensor_xy(coords: torch.Tensor) -> torch.Tensor:
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
    return y | (x << 1)


if __name__ == "__main__":
    dss = DSS(
        drop_prob=0.1,
        mamba_version="DSSMamba_Gigantic",
        batch_first=True,
        num_layers=3,
        use_rope=True,
        morton_rearrange=True,
        conv_path=False,
        xy=True
    ).cuda()
    query = torch.randn(2, 900, 256).cuda()
    query_pos = torch.randn(2, 900, 256).cuda()
    reference_points = torch.rand(2, 900, 2).cuda()
    grid_size = 180
    output = dss(query, query_pos, reference_points, grid_size).cuda()