import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from mamba_ssm import selective_scan_fn
from timm.layers import DropPath
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN


class DSSMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        expand,
        # ---- 1.是否使用morton进行重排，2.是否在ssm前使用conv1d，3.是否使用纵横交错的morton ----
        use_morton=False,       # 是否使用morton进行重排
        use_conv=False,         # 是否在ssm前使用conv1d
        use_xy=False,           # 是否使用横纵交错的morton
        # ---- 通常不进行修改的参数 -----------------------------------------------------------
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.use_morton = use_morton
        self.use_conv = use_conv
        self.use_xy = use_xy
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        if self.use_xy:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 4, bias=bias, **factory_kwargs)
            self.in_proj_xy = nn.Linear(self.d_model, self.d_inner * 4, bias=bias, **factory_kwargs)
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 4, bias=bias, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        dt = torch.exp(torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        
        self.act = nn.SiLU()
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

        if self.use_xy:
            self.act_xy = nn.SiLU()
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

        if self.use_conv and self.use_xy:
            self.conv1d_x_f = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_z_f = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_x_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_z_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_x_xy_f = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_z_xy_f = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_x_xy_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_z_xy_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
        elif self.use_conv and not self.use_xy:
            self.conv1d_x_f = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_z_f = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_x_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_z_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
        
        if self.use_xy:
            self.out_proj = nn.Linear(self.d_inner * 4, self.d_model, bias=bias, **factory_kwargs)
            self.global_proj = nn.Linear(self.d_inner * 4, self.d_inner * 4)
        else:
            self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs)
            self.global_proj = nn.Linear(self.d_inner * 2, self.d_inner * 2)
        
    def forward(self, hidden_states, reference_points=None, grid_size=None):
        """
            1. reference_points * grid_size应与BEV范围一致
            2. hidden_states的形状为(batch, seqlen, dim)
        """
        if not self.use_morton:
            assert reference_points is None, "当不使用morton重排序列时，输入不需要reference_points"
            assert grid_size is None, "当不使用morton重排序列时，输入不需要grid_size"
        batch, seqlen, dim = hidden_states.shape
        assert seqlen == 900, "测试代码（请忽略）：确保输入的seqlen为900"
        if self.use_morton and self.use_xy:
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
            grid_coords_xy = (reference_points * grid_size).floor().long()
            grid_coords_xy = grid_coords_xy.clamp(0, grid_size-1)
            morton_codes_xy = morton_encode_2d_tensor_xy(grid_coords_xy)
            sort_idx_xy = torch.argsort(morton_codes_xy, dim=1)
            unsort_idx_xy = torch.argsort(sort_idx_xy, dim=1)
            hidden_states_xy = torch.gather(
                hidden_states,
                dim=1,
                index=sort_idx_xy.unsqueeze(-1).expand(-1, -1, dim)
            )
        elif self.use_morton and not self.use_xy:
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
        elif self.use_xy and not self.use_morton:
            hidden_states_xy = hidden_states
        else:
            pass

        # ---- 基于生成的morton顺序重排 ----------------------------------------------------------
        if self.use_xy:
            xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            xz_xy = rearrange(self.in_proj_xy.weight @ rearrange(hidden_states_xy, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
            if self.in_proj_xy.bias is not None:
                xz_xy = xz_xy + rearrange(self.in_proj_xy.bias.to(dtype=xz_xy.dtype), "d -> d 1")
        else:
            xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        if self.use_xy:
            xz_f, xz_b = torch.chunk(xz, 2, dim=1)
            assert xz_f.size(-1) == 900 and xz_b.size(-1) == 900, "测试代码（请忽略）：确保seqlen为900"
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
            if self.use_conv:
                x_f = self.act(F.conv1d(input=x_f, weight=self.conv1d_x_f.weight, bias=self.conv1d_x_f.bias, padding='same', groups=self.d_inner))
                x_b = self.act(F.conv1d(input=x_b, weight=self.conv1d_x_b.weight, bias=self.conv1d_x_b.bias, padding='same', groups=self.d_inner))
                z_f = self.act(F.conv1d(input=z_f, weight=self.conv1d_z_f.weight, bias=self.conv1d_z_f.bias, padding='same', groups=self.d_inner))
                z_b = self.act(F.conv1d(input=z_b, weight=self.conv1d_z_b.weight, bias=self.conv1d_z_b.bias, padding='same', groups=self.d_inner))
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
                x_b,
                delta_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=z_b,
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            ).flip([-1])
            y_f = rearrange(y_f, "b d l -> b l d")
            y_b = rearrange(y_b, "b d l -> b l d")

            xz_f_xy, xz_b_xy = torch.chunk(xz_xy, 2, dim=1)
            assert xz_f_xy.size(-1) == 900 and xz_b_xy.size(-1) == 900, "测试代码（请忽略）：确保seqlen为900"
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
            if self.use_conv:
                x_f_xy = self.act_xy(F.conv1d(input=x_f_xy, weight=self.conv1d_x_xy_f.weight, bias=self.conv1d_x_xy_f.bias, padding='same', groups=self.d_inner))
                x_b_xy = self.act_xy(F.conv1d(input=x_b_xy, weight=self.conv1d_x_xy_b.weight, bias=self.conv1d_x_xy_b.bias, padding='same', groups=self.d_inner))
                z_f_xy = self.act_xy(F.conv1d(input=z_f_xy, weight=self.conv1d_z_xy_f.weight, bias=self.conv1d_z_xy_f.bias, padding='same', groups=self.d_inner))
                z_b_xy = self.act_xy(F.conv1d(input=z_b_xy, weight=self.conv1d_z_xy_b.weight, bias=self.conv1d_z_xy_b.bias, padding='same', groups=self.d_inner))
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

            if self.use_morton:
                y_f = torch.gather(y_f, dim=1,index=unsort_idx.unsqueeze(-1).expand(-1, -1, self.d_inner))
                y_b = torch.gather(y_b, dim=1,index=unsort_idx.unsqueeze(-1).expand(-1, -1, self.d_inner))
                y_f_xy = torch.gather(y_f_xy, dim=1,index=unsort_idx_xy.unsqueeze(-1).expand(-1, -1, self.d_inner))
                y_b_xy = torch.gather(y_b_xy, dim=1,index=unsort_idx_xy.unsqueeze(-1).expand(-1, -1, self.d_inner))

            y = torch.cat([y_f, y_b, y_f_xy, y_b_xy], dim=-1)
            # global_y = torch.mean(y, dim=1)
            # global_y = self.global_proj(global_y)
            # global_y = global_y.unsqueeze(1).expand(-1, y.shape[1], -1)
            # y = y + global_y

            out = self.out_proj(y)
            return out
        else:
            xz_f, xz_b = torch.chunk(xz, 2, dim=1)
            assert xz_f.size(-1) == 900 and xz_b.size(-1) == 900, "测试代码（请忽略）：确保seqlen为900"
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
            if self.use_conv:
                x_f = self.act(F.conv1d(input=x_f, weight=self.conv1d_x_f.weight, bias=self.conv1d_x_f.bias, padding='same', groups=self.d_inner))
                x_b = self.act(F.conv1d(input=x_b, weight=self.conv1d_x_b.weight, bias=self.conv1d_x_b.bias, padding='same', groups=self.d_inner))
                z_f = self.act(F.conv1d(input=z_f, weight=self.conv1d_z_f.weight, bias=self.conv1d_z_f.bias, padding='same', groups=self.d_inner))
                z_b = self.act(F.conv1d(input=z_b, weight=self.conv1d_z_b.weight, bias=self.conv1d_z_b.bias, padding='same', groups=self.d_inner))
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
                x_b,
                delta_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=z_b,
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            ).flip([-1])
            y_f = rearrange(y_f, "b d l -> b l d")
            y_b = rearrange(y_b, "b d l -> b l d")
            
            if self.use_morton:
                y_f = torch.gather(y_f, dim=1,index=unsort_idx.unsqueeze(-1).expand(-1, -1, self.d_inner))
                y_b = torch.gather(y_b, dim=1,index=unsort_idx.unsqueeze(-1).expand(-1, -1, self.d_inner))

            y = torch.cat([y_f, y_b], dim=-1)
            # global_y = torch.mean(y, dim=1)
            # global_y = self.global_proj(global_y)
            # global_y = global_y.unsqueeze(1).expand(-1, y.shape[1], -1)
            # y = y + global_y

            out = self.out_proj(y)
            return out


MAMBA_BUILDERS = {
    "DSSMamba_Gigantic": partial(DSSMamba, d_state=128, expand=1),
    "DSSMamba_Huge": partial(DSSMamba, d_state=64, expand=1),
    "DSSMamba_Large": partial(DSSMamba, d_state=32, expand=1),
    "DSSMamba_Middle": partial(DSSMamba, d_state=16, expand=1),
    "DSSMamba_Small": partial(DSSMamba, d_state=8, expand=1),
    "DSSMamba_Tiny": partial(DSSMamba, d_state=4, expand=1),
    "DSSMamba_Nano": partial(DSSMamba, d_state=2, expand=1),
    "DSSMamba_Pico": partial(DSSMamba, d_state=1, expand=1),
    "DSSMamba_Huge_EP2": partial(DSSMamba, d_state=64, expand=2),
}


class DSS(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        drop_prob: float = None,
        mamba_version: str = None,
        num_layers: int = None,
        use_morton: bool = False,
        use_conv: bool = False,
        use_xy: bool = False,
        # ---- 新增RoPE参数 -------------------
        use_rope: bool = False,
        rope_fraction: float = 0.5,
        rope_base: float = 10000.0,
        rope_max_seq_len: int = 900,
        device=None,
        dtype=None
    ):
        super(DSS, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        drop_path_rate = torch.linspace(0, drop_prob, num_layers).tolist()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_before": RMSNorm(d_model),
                "mamba": MAMBA_BUILDERS[mamba_version](
                    d_model=d_model,
                    use_morton=use_morton,
                    use_conv=use_conv,
                    use_xy=use_xy,
                    **factory_kwargs),
                "dropout": DropPath(drop_rate) if drop_rate > 0 else nn.Identity(),
                "norm": RMSNorm(d_model),
                "mlp": MLP(d_model, d_model * 4),
                "mlp_norm": RMSNorm(d_model) if i < num_layers - 1 else nn.Identity()
            }) for i, drop_rate in enumerate(drop_path_rate)
        ])
        self.use_rope = use_rope
        if self.use_rope:
            rope_dim = int(d_model * rope_fraction)
            if rope_dim % 2 != 0:
                rope_dim -= 1
            if rope_dim > 0:
                self.rope = RotaryEmbedding(
                    dim=rope_dim,
                    max_seq_len=rope_max_seq_len,
                    base=rope_base,
                    **factory_kwargs
                )
            else:
                self.rope = nn.Identity()
                self.use_rope = False
        else:
            self.rope = nn.Identity()

    def forward(self, query, query_pos, reference_points=None, grid_size=None):
        """
            1. reference_points * grid_size应与BEV范围一致
            2. hidden_states的形状为(batch, seqlen, dim)
        """
        assert query.size(1) == 900 and query_pos.size(1) == 900, "测试代码（请忽略）：确保seqlen为900"
        if self.use_rope:
            query_rope = self.rope(query, seq_dim=1)
        else:
            query_rope = query
        x = query_rope + query_pos
        for i, layer in enumerate(self.layers):
            residual = x
            x_norm_before = layer["norm_before"](x)
            if layer["mamba"].use_morton:
                x_mamba = layer["mamba"](x_norm_before, reference_points, grid_size)
                # x_mamba = checkpoint(
                #     lambda current_x, ref_pts, g_size: layer["mamba"](current_x, ref_pts, g_size),
                #     x_norm_before,
                #     reference_points,
                #     grid_size,
                #     use_reentrant=True
                # )
            else:
                x_mamba = layer["mamba"](x_norm_before)
                # x_mamba = checkpoint(
                #     layer["mamba"],
                #     x_norm_before,
                #     use_reentrant=True
                # )
            x = layer["dropout"](x_mamba)
            x = residual + x
            x = layer["norm"](x)

            residual_mlp = x
            x = layer["mlp"](x)
            x = x + residual_mlp
            x = layer["mlp_norm"](x)
        return x


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
    def __init__(self, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN['silu']

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000, device=None, dtype=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.device_cached = device
        self.dtype_cached = dtype
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len, device, dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        self.device_cached = device
        self.dtype_cached = dtype
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]
        if x.ndim == 3 and seq_dim == 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        elif x.ndim == 4 and seq_dim == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input ndim {x.ndim} or seq_dim {seq_dim}")
        x_rope = x[..., :self.dim]
        x_pass_through = x[..., self.dim:]
        x1 = x_rope[..., 0::2]
        x2 = x_rope[..., 1::2]
        cos_val_pairs = cos[..., 0::2]
        sin_val_pairs = sin[..., 0::2]
        rotated_x1 = x1 * cos_val_pairs - x2 * sin_val_pairs
        rotated_x2 = x1 * sin_val_pairs + x2 * cos_val_pairs
        rotated_parts = torch.zeros_like(x_rope)
        rotated_parts[..., 0::2] = rotated_x1
        rotated_parts[..., 1::2] = rotated_x2
        if x_pass_through.shape[-1] > 0:
            return torch.cat((rotated_parts, x_pass_through), dim=-1)
        else:
            return rotated_parts


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
    query = torch.randn(2, 900, 256).cuda()
    query_pos = torch.randn(2, 900, 256).cuda()
    reference_points = torch.rand(2, 900, 2).cuda()
    grid_size = 180
    model_1 = DSS(
        d_model=256,
        drop_prob=0.5,
        mamba_version="DSSMamba_Pico",
        num_layers=6,
        use_morton=False,
        use_conv=False,
        use_xy=False
    ).cuda()
    model_2 = DSS(
        d_model=256,
        drop_prob=0.5,
        mamba_version="DSSMamba_Pico",
        num_layers=6,
        use_morton=False,
        use_conv=False,
        use_xy=True
    ).cuda()
    model_3 = DSS(
        d_model=256,
        drop_prob=0.5,
        mamba_version="DSSMamba_Pico",
        num_layers=6,
        use_morton=False,
        use_conv=True,
        use_xy=False
    ).cuda()
    model_4 = DSS(
        d_model=256,
        drop_prob=0.5,
        mamba_version="DSSMamba_Pico",
        num_layers=6,
        use_morton=False,
        use_conv=True,
        use_xy=True
    ).cuda()
    model_5 = DSS(
        d_model=256,
        drop_prob=0.5,
        mamba_version="DSSMamba_Pico",
        num_layers=6,
        use_morton=True,
        use_conv=False,
        use_xy=False
    ).cuda()
    model_6 = DSS(
        d_model=256,
        drop_prob=0.5,
        mamba_version="DSSMamba_Pico",
        num_layers=6,
        use_morton=True,
        use_conv=False,
        use_xy=True
    ).cuda()
    model_7 = DSS(
        d_model=256,
        drop_prob=0.5,
        mamba_version="DSSMamba_Pico",
        num_layers=6,
        use_morton=True,
        use_conv=True,
        use_xy=False
    ).cuda()
    model_8 = DSS(
        d_model=256,
        drop_prob=0.5,
        mamba_version="DSSMamba_Pico",
        num_layers=6,
        use_morton=True,
        use_conv=True,
        use_xy=True
    ).cuda()
    model_rope_1 = DSS(
        d_model=256,
        drop_prob=0.5,
        mamba_version="DSSMamba_Pico",
        num_layers=6,
        use_morton=True,
        use_conv=True,
        use_xy=True,
        use_rope=True,
        rope_fraction=0.5,
        rope_base=10000.0,
        rope_max_seq_len=900,
    ).cuda()
    model_rope_2 = DSS(
        d_model=256,
        drop_prob=0.5,
        mamba_version="DSSMamba_Pico",
        num_layers=6,
        use_morton=True,
        use_conv=True,
        use_xy=True,
        use_rope=True,
        rope_fraction=1.0,
        rope_base=10000.0,
        rope_max_seq_len=900,
    ).cuda()
    
    output_1 = model_1(query, query_pos, reference_points, grid_size).cuda()
    output_2 = model_2(query, query_pos, reference_points, grid_size).cuda()
    output_3 = model_3(query, query_pos, reference_points, grid_size).cuda()
    output_4 = model_4(query, query_pos, reference_points, grid_size).cuda()
    output_5 = model_5(query, query_pos, reference_points, grid_size).cuda()
    output_6 = model_6(query, query_pos, reference_points, grid_size).cuda()
    output_7 = model_7(query, query_pos, reference_points, grid_size).cuda()
    output_8 = model_8(query, query_pos, reference_points, grid_size).cuda()
    output_rope_1 = model_rope_1(query, query_pos, reference_points, grid_size).cuda()
    output_rope_2 = model_rope_2(query, query_pos, reference_points, grid_size).cuda()

    target = torch.randn(2, 900, 256).cuda()
    loss = F.mse_loss(output_1, target)
    loss.backward()
    print("Loss:", loss.item())
