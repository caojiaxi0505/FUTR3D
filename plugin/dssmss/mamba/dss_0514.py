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
        mode=False,             # 单morton模式是否反转
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
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
            self.in_proj_xy = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        dt = torch.exp(torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
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
            self.x_proj_xy = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            self.dt_proj_xy = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            if dt_init == "constant":
                nn.init.constant_(self.dt_proj_xy.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_proj_xy.weight, -dt_init_std, dt_init_std)
            with torch.no_grad():
                self.dt_proj_xy.bias.copy_(inv_dt)
            self.dt_proj_xy.bias._no_reinit = True
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
            self.conv1d_x = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_z = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_x_xy = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_z_xy = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
        elif self.use_conv and not self.use_xy:
            self.conv1d_x = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=4,
                groups=self.d_inner,
                **factory_kwargs,
            )
            self.conv1d_z = nn.Conv1d(
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
            self.y_gate_proj = nn.Linear(self.d_inner * 4, self.d_inner * 4, bias=True, **factory_kwargs)
        else:
            self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs)
            self.global_proj = nn.Linear(self.d_inner * 2, self.d_inner * 2)
            self.y_gate_proj = nn.Linear(self.d_inner * 2, self.d_inner * 2, bias=True, **factory_kwargs)
        self.mode=mode
        
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
            if self.mode:
                morton_codes = morton_encode_2d_tensor_xy(grid_coords)
            else:
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
            A_f = -torch.exp(self.A_log_f.float())
            A_b = -torch.exp(self.A_log_b.float())
            L = xz.shape[-1]
            delta_rank = self.dt_proj.weight.shape[1]
            d_state = A_f.shape[-1] * (1 if not A_f.is_complex() else 2)
            x, z = xz.chunk(2, dim=1)
            if self.use_conv:
                x = self.act(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner))
                z = self.act(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner))
            x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), self.x_proj.weight)
            delta = self.dt_proj.weight @ x_dbl[:, :delta_rank].t()
            delta = rearrange(delta, "d (b l) -> b d l", l=L)
            B = x_dbl[:, delta_rank : delta_rank + d_state]
            C = x_dbl[:, -d_state:]
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
            y_f = selective_scan_fn(
                x,
                delta,
                A_f,
                B,
                C,
                self.D_f.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            y_b = selective_scan_fn(
                x.flip(-1),
                delta.flip(-1),
                A_b,
                B.flip(-1),
                C.flip(-1),
                self.D_b.float(),
                z=z.flip(-1),
                delta_bias=self.dt_proj.bias.float().flip(0),
                delta_softplus=True,
            ).flip([-1])
            y_f = rearrange(y_f, "b d l -> b l d")
            y_b = rearrange(y_b, "b d l -> b l d")

            A_f_xy = -torch.exp(self.A_log_f_xy.float())
            A_b_xy = -torch.exp(self.A_log_b_xy.float())
            L_xy = xz_xy.shape[-1]
            delta_rank_xy = self.dt_proj_xy.weight.shape[1]
            d_state_xy = A_f_xy.shape[-1] * (1 if not A_f_xy.is_complex() else 2)
            x_xy, z_xy = xz_xy.chunk(2, dim=1)
            if self.use_conv:
                x_xy = self.act_xy(F.conv1d(input=x_xy, weight=self.conv1d_x_xy.weight, bias=self.conv1d_x_xy.bias, padding='same', groups=self.d_inner))
                z_xy = self.act_xy(F.conv1d(input=z_xy, weight=self.conv1d_z_xy.weight, bias=self.conv1d_z_xy.bias, padding='same', groups=self.d_inner))
            x_dbl_xy = F.linear(rearrange(x_xy, "b d l -> (b l) d"), self.x_proj_xy.weight)
            delta_xy = self.dt_proj_xy.weight @ x_dbl_xy[:, :delta_rank_xy].t()
            delta_xy = rearrange(delta_xy, "d (b l) -> b d l", l=L_xy)
            B_xy = x_dbl_xy[:, delta_rank_xy : delta_rank_xy + d_state_xy]
            C_xy = x_dbl_xy[:, -d_state_xy:]
            B_xy = rearrange(B_xy, "(b l) dstate -> b dstate l", l=L_xy).contiguous()
            C_xy = rearrange(C_xy, "(b l) dstate -> b dstate l", l=L_xy).contiguous()
            y_f_xy = selective_scan_fn(
                x_xy,
                delta_xy,
                A_f_xy,
                B_xy,
                C_xy,
                self.D_f_xy.float(),
                z=z_xy,
                delta_bias=self.dt_proj_xy.bias.float(),
                delta_softplus=True,
            )
            y_b_xy = selective_scan_fn(
                x_xy.flip(-1),
                delta_xy.flip(-1),
                A_b_xy,
                B_xy.flip(-1),
                C_xy.flip(-1),
                self.D_b_xy.float(),
                z=z_xy.flip(-1),
                delta_bias=self.dt_proj_xy.bias.float().flip(0),
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
            global_y_avg = torch.mean(y, dim=1)
            global_y_transformed = self.global_proj(global_y_avg)
            global_y_expanded = global_y_transformed.unsqueeze(1).expand_as(y)
            gate = torch.sigmoid(self.y_gate_proj(y)) 
            y = y + gate * global_y_expanded

            out = self.out_proj(y)
            return out
        else:
            A_f = -torch.exp(self.A_log_f.float())
            A_b = -torch.exp(self.A_log_b.float())
            L = xz.shape[-1]
            delta_rank = self.dt_proj.weight.shape[1]
            d_state = A_f.shape[-1] * (1 if not A_f.is_complex() else 2)
            x, z = xz.chunk(2, dim=1)
            if self.use_conv:
                x = self.act(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner))
                z = self.act(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner))
            x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), self.x_proj.weight)
            delta = self.dt_proj.weight @ x_dbl[:, :delta_rank].t()
            delta = rearrange(delta, "d (b l) -> b d l", l=L)
            B = x_dbl[:, delta_rank : delta_rank + d_state]
            C = x_dbl[:, -d_state:]
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
            y_f = selective_scan_fn(
                x,
                delta,
                A_f,
                B,
                C,
                self.D_f.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            y_b = selective_scan_fn(
                x.flip(-1),
                delta.flip(-1),
                A_b,
                B.flip(-1),
                C.flip(-1),
                self.D_b.float(),
                z=z.flip(-1),
                delta_bias=self.dt_proj.bias.float().flip(0),
                delta_softplus=True,
            ).flip([-1])
            y_f = rearrange(y_f, "b d l -> b l d")
            y_b = rearrange(y_b, "b d l -> b l d")
            
            if self.use_morton:
                y_f = torch.gather(y_f, dim=1,index=unsort_idx.unsqueeze(-1).expand(-1, -1, self.d_inner))
                y_b = torch.gather(y_b, dim=1,index=unsort_idx.unsqueeze(-1).expand(-1, -1, self.d_inner))

            y = torch.cat([y_f, y_b], dim=-1)
            global_y_avg = torch.mean(y, dim=1)
            global_y_transformed = self.global_proj(global_y_avg)
            global_y_expanded = global_y_transformed.unsqueeze(1).expand_as(y)
            gate = torch.sigmoid(self.y_gate_proj(y)) 
            y = y + gate * global_y_expanded

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

            # residual_mlp = x
            # x = layer["mlp"](x)
            # x = x + residual_mlp
            # x = layer["mlp_norm"](x)
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
