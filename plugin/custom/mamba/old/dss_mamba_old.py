import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch.utils.checkpoint as cp

class DSSMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj_f = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.in_proj_b = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.activation = "silu"
        self.act_f = nn.SiLU()
        self.act_b = nn.SiLU()
        self.x_proj_f = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_f = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_f.weight, dt_init_std)
            nn.init.constant_(self.dt_proj_b.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_f.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_proj_b.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj_f.bias.copy_(inv_dt)
            self.dt_proj_b.bias.copy_(inv_dt)
        self.dt_proj_f.bias._no_reinit = True
        self.dt_proj_b.bias._no_reinit = True
        A_f = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
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
        self.out_proj = nn.Linear(
            self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, hidden_states):
        batch, seqlen, dim = hidden_states.shape
        xz_f = rearrange(
            self.in_proj_f.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        xz_b = rearrange(
            self.in_proj_b.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        ).flip([-1])
        if self.in_proj_f.bias is not None:
            xz_f = xz_f + rearrange(
                self.in_proj_f.bias.to(dtype=xz_f.dtype), "d -> d 1"
            )
        if self.in_proj_b.bias is not None:
            xz_b = xz_b + rearrange(
                self.in_proj_b.bias.to(dtype=xz_b.dtype), "d -> d 1"
            )
        A_f = -torch.exp(self.A_log_f.float())
        A_b = -torch.exp(self.A_log_b.float())
        # ---------------- custom implementation ----------------
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
