import math
import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class DSSMamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=256,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.world_size = 1
        self.d_inner = (self.expand * self.d_model) // self.world_size
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        self.ngroups = ngroups // self.world_size
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        # Asserts
        assert self.d_inner * self.world_size == self.expand * self.d_model
        assert ngroups % self.world_size == 0
        assert self.d_ssm % self.headdim == 0
        # In proj
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj_f = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.in_proj_b = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        # Conv1d
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d_f = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_b = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        # Activation
        self.act = nn.SiLU()
        # dt
        dt_f = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_b = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_f = torch.clamp(dt_f, min=dt_init_floor)
        dt_b = torch.clamp(dt_b, min=dt_init_floor)
        inv_dt_f = dt_f + torch.log(-torch.expm1(-dt_f))
        inv_dt_b = dt_b + torch.log(-torch.expm1(-dt_b))
        self.dt_bias_f = nn.Parameter(inv_dt_f)
        self.dt_bias_b = nn.Parameter(inv_dt_b)
        self.dt_bias_f._no_weight_decay = True
        self.dt_bias_b._no_weight_decay = True
        # A
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A_f = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_b = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log_f = torch.log(A_f).to(dtype=dtype)
        A_log_b = torch.log(A_b).to(dtype=dtype)
        self.A_log_f = nn.Parameter(A_log_f)
        self.A_log_b = nn.Parameter(A_log_b)
        self.A_log_f._no_weight_decay = True
        self.A_log_b._no_weight_decay = True
        # D
        self.D_f = nn.Parameter(
            torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device)
        )
        self.D_b = nn.Parameter(
            torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device)
        )
        self.D_f._no_weight_decay = True
        self.D_b._no_weight_decay = True
        # Norm
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm_f = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )
            self.norm_b = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )
        # Out proj
        self.out_proj_f = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.out_proj_b = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        # Output
        self.out_act = nn.SiLU()
        self.out_proj = nn.Linear(
            self.d_model * 2, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, u, seq_idx=None):
        batch, seqlen, dim = u.shape
        zxbcdt_f = self.in_proj_f(u)
        zxbcdt_b = self.in_proj_b(u.flip([1]))
        A_f = -torch.exp(self.A_log_f.float())
        A_b = -torch.exp(self.A_log_b.float())
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )
        out_f = mamba_split_conv1d_scan_combined(
            zxbcdt_f,
            rearrange(self.conv1d_f.weight, "d 1 w -> d w"),
            self.conv1d_f.bias,
            self.dt_bias_f,
            A_f,
            D=rearrange(self.D_f, "(h p) -> h p", p=self.headdim)
            if self.D_has_hdim
            else self.D_f,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm_f.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm_f.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj_f.weight,
            outproj_bias=self.out_proj_f.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
        )
        out_b = mamba_split_conv1d_scan_combined(
            zxbcdt_b,
            rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
            self.conv1d_b.bias,
            self.dt_bias_b,
            A_b,
            D=rearrange(self.D_b, "(h p) -> h p", p=self.headdim)
            if self.D_has_hdim
            else self.D_b,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm_b.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm_b.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj_b.weight,
            outproj_bias=self.out_proj_b.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
        )
        out = self.out_act(torch.cat([out_f, out_b.flip([1])], dim=-1))
        out = self.out_proj(out)
        return out
