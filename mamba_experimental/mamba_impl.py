"""
最小化带有RoPE的2D Mamba实现，带有Z order功能，带有Conv2d功能，带有4向扫描功能
"""
import math
import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import ssd_selective_scan


class RoPEMamba(nn.Module):
    """2D Mamba model with RoPE."""
    def __init__(self,
                 # --------------------------------
                 use_conv=True,
                 d_model=256,
                 d_state=1,
                 headdim=32,
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
                 W=180,
                 # --------------------------------
                 attn_drop=0.,
                 proj_drop=0.):
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
        d_in_proj = (2 * self.d_inner + 2 * self.d_state + self.nheads) * 4
        self.d_in_proj = d_in_proj
        # self.in_proj_H = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        # self.in_proj_V = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.act = nn.SiLU()
        # --- HF Path ---
        dt_HF = torch.clamp(
            torch.exp(
                torch.rand(self.nheads, **factory_kwargs)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ),
            min=dt_init_floor,
        )
        inv_dt_HF = dt_HF + torch.log(-torch.expm1(-dt_HF))
        self.dt_bias_HF = nn.Parameter(inv_dt_HF)
        self.dt_bias_HF._no_weight_decay = True

        # --- HB Path ---
        dt_HB = torch.clamp(
            torch.exp(
                torch.rand(self.nheads, **factory_kwargs)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ),
            min=dt_init_floor,
        )
        inv_dt_HB = dt_HB + torch.log(-torch.expm1(-dt_HB))
        self.dt_bias_HB = nn.Parameter(inv_dt_HB)
        self.dt_bias_HB._no_weight_decay = True

        # --- VH Path ---
        dt_VH = torch.clamp(
            torch.exp(
                torch.rand(self.nheads, **factory_kwargs)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ),
            min=dt_init_floor,
        )
        inv_dt_VH = dt_VH + torch.log(-torch.expm1(-dt_VH))
        self.dt_bias_VH = nn.Parameter(inv_dt_VH)
        self.dt_bias_VH._no_weight_decay = True

        # --- VB Path ---
        dt_VB = torch.clamp(
            torch.exp(
                torch.rand(self.nheads, **factory_kwargs)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ),
            min=dt_init_floor,
        )
        inv_dt_VB = dt_VB + torch.log(-torch.expm1(-dt_VB))
        self.dt_bias_VB = nn.Parameter(inv_dt_VB)
        self.dt_bias_VB._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]

        # --- A_log for HF Path ---
        A_HF = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log_HF = torch.log(A_HF).to(dtype=dtype)
        self.A_log_HF = nn.Parameter(A_log_HF)
        self.A_log_HF._no_weight_decay = True

        # --- A_log for HB Path ---
        A_HB = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log_HB = torch.log(A_HB).to(dtype=dtype)
        self.A_log_HB = nn.Parameter(A_log_HB)
        self.A_log_HB._no_weight_decay = True

        # --- A_log for VH Path ---
        A_VH = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log_VH = torch.log(A_VH).to(dtype=dtype)
        self.A_log_VH = nn.Parameter(A_log_VH)
        self.A_log_VH._no_weight_decay = True

        # --- A_log for VB Path ---
        A_VB = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log_VB = torch.log(A_VB).to(dtype=dtype)
        self.A_log_VB = nn.Parameter(A_log_VB)
        self.A_log_VB._no_weight_decay = True

        # --- D for HF Path ---
        self.D_HF = nn.Parameter(torch.ones(self.nheads, device=device)) # Consider adding dtype=dtype if needed
        self.D_HF._no_weight_decay = True

        # --- D for HB Path ---
        self.D_HB = nn.Parameter(torch.ones(self.nheads, device=device)) # Consider adding dtype=dtype if needed
        self.D_HB._no_weight_decay = True

        # --- D for VH Path ---
        self.D_VH = nn.Parameter(torch.ones(self.nheads, device=device)) # Consider adding dtype=dtype if needed
        self.D_VH._no_weight_decay = True

        # --- D for VB Path ---
        self.D_VB = nn.Parameter(torch.ones(self.nheads, device=device)) # Consider adding dtype=dtype if needed
        self.D_VB._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_model * 4, self.d_model, bias=bias, **factory_kwargs)
        self.out_norm = nn.LayerNorm(self.d_model, eps=1e-6, **factory_kwargs)
        self.out_drop = nn.Dropout(proj_drop)
        self.use_conv = use_conv
        if self.use_conv:
            self.conv1d_HF = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False,
            )
            self.conv1d_HB = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False,
            )
            self.conv1d_VF = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False,
            )
            self.conv1d_VB = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False,
            )
            init_conv(self.conv1d_HF)
            init_conv(self.conv1d_HB)
            init_conv(self.conv1d_VF)
            init_conv(self.conv1d_VB)
            mask_low_resolution = torch.ones(
                1, 1, H // 2, W // 2, device=device if device is not None else "cpu"
            )
            morton_H_indices_low, morton_V_indices_low = self.morton_code_extraction(mask_low_resolution)
            inverse_H_indices_low = torch.empty_like(morton_H_indices_low)
            inverse_H_indices_low[morton_H_indices_low] = torch.arange(morton_H_indices_low.size(0), device=morton_H_indices_low.device)
            inverse_V_indices_low = torch.empty_like(morton_V_indices_low)
            inverse_V_indices_low[morton_V_indices_low] = torch.arange(morton_V_indices_low.size(0), device=morton_V_indices_low.device)
            self.register_buffer('morton_H_indices_low', morton_H_indices_low)
            self.register_buffer('morton_V_indices_low', morton_V_indices_low)
            self.register_buffer('inverse_H_indices_low', inverse_H_indices_low)
            self.register_buffer('inverse_V_indices_low', inverse_V_indices_low)
        self.H = H
        self.W = W
        mask = torch.ones(1, 1, H, W, device=device if device is not None else "cpu")
        morton_H_indices, morton_V_indices = self.morton_code_extraction(mask)
        inverse_H_indices = torch.empty_like(morton_H_indices)
        inverse_H_indices[morton_H_indices] = torch.arange(morton_H_indices.size(0), device=morton_H_indices.device)
        inverse_V_indices = torch.empty_like(morton_V_indices)
        inverse_V_indices[morton_V_indices] = torch.arange(morton_V_indices.size(0), device=morton_V_indices.device)
        self.register_buffer('morton_H_indices', morton_H_indices)
        self.register_buffer('morton_V_indices', morton_V_indices)
        self.register_buffer('inverse_H_indices', inverse_H_indices)
        self.register_buffer('inverse_V_indices', inverse_V_indices)
        conv_dim = self.d_inner + 2 * self.d_state
        self.conv_dim = conv_dim
        # --- Conv1d for x component ---
        # Horizontal Forward for x
        # self.conv1d_hf_x = nn.Conv1d(
        #     in_channels=conv_dim,
        #     out_channels=conv_dim,
        #     bias=False,
        #     kernel_size=4,
        #     groups=conv_dim,
        #     **factory_kwargs,
        # )
        # # Horizontal Backward for x
        # self.conv1d_hb_x = nn.Conv1d(
        #     in_channels=conv_dim,
        #     out_channels=conv_dim,
        #     bias=False,
        #     kernel_size=4,
        #     groups=conv_dim,
        #     **factory_kwargs,
        # )
        # # Vertical Forward (Height-wise) for x
        # self.conv1d_vh_x = nn.Conv1d(
        #     in_channels=conv_dim,
        #     out_channels=conv_dim,
        #     bias=False,
        #     kernel_size=4,
        #     groups=conv_dim,
        #     **factory_kwargs,
        # )
        # # Vertical Backward (Width-wise) for x
        # self.conv1d_vb_x = nn.Conv1d(
        #     in_channels=conv_dim,
        #     out_channels=conv_dim,
        #     bias=False,
        #     kernel_size=4,
        #     groups=conv_dim,
        #     **factory_kwargs,
        # )

        # # --- Conv1d for z component ---
        # # Horizontal Forward for z
        # self.conv1d_hf_z = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=False,
        #     kernel_size=4,
        #     groups=self.d_inner,
        #     **factory_kwargs,
        # )
        # # Horizontal Backward for z
        # self.conv1d_hb_z = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=False,
        #     kernel_size=4,
        #     groups=self.d_inner,
        #     **factory_kwargs,
        # )
        # # Vertical Forward (Height-wise) for z
        # self.conv1d_vh_z = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=False,
        #     kernel_size=4,
        #     groups=self.d_inner,
        #     **factory_kwargs,
        # )
        # # Vertical Backward (Width-wise) for z
        # self.conv1d_vb_z = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=False,
        #     kernel_size=4,
        #     groups=self.d_inner,
        #     **factory_kwargs,
        # )
        # ------------------------
        d_conv2d = d_model * 2
        self.conv2d_hf = nn.Sequential(
            nn.Conv2d(d_conv2d,d_conv2d,kernel_size=3,stride=1,padding=1,bias=False),
            nn.GELU(),
            nn.BatchNorm2d(d_conv2d, eps=1e-3, momentum=0.01))
        self.conv2d_hb = nn.Sequential(
            nn.Conv2d(d_conv2d,d_conv2d,kernel_size=3,stride=1,padding=1,bias=False),
            nn.GELU(),
            nn.BatchNorm2d(d_conv2d, eps=1e-3, momentum=0.01))
        self.conv2d_vf = nn.Sequential(
            nn.Conv2d(d_conv2d,d_conv2d,kernel_size=3,stride=1,padding=1,bias=False),
            nn.GELU(),
            nn.BatchNorm2d(d_conv2d, eps=1e-3, momentum=0.01))
        self.conv2d_vb = nn.Sequential(
            nn.Conv2d(d_conv2d,d_conv2d,kernel_size=3,stride=1,padding=1,bias=False),
            nn.GELU(),
            nn.BatchNorm2d(d_conv2d, eps=1e-3, momentum=0.01))
        
    def forward(self, x, freqs_cis):
        B, C, H, W = x.shape
        morton_H_indices = self.morton_H_indices.to(x.device)
        morton_V_indices = self.morton_V_indices.to(x.device)
        inverse_H_indices = self.inverse_H_indices.to(x.device)
        inverse_V_indices = self.inverse_V_indices.to(x.device)
        if self.use_conv:
            inverse_H_indices_low = self.inverse_H_indices_low.to(x.device)
            inverse_V_indices_low = self.inverse_V_indices_low.to(x.device)
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        # ---- 重排在in_proj后实现
        # x_morton_H = x_flat[:, :, morton_H_indices].permute(0, 2, 1)
        # x_morton_V = x_flat[:, :, morton_V_indices].permute(0, 2, 1)
        zxbcdt = self.in_proj(x_flat)

        A_HF = -torch.exp(self.A_log_HF.float())
        A_HB = -torch.exp(self.A_log_HB.float())
        A_VF = -torch.exp(self.A_log_VF.float())
        A_VB = -torch.exp(self.A_log_VB.float())

        dim = self.d_ssm

        z_HF, xBC_HF, dt_HF, z_HB, xBC_HB, dt_HB, z_VF, xBC_VF, dt_VF, z_VB, xBC_VB, dt_VB = torch.split(
            zxbcdt, [dim, dim + 2 * self.d_state, self.nheads, dim, dim + 2 * self.d_state, self.nheads, dim, dim + 2 * self.d_state, self.nheads, dim, dim + 2 * self.d_state, self.nheads], dim=-1
        )
        # x_shape: (B, N, C)
        x_HF, B_HF, C_HF = torch.split(xBC_HF, [dim, self.d_state, self.d_state], dim=-1)
        x_HB, B_HB, C_HB = torch.split(xBC_HB, [dim, self.d_state, self.d_state], dim=-1)
        x_VF, B_VF, C_VF = torch.split(xBC_VF, [dim, self.d_state, self.d_state], dim=-1)
        x_VB, B_VB, C_VB = torch.split(xBC_VB, [dim, self.d_state, self.d_state], dim=-1)

        xz_HF = torch.cat([x_HF, z_HF], dim=-1).permute(0, 2, 1).view(B,C*2,H,W)
        xz_HB = torch.cat([x_HB, z_HB], dim=-1).permute(0, 2, 1).view(B,C*2,H,W)
        xz_VF = torch.cat([x_VF, z_VF], dim=-1).permute(0, 2, 1).view(B,C*2,H,W)
        xz_VB = torch.cat([x_VB, z_VB], dim=-1).permute(0, 2, 1).view(B,C*2,H,W)

        xz_HF = self.conv2d_hf(xz_HF)
        xz_HB = self.conv2d_hb(xz_HB)
        xz_VF = self.conv2d_vf(xz_VF)
        xz_VB = self.conv2d_vb(xz_VB)

        x_HF, z_HF = torch.split(xz_HF.view(B, C*2, -1).permute(0, 2, 1),[C,C], dim=-1)
        x_HB, z_HB = torch.split(xz_HB.view(B, C*2, -1).permute(0, 2, 1),[C,C], dim=-1)
        x_VF, z_VF = torch.split(xz_VF.view(B, C*2, -1).permute(0, 2, 1),[C,C], dim=-1)
        x_VB, z_VB = torch.split(xz_VB.view(B, C*2, -1).permute(0, 2, 1),[C,C], dim=-1)

        x_HF = x_HF[:, morton_H_indices, :]
        x_HB = x_HB[:, morton_H_indices, :]
        x_VF = x_VF[:, morton_V_indices, :]
        x_VB = x_VB[:, morton_V_indices, :]
        z_HF = z_HF[:, morton_H_indices, :]
        z_HB = z_HB[:, morton_H_indices, :]
        z_VF = z_VF[:, morton_V_indices, :]
        z_VB = z_VB[:, morton_V_indices, :]
        B_HF = B_HF[:, morton_H_indices, :]
        B_HB = B_HB[:, morton_H_indices, :]
        B_VF = B_VF[:, morton_V_indices, :]
        B_VB = B_VB[:, morton_V_indices, :]
        C_HF = C_HF[:, morton_H_indices, :]
        C_HB = C_HB[:, morton_H_indices, :]
        C_VF = C_VF[:, morton_V_indices, :]
        C_VB = C_VB[:, morton_V_indices, :]
        dt_HF = dt_HF[:, morton_H_indices]
        dt_HB = dt_HB[:, morton_H_indices]
        dt_VF = dt_VF[:, morton_V_indices]
        dt_VB = dt_VB[:, morton_V_indices]

        x_HF = apply_rotary_emb_mamba(x_HF, freqs_cis=freqs_cis)
        x_HB = apply_rotary_emb_mamba(x_HB, freqs_cis=freqs_cis)
        x_VF = apply_rotary_emb_mamba(x_VF, freqs_cis=freqs_cis)
        x_VB = apply_rotary_emb_mamba(x_VB, freqs_cis=freqs_cis)

        x_HF = rearrange(x_HF, "b l (h p) -> b l h p", h=self.nheads).contiguous()
        B_HF = rearrange(B_HF, "b l (g n) -> b l g n", g=1).contiguous()
        C_HF = rearrange(C_HF, "b l (g n) -> b l g n", g=1).contiguous()
        z_HF = rearrange(z_HF, "b l (h p) -> b l h p", h=self.nheads).contiguous()

        x_HB = rearrange(x_HB, "b l (h p) -> b l h p", h=self.nheads).contiguous()
        B_HB = rearrange(B_HB, "b l (g n) -> b l g n", g=1).contiguous()
        C_HB = rearrange(C_HB, "b l (g n) -> b l g n", g=1).contiguous()
        z_HB = rearrange(z_HB, "b l (h p) -> b l h p", h=self.nheads).contiguous()

        x_VF = rearrange(x_VF, "b l (h p) -> b l h p", h=self.nheads).contiguous()
        B_VF = rearrange(B_VF, "b l (g n) -> b l g n", g=1).contiguous()
        C_VF = rearrange(C_VF, "b l (g n) -> b l g n", g=1).contiguous()
        z_VF = rearrange(z_VF, "b l (h p) -> b l h p", h=self.nheads).contiguous()

        x_VB = rearrange(x_VB, "b l (h p) -> b l h p", h=self.nheads).contiguous()
        B_VB = rearrange(B_VB, "b l (g n) -> b l g n", g=1).contiguous()
        C_VB = rearrange(C_VB, "b l (g n) -> b l g n", g=1).contiguous()
        z_VB = rearrange(z_VB, "b l (h p) -> b l h p", h=self.nheads).contiguous()

        out_HF = ssd_selective_scan(
            x_HF,
            dt_HF.to(x_HF.dtype),
            A_HF,
            B_HF,
            C_HF,
            D=self.D_HF.float(),
            z=z_HF,
            dt_bias=self.dt_bias_HF,
            dt_softplus=True,
            dt_limit=self.dt_limit,
        )
        out_HB = ssd_selective_scan(
            x_HB.flip(1),
            dt_HB.to(x_HB.dtype).flip(1),
            A_HB,
            B_HB.flip(1),
            C_HB.flip(1),
            D=self.D_HB.float(),
            z=z_HB.flip(1),
            dt_bias=self.dt_bias_HB,
            dt_softplus=True,
            dt_limit=self.dt_limit,
        ).flip(1)
        out_VF = ssd_selective_scan(
            x_VF,
            dt_VF.to(x_VF.dtype),
            A_VF,
            B_VF,
            C_VF,
            D=self.D_VF.float(),
            z=z_VF,
            dt_bias=self.dt_bias_VF,
            dt_softplus=True,
            dt_limit=self.dt_limit,
        )
        out_VB = ssd_selective_scan(
            x_VB.flip(1),
            dt_VB.to(x_VB.dtype).flip(1),
            A_VB,
            B_VB.flip(1),
            C_VB.flip(1),
            D=self.D_VB.float(),
            z=z_VB.flip(1),
            dt_bias=self.dt_bias_VB,
            dt_softplus=True,
            dt_limit=self.dt_limit,
        ).flip(1)

        out_HF = rearrange(out_HF, "b s h p -> b s (h p)")
        out_HB = rearrange(out_HB, "b s h p -> b s (h p)")
        out_VF = rearrange(out_VF, "b s h p -> b s (h p)")
        out_VB = rearrange(out_VB, "b s h p -> b s (h p)")

        if self.use_conv:
            out_HF = self.conv1d_HF(out_HF.permute(0, 2, 1)).permute(0, 2, 1)
            out_HB = self.conv1d_HB(out_HB.permute(0, 2, 1)).permute(0, 2, 1)
            out_VF = self.conv1d_VF(out_VF.permute(0, 2, 1)).permute(0, 2, 1)
            out_VB = self.conv1d_VB(out_VB.permute(0, 2, 1)).permute(0, 2, 1)
            
            out_HF_inverse = out_HF[:, inverse_H_indices_low, :]
            out_HB_inverse = out_HB[:, inverse_H_indices_low, :]
            out_VF_inverse = out_VF[:, inverse_V_indices_low, :]
            out_VB_inverse = out_VB[:, inverse_V_indices_low, :]
        else:
            out_HF_inverse = out_HF[:, inverse_H_indices, :]
            out_HB_inverse = out_HB[:, inverse_H_indices, :]
            out_VF_inverse = out_VF[:, inverse_V_indices, :]
            out_VB_inverse = out_VB[:, inverse_V_indices, :]

        out = self.out_proj(
            torch.cat([out_HF_inverse, out_HB_inverse, out_VF_inverse, out_VB_inverse], dim=-1).contiguous()
        )

        out = self.out_norm(out)
        out = self.out_drop(out)
        if self.use_conv:
            out = out.view(B, C, H // 2, W // 2)
        else:
            out = out.view(B, C, H, W)

        return out
    
    def morton_code_extraction(self, mask):
        device = mask.device
        h, w = mask[0][0].shape
        """
        说明：
            row_indices[2,3]的值为2，表示该位置的行索引为2
            col_indices[2,3]的值为3，表示该位置的列索引为3
        """
        row_indices, col_indices = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        row_indices = row_indices.flatten()
        col_indices = col_indices.flatten()
        valid_indices = mask[0][0].flatten() != 0
        row_indices = row_indices[valid_indices]
        col_indices = col_indices[valid_indices]
        morton_codes_1 = self.interleave_bits(col_indices, row_indices)
        morton_codes_2 = self.interleave_bits_x_last(col_indices, row_indices)
        sorted_indices_1 = torch.argsort(morton_codes_1)
        sorted_indices_2 = torch.argsort(morton_codes_2)
        linear_indices_1 = (
            row_indices[sorted_indices_1] * w + col_indices[sorted_indices_1]
        )
        linear_indices_2 = (
            row_indices[sorted_indices_2] * w + col_indices[sorted_indices_2]
        )
        return linear_indices_1, linear_indices_2

    def interleave_bits(self, x, y):
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

    def interleave_bits_x_last(self, x, y):
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

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb_mamba(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x).to(x.device)



def init_conv(conv):
    """Initialize convolution layers."""
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)