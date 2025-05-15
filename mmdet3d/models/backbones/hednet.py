import math
import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import ssd_selective_scan
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import load_checkpoint
from mmdet.models import BACKBONES
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

def build_2d_sincos_pos_embed(h, w, embed_dim, temperature=10000.0, device=None):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    num_pos_feats = embed_dim // 2
    inv_freq = 1.0 / (
        temperature
        ** (torch.arange(0, num_pos_feats, 2, device=device).float() / num_pos_feats)
    )
    pos_x = grid_x[:, :, None] * inv_freq[None, None, :]
    pos_y = grid_y[:, :, None] * inv_freq[None, None, :]
    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
    pos_embed = torch.cat((pos_y, pos_x), dim=-1)
    pos_embed = pos_embed.flatten(0, 1).unsqueeze(0)
    return pos_embed


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample_layer(x)
        out += identity
        out = self.relu2(out)
        return out


class DEDBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        num_SBB = model_cfg["NUM_SBB"]  # 2,1,1
        down_strides = model_cfg["DOWN_STRIDES"]  # 1,2,2
        dim = model_cfg["FEATURE_DIM"]  # 256
        assert len(num_SBB) == len(down_strides)
        num_levels = len(down_strides)  # 3
        first_block = []
        if input_channels != dim:
            first_block.append(
                BasicBlock(input_channels, dim, down_strides[0], 1, True)
            )
        first_block += [BasicBlock(dim, dim) for _ in range(num_SBB[0])]
        self.encoder = nn.ModuleList([nn.Sequential(*first_block)])
        for idx in range(1, num_levels):  # 1,2
            cur_layers = [BasicBlock(dim, dim, down_strides[idx], 1, True)]
            cur_layers.extend([BasicBlock(dim, dim) for _ in range(num_SBB[idx])])
            self.encoder.append(nn.Sequential(*cur_layers))
        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):  # 2,1
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        dim, dim, down_strides[idx], down_strides[idx], bias=False
                    ),
                    nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )
            self.decoder_norm.append(nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01))
        self.num_bev_features = dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, a=0, mode="fan_out", nonlinearity="relu"
                )
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder[0](x)
        feats = [x]
        for conv in self.encoder[1:]:
            x = conv(x)
            feats.append(x)
        for deconv, norm, up_x in zip(
            self.decoder, self.decoder_norm, feats[:-1][::-1]
        ):
            x = norm(deconv(x) + up_x)
        return x


@BACKBONES.register_module()
class CascadeDEDBackbone(nn.Module):
    def __init__(self, model_cfg, in_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        for idx in range(model_cfg["NUM_LAYERS"]):
            input_dim = in_channels if idx == 0 else model_cfg["FEATURE_DIM"]
            self.layers.append(DEDBackbone(model_cfg, input_dim))
        self.num_bev_features = self.layers[0].num_bev_features
        if model_cfg["USE_SECONDMAMBA"]:
            self.use_seconddmamba = True
            self.secondmamba = nn.Sequential(*[SECONDMamba() for _ in range(model_cfg["SECONDMAMBA_NUM_LAYERS"])])
        else:
            self.use_seconddmamba = False

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.use_seconddmamba:
            x = self.secondmamba(x)
        if isinstance(x, tuple):
            return x
        else:
            return [x]


class SECONDMamba(nn.Module):
    def __init__(self):
        super(SECONDMamba, self).__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SECONDMambaBlock(use_conv=False, d_model=256, H=180, W=180),
                    # nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
                    # nn.ReLU(),
                    # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
                    # nn.ReLU(),
                    # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
                    # nn.ReLU(),
                    # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    # nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
                    # nn.ReLU(),
                ),
            #     nn.Sequential(
            #         SECONDMambaBlock(use_conv=True, d_model=256, H=180, W=180),
            #         # nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
            #         nn.ReLU(),
            #         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #         nn.BatchNorm2d(
            #             256,
            #             eps=0.001,
            #             momentum=0.01,
            #         ),
            #         nn.ReLU(),
            #         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #         nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
            #         nn.ReLU(),
            #         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #         nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
            #         nn.ReLU(),
            #     ),
            #     nn.Sequential(
            #         SECONDMambaBlock(use_conv=True, d_model=256, H=90, W=90),
            #         # nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
            #         nn.ReLU(),
            #         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #         nn.BatchNorm2d(
            #             256,
            #             eps=0.001,
            #             momentum=0.01,
            #         ),
            #         nn.ReLU(),
            #         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #         nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
            #         nn.ReLU(),
            #         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #         nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
            #         nn.ReLU(),
            #     ),
            #     nn.Sequential(
            #         nn.ConvTranspose2d(256, 256, 2, 2, bias=False),
            #         nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
            #         nn.ReLU(),
            #     ),
            #     nn.Sequential(
            #         nn.ConvTranspose2d(256, 256, 2, 2, bias=False),
            #         nn.BatchNorm2d(256, eps=0.001, momentum=0.01),
            #         nn.ReLU(),
            #     ),
            ]
        )
        # self.norm_1 = nn.BatchNorm2d(256, eps=0.001, momentum=0.01)
        # self.norm_2 = nn.BatchNorm2d(256, eps=0.001, momentum=0.01)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        residual = x
        x = self.blocks[0](x)
        x = residual + self.dropout(x)
        return x
        outs = []
        x1 = self.blocks[0](x)
        x2 = self.blocks[1](x1)
        x3 = self.blocks[2](x2)
        x4 = self.norm_1(x2 + self.blocks[3](x3))
        x5 = self.norm_2(x1 + self.blocks[4](x4))
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(f"  - Parameter '{name}' has no gradient.")
        # import pdb; pdb.set_trace()
        return x5


def init_conv(conv):
    """Initialize convolution layers."""
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class SECONDMambaBlock(nn.Module):
    def __init__(
        self,
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
        before_ssm_conv=False,
        H=180,
        W=180,
    ):
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
        dt_H = torch.clamp(
            torch.exp(
                torch.rand(self.nheads, **factory_kwargs)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ),
            min=dt_init_floor,
        )
        inv_dt_H = dt_H + torch.log(-torch.expm1(-dt_H))
        dt_V = torch.clamp(
            torch.exp(
                torch.rand(self.nheads, **factory_kwargs)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ),
            min=dt_init_floor,
        )
        inv_dt_V = dt_V + torch.log(-torch.expm1(-dt_V))
        self.dt_bias_H = nn.Parameter(inv_dt_H)
        self.dt_bias_V = nn.Parameter(inv_dt_V)
        self.dt_bias_H._no_weight_decay = True
        self.dt_bias_V._no_weight_decay = True
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A_H = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_V = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
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
        assert not (
            channel_reduct and channel_increase
        ), "channel_reduct and channel_increase cannot both be True"
        if channel_reduct:
            self.out_proj = nn.Linear(
                self.d_model * 2, self.d_model // 2, bias=bias, **factory_kwargs
            )
        elif channel_increase:
            self.out_proj = nn.Linear(
                self.d_model * 2, self.d_model * 2, bias=bias, **factory_kwargs
            )
        else:
            self.out_proj = nn.Linear(
                self.d_model * 2, self.d_model, bias=bias, **factory_kwargs
            )
        self.out_norm = nn.LayerNorm(self.d_model, eps=1e-6, **factory_kwargs)
        self.use_conv = use_conv
        self.channel_reduct = channel_reduct
        self.channel_increase = channel_increase
        if self.use_conv:
            self.conv1d_H = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False,
            )
            self.conv1d_V = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=False,
            )
            init_conv(self.conv1d_H)
            init_conv(self.conv1d_V)
            mask_low_resolution = torch.ones(
                1, 1, H // 2, W // 2, device=device if device is not None else "cpu"
            )
            morton_H_indices_low, morton_V_indices_low = self.morton_code_extraction(
                mask_low_resolution
            )
            inverse_H_indices_low = torch.empty_like(morton_H_indices_low)
            inverse_H_indices_low[morton_H_indices_low] = torch.arange(
                morton_H_indices_low.size(0), device=morton_H_indices_low.device
            )
            inverse_V_indices_low = torch.empty_like(morton_V_indices_low)
            inverse_V_indices_low[morton_V_indices_low] = torch.arange(
                morton_V_indices_low.size(0), device=morton_V_indices_low.device
            )
            self.register_buffer('morton_H_indices_low', morton_H_indices_low)
            self.register_buffer('morton_V_indices_low', morton_V_indices_low)
            self.register_buffer('inverse_H_indices_low', inverse_H_indices_low)
            self.register_buffer('inverse_V_indices_low', inverse_V_indices_low)
        # >>> 新增部分: 计算并保存morton和inverse索引 >>>
        self.H = H
        self.W = W
        mask = torch.ones(1, 1, H, W, device=device if device is not None else "cpu")
        morton_H_indices, morton_V_indices = self.morton_code_extraction(mask)
        inverse_H_indices = torch.empty_like(morton_H_indices)
        inverse_H_indices[morton_H_indices] = torch.arange(
            morton_H_indices.size(0), device=morton_H_indices.device
        )
        inverse_V_indices = torch.empty_like(morton_V_indices)
        inverse_V_indices[morton_V_indices] = torch.arange(
            morton_V_indices.size(0), device=morton_V_indices.device
        )
        self.register_buffer('morton_H_indices', morton_H_indices)
        self.register_buffer('morton_V_indices', morton_V_indices)
        self.register_buffer('inverse_H_indices', inverse_H_indices)
        self.register_buffer('inverse_V_indices', inverse_V_indices)

        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, H, W))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.sincos_proj = nn.Linear(d_model, d_model)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_conv(m)
        conv_dim = self.d_inner + 2 * self.d_state
        self.conv_dim = conv_dim
        self.conv1d_h_x = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=False,
            kernel_size=4,
            groups=conv_dim,
            **factory_kwargs,
        )
        self.conv1d_w_x = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=False,
            kernel_size=4,
            groups=conv_dim,
            **factory_kwargs,
        )
        self.conv1d_h_z = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=False,
            kernel_size=4,
            groups=self.d_inner,
            **factory_kwargs,
        )
        self.conv1d_w_z = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=False,
            kernel_size=4,
            groups=self.d_inner,
            **factory_kwargs,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # if (H, W) != (self.pos_embed.shape[2], self.pos_embed.shape[3]):
        #     pos_embed = self.pos_embed
        #     pos_embed = torch.nn.functional.interpolate(
        #         pos_embed, size=(H, W), mode="bicubic", align_corners=False
        #     )
        # else:
        #     pos_embed = self.pos_embed
        # sincos_pe = build_2d_sincos_pos_embed(H, W, self.d_model, device=x.device)
        # sincos_pe = self.sincos_proj(sincos_pe)
        # sincos_pe = sincos_pe.transpose(1, 2).view(1, self.d_model, H, W)
        # x = x + sincos_pe
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
        x_morton_H = x_flat[:, :, morton_H_indices].permute(0, 2, 1)
        x_morton_V = x_flat[:, :, morton_V_indices].permute(0, 2, 1)
        zxbcdt_H = self.in_proj_H(x_morton_H)
        zxbcdt_V = self.in_proj_V(x_morton_V)
        A_H = -torch.exp(self.A_log_H.float())
        A_V = -torch.exp(self.A_log_V.float())

        dim = self.d_ssm

        z_H, xBC_H, dt_H = torch.split(
            zxbcdt_H, [dim, dim + 2 * self.d_state, self.nheads], dim=-1
        )
        z_H = F.silu(F.conv1d(input=z_H.transpose(1, 2), weight=self.conv1d_h_z.weight, bias=self.conv1d_h_z.bias, padding='same', groups=self.d_inner).transpose(1, 2))
        # xBC_H = self.act(xBC_H)
        xBC_H = F.silu(F.conv1d(input=xBC_H.transpose(1, 2), weight=self.conv1d_h_x.weight, bias=self.conv1d_h_x.bias, padding='same', groups=self.conv_dim).transpose(1, 2))
        x_H, B_H, C_H = torch.split(xBC_H, [dim, self.d_state, self.d_state], dim=-1)
        x_H = rearrange(x_H, "b l (h p) -> b l h p", h=self.nheads).contiguous()
        B_H = rearrange(B_H, "b l (g n) -> b l g n", g=1).contiguous()
        C_H = rearrange(C_H, "b l (g n) -> b l g n", g=1).contiguous()
        z_H = rearrange(z_H, "b l (h p) -> b l h p", h=self.nheads).contiguous()
        out_H = ssd_selective_scan(
            x_H,
            dt_H.to(x_H.dtype),
            A_H,
            B_H,
            C_H,
            D=self.D_H.float(),
            z=z_H,
            dt_bias=self.dt_bias_H,
            dt_softplus=True,
            dt_limit=self.dt_limit,
        )
        out_H = rearrange(out_H, "b s h p -> b s (h p)")

        z_V, xBC_V, dt_V = torch.split(
            zxbcdt_V, [dim, dim + 2 * self.d_state, self.nheads], dim=-1
        )
        z_V = F.silu(F.conv1d(input=z_V.transpose(1, 2), weight=self.conv1d_w_z.weight, bias=self.conv1d_w_z.bias, padding='same', groups=self.d_inner).transpose(1, 2))
        # xBC_V = self.act(xBC_V)
        xBC_V = F.silu(F.conv1d(input=xBC_V.transpose(1, 2), weight=self.conv1d_w_x.weight, bias=self.conv1d_w_x.bias, padding='same', groups=self.conv_dim).transpose(1, 2))
        x_V, B_V, C_V = torch.split(xBC_V, [dim, self.d_state, self.d_state], dim=-1)
        x_V = rearrange(x_V, "b l (h p) -> b l h p", h=self.nheads).contiguous()
        B_V = rearrange(B_V, "b l (g n) -> b l g n", g=1).contiguous()
        C_V = rearrange(C_V, "b l (g n) -> b l g n", g=1).contiguous()
        z_V = rearrange(z_V, "b l (h p) -> b l h p", h=self.nheads).contiguous()
        out_V = ssd_selective_scan(
            x_V,
            dt_V.to(x_V.dtype),
            A_V,
            B_V,
            C_V,
            D=self.D_V.float(),
            z=z_V,
            dt_bias=self.dt_bias_V,
            dt_softplus=True,
            dt_limit=self.dt_limit,
        )
        out_V = rearrange(out_V, "b s h p -> b s (h p)")

        if self.use_conv:
            out_H = self.conv1d_H(out_H.permute(0, 2, 1)).permute(0, 2, 1)
            out_V = self.conv1d_V(out_V.permute(0, 2, 1)).permute(0, 2, 1)
            out_H_inverse = out_H[:, inverse_H_indices_low, :]
            out_V_inverse = out_V[:, inverse_V_indices_low, :]
        else:
            out_H_inverse = out_H[:, inverse_H_indices, :]
            out_V_inverse = out_V[:, inverse_V_indices, :]
        out = self.out_proj(
            torch.cat([out_H_inverse, out_V_inverse], dim=-1).contiguous()
        )
        out = self.out_norm(out).permute(0, 2, 1)
        if self.channel_reduct:
            out_C = C // 2
        elif self.channel_increase:
            out_C = C * 2
        else:
            out_C = C
        if self.use_conv:
            out = out.view(B, out_C, H // 2, W // 2)
        else:
            out = out.view(B, out_C, H, W)
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


# if __name__ == "__main__":
#     import torch
#     # input = torch.tensor([[0.,0.,3.,4.],[0.,0.,7.,8.],[9.,10.,11.,12.],[13.,14.,15.,16.]]).unsqueeze(0).unsqueeze(0).repeat(1,256,1,1).cuda()
#     input = torch.randn(1,256,180,180).cuda()
#     model = SECONDMamba().cuda()
#     # model0 = SECONDMambaBlock(use_conv=False,H=4,W=4).cuda()
#     # model1 = SECONDMambaBlock(use_conv=True,H=4,W=4).cuda()
#     # o1 = model0(input)
#     # o2 = model1(input)
#     o = model(input)
#     print('end')
