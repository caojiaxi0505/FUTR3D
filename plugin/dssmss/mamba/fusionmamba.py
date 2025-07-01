import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import Mamba
from einops import rearrange
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class LidarCameraFusionMambaV1(Mamba):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        bias=False,
        fusion_style='add',
        **kwargs
    ):
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            bias=bias,
            **kwargs
        )
        factory_kwargs = {"device": kwargs.get("device"), "dtype": kwargs.get("dtype")}
        self.in_proj_lidar = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.fusion_style = fusion_style
        if self.fusion_style == 'concat':
            self.fusion_linear = nn.Linear(self.d_inner * 2, self.d_inner, bias=bias, **factory_kwargs)
        elif self.fusion_style == 'attention':
            self.q_proj = nn.Linear(self.d_inner, self.d_inner, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(self.d_inner, self.d_inner, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(self.d_inner, self.d_inner, bias=bias, **factory_kwargs)
        elif self.fusion_style != 'add':
            raise ValueError(f"未知的融合方式: {self.fusion_style}. 可选项为 'add', 'concat', 'attention'.")

    def forward(
        self,
        camera_query: torch.Tensor,
        lidar_query: torch.Tensor
    ):
        batch, seqlen, dim = camera_query.shape
        xz_cam = rearrange(self.in_proj(camera_query), "b l d -> b d l")
        x_cam, z_cam = xz_cam.chunk(2, dim=1)
        x_lidar = rearrange(self.in_proj_lidar(lidar_query), "b l d -> b d l")
        if self.fusion_style == 'add':
            x_fused = x_cam + x_lidar
        elif self.fusion_style == 'concat':
            x_concat = torch.cat([x_cam, x_lidar], dim=1)
            x_fused = rearrange(self.fusion_linear(rearrange(x_concat, 'b d l -> (b l) d')), '(b l) d -> b d l', l=seqlen)
        elif self.fusion_style == 'attention':
            cam_re = rearrange(x_cam, 'b d l -> b l d')
            lidar_re = rearrange(x_lidar, 'b d l -> b l d')
            q = self.q_proj(cam_re)
            k = self.k_proj(lidar_re)
            v = self.v_proj(lidar_re)
            attn_out = F.scaled_dot_product_attention(q, k, v)
            fused_re = cam_re + attn_out
            x_fused = rearrange(fused_re, 'b l d -> b d l')
        conv_x_cam = causal_conv1d_fn(x=x_cam, weight=rearrange(self.conv1d.weight, "d 1 w -> d w"), bias=self.conv1d.bias, activation=self.activation)
        conv_x_fused = causal_conv1d_fn(x=x_fused, weight=rearrange(self.conv1d.weight, "d 1 w -> d w"), bias=self.conv1d.bias, activation=self.activation)
        x_dbl_fused = self.x_proj(rearrange(conv_x_fused, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl_fused, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        A = -torch.exp(self.A_log.float())
        y = selective_scan_fn(
            conv_x_cam,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z_cam,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class BiFusionMamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        fusion_style: str = 'add',
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": kwargs.get("device"), "dtype": kwargs.get("dtype")}
        self.forward_mamba = LidarCameraFusionMambaV1(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            fusion_style=fusion_style,
            **factory_kwargs,
        )
        self.backward_mamba = LidarCameraFusionMambaV1(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            fusion_style=fusion_style,
            **factory_kwargs,
        )
        self.proj = nn.Linear(2 * d_model, d_model, **factory_kwargs)
        self.act = nn.SiLU()

    def forward(self, camera_query: torch.Tensor, lidar_query: torch.Tensor) -> torch.Tensor:
        """
        输入:
            camera_query (B, L, D): 相机特征
            lidar_query (B, L, D): 激光雷达特征
        输出:
            双向处理后的相机特征 (B, L, D)
        """
        residual = camera_query
        
        # 前向处理
        y_forward = self.forward_mamba(camera_query, lidar_query)
        
        # 后向处理
        cam_rev = torch.flip(camera_query, dims=[1])
        lidar_rev = torch.flip(lidar_query, dims=[1])
        y_rev = self.backward_mamba(cam_rev, lidar_rev)
        y_backward = torch.flip(y_rev, dims=[1])
        
        # 合并与残差连接
        y = torch.cat([y_forward, y_backward], dim=-1)
        y = self.act(y)
        y = self.proj(y)
        y = y + residual
        return y


class FusionMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        fusion_style: str = 'add',
        bi_directional: bool = False,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": kwargs.get("device"), "dtype": kwargs.get("dtype")}
        if bi_directional:
            self.mamba = BiFusionMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                fusion_style=fusion_style,
                **factory_kwargs,
            )
        else:
            self.mamba = LidarCameraFusionMambaV1(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                fusion_style=fusion_style,
                **factory_kwargs,
            )
        self.norm = nn.LayerNorm(d_model, **factory_kwargs)

    def forward(self, camera_query: torch.Tensor, lidar_query: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        输入:
            camera_query (B, L, D): 待处理的相机特征
            lidar_query (B, L, D): 用于指导的激光雷达特征
            residual (B, L, D): 用于残差连接的相机特征
        输出:
            处理后的相机特征 (B, L, D)
        """
        # 只对主干特征 camera_query 进行归一化
        normed_camera_query = self.norm(camera_query)
        
        # lidar_query 作为指导信息传入，不参与归一化和残差连接
        hidden_states = self.mamba(normed_camera_query, lidar_query)
        
        return hidden_states + residual


class FusionMambaLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        fusion_style: str = 'add',
        bi_directional: bool = False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                FusionMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    fusion_style=fusion_style,
                    bi_directional=bi_directional,
                    **factory_kwargs,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(d_model, **factory_kwargs)

    def forward(self, camera_query: torch.Tensor, lidar_query: torch.Tensor) -> torch.Tensor:
        """
        输入:
            camera_query (B, L, D): 初始相机特征
            lidar_query (B, L, D): 用于指导的激光雷达特征
        输出:
            经过N层处理后的相机特征 (B, L, D)
        """
        for layer in self.layers:
            # lidar_query 在每层中都作为指导信号传入
            # 残差连接作用于 camera_query
            camera_query = layer(camera_query, lidar_query, residual=camera_query)
        
        camera_query = self.norm_f(camera_query)
        return camera_query



