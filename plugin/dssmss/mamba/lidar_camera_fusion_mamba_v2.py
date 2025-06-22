import torch.nn as nn
import math
import torch
from einops import repeat, rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import DropPath

class LidarCameraFusionMambaBlockV4(nn.Module):
    def __init__(
        self,
        num_layer,
        # layer_type,
        d_model,
        d_state=16,
        expand=2,
        drop_prob=0.2,
        batch_first=True,
        prenorm=True,
        device=None,
        dtype=None
    ):
        super(LidarCameraFusionMambaBlockV4, self).__init__()
        # 要求输入的camera_feats和lidar_feats的通道维度均为d_model
        self.lidar_camera_fuse_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model*2,d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model*4),
                    nn.SiLU(),
                    nn.Linear(d_model*4, d_model),
                )
            ]
        )
        # x(t+1) = A * x(t) + B * u(t)
        # y(t) = C * x(t) + D * u(t)
        # B决定状态更新，C决定输出，dt影响A和B
        # dt和B影响状态更新，C影响输出
        # C对应注意力机制中V
        # dt和B对应注意力机制中Q和K
        # ---- ---- ----
        # 1. lidar query引导camera query更新
        # lidar生成门控，控制camera生成的x
        self.lidar_guide_camera_fusion = nn.ModuleList(
            [
                LidarCameraFusionMambaV4(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    device=device,
                    dtype=dtype,
                ) for _ in range(num_layer)
            ]
        )
        self.norm_fusion_fuse = nn.ModuleList(
            [
                nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True, device=device, dtype=dtype)
                for _ in range(num_layer)
            ]
        )
        self.norm_fusion_camera = nn.ModuleList(
            [
                nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True, device=device, dtype=dtype)
                for _ in range(num_layer)
            ]
        )
        self.dropout = nn.ModuleList(
            [
                DropPath(drop_prob) for _ in range(num_layer)
            ]
        )
        self.dropout_output = DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()
        self.num_layer = num_layer
        self.prenorm = prenorm
        self.norm_output = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True, device=device, dtype=dtype) if prenorm else nn.Identity()
        # for _ in range(num_layer):
    
    def forward(
        self,
        pts_query,
        camera_query,
    ):
        fuse_query = pts_query
        # identity_cam = camera_query
        for layer_idx in range(self.num_layer):
            # pts_query: (batch, seqlen, d_model)
            # camera_query: (batch, seqlen, d_model)
            if self.prenorm:
                fuse_query = self.lidar_camera_fuse_layer[layer_idx](torch.cat([fuse_query, camera_query], dim=-1))
                residual_cam = camera_query
                fuse_query = self.norm_fusion_fuse[layer_idx](fuse_query)
                camera_query = self.norm_fusion_camera[layer_idx](camera_query)
                camera_query = self.lidar_guide_camera_fusion[layer_idx](fuse_query, camera_query)
                camera_query = self.dropout[layer_idx](camera_query) + residual_cam
                camera_query = self.dropout[layer_idx](camera_query)
            
            else:
                residual_fuse = fuse_query
                fuse_query = self.lidar_camera_fuse_layer[layer_idx](torch.cat([fuse_query, camera_query], dim=-1))
                residual_cam = camera_query
                fuse_query = self.norm_fusion_fuse[layer_idx](fuse_query + residual_fuse)
                camera_query = self.lidar_guide_camera_fusion[layer_idx](fuse_query, camera_query)
                camera_query = self.dropout[layer_idx](camera_query) + residual_cam
                camera_query = self.dropout[layer_idx](camera_query)
                camera_query = self.norm_fusion_camera[layer_idx](camera_query)
        # camera_query = identity_cam + self.dropout_output(camera_query)
        # camera_query = self.dropout_output(camera_query)
        camera_query = self.norm_output(camera_query)
        return camera_query

class LidarCameraFusionMambaV4(nn.Module):
    # 模式一：跨模态门控 (Cross-Modal Gating)
    # 这是最能体现SSM/Mamba设计精髓的方法。我们不直接融合特征，而是让一个模态（LiDAR）为另一个模态（Camera）生成SSM的状态更新参数。
    # 工作流程:
    # 输入:
    # lidar_query: LiDAR特征序列，L = (l_1, l_2, ..., l_n)
    # camera_query: Camera特征序列，C = (c_1, c_2, ..., c_n)
    # LiDAR作为“门控生成器”:
    # 将lidar_query序列通过一个小的神经网络（比如一个线性层或者一个迷你的SSM）来提取其“指导信息”。我们称这个输出为lidar_guidance。
    # lidar_guidance = MLP(lidar_query)
    # 生成动态参数:
    # 使用lidar_guidance来为Camera SSM生成其动态参数Δ、B和C。
    # Δ_cam = Linear_Δ(lidar_guidance)
    # B_cam = Linear_B(lidar_guidance)
    # C_cam = Linear_C(lidar_guidance)
    # （状态转换矩阵A通常是固定的，不可学习或全局学习）。
    # Camera SSM更新:
    # 现在，使用这些由LiDAR“指导”生成的参数来处理camera_query序列。
    # 对于Camera序列中的每一步t：
    # h_t_cam = Ā(A, Δ_cam_t) * h_{t-1}_cam + B̄(B_cam_t, Δ_cam_t) * c_t
    # output_t_cam = C_cam_t * h_t_cam
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
        self.activation_fn = nn.SiLU()
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.lidar_guidance_layer = MLP(input_dim=d_model, hidden_dim=self.d_inner, output_dim=self.d_inner, num_layers=2, activation=nn.SiLU())
        self.lidar_modulation_layer = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner), nn.SiLU()
        )

        # --- Head-to-Tail (Forward) Path Components ---
        self.x_proj_h2t = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_h2t = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        A_log_h2t = torch.log(repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=self.d_inner,
        ).contiguous())
        self.A_log_h2t = nn.Parameter(A_log_h2t)
        self.A_log_h2t._no_weight_decay = True
        self.D_h2t = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_h2t._no_weight_decay = True

        # --- Tail-to-Head (Backward) Path Components ---
        self.x_proj_t2h = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_t2h = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        A_log_t2h = torch.log(repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=self.d_inner,
        ).contiguous())
        self.A_log_t2h = nn.Parameter(A_log_t2h.clone()) # Clone for potentially different learning trajectories
        self.A_log_t2h._no_weight_decay = True
        self.D_t2h = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_t2h._no_weight_decay = True

        # --- Output Projection ---
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Initialize dt_projs (common logic for h2t and t2h)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        for dt_proj_layer in [self.dt_proj_h2t, self.dt_proj_t2h]:
            if dt_init == "constant":
                nn.init.constant_(dt_proj_layer.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj_layer.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError
            
            # Initialize bias
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt)) # Inverse of softplus
            with torch.no_grad():
                dt_proj_layer.bias.copy_(inv_dt)
            dt_proj_layer.bias._no_reinit = True
    
    def forward(self, lidar_feats, camera_feats):
        """
        lidar_feats: (batch, seqlen, d_model)
        camera_feats: (batch, seqlen, d_model)
        """
        batch, seqlen, dim = camera_feats.shape
        assert dim == self.d_model
        assert lidar_feats.shape == (batch, seqlen, dim)

        # 1. Project camera features to get camera_x_orig and camera_z_orig
        camera_xz_orig_flat = self.in_proj(rearrange(camera_feats, "b l d -> (b l) d"))
        camera_xz_orig = rearrange(camera_xz_orig_flat, "(b l) d_out -> b d_out l", l=seqlen, b=batch)
        camera_x_orig, camera_z_orig = camera_xz_orig.chunk(2, dim=1) # (B, D_inner, L)

        lidar_guidance = self.lidar_guidance_layer(rearrange(lidar_feats, "b l d -> (b l) d"))
        lidar_guidance = rearrange(lidar_guidance, "(b l) d_out -> b d_out l", l=seqlen, b=batch)
        lidar_modulation = self.lidar_modulation_layer(lidar_guidance)
        ssm_params_h2t = self.x_proj_h2t(rearrange(lidar_guidance, "b d l -> (b l) d"))
        ssm_params_t2h = self.x_proj_t2h(rearrange(lidar_guidance.flip(dims=[-1]), "b d l -> (b l) d"))
        dt_h2t, B_h2t, C_h2t = torch.split(
            ssm_params_h2t, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt_t2h, B_t2h, C_t2h = torch.split(
            ssm_params_t2h, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt_h2t = self.dt_proj_h2t.weight @ dt_h2t.t()
        dt_t2h = self.dt_proj_t2h.weight @ dt_t2h.t()
        dt_h2t = rearrange(dt_h2t, "d (b l) -> b d l", l=seqlen, b=batch)
        B_h2t = rearrange(B_h2t, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()
        C_h2t = rearrange(C_h2t, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()
        dt_t2h = rearrange(dt_t2h, "d (b l) -> b d l", l=seqlen, b=batch)
        B_t2h = rearrange(B_t2h, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()
        C_t2h = rearrange(C_t2h, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()
        A_h2t = -torch.exp(self.A_log_h2t.float())
        A_t2h = -torch.exp(self.A_log_t2h.float())

        y_h2t = selective_scan_fn(
            camera_x_orig,
            dt_h2t,
            A_h2t,
            B_h2t,
            C_h2t,
            self.D_h2t.float(),
            z=camera_z_orig,
            delta_bias=self.dt_proj_h2t.bias.float(),
            delta_softplus=True,
        )
        y_t2h = selective_scan_fn(
            camera_x_orig.flip(dims=[1]),
            dt_t2h,
            A_t2h,
            B_t2h,
            C_t2h,
            self.D_t2h.float(),
            z=camera_z_orig.flip(dims=[1]),
            delta_bias=self.dt_proj_t2h.bias.float(),
            delta_softplus=True,
        ).flip(dims=[1])
        y_fused = y_h2t + y_t2h
        y_fused = self.out_proj(rearrange(y_fused, "b d l -> b l d"))
        return y_fused



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, activation=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    
