import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# It's assumed that selective_scan_fn is available in the environment,
# typically from mamba_ssm.ops.selective_scan_interface.
# If 'mamba_ssm' is a local package or installed, this import should work.
# For self-contained execution without the actual mamba_ssm package,
# this function would need to be defined or mocked.
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    # Fallback or placeholder if mamba_ssm is not found.
    # This is for the code to be syntactically complete.
    # In a real environment, this import is expected to succeed.
    print("Warning: mamba_ssm.ops.selective_scan_interface.selective_scan_fn not found. Using a placeholder.")
    def selective_scan_fn(*args, **kwargs):
        raise NotImplementedError("selective_scan_fn is not available. Please ensure mamba_ssm is installed.")
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from timm.models.layers import DropPath



# --- LidarCameraFusionMambaBlock ---
class LidarCameraFusionMambaBlock(nn.Module):
    def __init__(self, 
                 num_layer, 
                 layer_type, # 'fusion_v1' or 'fusion_v2'
                 d_model, 
                 d_state=16, 
                 d_conv=4, # Added d_conv here
                 expand=2, # Added expand here
                 drop_prob=0.1, 
                 batch_first=True, 
                 prenorm=False,
                 # Removed pe_each_layer as per comment "没有pe"
                 device=None, # Pass device and dtype for Mamba layers
                 dtype=None):
        super(LidarCameraFusionMambaBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            if layer_type == 'fusion_v1':
                mamba_layer = LidarCameraFusionMamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    device=device, # Pass device
                    dtype=dtype   # Pass dtype
                )
            elif layer_type == 'fusion_v2':
                mamba_layer = LidarCameraFusionMambaV2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    device=device, # Pass device
                    dtype=dtype   # Pass dtype
                )
            else:
                raise ValueError(f"Unsupported layer_type: {layer_type}. Choose 'fusion_v1' or 'fusion_v2'.")
            self.layers.append(mamba_layer)

        self.norm = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layer)
        ])
        self.dropout = nn.ModuleList([
            DropPath(drop_prob) for _ in range(num_layer)
        ])
        self.num_layer = num_layer
        self.batch_first = batch_first
        self.prenorm = prenorm

    def forward(self, lidar_feats, camera_feats):
        """
        lidar_feats: (batch, seqlen, d_model) or (seqlen, batch, d_model)
        camera_feats: (batch, seqlen, d_model) or (seqlen, batch, d_model)
        Output: updated camera_feats with the same shape as input camera_feats
        """
        if not self.batch_first:
            # Assuming lidar_feats and camera_feats have the same batch_first status
            lidar_feats = lidar_feats.transpose(0, 1)
            camera_feats = camera_feats.transpose(0, 1)

        # The block operates by updating camera_feats using lidar_feats for guidance
        current_camera_feats = camera_feats

        for i in range(self.num_layer):
            residual = current_camera_feats
            
            if self.prenorm:
                # Normalize camera features before Mamba fusion layer
                # Lidar features are used as is by the fusion layer
                normed_camera_feats = self.norm[i](current_camera_feats)
                # The Mamba layer itself will handle lidar_feats
                fused_camera_feats = self.layers[i](lidar_feats, normed_camera_feats)
                current_camera_feats = self.dropout[i](fused_camera_feats) + residual
            else: # Post-norm
                # The Mamba layer itself will handle lidar_feats
                fused_camera_feats = self.layers[i](lidar_feats, current_camera_feats)
                # Apply dropout and residual to the output of Mamba
                current_camera_feats = self.dropout[i](fused_camera_feats) + residual
                # Normalize after adding residual
                current_camera_feats = self.norm[i](current_camera_feats)
        
        if not self.batch_first:
            current_camera_feats = current_camera_feats.transpose(0, 1)
            # lidar_feats is not modified by this block, so no need to transpose it back unless consumed later
            
        return current_camera_feats


# --- LidarCameraFusionMambaBlockV2 ---
class LidarCameraFusionMambaBlockV2(nn.Module):
    def __init__(self, 
                 num_layer, 
                 layer_type, # 'fusion_v1' or 'fusion_v2'
                 d_model, 
                 d_state=16, 
                 d_conv=4, # Added d_conv here
                 expand=2, # Added expand here
                 drop_prob=0.1, 
                 batch_first=True, 
                 prenorm=False,
                 # Removed pe_each_layer as per comment "没有pe"
                 device=None, # Pass device and dtype for Mamba layers
                 dtype=None):
        super(LidarCameraFusionMambaBlockV2, self).__init__()
        
        self.layers_LGC = nn.ModuleList()
        self.layers_CGL = nn.ModuleList()
        for _ in range(num_layer):
            if layer_type == 'fusion_v1':
                raise ValueError("layer_type 'fusion_v1' is not supported in LidarCameraFusionMambaBlockV2. Use 'fusion_v2' or 'fusion_v3'.")
            elif layer_type == 'fusion_v2':
                mamba_layer_LGC = LidarCameraFusionMambaV2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    device=device, # Pass device
                    dtype=dtype   # Pass dtype
                )
                mamba_layer_CGL = LidarCameraFusionMambaV2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    device=device, # Pass device
                    dtype=dtype   # Pass dtype
                )
                
            elif layer_type == 'fusion_v3':
                mamba_layer_LGC = LidarCameraFusionMambaV3(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    device=device, # Pass device
                    dtype=dtype   # Pass dtype
                )
                mamba_layer_CGL = LidarCameraFusionMambaV3(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    device=device, # Pass device
                    dtype=dtype   # Pass dtype
                )
            else:
                raise ValueError(f"Unsupported layer_type: {layer_type}. Choose 'fusion_v3'")
            self.layers_LGC.append(mamba_layer_LGC)
            self.layers_CGL.append(mamba_layer_CGL)

        self.norm_L = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layer)
        ])
        self.norm_C = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layer)
        ])
        self.dropout_L = nn.ModuleList([
            DropPath(drop_prob) for _ in range(num_layer)
        ])
        self.dropout_C = nn.ModuleList([
            DropPath(drop_prob) for _ in range(num_layer)
        ])
        self.num_layer = num_layer
        self.batch_first = batch_first
        self.prenorm = prenorm

        ffn_dropout = 0.1
        self.ffn_block_L = nn.ModuleList(
            FeedForwardNetwork(d_model, d_model*4, ffn_dropout) for _ in range(num_layer)
        )
        self.ffn_block_C = nn.ModuleList(
            FeedForwardNetwork(d_model, d_model*4, ffn_dropout) for _ in range(num_layer)
        )
        self.ffn_norm_L = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layer)
        ])
        self.ffn_norm_C = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layer)
        ])

    def forward(self, lidar_feats, camera_feats):
        """
        lidar_feats: (batch, seqlen, d_model) or (seqlen, batch, d_model)
        camera_feats: (batch, seqlen, d_model) or (seqlen, batch, d_model)
        Output: updated camera_feats with the same shape as input camera_feats
        """
        if not self.batch_first:
            # Assuming lidar_feats and camera_feats have the same batch_first status
            lidar_feats = lidar_feats.transpose(0, 1)
            camera_feats = camera_feats.transpose(0, 1)

        # The block operates by updating camera_feats using lidar_feats for guidance
        current_lidar_feats = lidar_feats
        current_camera_feats = camera_feats
        

        for i in range(self.num_layer):
            # --- 1. Mamba 融合子块 ---
            if self.prenorm:
                # 存储Mamba子块的输入，用于其残差连接
                residual_L_mamba = current_lidar_feats
                residual_C_mamba = current_camera_feats
                
                # Pre-Norm
                normed_lidar_feats = self.norm_L[i](current_lidar_feats)
                normed_camera_feats = self.norm_C[i](current_camera_feats)
                
                # Mamba Fusion
                fused_lidar_feats = self.layers_CGL[i](normed_camera_feats, normed_lidar_feats)
                fused_camera_feats = self.layers_LGC[i](normed_lidar_feats, normed_camera_feats)
                
                # Dropout and first Residual Connection
                current_lidar_feats = self.dropout_L[i](fused_lidar_feats) + residual_L_mamba
                current_camera_feats = self.dropout_C[i](fused_camera_feats) + residual_C_mamba

            else: # Post-norm
                # 存储Mamba子块的输入，用于其残差连接
                residual_L_mamba = current_lidar_feats
                residual_C_mamba = current_camera_feats

                # Mamba Fusion
                fused_lidar_feats = self.layers_CGL[i](current_camera_feats, current_lidar_feats)
                fused_camera_feats = self.layers_LGC[i](current_lidar_feats, current_camera_feats)

                # Dropout and first Residual Connection
                current_lidar_feats = self.dropout_L[i](fused_lidar_feats) + residual_L_mamba
                current_camera_feats = self.dropout_C[i](fused_camera_feats) + residual_C_mamba

                # Post-Norm
                current_lidar_feats = self.norm_L[i](current_lidar_feats)
                current_camera_feats = self.norm_C[i](current_camera_feats)

            # --- 2. FFN (前馈网络) 子块 ---
            if self.prenorm:
                # 存储FFN子块的输入（即Mamba子块的输出），用于其残差连接
                residual_L_ffn = current_lidar_feats
                residual_C_ffn = current_camera_feats

                # Pre-Norm for FFN
                normed_lidar_feats = self.ffn_norm_L[i](current_lidar_feats)
                normed_camera_feats = self.ffn_norm_C[i](current_camera_feats)

                # FFN
                ffn_out_L = self.ffn_block_L[i](normed_lidar_feats)
                ffn_out_C = self.ffn_block_C[i](normed_camera_feats)
                
                # Dropout and second Residual Connection
                # 这里没有再加dropout，因为FeedForwardNetwork内部通常已经包含了dropout
                current_lidar_feats = ffn_out_L + residual_L_ffn
                current_camera_feats = ffn_out_C + residual_C_ffn

            else: # Post-norm
                # 存储FFN子块的输入，用于其残差连接
                residual_L_ffn = current_lidar_feats
                residual_C_ffn = current_camera_feats

                # FFN
                ffn_out_L = self.ffn_block_L[i](current_lidar_feats)
                ffn_out_C = self.ffn_block_C[i](current_camera_feats)

                # Dropout/Residual is typically handled inside FFN or right after
                # Assuming no extra dropout here
                current_lidar_feats = ffn_out_L + residual_L_ffn
                current_camera_feats = ffn_out_C + residual_C_ffn

                # Post-Norm for FFN
                current_lidar_feats = self.ffn_norm_L[i](current_lidar_feats)
                current_camera_feats = self.ffn_norm_C[i](current_camera_feats)
        
        if not self.batch_first:
            current_lidar_feats = current_lidar_feats.transpose(0, 1)
            current_camera_feats = current_camera_feats.transpose(0, 1)
            # lidar_feats is not modified by this block, so no need to transpose it back unless consumed later
        
        return current_lidar_feats, current_camera_feats


class LidarCameraFusionMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,  # d_conv is part of BiMambaShare's init, kept for consistency
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True, # conv_bias is part of BiMambaShare's init
        bias=False,
        layer_idx=None, # layer_idx is part of BiMambaShare's init, not used here but kept for consistency
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Convolution layer (defined as in BiMambaShare, though not used in its direct fwd path)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        
        self.activation = "silu" # As in BiMambaShare

        # Projection for SSM parameters (x_proj)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        # Projection for delta_t (dt_proj)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt_proj.weight and dt_proj.bias as in BiMambaShare
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # Inverse of softplus: log(exp(x) - 1)
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True # Flag to prevent reinitialization

        # S4D A matrix (log space)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log_h2t = nn.Parameter(A_log)
        self.A_log_h2t._no_weight_decay = True
        self.A_log_t2h = nn.Parameter(A_log.clone()) # For bi-directional
        self.A_log_t2h._no_weight_decay = True

        # S4D D matrix
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # New layer for lidar guidance projection
        self.lidar_guidance_proj = nn.Linear(
            self.d_inner,
            self.d_inner,
            bias=True,
            **factory_kwargs
        )

    def forward(self, lidar_feats, camera_feats):
        """
        lidar_feats: (batch, seqlen, d_model)
        camera_feats: (batch, seqlen, d_model)
        """
        batch, seqlen, dim = camera_feats.shape
        assert dim == self.d_model
        assert lidar_feats.shape == (batch, seqlen, dim)

        # Project camera features
        camera_feats_flat = rearrange(camera_feats, "b l d -> (b l) d")
        camera_xz_flat = self.in_proj(camera_feats_flat) # (B*L, D_inner*2)
        camera_xz = rearrange(camera_xz_flat, "(b l) d_out -> b d_out l", l=seqlen, b=batch) # (B, D_inner*2, L)
        camera_x, camera_z = camera_xz.chunk(2, dim=1) # camera_x, camera_z: (B, D_inner, L)

        # Project lidar features
        lidar_feats_flat = rearrange(lidar_feats, "b l d -> (b l) d")
        lidar_xz_flat = self.in_proj(lidar_feats_flat) # (B*L, D_inner*2)
        lidar_xz = rearrange(lidar_xz_flat, "(b l) d_out -> b d_out l", l=seqlen, b=batch) # (B, D_inner*2, L)
        lidar_x, _ = lidar_xz.chunk(2, dim=1) # lidar_x: (B, D_inner, L), lidar_z is not directly used for guidance signal here

        # Generate lidar guidance signal
        # lidar_x is (B, D_inner, L)
        # Rearrange lidar_x for nn.Linear: (B, L, D_inner)
        lidar_x_permuted = rearrange(lidar_x, 'b d l -> b l d')
        lidar_x_proj = self.lidar_guidance_proj(lidar_x_permuted) # (B, L, D_inner)
        # Rearrange back to (B, D_inner, L) for element-wise multiplication
        lidar_x_proj_permuted = rearrange(lidar_x_proj, 'b l d -> b d l')
        lidar_guidance_signal = torch.sigmoid(lidar_x_proj_permuted) # (B, D_inner, L)

        # Modulate camera_x with lidar guidance
        camera_x_guided = camera_x * lidar_guidance_signal # (B, D_inner, L)

        # Derive SSM parameters (dt, B, C) from original camera_x
        # This means the SSM's state dynamics are primarily driven by camera features,
        # while lidar gates the input to the SSM.
        cam_x_for_params_flat = rearrange(camera_x, "b d l -> (b l) d") # (B*L, D_inner)
        ssm_params_cam_flat = self.x_proj(cam_x_for_params_flat) # (B*L, dt_rank + 2*d_state)
        
        dt_cam_flat, B_cam_flat, C_cam_flat = torch.split(
            ssm_params_cam_flat, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # dt_cam: (B, D_inner, L)
        dt_cam = self.dt_proj.weight @ dt_cam_flat.t()
        dt_cam = rearrange(dt_cam, "d (b l) -> b d l", l=seqlen, b=batch)
        
        # B_cam, C_cam: (B, D_state, L)
        B_cam = rearrange(B_cam_flat, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()
        C_cam = rearrange(C_cam_flat, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()

        # Prepare A matrices
        A_h2t = -torch.exp(self.A_log_h2t.float()) # (D_inner, D_state)
        A_t2h = -torch.exp(self.A_log_t2h.float()) # (D_inner, D_state)

        # Forward scan for camera features (guided)
        # Input to selective_scan_fn is camera_x_guided
        y_fwd = selective_scan_fn(
            camera_x_guided, dt_cam, A_h2t, B_cam, C_cam, self.D.float(),
            z=camera_z, # Use original camera_z for gating inside SSM
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        ) # (B, D_inner, L)

        # Backward scan for camera features (guided)
        y_bwd = selective_scan_fn(
            camera_x_guided.flip(dims=[-1]), 
            dt_cam.flip(dims=[-1]), 
            A_t2h, # Use A_t2h for backward
            B_cam.flip(dims=[-1]), 
            C_cam.flip(dims=[-1]), 
            self.D.float(),
            z=camera_z.flip(dims=[-1]), # Use original camera_z flipped
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        ).flip(dims=[-1]) # (B, D_inner, L)

        # Combine forward and backward paths
        y_fused = y_fwd + y_bwd # (B, D_inner, L)
        
        # Rearrange and output projection
        y_fused_rearranged = rearrange(y_fused, "b d l -> b l d") # (B, L, D_inner)
        output_camera_feats = self.out_proj(y_fused_rearranged) # (B, L, D_model)

        return output_camera_feats


class LidarCameraFusionMambaV2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        # layer_idx is not used in BiMamba's core logic here, kept for consistency if needed later
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.activation_fn = nn.SiLU() # Using nn.SiLU for clarity

        # Input projection (common for camera and lidar to get x, z components)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # --- Head-to-Tail (Forward) Path Components ---
        self.conv1d_h2t = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj_h2t = nn.Linear( # Projects camera_x (not guided) for dt, B, C derivation
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
        self.conv1d_t2h = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj_t2h = nn.Linear( # Projects camera_x (not guided) for dt, B, C derivation
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_t2h = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        A_log_t2h = torch.log(repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=self.d_inner,
        ).contiguous()) # Can be .clone() if independent learning is desired from start
        self.A_log_t2h = nn.Parameter(A_log_t2h.clone()) # Clone for potentially different learning trajectories
        self.A_log_t2h._no_weight_decay = True
        self.D_t2h = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_t2h._no_weight_decay = True

        # --- Common Lidar Guidance Projection ---
        self.lidar_guidance_proj = nn.Linear(
            self.d_inner, # Input from lidar_x
            self.d_inner, # Output to modulate camera_x_guided
            bias=True,
            **factory_kwargs
        )

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

        # 2. Project lidar features to get lidar_x for guidance
        lidar_xz_flat = self.in_proj(rearrange(lidar_feats, "b l d -> (b l) d"))
        lidar_xz = rearrange(lidar_xz_flat, "(b l) d_out -> b d_out l", l=seqlen, b=batch)
        lidar_x, _ = lidar_xz.chunk(2, dim=1) # (B, D_inner, L), lidar_z is not used for guidance here

        # 3. Generate lidar guidance signal
        lidar_x_permuted = rearrange(lidar_x, 'b d l -> b l d')
        lidar_guidance_sig_proj = self.lidar_guidance_proj(lidar_x_permuted)
        lidar_guidance_sig = torch.sigmoid(rearrange(lidar_guidance_sig_proj, 'b l d -> b d l')) # (B, D_inner, L)

        # 4. Modulate original camera_x with lidar guidance to get camera_x_guided
        # This camera_x_guided will be the 'x' input to conv1d in both Mamba directions
        camera_x_guided = camera_x_orig * lidar_guidance_sig # (B, D_inner, L)

        # --- H2T (Forward) Path ---
        # Convolution and Activation for h2t path (using camera_x_guided)
        if causal_conv1d_fn is None:
            x_conv_h2t = self.conv1d_h2t(camera_x_guided)[..., :seqlen] # (B, D_inner, L)
            x_activated_h2t = self.activation_fn(x_conv_h2t)
        else:
            # Assumes causal_conv1d_fn takes x and applies conv + activation
            # This part might need adjustment based on the exact signature of causal_conv1d_fn
            x_activated_h2t = causal_conv1d_fn(
                x=camera_x_guided,
                weight=rearrange(self.conv1d_h2t.weight, "d 1 w -> d w"), # if causal_conv1d_fn expects conv weights directly
                bias=self.conv1d_h2t.bias,
                activation="silu" # or self.activation
            )

        # Derive SSM parameters (dt, B, C) for h2t from original camera_x_orig
        # This keeps SSM dynamics primarily driven by camera, while Lidar gates the SSM input state
        ssm_params_cam_flat_h2t = self.x_proj_h2t(rearrange(camera_x_orig, "b d l -> (b l) d"))
        dt_cam_h2t_flat, B_cam_h2t_flat, C_cam_h2t_flat = torch.split(
            ssm_params_cam_flat_h2t, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        dt_cam_h2t = self.dt_proj_h2t.weight @ dt_cam_h2t_flat.t()
        dt_cam_h2t = rearrange(dt_cam_h2t, "d (b l) -> b d l", l=seqlen, b=batch)
        
        B_cam_h2t = rearrange(B_cam_h2t_flat, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()
        C_cam_h2t = rearrange(C_cam_h2t_flat, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()

        A_h2t = -torch.exp(self.A_log_h2t.float())
        
        y_h2t = selective_scan_fn(
            x_activated_h2t, # Input to SSM is Lidar-guided, conv-ed, activated camera feature
            dt_cam_h2t,
            A_h2t,
            B_cam_h2t,
            C_cam_h2t,
            self.D_h2t.float(),
            z=camera_z_orig, # Gating signal from original camera features
            delta_bias=self.dt_proj_h2t.bias.float(),
            delta_softplus=True,
        )

        # --- T2H (Backward) Path ---
        camera_x_guided_flipped = torch.flip(camera_x_guided, dims=[-1])
        camera_x_orig_flipped = torch.flip(camera_x_orig, dims=[-1])
        camera_z_orig_flipped = torch.flip(camera_z_orig, dims=[-1])

        # Convolution and Activation for t2h path (using camera_x_guided_flipped)
        if causal_conv1d_fn is None:
            x_conv_t2h = self.conv1d_t2h(camera_x_guided_flipped)[..., :seqlen]
            x_activated_t2h = self.activation_fn(x_conv_t2h)
        else:
            x_activated_t2h = causal_conv1d_fn(
                x=camera_x_guided_flipped,
                weight=rearrange(self.conv1d_t2h.weight, "d 1 w -> d w"),
                bias=self.conv1d_t2h.bias,
                activation="silu"
            )

        # Derive SSM parameters (dt, B, C) for t2h from original camera_x_orig_flipped
        ssm_params_cam_flat_t2h = self.x_proj_t2h(rearrange(camera_x_orig_flipped, "b d l -> (b l) d"))
        dt_cam_t2h_flat, B_cam_t2h_flat, C_cam_t2h_flat = torch.split(
            ssm_params_cam_flat_t2h, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt_cam_t2h = self.dt_proj_t2h.weight @ dt_cam_t2h_flat.t()
        dt_cam_t2h = rearrange(dt_cam_t2h, "d (b l) -> b d l", l=seqlen, b=batch)

        B_cam_t2h = rearrange(B_cam_t2h_flat, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()
        C_cam_t2h = rearrange(C_cam_t2h_flat, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()

        A_t2h = -torch.exp(self.A_log_t2h.float())

        y_t2h_flipped = selective_scan_fn(
            x_activated_t2h, # Input to SSM is Lidar-guided, conv-ed, activated camera feature (flipped)
            dt_cam_t2h,
            A_t2h,
            B_cam_t2h,
            C_cam_t2h,
            self.D_t2h.float(),
            z=camera_z_orig_flipped, # Gating signal from original camera features (flipped)
            delta_bias=self.dt_proj_t2h.bias.float(),
            delta_softplus=True,
        )
        y_t2h = torch.flip(y_t2h_flipped, dims=[-1])

        # --- Combine and Output ---
        y_fused = y_h2t + y_t2h # (B, D_inner, L)
        
        output_camera_feats = self.out_proj(rearrange(y_fused, "b d l -> b l d")) # (B, L, D_model)

        return output_camera_feats


class LidarCameraFusionMambaV3(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        # layer_idx is not used in BiMamba's core logic here, kept for consistency if needed later
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.activation_fn = nn.SiLU() # Using nn.SiLU for clarity

        # Input projection (common for camera and lidar to get x, z components)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_lidar = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # --- Head-to-Tail (Forward) Path Components ---
        self.conv1d_h2t = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj_h2t = nn.Linear( # Projects camera_x (not guided) for dt, B, C derivation
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
        self.conv1d_t2h = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj_t2h = nn.Linear( # Projects camera_x (not guided) for dt, B, C derivation
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_t2h = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )
        A_log_t2h = torch.log(repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=self.d_inner,
        ).contiguous()) # Can be .clone() if independent learning is desired from start
        self.A_log_t2h = nn.Parameter(A_log_t2h.clone()) # Clone for potentially different learning trajectories
        self.A_log_t2h._no_weight_decay = True
        self.D_t2h = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D_t2h._no_weight_decay = True

        # --- Common Lidar Guidance Projection ---
        self.lidar_guidance_proj = nn.Linear(
            self.d_inner, # Input from lidar_x
            self.d_inner, # Output to modulate camera_x_guided
            bias=True,
            **factory_kwargs
        )

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

        # 2. Project lidar features to get lidar_x for guidance
        lidar_xz_flat = self.in_proj_lidar(rearrange(lidar_feats, "b l d -> (b l) d"))
        lidar_x = rearrange(lidar_xz_flat, "(b l) d_out -> b d_out l", l=seqlen, b=batch)

        # 3. Generate lidar guidance signal
        lidar_x_permuted = rearrange(lidar_x, 'b d l -> b l d')
        lidar_guidance_sig_proj = self.lidar_guidance_proj(lidar_x_permuted)
        lidar_guidance_sig = torch.sigmoid(rearrange(lidar_guidance_sig_proj, 'b l d -> b d l')) # (B, D_inner, L)

        # 4. Modulate original camera_x with lidar guidance to get camera_x_guided
        # This camera_x_guided will be the 'x' input to conv1d in both Mamba directions
        camera_x_guided = camera_x_orig * lidar_guidance_sig # (B, D_inner, L)

        # --- H2T (Forward) Path ---
        # Convolution and Activation for h2t path (using camera_x_guided)
        if causal_conv1d_fn is None:
            x_conv_h2t = self.conv1d_h2t(camera_x_guided)[..., :seqlen] # (B, D_inner, L)
            x_activated_h2t = self.activation_fn(x_conv_h2t)
        else:
            # Assumes causal_conv1d_fn takes x and applies conv + activation
            # This part might need adjustment based on the exact signature of causal_conv1d_fn
            x_activated_h2t = causal_conv1d_fn(
                x=camera_x_guided,
                weight=rearrange(self.conv1d_h2t.weight, "d 1 w -> d w"), # if causal_conv1d_fn expects conv weights directly
                bias=self.conv1d_h2t.bias,
                activation="silu" # or self.activation
            )

        # Derive SSM parameters (dt, B, C) for h2t from original camera_x_orig
        # This keeps SSM dynamics primarily driven by camera, while Lidar gates the SSM input state
        ssm_params_cam_flat_h2t = self.x_proj_h2t(rearrange(camera_x_orig, "b d l -> (b l) d"))
        dt_cam_h2t_flat, B_cam_h2t_flat, C_cam_h2t_flat = torch.split(
            ssm_params_cam_flat_h2t, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        dt_cam_h2t = self.dt_proj_h2t.weight @ dt_cam_h2t_flat.t()
        dt_cam_h2t = rearrange(dt_cam_h2t, "d (b l) -> b d l", l=seqlen, b=batch)
        
        B_cam_h2t = rearrange(B_cam_h2t_flat, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()
        C_cam_h2t = rearrange(C_cam_h2t_flat, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()

        A_h2t = -torch.exp(self.A_log_h2t.float())
        
        y_h2t = selective_scan_fn(
            x_activated_h2t, # Input to SSM is Lidar-guided, conv-ed, activated camera feature
            dt_cam_h2t,
            A_h2t,
            B_cam_h2t,
            C_cam_h2t,
            self.D_h2t.float(),
            z=camera_z_orig, # Gating signal from original camera features
            delta_bias=self.dt_proj_h2t.bias.float(),
            delta_softplus=True,
        )

        # --- T2H (Backward) Path ---
        camera_x_guided_flipped = torch.flip(camera_x_guided, dims=[-1])
        camera_x_orig_flipped = torch.flip(camera_x_orig, dims=[-1])
        camera_z_orig_flipped = torch.flip(camera_z_orig, dims=[-1])

        # Convolution and Activation for t2h path (using camera_x_guided_flipped)
        if causal_conv1d_fn is None:
            x_conv_t2h = self.conv1d_t2h(camera_x_guided_flipped)[..., :seqlen]
            x_activated_t2h = self.activation_fn(x_conv_t2h)
        else:
            x_activated_t2h = causal_conv1d_fn(
                x=camera_x_guided_flipped,
                weight=rearrange(self.conv1d_t2h.weight, "d 1 w -> d w"),
                bias=self.conv1d_t2h.bias,
                activation="silu"
            )

        # Derive SSM parameters (dt, B, C) for t2h from original camera_x_orig_flipped
        ssm_params_cam_flat_t2h = self.x_proj_t2h(rearrange(camera_x_orig_flipped, "b d l -> (b l) d"))
        dt_cam_t2h_flat, B_cam_t2h_flat, C_cam_t2h_flat = torch.split(
            ssm_params_cam_flat_t2h, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt_cam_t2h = self.dt_proj_t2h.weight @ dt_cam_t2h_flat.t()
        dt_cam_t2h = rearrange(dt_cam_t2h, "d (b l) -> b d l", l=seqlen, b=batch)

        B_cam_t2h = rearrange(B_cam_t2h_flat, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()
        C_cam_t2h = rearrange(C_cam_t2h_flat, "(b l) dstate -> b dstate l", l=seqlen, b=batch).contiguous()

        A_t2h = -torch.exp(self.A_log_t2h.float())

        y_t2h_flipped = selective_scan_fn(
            x_activated_t2h, # Input to SSM is Lidar-guided, conv-ed, activated camera feature (flipped)
            dt_cam_t2h,
            A_t2h,
            B_cam_t2h,
            C_cam_t2h,
            self.D_t2h.float(),
            z=camera_z_orig_flipped, # Gating signal from original camera features (flipped)
            delta_bias=self.dt_proj_t2h.bias.float(),
            delta_softplus=True,
        )
        y_t2h = torch.flip(y_t2h_flipped, dims=[-1])

        # --- Combine and Output ---
        y_fused = y_h2t + y_t2h # (B, D_inner, L)
        
        output_camera_feats = self.out_proj(rearrange(y_fused, "b d l -> b l d")) # (B, L, D_model)

        return output_camera_feats


class FeedForwardNetwork(nn.Module):
    """
    一个标准的前馈神经网络模块
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: nn.Module = nn.ReLU()):
        """
        初始化函数

        参数:
        d_model (int): 输入和输出的维度
        d_ff (int): 隐藏层的维度 (通常是 d_model 的 2 倍或 4 倍)
        dropout (float): Dropout 的比例，默认为 0.1
        activation (nn.Module): 激活函数，默认为 ReLU
        """
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        return self.ffn(x)



class LidarCameraFusionMambaBlockV4(nn.Module):
    def __init__(
        self,
        num_layer,
        # layer_type,
        d_model,
        d_state=16,
        expand=2,
        drop_prob=0.2,
        ffn_dropout=0.1,
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
                for _ in range(num_layer)
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
        self.batch_first = batch_first
        self.camera_ffn = nn.ModuleList(
            [FeedForwardNetwork(d_model, d_model*4, ffn_dropout) for _ in range(num_layer)]
        )
        self.camera_ffn_norm = nn.ModuleList(
            [nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True, device=device, dtype=dtype) for _ in range(num_layer)]
        )
        # for _ in range(num_layer):
    
    def forward(
        self,
        pts_query,
        camera_query,
    ):
        if not self.batch_first:
            pts_query = pts_query.transpose(0, 1)
            camera_query = camera_query.transpose(0, 1)
        fuse_query = pts_query
        identity_cam = camera_query
        for layer_idx in range(self.num_layer):
            # pts_query: (batch, seqlen, d_model)
            # camera_query: (batch, seqlen, d_model)
            if self.prenorm:
                fuse_query = self.lidar_camera_fuse_layer[layer_idx](torch.cat([fuse_query, camera_query], dim=-1))
                # residual_cam = camera_query
                fuse_query = self.norm_fusion_fuse[layer_idx](fuse_query)
                camera_query = self.norm_fusion_camera[layer_idx](camera_query)
                camera_query = self.lidar_guide_camera_fusion[layer_idx](fuse_query, camera_query)
                # camera_query = self.dropout[layer_idx](camera_query) + residual_cam
                camera_query = self.dropout[layer_idx](camera_query)
                camera_query = self.camera_ffn_norm[layer_idx](camera_query)
                camera_query = self.camera_ffn[layer_idx](camera_query)
            else:
                residual_fuse = fuse_query
                fuse_query = self.lidar_camera_fuse_layer[layer_idx](torch.cat([fuse_query, camera_query], dim=-1))
                # residual_cam = camera_query
                fuse_query = self.norm_fusion_fuse[layer_idx](fuse_query + residual_fuse)
                camera_query = self.lidar_guide_camera_fusion[layer_idx](fuse_query, camera_query)
                # camera_query = self.dropout[layer_idx](camera_query) + residual_cam
                camera_query = self.dropout[layer_idx](camera_query)
                camera_query = self.norm_fusion_camera[layer_idx](camera_query)
                camera_query = self.camera_ffn[layer_idx](camera_query)
                camera_query = self.camera_ffn_norm[layer_idx](camera_query)
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
        self.lidar_guidance_layer = MLP(input_dim=d_model, hidden_dim=self.d_inner, output_dim=self.d_inner, num_layers=2, activation=nn.SiLU)
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
        lidar_modulation = self.lidar_modulation_layer(lidar_guidance.permute(0,2,1)).permute(0,2,1)
        lidar_modulation_flip = lidar_modulation.flip(dims=[-1])
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
            camera_x_orig*lidar_modulation,
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
            camera_x_orig.flip(dims=[1])*lidar_modulation_flip,
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
    
    



if __name__ == '__main__':
    # Example Usage (requires selective_scan_fn to be available)
    # Mock selective_scan_fn if mamba_ssm is not installed for testing structure
    if not hasattr(torch.ops, "mamba_ssm") or not hasattr(torch.ops.mamba_ssm, "selective_scan_fn"):
        print("Mocking selective_scan_fn for basic structural test.")
        def mock_selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                                   return_last_state=False, P_SEQLEN_MAX=0): # Added P_SEQLEN_MAX for newer mamba_ssm
            # u, z: (batch, dim, seqlen)
            # delta: (batch, dim, seqlen)
            # A: (dim, dstate)
            # B: (batch, dstate, seqlen)
            # C: (batch, dstate, seqlen)
            # D: (dim)
            # delta_bias: (dim)
            if z is not None:
                return u * z # Simplified mock: element-wise product if z is present
            return u # Simplified mock: pass through u

        _selective_scan_fn_original = None
        if 'selective_scan_fn' in globals():
            _selective_scan_fn_original = selective_scan_fn
        
        selective_scan_fn = mock_selective_scan_fn


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 2
    seq_len = 100
    d_model_size = 64
    d_state_size = 8
    expand_factor = 2

    fusion_mamba = LidarCameraFusionMamba(
        d_model=d_model_size,
        d_state=d_state_size,
        expand=expand_factor,
        device=device,
        dtype=torch.float32
    ).to(device)

    lidar_features = torch.randn(batch_size, seq_len, d_model_size, device=device, dtype=torch.float32)
    camera_features = torch.randn(batch_size, seq_len, d_model_size, device=device, dtype=torch.float32)

    try:
        output_features = fusion_mamba(lidar_features, camera_features)
        print("LidarCameraFusionMamba instantiated and forward pass completed.")
        print("Input camera shape:", camera_features.shape)
        print("Output camera shape:", output_features.shape)
        assert output_features.shape == camera_features.shape
        print("Shape assertion passed.")
    except Exception as e:
        print(f"An error occurred during example usage: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original selective_scan_fn if it was mocked
        if '_selective_scan_fn_original' in globals() and _selective_scan_fn_original is not None:
            selective_scan_fn = _selective_scan_fn_original
            del _selective_scan_fn_original # Clean up
        elif 'selective_scan_fn' in globals() and selective_scan_fn.__name__ == 'mock_selective_scan_fn':
             # If only mock was defined, remove it to avoid polluting global scope if script is imported
            del selective_scan_fn


