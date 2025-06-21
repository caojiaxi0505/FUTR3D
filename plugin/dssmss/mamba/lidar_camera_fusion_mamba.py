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

if __name__ == '__main__':
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 2
    seq_len = 50 # Reduced for faster mock testing
    d_model_size = 64
    d_state_size = 8
    d_conv_size = 4
    expand_factor = 2

    # Instantiate LidarCameraFusionMambaV2
    fusion_mamba_v2 = LidarCameraFusionMambaV2(
        d_model=d_model_size,
        d_state=d_state_size,
        d_conv=d_conv_size,
        expand=expand_factor,
        device=device,
        dtype=torch.float32
    ).to(device)

    lidar_features = torch.randn(batch_size, seq_len, d_model_size, device=device, dtype=torch.float32)
    camera_features = torch.randn(batch_size, seq_len, d_model_size, device=device, dtype=torch.float32)

    try:
        print("\nTesting LidarCameraFusionMambaV2...")
        output_features_v2 = fusion_mamba_v2(lidar_features, camera_features)
        print("LidarCameraFusionMambaV2 instantiated and forward pass completed.")
        print("Input camera shape:", camera_features.shape)
        print("Output camera shape (V2):", output_features_v2.shape)
        assert output_features_v2.shape == camera_features.shape
        print("Shape assertion passed for V2.")

        # Test with a previous LidarCameraFusionMamba if available (for comparison or context)
        # Assuming LidarCameraFusionMamba (V1) is defined in the same scope or imported
        # from your previous code block.
        # For this test, we need its definition.
        # If LidarCameraFusionMamba is not defined, this part will error or should be skipped.
        try:
            from plugin.dssmss.mamba.lidar_camera_fusion_mamba import LidarCameraFusionMamba # Assuming V1 is here
            print("\nTesting LidarCameraFusionMamba (V1 for comparison, if available)...")
            fusion_mamba_v1 = LidarCameraFusionMamba(
                d_model=d_model_size,
                d_state=d_state_size,
                d_conv=d_conv_size, # V1 also uses d_conv
                expand=expand_factor,
                device=device,
                dtype=torch.float32
            ).to(device)
            output_features_v1 = fusion_mamba_v1(lidar_features, camera_features)
            print("LidarCameraFusionMamba (V1) forward pass completed.")
            print("Output camera shape (V1):", output_features_v1.shape)
            assert output_features_v1.shape == camera_features.shape
        except ImportError:
            print("Skipping V1 comparison as LidarCameraFusionMamba definition not found.")
        except NameError:
             print("Skipping V1 comparison as LidarCameraFusionMamba definition not found.")
        except Exception as e_v1:
            print(f"Error during V1 test: {e_v1}")


    except NotImplementedError as e:
        print(f"NotImplementedError during V2 example usage: {e}. This likely means selective_scan_fn is mocked and hit a path not fully implemented in mock.")
    except Exception as e:
        print(f"An error occurred during V2 example usage: {e}")
        import traceback
        traceback.print_exc()


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