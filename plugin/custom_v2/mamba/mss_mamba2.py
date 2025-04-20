import math
import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from plugin.custom_v2.models.utils import get_sorted_indices, extract_features_by_indices, remap_vertical_to_horizontal, insert_features_back_batched


class MSSMamba2(nn.Module):
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
        self.in_proj_f_horizen = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.in_proj_b_horizen = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.in_proj_f_vertical = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.in_proj_b_vertical = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        # Conv1d
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d_f_horizen = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_b_horizen = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_f_vertical = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_b_vertical = nn.Conv1d(
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
        dt_f_horizen = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_b_horizen = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_f_vertical = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_b_vertical = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt_f_horizen = torch.clamp(dt_f_horizen, min=dt_init_floor)
        dt_b_horizen = torch.clamp(dt_b_horizen, min=dt_init_floor)
        dt_f_vertical = torch.clamp(dt_f_vertical, min=dt_init_floor)
        dt_b_vertical = torch.clamp(dt_b_vertical, min=dt_init_floor)
        inv_dt_f_horizen = dt_f_horizen + torch.log(-torch.expm1(-dt_f_horizen))
        inv_dt_b_horizen = dt_b_horizen + torch.log(-torch.expm1(-dt_b_horizen))
        inv_dt_f_vertical = dt_f_vertical + torch.log(-torch.expm1(-dt_f_vertical))
        inv_dt_b_vertical = dt_b_vertical + torch.log(-torch.expm1(-dt_b_vertical))
        self.dt_bias_f_horizen = nn.Parameter(inv_dt_f_horizen)
        self.dt_bias_b_horizen = nn.Parameter(inv_dt_b_horizen)
        self.dt_bias_f_vertical = nn.Parameter(inv_dt_f_vertical)
        self.dt_bias_b_vertical = nn.Parameter(inv_dt_b_vertical)
        self.dt_bias_f_horizen._no_weight_decay = True
        self.dt_bias_b_horizen._no_weight_decay = True
        self.dt_bias_f_vertical._no_weight_decay = True
        self.dt_bias_b_vertical._no_weight_decay = True
        # A
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A_f_horizen = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_b_horizen = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_f_vertical = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_b_vertical = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log_f_horizen = torch.log(A_f_horizen).to(dtype=dtype)
        A_log_b_horizen = torch.log(A_b_horizen).to(dtype=dtype)
        A_log_f_vertical = torch.log(A_f_vertical).to(dtype=dtype)
        A_log_b_vertical = torch.log(A_b_vertical).to(dtype=dtype)
        self.A_log_f_horizen = nn.Parameter(A_log_f_horizen)
        self.A_log_b_horizen = nn.Parameter(A_log_b_horizen)
        self.A_log_f_vertical = nn.Parameter(A_log_f_vertical)
        self.A_log_b_vertical = nn.Parameter(A_log_b_vertical)
        self.A_log_f_horizen._no_weight_decay = True
        self.A_log_b_horizen._no_weight_decay = True
        self.A_log_f_vertical._no_weight_decay = True
        self.A_log_b_vertical._no_weight_decay = True
        # D
        self.D_f_horizen = nn.Parameter(
            torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device)
        )
        self.D_b_horizen = nn.Parameter(
            torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device)
        )
        self.D_f_vertical = nn.Parameter(
            torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device)
        )
        self.D_b_vertical = nn.Parameter(
            torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device)
        )
        self.D_f_horizen._no_weight_decay = True
        self.D_b_horizen._no_weight_decay = True
        self.D_f_vertical._no_weight_decay = True
        self.D_b_vertical._no_weight_decay = True
        # Norm
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm_f_horizen = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )
            self.norm_b_horizen = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )
            self.norm_f_vertical = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )
            self.norm_b_vertical = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )
        # Out proj
        self.out_proj_f_horizen = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.out_proj_b_horizen = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.out_proj_f_vertical = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.out_proj_b_vertical = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        # Output
        self.out_act = nn.SiLU()
        self.out_proj = nn.Linear(
            self.d_model * 4, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, multi_scale_pts_feats, foreground_predictions, seq_idx=None):
        horizen_indices, vertical_indices = get_sorted_indices(foreground_predictions)
        h_features, v_features = extract_features_by_indices(multi_scale_pts_feats, horizen_indices, vertical_indices)
        heights = []
        widths = []
        for i in range(len(multi_scale_pts_feats)):
            heights.append(multi_scale_pts_feats[i].shape[2])
            widths.append(multi_scale_pts_feats[i].shape[3])
        # h_features = torch.stack(h_features, dim=0)
        # v_features = torch.stack(v_features, dim=0)
        # h_features = h_features.permute(0, 2, 1).contiguous()
        # v_features = v_features.permute(0, 2, 1).contiguous()
        u_f_horizen = h_features
        u_b_horizen = h_features.flip([1])
        u_f_vertical = v_features
        u_b_vertical = v_features.flip([1])
        zxbcdt_f_horizen = self.in_proj_f_horizen(u_f_horizen)
        zxbcdt_b_horizen = self.in_proj_b_horizen(u_b_horizen)
        zxbcdt_f_vertical = self.in_proj_f_vertical(u_f_vertical)
        zxbcdt_b_vertical = self.in_proj_b_vertical(u_b_vertical)
        A_f_horizen = -torch.exp(self.A_log_f_horizen.float())
        A_b_horizen = -torch.exp(self.A_log_b_horizen.float())
        A_f_vertical = -torch.exp(self.A_log_f_vertical.float())
        A_b_vertical = -torch.exp(self.A_log_b_vertical.float())
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )
        out_f_horizen = mamba_split_conv1d_scan_combined(
            zxbcdt_f_horizen,
            rearrange(self.conv1d_f_horizen.weight, "d 1 w -> d w"),
            self.conv1d_f_horizen.bias,
            self.dt_bias_f_horizen,
            A_f_horizen,
            D=rearrange(self.D_f_horizen, "(h p) -> h p", p=self.headdim)
            if self.D_has_hdim
            else self.D_f_horizen,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm_f_horizen.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm_f_horizen.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj_f_horizen.weight,
            outproj_bias=self.out_proj_f_horizen.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
        )
        out_b_horizen = mamba_split_conv1d_scan_combined(
            zxbcdt_b_horizen,
            rearrange(self.conv1d_b_horizen.weight, "d 1 w -> d w"),
            self.conv1d_b_horizen.bias,
            self.dt_bias_b_horizen,
            A_b_horizen,
            D=rearrange(self.D_b_horizen, "(h p) -> h p", p=self.headdim)
            if self.D_has_hdim
            else self.D_b_horizen,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm_b_horizen.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm_b_horizen.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj_b_horizen.weight,
            outproj_bias=self.out_proj_b_horizen.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
        )
        out_f_vertical = mamba_split_conv1d_scan_combined(
            zxbcdt_f_vertical,
            rearrange(self.conv1d_f_vertical.weight, "d 1 w -> d w"),
            self.conv1d_f_vertical.bias,
            self.dt_bias_f_vertical,
            A_f_vertical,
            D=rearrange(self.D_f_vertical, "(h p) -> h p", p=self.headdim)
            if self.D_has_hdim
            else self.D_f_vertical,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm_f_vertical.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm_f_vertical.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj_f_vertical.weight,
            outproj_bias=self.out_proj_f_vertical.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
        )
        out_b_vertical = mamba_split_conv1d_scan_combined(
            zxbcdt_b_vertical,
            rearrange(self.conv1d_b_vertical.weight, "d 1 w -> d w"),
            self.conv1d_b_vertical.bias,
            self.dt_bias_b_vertical,
            A_b_vertical,
            D=rearrange(self.D_b_vertical, "(h p) -> h p", p=self.headdim)
            if self.D_has_hdim
            else self.D_b_vertical,
            chunk_size=self.chunk_size,
            seq_idx=seq_idx,
            activation=self.activation,
            rmsnorm_weight=self.norm_b_vertical.weight if self.rmsnorm else None,
            rmsnorm_eps=self.norm_b_vertical.eps if self.rmsnorm else 1e-6,
            outproj_weight=self.out_proj_b_vertical.weight,
            outproj_bias=self.out_proj_b_vertical.bias,
            headdim=None if self.D_has_hdim else self.headdim,
            ngroups=self.ngroups,
            norm_before_gate=self.norm_before_gate,
            **dt_limit_kwargs,
        )
        out = self.out_act(torch.cat([out_f_horizen, out_b_horizen.flip([1]), remap_vertical_to_horizontal(out_f_vertical, vertical_indices, horizen_indices), remap_vertical_to_horizontal(out_b_vertical.flip([1]), vertical_indices, horizen_indices)], dim=-1))
        out = self.out_proj(out)
        out = insert_features_back_batched(out, horizen_indices, heights, widths)
        return out

if __name__ == "__main__":
    input_data = [
        torch.tensor([
            # batch 1
            [[1, 2],
            [3, 4]], 
            # batch 2
            [[5, 6],
            [7, 8]]
        ]).float().unsqueeze(1).to("cuda"),  # (2,1,2,2)     
        
        torch.tensor([
            # batch 1
            [[10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]],
            # batch 2
            [[20, 21, 22],
            [23, 24, 25],
            [26, 27, 28]]
        ]).float().unsqueeze(1).to("cuda")  # (2,1,3,3)
    ]

    foreground_predictions = [
        torch.tensor([
            [[1, 0],
            [1, 1]],
            [[1, 1],
            [0, 1]]
        ]).float().unsqueeze(1).to("cuda"),  # (2,1,2,2)
        
        torch.tensor([
            [[1, 1, 1],
            [0, 1, 0],
            [1, 1, 1]],
            [[1, 0, 1],
            [1, 1, 1],
            [0, 1, 0]]
        ]).float().unsqueeze(1).to("cuda")  # (2,1,3,3)
    ]
    def expand_channels(input_list, target_channels=256):
        """
        将输入tensor列表的通道维度扩展到指定大小，保持每个通道的值相同
        Args:
            input_list: List[Tensor] 输入tensor列表
            target_channels: int 目标通道数
        Returns:
            List[Tensor] 扩展通道后的tensor列表
        """
        expanded_list = []
        
        for tensor in input_list:
            # 获取原始形状
            b, _, h, w = tensor.shape
            
            # 扩展通道维度
            expanded = tensor.expand(b, target_channels, h, w)
            expanded_list.append(expanded)
        
        return expanded_list
    # 扩展通道维度到256
    input_expanded = expand_channels(input_data, 256)
    model = MSSMamba2(d_model=256).to("cuda")
    output = model(input_expanded, foreground_predictions)
    import ipdb; ipdb.set_trace()

