import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import ssd_selective_scan

class ForePredNet(nn.Module):
    def __init__(self, in_channels=256, intermediate_channels=16, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # self._init_weights()

    def forward(self, x):
        return self.net(x)

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)

class MSSMamba(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_state=8,
        expand=1,
        # >>> 一般不做修改的参数 >>>
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        chunk_size=256,
        # <<< 一般不做修改的参数 <<<
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = (self.expand * self.d_model)
        self.headdim = headdim
        self.d_ssm = self.d_inner
        self.ngroups = ngroups
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        assert self.d_ssm % self.headdim == 0
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj_H = nn.Linear(self.d_model, d_in_proj*2, bias=bias, **factory_kwargs)
        self.in_proj_V = nn.Linear(self.d_model, d_in_proj*2, bias=bias, **factory_kwargs)
        self.act = nn.SiLU()
        dt_HF = torch.clamp(torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)), min=dt_init_floor)
        dt_HB = torch.clamp(torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)), min=dt_init_floor)
        dt_VF = torch.clamp(torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)), min=dt_init_floor)
        dt_VB = torch.clamp(torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)), min=dt_init_floor)
        inv_dt_HF = dt_HF + torch.log(-torch.expm1(-dt_HF))
        inv_dt_HB = dt_HB + torch.log(-torch.expm1(-dt_HB))
        inv_dt_VF = dt_VF + torch.log(-torch.expm1(-dt_VF))
        inv_dt_VB = dt_VB + torch.log(-torch.expm1(-dt_VB))
        self.dt_bias_HF = nn.Parameter(inv_dt_HF)
        self.dt_bias_HB = nn.Parameter(inv_dt_HB)
        self.dt_bias_VF = nn.Parameter(inv_dt_VF)
        self.dt_bias_VB = nn.Parameter(inv_dt_VB)
        self.dt_bias_HF._no_weight_decay = True
        self.dt_bias_HB._no_weight_decay = True
        self.dt_bias_VF._no_weight_decay = True
        self.dt_bias_VB._no_weight_decay = True
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A_HF = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_HB = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_VF = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_VB = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log_HF = torch.log(A_HF).to(dtype=dtype)
        A_log_HB = torch.log(A_HB).to(dtype=dtype)
        A_log_VF = torch.log(A_VF).to(dtype=dtype)
        A_log_VB = torch.log(A_VB).to(dtype=dtype)
        self.A_log_HF = nn.Parameter(A_log_HF)
        self.A_log_HB = nn.Parameter(A_log_HB)
        self.A_log_VF = nn.Parameter(A_log_VF)
        self.A_log_VB = nn.Parameter(A_log_VB)
        self.A_log_HF._no_weight_decay = True
        self.A_log_HB._no_weight_decay = True
        self.A_log_VF._no_weight_decay = True
        self.A_log_VB._no_weight_decay = True
        self.D_HF = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_HB = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_VF = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_VB = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_HF._no_weight_decay = True
        self.D_HB._no_weight_decay = True
        self.D_VF._no_weight_decay = True
        self.D_VB._no_weight_decay = True
        self.out_proj_HF = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_HB = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_VF = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_VB = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_model*4, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, pts_feats: list, forepred: list) -> list:
        morton_H_indices, morton_V_indices = self.get_indices(forepred)
        H_feats, V_feats = self.extract_features_by_indices(pts_feats, morton_H_indices, morton_V_indices)
        heights = []
        widths = []
        for i in range(len(pts_feats)):
            heights.append(pts_feats[i].shape[2])
            widths.append(pts_feats[i].shape[3])
        if H_feats.shape[1] == 0:
            out = []
            for i in range(len(pts_feats)):
                out.append(torch.zeros_like(pts_feats[i]))
            return out
        zxbcdt_HF, zxbcdt_HB = self.in_proj_H(H_feats).chunk(2, dim=2)
        zxbcdt_VF, zxbcdt_VB = self.in_proj_V(V_feats).chunk(2, dim=2)
        zxbcdt_HB = zxbcdt_HB.flip(dims=[1])
        zxbcdt_VB = zxbcdt_VB.flip(dims=[1])
        A_HF = -torch.exp(self.A_log_HF.float())
        A_HB = -torch.exp(self.A_log_HB.float())
        A_VF = -torch.exp(self.A_log_VF.float())
        A_VB = -torch.exp(self.A_log_VB.float())

        dim = self.nheads * self.headdim

        z_HF, xBC_HF, dt_HF = torch.split(zxbcdt_HF, [dim, dim + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        xBC_HF = self.act(xBC_HF)
        x_HF, B_HF, C_HF = torch.split(xBC_HF, [dim, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x_HF = rearrange(x_HF, "b l (h p) -> b l h p", h=self.nheads)
        B_HF = rearrange(B_HF, "b l (g n) -> b l g n", g=self.ngroups)
        C_HF = rearrange(C_HF, "b l (g n) -> b l g n", g=self.ngroups)
        z_HF = rearrange(z_HF, "b l (h p) -> b l h p", h=self.nheads)
        out_HF = ssd_selective_scan(x_HF, dt_HF.to(x_HF.dtype), A_HF, B_HF, C_HF, D=self.D_HF.float(), z=z_HF, dt_bias=self.dt_bias_HF, dt_softplus=True, dt_limit=(0.0, float("inf")))
        out_HF = rearrange(out_HF, "b s h p -> b s (h p)")
        out_HF = self.out_proj_HB(out_HF)

        z_HB, xBC_HB, dt_HB = torch.split(zxbcdt_HB, [dim, dim + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        xBC_HB = self.act(xBC_HB)
        x_HB, B_HB, C_HB = torch.split(xBC_HB, [dim, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x_HB = rearrange(x_HB, "b l (h p) -> b l h p", h=self.nheads)
        B_HB = rearrange(B_HB, "b l (g n) -> b l g n", g=self.ngroups)
        C_HB = rearrange(C_HB, "b l (g n) -> b l g n", g=self.ngroups)
        z_HB = rearrange(z_HB, "b l (h p) -> b l h p", h=self.nheads)
        out_HB = ssd_selective_scan(x_HB, dt_HB.to(x_HB.dtype), A_HB, B_HB, C_HB, D=self.D_HB.float(), z=z_HB, dt_bias=self.dt_bias_HB, dt_softplus=True, dt_limit=(0.0, float("inf")))
        out_HB = rearrange(out_HB, "b s h p -> b s (h p)")
        out_HB = self.out_proj_HB(out_HB)
        out_HB = out_HB.flip(dims=[1])

        z_VF, xBC_VF, dt_VF = torch.split(zxbcdt_VF, [dim, dim + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        xBC_VF = self.act(xBC_VF)
        x_VF, B_VF, C_VF = torch.split(xBC_VF, [dim, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x_VF = rearrange(x_VF, "b l (h p) -> b l h p", h=self.nheads)
        B_VF = rearrange(B_VF, "b l (g n) -> b l g n", g=self.ngroups)
        C_VF = rearrange(C_VF, "b l (g n) -> b l g n", g=self.ngroups)
        z_VF = rearrange(z_VF, "b l (h p) -> b l h p", h=self.nheads)
        out_VF = ssd_selective_scan(x_VF, dt_VF.to(x_VF.dtype), A_VF, B_VF, C_VF, D=self.D_VF.float(), z=z_VF, dt_bias=self.dt_bias_VF, dt_softplus=True, dt_limit=(0.0, float("inf")))
        out_VF = rearrange(out_VF, "b s h p -> b s (h p)")
        out_VF = self.out_proj_VF(out_VF)

        z_VB, xBC_VB, dt_VB = torch.split(zxbcdt_VB, [dim, dim + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        xBC_VB = self.act(xBC_VB)
        x_VB, B_VB, C_VB = torch.split(xBC_VB, [dim, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x_VB = rearrange(x_VB, "b l (h p) -> b l h p", h=self.nheads)
        B_VB = rearrange(B_VB, "b l (g n) -> b l g n", g=self.ngroups)
        C_VB = rearrange(C_VB, "b l (g n) -> b l g n", g=self.ngroups)
        z_VB = rearrange(z_VB, "b l (h p) -> b l h p", h=self.nheads)
        out_VB = ssd_selective_scan(x_VB, dt_VB.to(x_VB.dtype), A_VB, B_VB, C_VB, D=self.D_VB.float(), z=z_VB, dt_bias=self.dt_bias_VB, dt_softplus=True, dt_limit=(0.0, float("inf")))
        out_VB = rearrange(out_VB, "b s h p -> b s (h p)")
        out_VB = self.out_proj_VB(out_VB)
        out_VB = out_VB.flip(dims=[1])

        out = self.out_act(torch.cat([out_HF, out_HB, remap_vertical_to_horizontal(out_VF, morton_V_indices, morton_H_indices), remap_vertical_to_horizontal(out_VB, morton_V_indices, morton_H_indices)], dim=-1))
        out = self.out_proj(out)
        out = insert_features_back_batched(out, morton_H_indices, heights, widths)
        return out
    
    def extract_features_by_indices(self, features, morton_H_indices, morton_V_indices):
        bs, c, device = features[0].shape[0], features[0].shape[1], features[0].device
        h_total_lens = torch.tensor([sum(h_indices[b].numel() for h_indices in morton_H_indices) for b in range(bs)], device=device)
        v_total_lens = torch.tensor([sum(v_indices[b].numel() for v_indices in morton_V_indices) for b in range(bs)], device=device)
        h_features = torch.zeros((bs, h_total_lens.max().item(), c), device=device)
        v_features = torch.zeros((bs, v_total_lens.max().item(), c), device=device)
        h_curr_pos = torch.zeros(bs, dtype=torch.long, device=device)
        v_curr_pos = torch.zeros(bs, dtype=torch.long, device=device)
        for feat, h_indices_scale, v_indices_scale in zip(features, morton_H_indices, morton_V_indices):
            feat_flat = feat.view(bs, c, -1)  # (B,C,H*W)
            for b in range(bs):
                h_idx = h_indices_scale[b]
                if h_idx.numel() > 0:
                    curr_pos = h_curr_pos[b]
                    next_pos = curr_pos + h_idx.numel()
                    h_features[b, curr_pos:next_pos] = feat_flat[b, :, h_idx].T  # (N, C)
                    h_curr_pos[b] = next_pos
            for b in range(bs):
                v_idx = v_indices_scale[b]
                if v_idx.numel() > 0:
                    curr_pos = v_curr_pos[b]
                    next_pos = curr_pos + v_idx.numel()
                    v_features[b, curr_pos:next_pos] = feat_flat[b, :, v_idx].T  # (N, C)
                    v_curr_pos[b] = next_pos
        return h_features, v_features

    def get_indices(self, forepred: list) -> tuple:
        morton_indices_list_1 = []
        morton_indices_list_2 = []
        batch_size = forepred[0].shape[0]
        for pred_mask in forepred:  # 不同尺度
            morton_indices_batch_1 = []
            morton_indices_batch_2 = []
            for b in range(batch_size):
                mask = pred_mask[b, 0] > 0.5  # 所有大于0.5的位置都是前景点
                idx_1, idx_2 = self.morton_code_extraction(mask)
                morton_indices_batch_1.append(idx_1)
                morton_indices_batch_2.append(idx_2)
            morton_indices_list_1.append(morton_indices_batch_1)
            morton_indices_list_2.append(morton_indices_batch_2)
        return morton_indices_list_1, morton_indices_list_2

    def morton_code_extraction(self, mask: torch.Tensor) -> tuple:
        device = mask.device
        h, w = mask.shape
        row_indices, col_indices = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')    # row_indices[2,3]的值为2，表示该位置的行索引为2，col_indices[2,3]的值为3，表示该位置的列索引为3
        row_indices = row_indices.flatten()
        col_indices = col_indices.flatten()
        valid_indices = mask.flatten() != 0
        row_indices = row_indices[valid_indices]
        col_indices = col_indices[valid_indices]
        morton_codes_1 = self.interleave_bits(col_indices, row_indices)
        morton_codes_2 = self.interleave_bits_x_last(col_indices, row_indices)
        sorted_indices_1 = torch.argsort(morton_codes_1)
        sorted_indices_2 = torch.argsort(morton_codes_2)
        linear_indices_1 = row_indices[sorted_indices_1] * w + col_indices[sorted_indices_1]
        linear_indices_2 = row_indices[sorted_indices_2] * w + col_indices[sorted_indices_2]
        return linear_indices_1, linear_indices_2

    def interleave_bits(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    def interleave_bits_x_last(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
    
def remap_vertical_to_horizontal(v_features: torch.Tensor, v_indices: list, h_indices: list) -> torch.Tensor:
    b, _, c = v_features.shape
    device = v_features.device
    batch_lengths = []
    v_start_indices = []
    h_start_indices = []
    for batch_idx in range(b):
        curr_length = sum(h_indices[i][batch_idx].numel() for i in range(len(h_indices)))
        batch_lengths.append(curr_length)
        v_starts = [0]
        for i in range(len(v_indices)-1):
            v_starts.append(v_starts[-1] + v_indices[i][batch_idx].numel())
        v_start_indices.append(v_starts)
        h_starts = [0]
        for i in range(len(h_indices)-1):
            h_starts.append(h_starts[-1] + h_indices[i][batch_idx].numel())
        h_start_indices.append(h_starts)
    max_length = max(batch_lengths)
    output = torch.zeros((b, max_length, c), device=device)
    for batch_idx in range(b):
        if batch_lengths[batch_idx] == 0:
            continue
        src_indices = []
        dst_indices = []
        for scale_idx in range(len(v_indices)):
            v_idx = v_indices[scale_idx][batch_idx]
            h_idx = h_indices[scale_idx][batch_idx]
            if v_idx.numel() > 0:
                v_start = v_start_indices[batch_idx][scale_idx]
                src_pos = torch.arange(v_idx.numel(), device=device) + v_start
                h_start = h_start_indices[batch_idx][scale_idx]
                _, sort_indices = h_idx.sort()
                dst_pos = torch.arange(h_idx.numel(), device=device) + h_start
                src_indices.append(src_pos)
                dst_indices.append(dst_pos[sort_indices])
        if src_indices:
            src_indices = torch.cat(src_indices)
            dst_indices = torch.cat(dst_indices)
            dst_expanded = dst_indices.unsqueeze(-1).expand(dst_indices.size(0), c)
            output[batch_idx].index_copy_(0, dst_indices, v_features[batch_idx, src_indices])
    return output

def insert_features_back_batched(features: torch.Tensor, h_indices: list, heights: list, widths: list) -> list:
    b, _, c = features.shape
    device = features.device
    output_features = []
    for scale_idx, (scale_indices, h, w) in enumerate(zip(h_indices, heights, widths)):
        scale_output = []
        for batch_idx in range(b):
            curr_output = torch.zeros((c, h * w), device=device)
            curr_indices = scale_indices[batch_idx]
            if curr_indices.numel() > 0:
                start_idx = sum(indices[batch_idx].shape[0] for indices in h_indices[:scale_idx])
                end_idx = start_idx + curr_indices.shape[0]
                curr_features = features[batch_idx, start_idx:end_idx]
                curr_output[:, curr_indices.long()] = curr_features.t()
            curr_output = curr_output.reshape(c, h, w)
            scale_output.append(curr_output)
        output_features.append(torch.stack(scale_output))
    return output_features

def generate_foregt(multi_scale_batch_bev_feats: list, gt_bboxes: list, bev_scales: tuple) -> list:
    bs = len(gt_bboxes)
    bev_x_min, bev_y_min, bev_x_max, bev_y_max = bev_scales
    bev_width = bev_x_max - bev_x_min
    bev_height = bev_y_max - bev_y_min
    device = multi_scale_batch_bev_feats[0].device
    def get_rotation_matrices(yaws: torch.Tensor) -> torch.Tensor:
        yaws = torch.pi - yaws  # 修正 yaw，使其与特征图的 row-col 坐标系对齐
        cos_yaws = torch.cos(yaws)
        sin_yaws = torch.sin(yaws)
        R = torch.stack([
            torch.stack([cos_yaws, -sin_yaws], dim=-1),
            torch.stack([sin_yaws, cos_yaws], dim=-1)
        ], dim=-2)
        return R
    gt_foreground = []
    for bev_feat in multi_scale_batch_bev_feats:
        _, _, feat_row, feat_col = bev_feat.shape
        scale_col = feat_col / bev_width
        scale_row = feat_row / bev_height
        foreground_mask = torch.zeros((bs, 1, feat_row, feat_col), device=device)
        for b_idx in range(bs):
            bboxes = gt_bboxes[b_idx]
            if bboxes.shape[0] == 0:
                continue
            centers = bboxes[:, :2]
            dims = bboxes[:, 2:4]
            yaws = bboxes[:, 4]
            R = get_rotation_matrices(yaws)
            grid_x, grid_y = torch.meshgrid(
                torch.arange(feat_col, device=device),
                torch.arange(feat_row, device=device), indexing="ij")
            grid_points = torch.stack([
                bev_x_min + grid_x.float() / scale_row,
                bev_y_min + grid_y.float() / scale_col], dim=-1)
            for bbox_idx in range(bboxes.shape[0]):
                rel_points = grid_points - centers[bbox_idx]
                rel_points_rotated = torch.matmul(rel_points.view(-1, 2), R[bbox_idx].t()).view(grid_x.shape + (2,))
                inside_mask = ((rel_points_rotated[..., 0].abs() <= dims[bbox_idx, 0] / 2) & (rel_points_rotated[..., 1].abs() <= dims[bbox_idx, 1] / 2))
                foreground_mask[b_idx, 0] = torch.logical_or(foreground_mask[b_idx, 0], inside_mask)
        gt_foreground.append(foreground_mask)
    return gt_foreground

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_gt_in_bev_space(gt_bboxes, bev_scales):
    bev_x_min, bev_y_min, bev_x_max, bev_y_max = bev_scales
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bev_x_min, bev_x_max)
    ax.set_ylim(bev_y_min, bev_y_max)
    ax.set_aspect('equal', 'box')
    for bboxes in gt_bboxes:
        for bbox in bboxes:
            center_x, center_y, width, height, yaw = bbox.cpu().numpy()
            corners = get_corners_from_bbox(center_x, center_y, width, height, yaw)
            ax.plot(
                [corners[0, 0], corners[1, 0], corners[2, 0], corners[3, 0], corners[0, 0]],
                [corners[0, 1], corners[1, 1], corners[2, 1], corners[3, 1], corners[0, 1]],
                color='r', linewidth=2)
    ax.set_title("Ground Truth Bounding Boxes in BEV Space")
    ax.grid(True)
    plt.show()

def plot_foreground_mask(foreground_masks):
    for i, mask in enumerate(foreground_masks):
        mask_numpy = mask[0, 0].cpu().numpy()  # 转换到 numpy 格式
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_numpy.T, cmap="gray", origin='lower')  # 设置原点在左下角
        plt.title(f"Foreground Mask for Scale {i + 1}")
        plt.colorbar()
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")
        plt.show()

def get_corners_from_bbox(center_x, center_y, width, height, yaw):
    half_width = width / 2
    half_height = height / 2
    local_corners = np.array([
        [ half_width,  half_height],
        [-half_width,  half_height],
        [-half_width, -half_height],
        [ half_width, -half_height]])
    rotation_matrix = np.array([
        [ np.cos(yaw), -np.sin(yaw)],
        [ np.sin(yaw),  np.cos(yaw)]])
    rotated_corners = np.dot(local_corners, rotation_matrix.T)  # dot 是二维旋转的线性变换
    corners = rotated_corners + [center_x, center_y]
    return corners

def test_generate_foreground():
    bev_feat = torch.zeros((1, 64, 100, 100), device='cuda')  # 单层特征图，Shape: (1, C, H, W)
    multi_scale_batch_bev_feats = [bev_feat]
    gt_bboxes = [
        torch.tensor([
            [0, 30, 20, 10, torch.pi / 6],
            [70, 70, 15, 25, torch.pi / 6]], device='cuda')]
    bev_scales = (-54, -54, 54, 54)
    foreground_masks = generate_foregt(multi_scale_batch_bev_feats, gt_bboxes, bev_scales)
    plot_gt_in_bev_space(gt_bboxes, bev_scales)
    plot_foreground_mask(foreground_masks)


if __name__ == "__main__":
    test_generate_foreground()