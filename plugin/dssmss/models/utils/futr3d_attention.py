import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from mmcv.cnn import constant_init, xavier_init
from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import (
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    build_transformer_layer_sequence,
)
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch,
)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE
from mmdet.models.utils.builder import TRANSFORMER
from plugin.dssmss.mamba.lidar_camera_fusion_mamba import LidarCameraFusionMamba, LidarCameraFusionMambaV2, LidarCameraFusionMambaBlock

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@ATTENTION.register_module(force=True)
class FUTR3DAttention(BaseModule):
    def __init__(
        self,
        use_lidar=True,
        use_camera=False,
        use_radar=False,
        embed_dims=256,
        radar_dims=64,
        num_cams=6,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        pc_range=None,
        rad_cuda=True,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar
        self.pc_range = pc_range
        self.num_cams = num_cams
        self.rad_cuda = rad_cuda

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in MultiScaleDeformAttention to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation."
            )
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.fused_embed = 0
        if self.use_lidar:
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2
            )
            self.attention_weights = nn.Linear(
                embed_dims, num_heads * num_levels * num_points
            )
            self.value_proj = nn.Linear(embed_dims, embed_dims)
            self.output_proj = nn.Linear(embed_dims, embed_dims)
            self.fused_embed += embed_dims
        if self.use_camera:
            self.img_attention_weights = nn.Linear(embed_dims, num_cams * num_levels)
            # 专为petr设计
            # self.img_attention_weights = nn.Linear(embed_dims, num_cams * 2)
            self.img_output_proj = nn.Linear(embed_dims, embed_dims)
            self.position_encoder = nn.Sequential(
                nn.Linear(3, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )
            self.weight_dropout = nn.Dropout(0.0)
            self.fused_embed += embed_dims
        if self.use_radar:
            self.rad_sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_points * 2
            )
            self.rad_attention_weights = nn.Linear(embed_dims, num_heads * num_points)
            self.rad_value_proj = nn.Linear(radar_dims, radar_dims)
            self.rad_output_proj = nn.Linear(radar_dims, radar_dims)
            self.fused_embed += radar_dims

        if self.fused_embed > embed_dims:
            self.fused_embed += embed_dims
            self.modality_fusion_layer = nn.Sequential(
                nn.Linear(self.fused_embed, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=False),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            )
        self.init_weights()
        self.camera_mixer = LidarCameraFusionMambaBlock(num_layer=2, layer_type='fusion_v2', d_model=256)

    def init_weights(self):
        device = next(self.parameters()).device
        thetas = torch.arange(self.num_heads, dtype=torch.float32, device=device) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        if self.use_lidar:
            constant_init(self.sampling_offsets, 0.0)
            self.sampling_offsets.bias.data = grid_init.view(-1)
            constant_init(self.attention_weights, val=0.0, bias=0.0)
            xavier_init(self.value_proj, distribution="uniform", bias=0.0)
            xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        if self.use_camera:
            constant_init(self.img_attention_weights, val=0.0, bias=0.0)
            xavier_init(self.img_output_proj, distribution="uniform", bias=0.0)
        if self.use_radar:
            constant_init(self.rad_sampling_offsets, 0.0)
            self.rad_sampling_offsets.bias.data = grid_init[:, 0].reshape(-1)
            constant_init(self.rad_attention_weights, val=0.0, bias=0.0)
            xavier_init(self.rad_value_proj, distribution="uniform", bias=0.0)
            xavier_init(self.rad_output_proj, distribution="uniform", bias=0.0)
        self._is_init = True

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        pts_feats=None,
        img_feats=None,
        rad_feats=None,
        rad_key_padding_mask=None,
        rad_spatial_shapes=None,
        rad_level_start_index=None,
        **kwargs,
    ):
        if value is None:
            value = query       # value赋值query
        if identity is None:
            identity = query    # identity赋值初始query
        if query_pos is not None:
            query = query + query_pos   # 为query添加pe，pe是reference_points的sincospe+mlp
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, _ = query.shape
        if self.use_lidar:
            value = pts_feats   # with shape [bs, sum{hi*wi}, embed_dims], embed_dims=256
            bs, num_value, _ = value.shape
            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
            value = self.value_proj(value)  # 对lidar_feats进行线性变换
            if key_padding_mask is not None:    # key_padding_mask全为False，实际上并没有进行padding操作
                value = value.masked_fill(key_padding_mask[..., None], 0.0)
            value = value.view(bs, num_value, self.num_heads, -1)   # 转化为8头，每个头单独处理32维度
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
            )   # 使用query生成偏移量，query的shape为bs,nq,c，经过linear变换后为bs,nq,c，reshape为bs,nq,nh,nl,np,2，注意c为256，nh*nl*np*2=8*4*4*2=256=c
            attention_weights = self.attention_weights(query).view(
                bs, num_query, self.num_heads, self.num_levels * self.num_points
            )   # 使用query生成注意力权重，即每个采样点提取特征的权重，经过linear变换后为bs,nq,c/2，reshape为bs,nh,nq,nl,np，注意nh*nq*nl*np=8*4*4=128=c/2
            attention_weights = attention_weights.softmax(-1)   # 对nl,np维度做softmax，即多尺度提取出的特征权重和为1
            attention_weights = attention_weights.view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points
            )
            ref_points = reference_points.unsqueeze(2).expand(
                -1, -1, self.num_levels, -1
            )   # 这里的ref_points形状为bs,nq,3，表示三维坐标，已经经过sigmoid处理，范围在0-1之间
            ref_points = ref_points[..., :2]
            if ref_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
                )
                sampling_locations = (
                    ref_points[:, :, None, :, None, :]
                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                )
            else:
                raise ValueError(
                    f"Last dim of reference_points must be 2, but get {reference_points.shape[-1]} instead."
                )
            if (IS_CUDA_AVAILABLE and value.is_cuda) or (
                IS_MLU_AVAILABLE and value.is_mlu
            ):
                output = MultiScaleDeformableAttnFunction.apply(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            else:
                output = multi_scale_deformable_attn_pytorch(
                    value, spatial_shapes, sampling_locations, attention_weights
                )
            pts_output = self.output_proj(output)   # 将输出再经过linear进行线性变换
        if self.use_camera:
            img_attention_weights = self.img_attention_weights(query).view(
                bs, 1, num_query, self.num_cams, 1, self.num_levels
            )   # 与lidar部分类似，得到权重，权重的形状为bs,1,nq,6,1,nl
            # # 专为petr设计
            # img_attention_weights = self.img_attention_weights(query).view(
            #     bs, 1, num_query, self.num_cams, 1, 2
            # )   # 与lidar部分类似，得到权重，权重的形状为bs,1,nq,6,1,nl
            reference_points_3d, img_output, mask = feature_sampling(
                img_feats, reference_points, self.pc_range, kwargs["img_metas"]
            )   # img_output的形状为bs,c,nq,6,1,nl，其中c=256，nq=4，6为相机数量，nl=4为特征层数
            img_output = torch.nan_to_num(img_output)
            mask = torch.nan_to_num(mask)
            img_attention_weights = (
                self.weight_dropout(img_attention_weights.sigmoid()) * mask
            )
            img_output = img_output * img_attention_weights
            img_output = img_output.sum(-1).sum(-1).sum(-1)
            img_output = img_output.permute(0, 2, 1)
            img_output = self.img_output_proj(img_output)
        if self.use_radar:
            value = rad_feats
            bs, num_value, _ = value.shape
            assert (
                rad_spatial_shapes[:, 0] * rad_spatial_shapes[:, 1]
            ).sum() == num_value
            value = self.rad_value_proj(value)
            if rad_key_padding_mask is not None:
                value = value.masked_fill(rad_key_padding_mask[..., None], 0.0)
            value = value.view(bs, num_value, self.num_heads, -1)
            sampling_offsets = self.rad_sampling_offsets(query).view(
                bs, num_query, self.num_heads, 1, self.num_points, 2
            )
            attention_weights = self.rad_attention_weights(query).view(
                bs, num_query, self.num_heads, self.num_points
            )
            attention_weights = attention_weights.softmax(-1)
            attention_weights = attention_weights.view(
                bs, num_query, self.num_heads, 1, self.num_points
            )
            ref_points = reference_points.unsqueeze(2).expand(-1, -1, 1, -1)
            ref_points = ref_points[..., :2]
            if ref_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [rad_spatial_shapes[..., 1], rad_spatial_shapes[..., 0]], -1
                )
                sampling_locations = (
                    ref_points[:, :, None, :, None, :]
                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                )
            else:
                raise ValueError(
                    f"Last dim of reference_points must be 2, but get {reference_points.shape[-1]} instead."
                )
            if (IS_CUDA_AVAILABLE and value.is_cuda) or (
                IS_MLU_AVAILABLE and value.is_mlu
            ):
                output = MultiScaleDeformableAttnFunction.apply(
                    value,
                    rad_spatial_shapes,
                    rad_level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            else:
                output = multi_scale_deformable_attn_pytorch(
                    value, rad_spatial_shapes, sampling_locations, attention_weights
                )
            radar_output = self.rad_output_proj(output)
        if self.use_lidar and self.use_camera and self.use_radar:
            output = torch.cat((img_output, pts_output, radar_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif self.use_lidar and self.use_camera:
            img_output_processed = self.camera_mixer(pts_output, img_output)
            # output = torch.cat((img_output, pts_output), dim=2)
            output = torch.cat((img_output_processed, img_output, pts_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif self.use_camera and self.use_radar:
            output = torch.cat((img_output, radar_output), dim=2)
            output = self.modality_fusion_layer(output)
        elif self.use_lidar:
            output = pts_output
        elif self.use_camera:
            output = img_output
        elif self.use_radar:
            output = img_output
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity


def feature_sampling(mlvl_feats, reference_points, pc_range, img_metas):
    # TODO: img_metas使用第一个batch，检查batch不为1的时候，img_metas[i]["img_shape"][0][1]和img_metas[i]["img_shape"][0][0]是否不一致
    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta["lidar2img"])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = (
        reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )
    reference_points[..., 1:2] = (
        reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )
    reference_points[..., 2:3] = (
        reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    )
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1
    )
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = (
        reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    )
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
    eps = 1e-5
    mask = reference_points_cam[..., 2:3] > eps
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3],
        torch.ones_like(reference_points_cam[..., 2:3]) * eps,
    )
    reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
    reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (
        mask
        & (reference_points_cam[..., 0:1] > -1.0)
        & (reference_points_cam[..., 0:1] < 1.0)
        & (reference_points_cam[..., 1:2] > -1.0)
        & (reference_points_cam[..., 1:2] < 1.0)
    )
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B * N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask
