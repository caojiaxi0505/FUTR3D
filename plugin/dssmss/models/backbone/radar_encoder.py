import torch
from mmcv.cnn import build_norm_layer
from mmdet3d.models.builder import BACKBONES
from mmdet3d.models.builder import build_backbone
from torch import nn
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict

__all__ = ["RadarFeatureNet", "RadarEncoder"]


def get_paddings_indicator(actual_num, max_num, axis=0):
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class RFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        super().__init__()
        self.name = "RFNLayer"
        self.last_vfe = last_layer
        self.units = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg
        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        if self.last_vfe:
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            return x_max
        else:
            return x


@BACKBONES.register_module(force=True)
class RadarFeatureNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        feat_channels=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
    ):
        super().__init__()
        self.name = "RadarFeatureNet"
        assert len(feat_channels) > 0
        self.in_channels = in_channels
        in_channels += 2
        self._with_distance = with_distance
        feat_channels = [in_channels] + list(feat_channels)
        rfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            rfn_layers.append(
                RFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.rfn_layers = nn.ModuleList(rfn_layers)
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.pc_range = point_cloud_range

    def forward(self, features, num_voxels, coors):
        dtype = features.dtype
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 1].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )
        features[:, :, 0:1] = (features[:, :, 0:1] - self.pc_range[0]) / (
            self.pc_range[3] - self.pc_range[0]
        )
        features[:, :, 1:2] = (features[:, :, 1:2] - self.pc_range[1]) / (
            self.pc_range[4] - self.pc_range[1]
        )
        features[:, :, 2:3] = (features[:, :, 2:3] - self.pc_range[2]) / (
            self.pc_range[5] - self.pc_range[2]
        )
        features_ls = [features, f_center]
        features = torch.cat(features_ls, dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask
        for rfn in self.rfn_layers:
            features = rfn(features)
        return features.squeeze()


@BACKBONES.register_module(force=True)
class RadarEncoder(nn.Module):
    def __init__(
        self,
        pts_voxel_encoder: Dict[str, Any],
        pts_middle_encoder: Dict[str, Any],
        pts_bev_encoder=None,
        **kwargs,
    ):
        super().__init__()
        self.pts_voxel_encoder = build_backbone(pts_voxel_encoder)
        self.pts_middle_encoder = build_backbone(pts_middle_encoder)
        if pts_bev_encoder is not None:
            self.pts_bev_encoder = build_backbone(pts_bev_encoder)
        else:
            self.pts_bev_encoder = None

    def forward(self, feats, coords, batch_size, sizes):
        x = self.pts_voxel_encoder(feats, sizes, coords)
        x = self.pts_middle_encoder(x, coords, batch_size)
        if self.pts_bev_encoder is not None:
            x = self.pts_bev_encoder(x)
        return x
