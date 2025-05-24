import torch
from torch.utils.checkpoint import checkpoint
from mmcv.cnn import build_norm_layer
from torch import nn
from typing import Dict
from .dss_mamba import DSSMamba, dssmamba_g, dssmamba_h, dssmamba_l, dssmamba_m, dssmamba_s, dssmamba_s_morton_conv
# from .dss_mamba2 import DSSMamba2
from timm.layers import DropPath

class DSS(nn.Module):
    def __init__(
        self,
        batch_first: bool = None,
        drop_prob: float = None,
        mamba_prenorm: bool = None,
        mamba_cfg: Dict = None,
        mamba_version: str = None,  # dss_mamba 或 dss_mamba2
        num_layers: int = None,
    ):
        super(DSS, self).__init__()
        assert batch_first is not None, "batch_first must be provided"
        assert drop_prob is not None, "drop_prob must be provided"
        assert mamba_prenorm is not None, "mamba_prenorm must be provided"
        assert mamba_cfg is not None, "mamba_cfg must be provided"
        assert mamba_version is not None, "mamba_version must be provided"
        assert num_layers is not None, "num_layers must be provided"
        self.batch_first = batch_first
        self.mamba_prenorm = mamba_prenorm
        drop_path_rate = [x.item() for x in torch.linspace(0, drop_prob, num_layers)]
        if mamba_version == "dssmamba_s_morton_conv":
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "mamba": dssmamba_s_morton_conv(),
                            "dropout": (
                                DropPath(drop_path_rate[i])
                                if drop_path_rate[i] > 0
                                else nn.Identity()
                            ),
                            "norm": build_norm_layer(dict(type="LN"), 256)[1],
                        }
                    )
                    for i in range(num_layers)
                ]
            )
        elif mamba_version == "dssmamba_s":
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "mamba": dssmamba_s(),
                            "dropout": (
                                DropPath(drop_path_rate[i])
                                if drop_path_rate[i] > 0
                                else nn.Identity()
                            ),
                            "norm": build_norm_layer(dict(type="LN"), 256)[1],
                        }
                    )
                    for i in range(num_layers)
                ]
            )
        elif mamba_version == "dssmamba_m":
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "mamba": dssmamba_m(),
                            "dropout": (
                                DropPath(drop_path_rate[i])
                                if drop_path_rate[i] > 0
                                else nn.Identity()
                            ),
                            "norm": build_norm_layer(dict(type="LN"), 256)[1],
                        }
                    )
                    for i in range(num_layers)
                ]
            )
        elif mamba_version == "dssmamba_l":
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "mamba": dssmamba_l(),
                            "dropout": (
                                DropPath(drop_path_rate[i])
                                if drop_path_rate[i] > 0
                                else nn.Identity()
                            ),
                            "norm": build_norm_layer(dict(type="LN"), 256)[1],
                        }
                    )
                    for i in range(num_layers)
                ]
            )
        elif mamba_version == "dssmamba_h":
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "mamba": dssmamba_h(),
                            "dropout": (
                                DropPath(drop_path_rate[i])
                                if drop_path_rate[i] > 0
                                else nn.Identity()
                            ),
                            "norm": build_norm_layer(dict(type="LN"), 256)[1],
                        }
                    )
                    for i in range(num_layers)
                ]
            )
        elif mamba_version == "dssmamba_g":
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "mamba": dssmamba_g(),
                            "dropout": (
                                DropPath(drop_path_rate[i])
                                if drop_path_rate[i] > 0
                                else nn.Identity()
                            ),
                            "norm": build_norm_layer(dict(type="LN"), 256)[1],
                        }
                    )
                    for i in range(num_layers)
                ]
            )
        
        else:
            raise NotImplementedError

    def forward(self, query, query_pos, reference_points=None, grid_size=180, spatial_scale=1.):
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            query_pos = query_pos.permute(1, 0, 2)
        x = query + query_pos
        residual = torch.zeros_like(x)
        for i, layer in enumerate(self.layers):
            if self.mamba_prenorm:
                residual = x
                x = layer["norm"](x)
                if isinstance(layer["mamba"], DSSMamba) and layer["mamba"].morton_rearrange:
                    x = layer["mamba"](x, reference_points, grid_size, spatial_scale)
                else:
                    x = layer["mamba"](x)
                x = layer["dropout"](x)
                x = residual + x
            else:
                residual = x
                if isinstance(layer["mamba"], DSSMamba) and layer["mamba"].morton_rearrange:
                    x = layer["mamba"](x, reference_points, grid_size, spatial_scale)
                else:
                    x = layer["mamba"](x)
                x = layer["dropout"](x)
                x = residual + x
                x = layer["norm"](x) if i < len(self.layers) - 1 else x
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        return x
