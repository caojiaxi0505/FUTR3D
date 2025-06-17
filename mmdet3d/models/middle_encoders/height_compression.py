# 说明：该文件是LION的依赖
import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['NUM_BEV_FEATURES']

    def forward(self, batch_dict):
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict


class HeightCompression_None(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['NUM_BEV_FEATURES']

    def forward(self, batch_dict):
        batch_dict['spatial_features'] = batch_dict['multi_scale_2d_features']['x_conv5']
        return batch_dict
    

class HeightCompression_VoxelNext(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['NUM_BEV_FEATURES']

    def forward(self, batch_dict):
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict


class PseudoHeightCompression(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg['NUM_BEV_FEATURES']

    def forward(self, batch_dict):
        batch_dict['spatial_features'] = batch_dict['spatial_features_2d']
        return batch_dict
