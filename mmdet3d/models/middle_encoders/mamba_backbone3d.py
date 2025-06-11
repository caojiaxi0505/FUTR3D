# 说明：该文件是LION的依赖
import spconv.pytorch as spconv
from functools import partial
import torch.nn as nn


class Sparse2LayerResBlock3D(spconv.SparseModule):
    def __init__(self, dim, indice_key):
        super(Sparse2LayerResBlock3D, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            indice_key=indice_key,
        )
        self.bn1 = nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01)
        self.conv2 = spconv.SubMConv3d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            indice_key=indice_key,
        )
        self.bn2 = nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.relu(self.bn2(out.features) + x.features))
        return out


class SparseDownBlock3D(spconv.SparseModule):
    def __init__(self, dim, kernel_size, stride, num_res_block3d, indice_key):
        super(SparseDownBlock3D, self).__init__()
        # 用于下采样，仅stride>1时才使用
        sparse_conv3d_block = spconv.SparseSequential(
            spconv.SparseConv3d(
                in_channels=dim,
                out_channels=dim,
                kennel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                indice_key=f"spconv_{indice_key}"),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        block_list = [sparse_conv3d_block if stride > 1 else nn.Identity()]
        for _ in range(num_res_block3d):
            block_list.append(Sparse2LayerResBlock3D(dim, indice_key=indice_key))
        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, x):
        return self.blocks(x)


class SparseUNetBlock3D(spconv.SparseModule):
    def __init__(
        self,
        dim,
        kernel_size,
        stride,
        num_res_block3d,
        indice_key,
    ):
        super().__init__()
        # 第一层不能下采样
        assert stride[0] == 1
        self.encoder = nn.ModuleList()
        for idx in range(len(stride)):
            self.encoder.append(
                SparseDownBlock3D(
                    dim,
                    kernel_size[idx],
                    stride[idx],
                    num_res_block3d[idx],
                    f"{indice_key}_{idx}",
                )
            )
        # 下采样的次数，因为第一层不能下采样，所以从第二层开始算
        downsample_times = len(stride[1:])
        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx, kernel_size in enumerate(kernel_size[1:]):
            self.decoder.append(
                spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        in_channels=dim,
                        out_channels=dim,
                        kennel_size=kernel_size,
                        indice_key=f"spconv_{indice_key}_{downsample_times - idx}",
                        bias=False
                    ),
                    nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            )
            self.decoder_norm.append(nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01))

    def forward(self, x):
        # U-Net结构
        features = []
        for conv in self.encoder:
            x = conv(x)
            features.append(x)
        x = features[-1]
        for deconv, norm, up_x in zip(
            self.decoder, self.decoder_norm, features[:-1][::-1]
        ):
            x = deconv(x)
            x = x.replace_feature(norm(x.features + up_x.features))
        return x


class MambaBackbone3D(nn.Module):
    def __init__(
        self,
        sparse_shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        num_res_block3d,
        num_layers,
    ):
        super().__init__()
        self.sparse_shape = sparse_shape
        # conv1非U-Net结构
        self.conv1 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels=in_channels,
                    out_channels=16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    indice_key="subm1"),
                nn.BatchNorm1d(16, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            Sparse2LayerResBlock3D(16, indice_key="stem"),
            Sparse2LayerResBlock3D(16, indice_key="stem"),
            spconv.SparseSequential(
                spconv.SparseConv3d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    indice_key="spconv1"),
                nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )
        # conv2是U-Net结构
        self.conv2 = spconv.SparseSequential(
            SparseUNetBlock3D(
                dim=32,
                kernel_size=kernel_size,
                stride=stride,
                num_res_block3d=num_res_block3d,
                indice_key="sparseunetblock3d1",
            ),
            spconv.SparseSequential(
                spconv.SparseConv3d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    indice_key="spconv2"),
                nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )
        # spconv3是U-Net结构
        self.conv3 = spconv.SparseSequential(
            SparseUNetBlock3D(
                dim=64,
                kernel_size=kernel_size,
                stride=stride,
                num_res_block3d=num_res_block3d,
                indice_key="sparseunetblock3d2",
            ),
            spconv.SparseSequential(
                spconv.SparseConv3d(
                    in_channels=64,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                    indice_key="spconv3"),
                nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            conv = SparseUNetBlock3D(
                dim=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                num_res_block3d=num_res_block3d,
                indice_key=f"sparseunetblock3d{idx+3}",
            ),
            self.layers.append(conv)

        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(
                out_channels,
                out_channels,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=0,
                bias=False,
                indice_key="spconv4",
            ),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SparseConv3d(
                out_channels,
                out_channels,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=0,
                bias=False,
                indice_key="spconv5",
            ),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.init_weights()
    
    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (spconv.SubMConv3d, spconv.SparseConv3d)):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out", nonlinearity="relu")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, voxel_features, voxel_coords, batch_size):
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )
        # 非U-Net结构
        x = self.conv1(x)
        # U-Net结构
        x = self.conv2(x)
        # U-Net结构
        x = self.conv3(x)
        # U-Net结构
        for conv in self.layers:
            x = conv(x)
        # 非U-Net结构
        out = self.conv_out(x).dense()
        N, C, D, H, W = out.shape
        out = out.view(N, C * D, H, W)
        return out