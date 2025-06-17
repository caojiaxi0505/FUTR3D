# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
from mmcv.runner import save_checkpoint
from torch import nn as nn
from mmdet3d.apis import init_model

def fuse_conv_bn(conv, bn):
    """在推理过程中,批归一化层的功能被关闭,仅使用通道均值和方差,这提供了将其与前面的卷积层融合的机会,以节省计算并简化网络结构。"""
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w * factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv

def fuse_module(m):
    last_conv = None
    last_conv_name = None
    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if last_conv is None:
                continue
            fused_conv = fuse_conv_bn(last_conv, child)
            m._modules[last_conv_name] = fused_conv
            m._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_module(child)
    return m

def parse_args():
    parser = argparse.ArgumentParser(description='融合模型中的卷积和BN层')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='检查点文件路径')
    parser.add_argument('out', help='转换后模型的输出路径')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = init_model(args.config, args.checkpoint)
    fused_model = fuse_module(model)
    save_checkpoint(fused_model, args.out)

if __name__ == '__main__':
    main()
