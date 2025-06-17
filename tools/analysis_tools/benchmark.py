# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from tools.misc.fuse_conv_bn import fuse_module

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet基准测试模型')
    parser.add_argument('config', help='测试配置文件路径')
    parser.add_argument('checkpoint', help='检查点文件')
    parser.add_argument('--samples', default=2000, help='基准测试样本数')
    parser.add_argument('--log-interval', default=50, help='日志记录间隔')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='是否融合conv和bn,这会略微提高推理速度')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    num_warmup = 5
    pure_inf_time = 0

    for i, data in enumerate(data_loader):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'完成图像 [{i + 1:<3}/ {args.samples}], fps: {fps:.1f} 图/秒')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'总体 fps: {fps:.1f} 图/秒')
            break

if __name__ == '__main__':
    main()
