# Copyright (c) OpenMMLab. All rights reserved.
import argparse, os, warnings, time, numpy as np
import mmcv, torch
import mmdet
import random
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet3d.models import build_model
if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes
try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet测试(和评估)模型')
    parser.add_argument('config', help='测试配置文件路径')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='是否融合conv和bn,这会略微提高推理速度')
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='(已弃用,请使用--gpu-id)要使用的gpu id(仅适用于非分布式训练)')
    parser.add_argument('--gpu-id', type=int, default=0, help='要使用的gpu id(仅适用于非分布式测试)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--deterministic', action='store_true', help='是否为CUDNN后端设置确定性选项')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置文件中的一些设置,键值对格式为xxx=yyy,将合并到配置文件中。如果要覆盖的值是列表,应该像key="[a,b]"或key=a,b。也允许嵌套的列表/元组值,例如key="[(a,b),(c,d)]"。注意引号是必需的,且不允许有空格')
    parser.add_argument('--options', nargs='+', action=DictAction, help='评估的自定义选项,xxx=yyy格式的键值对将作为dataset.evaluate()函数的kwargs(已弃用),改为使用--eval-options')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
    cfg.model.pretrained = None
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids`已弃用,请使用`--gpu-id`。因为在非分布式测试中我们只支持单GPU模式。现在使用`gpu_ids`中的第一个GPU。')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    rank, _ = get_dist_info()
    random.seed(0)
    
    @torch.inference_mode()
    def measure(model, num_repeats=500, num_warmup=500):
        model.cuda()
        model.eval()
        latencies = []
        for k in range(num_repeats + num_warmup):
            img_size = [928, 1600]
            images = torch.randn(1, 6, 3, img_size[0], img_size[1]).cuda()
            radar = [torch.randn(800, 6).cuda()]
            img_meta = dict()
            img_meta['ori_shape'] = (900, 1600, 3, 6)
            img_meta['img_shape'] = []
            img_meta['lidar2img'] = []
            for i in range(6):
                img_meta['lidar2img'].append(np.random.rand(4, 4))
                img_meta['img_shape'].append((928, 1600, 3))
            img_metas = [img_meta]
            start = cuda_time()
            model.dummy_forward(img_metas, img=images, radar=radar)
            if k >= num_warmup:
                latencies.append((cuda_time() - start) * 1000)
        latencies = sorted(latencies)
        drop = int(len(latencies) * 0.25)
        return np.mean(latencies[drop:-drop])
    print(measure(model))

if __name__ == '__main__':
    main()
