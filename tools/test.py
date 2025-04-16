# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import importlib
import mmcv
import mmdet
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes
try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('config', help='测试配置文件路径')
    parser.add_argument('checkpoint', help='检查点文件')
    parser.add_argument('--out', help='以pickle格式输出结果文件')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='是否融合conv和bn，这将略微提高推理速度')
    parser.add_argument('--gpu-ids', type=int, nargs='+', help='(已弃用，请使用--gpu-id) 要使用的gpu ids（仅适用于非分布式训练）')
    parser.add_argument('--gpu-id', type=int, default=0, help='要使用的gpu ids（仅适用于非分布式测试）')
    parser.add_argument('--format-only', action='store_true', help='格式化输出结果而不执行eval。当您想要将结果格式化为特定格式并将其提交到测试服务器时，它很有用')
    parser.add_argument('--eval', type=str, nargs='+', help='评估指标，依赖于数据集，例如COCO的“bbox”、“segm”、“proposal”，以及PASCAL VOC的“mAP”、“recall”')
    parser.add_argument('--show', action='store_true', help='显示结果')
    parser.add_argument('--show-dir', help='保存结果的目录')
    parser.add_argument('--gpu-collect', action='store_true', help='是否使用gpu采集结果')
    parser.add_argument('--tmpdir', help='TMP目录用于从多个工作进程收集结果，当没有指定gpu-collect时可用')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--deterministic', action='store_true', help='是否为CUDNN后端设置确定性选项')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖使用的配置中的某些设置，格式为xxx=yyy的键值对将合并到配置文件中。如果要覆盖的值是一个列表，它应该像key="[a,b]"或key=a,b它还允许嵌套列表/元组值，例如key="[(a,b),(c,d)]"注意，引号是必要的，并且不允许空格')
    parser.add_argument('--options', nargs='+', action=DictAction, help='自定义的eval选项，xxx=yyy格式的键值对将是kwargs，用于dataset.evaluate()函数（已弃用），更改为--eval-options')
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help='自定义的eval选项，xxx=yyy格式的键值对将是dataset.evaluate()函数的kwargs')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.eval_options:
        raise ValueError('--options and --eval-options cannot be both specified --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show or args.show_dir, ('请使用参数"--out"、"--eval"、"--format-only"、"--show"或"--show-dir"指定至少一个操作（保存/eval/生成测试结果/显示结果/保存结果）')
    if args.eval and args.format_only:
        raise ValueError('--eval和--format_only不能同时指定')
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('输出文件必须是pkl文件')
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
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
        warnings.warn('`--gpu-ids`已被弃用，请使用`--gpu-id`。因为在非分布式测试中，我们只支持单GPU模式。现在使用`gpu_ids`中的第一个GPU')
    else:
        cfg.gpu_ids = [args.gpu_id]
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    test_dataloader_default_args = dict(samples_per_gpu=1, workers_per_gpu=6, dist=distributed, shuffle=False)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    test_loader_cfg = {**test_dataloader_default_args, **cfg.data.get('test_dataloader', {})}
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

if __name__ == '__main__':
    main()
