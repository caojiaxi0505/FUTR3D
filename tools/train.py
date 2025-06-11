# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import copy
import importlib
import mmcv
import os
import time
import torch
import torch.distributed as dist
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet import __version__ as mmdet_version
from mmdet.apis import set_random_seed
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmseg import __version__ as mmseg_version
from os import path as osp
try:
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description='训练一个检测器')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--work-dir', help='保存日志和模型的目录')
    parser.add_argument('--resume-from', help='从指定的检查点文件恢复训练')
    parser.add_argument('--auto-resume', action='store_true', help='自动从最新的检查点恢复训练')
    parser.add_argument('--no-validate', action='store_true', help='是否在训练期间不评估检查点')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, help='(已弃用，请使用--gpu-id) 要使用的gpu数量（仅适用于非分布式训练）')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='(已弃用，请使用--gpu-id) 要使用的gpu ids（仅适用于非分布式训练）')
    group_gpus.add_argument('--gpu-id', type=int, default=0, help='要使用的gpu ids（仅适用于非分布式训练）')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--diff-seed', action='store_true', help='是否为不同rank设置不同的种子')
    parser.add_argument('--deterministic', action='store_true', help='是否为CUDNN后端设置确定性选项')
    parser.add_argument('--options', nargs='+', action=DictAction, help='覆盖使用的配置中的某些设置，格式为xxx=yyy的键值对将合并到配置文件中。如果要覆盖的值是一个列表，它应该像key="[a,b]"或key=a,b它还允许嵌套列表/元组值，例如key="[(a,b),(c,d)]"注意，引号是必要的，并且不允许空格')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖使用的配置中的某些设置，格式为xxx=yyy的键值对将合并到配置文件中。如果要覆盖的值是一个列表，它应该像key="[a,b]"或key=a,b它还允许嵌套列表/元组值，例如key="[(a,b),(c,d)]"注意，引号是必要的，并且不允许空格')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true', help='自动缩放学习率与gpu数量')
    parser.add_argument('--extra-tag', default='default', help='添加额外说明，影响保存路径')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError('不能同时指定--options和--cfg-options，--options已弃用，请使用--cfg-options')
    if args.options:
        warnings.warn('--options已弃用，请使用--cfg-options')
        args.cfg_options = args.options
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
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
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0], args.extra_tag)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume`仅在mmdet版本>=2.20.0(3D检测模型)或mmsegmentation版本>=0.21.0(3D分割模型)时支持')
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus`已被弃用，因为在非分布式训练中我们只支持单GPU模式。现在使用`gpus=1`')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids`已被弃用，请使用`--gpu-id`。因为在非分布式训练中我们只支持单GPU模式。现在使用`gpu_ids`中的第一个GPU')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]
    if args.autoscale_lr:
        # 根据CUDA_VISIBLE_DEVICES和cfg.data.samples_per_gpu自动缩放学习率，默认8个GPU，4个样本/GPU
        # cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(os.getenv("CUDA_VISIBLE_DEVICES").split(",")) / 8 * cfg.data.samples_per_gpu / 4
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(os.getenv("CUDA_VISIBLE_DEVICES").split(",")) / 8 * cfg.data.samples_per_gpu / 1 * 10

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, name=logger_name)
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    logger.info(f'分布式训练: {distributed}')
    logger.info(f'配置:\n{cfg.pretty_text}')
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'设置随机种子为 {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.init_weights()
    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE
            if hasattr(datasets[0], 'PALETTE') else None)
    model.CLASSES = datasets[0].CLASSES
    train_model(model, datasets, cfg, distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta)


if __name__ == '__main__':
    main()
