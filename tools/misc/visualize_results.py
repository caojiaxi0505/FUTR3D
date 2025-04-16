# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet3D可视化结果')
    parser.add_argument('config', help='测试配置文件路径')
    parser.add_argument('--result', help='pickle格式的结果文件')
    parser.add_argument('--show-dir', help='可视化结果保存目录')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.result is not None and not args.result.endswith(('.pkl', '.pickle')):
        raise ValueError('结果文件必须是pkl格式')
    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result)
    if getattr(dataset, 'show', None) is not None:
        eval_pipeline = cfg.get('eval_pipeline', {})
        if eval_pipeline:
            dataset.show(results, args.show_dir, pipeline=eval_pipeline)
        else:
            dataset.show(results, args.show_dir)
    else:
        raise NotImplementedError('数据集{}未实现show方法!'.format(type(dataset).__name__))

if __name__ == '__main__':
    main()
