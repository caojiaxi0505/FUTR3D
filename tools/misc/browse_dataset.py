# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings
from os import path as osp
from pathlib import Path
import mmcv
import numpy as np
from mmcv import Config, DictAction, mkdir_or_exist
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode, DepthInstance3DBoxes, LiDARInstance3DBoxes)
from mmdet3d.core.visualizer import (show_multi_modality_result, show_result, show_seg_result)
from mmdet3d.datasets import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='浏览数据集')
    parser.add_argument('config', help='训练配置文件路径')
    parser.add_argument('--skip-type', type=str, nargs='+', default=['Normalize'], help='跳过一些无用的管道')
    parser.add_argument('--output-dir', default=None, type=str, help='如果没有显示界面,可以保存到此目录')
    parser.add_argument('--task', type=str, choices=['det', 'seg', 'multi_modality-det', 'mono-det'], help='根据任务确定可视化方法')
    parser.add_argument('--aug', action='store_true', help='是否可视化增强后的数据集或原始数据集')
    parser.add_argument('--online', action='store_true', help='是否进行在线可视化,注意通常需要显示器')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='覆盖配置文件中的一些设置,格式为xxx=yyy的键值对将被合并到配置文件中。如果要覆盖的值是列表,应该像key="[a,b]"或key=a,b。也允许嵌套的列表/元组值,例如key="[(a,b),(c,d)]"。注意引号是必需的,并且不允许有空格')
    args = parser.parse_args()
    return args

def build_data_cfg(config_path, skip_type, aug, cfg_options):
    """构建用于加载可视化数据的配置"""
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    if cfg.data.train['type'] == 'RepeatDataset':
        cfg.data.train = cfg.data.train.dataset
    if cfg.data.train['type'] == 'ConcatDataset':
        cfg.data.train = cfg.data.train.datasets[0]
    train_data_cfg = cfg.data.train

    if aug:
        show_pipeline = cfg.train_pipeline
    else:
        show_pipeline = cfg.eval_pipeline
        for i in range(len(cfg.train_pipeline)):
            if cfg.train_pipeline[i]['type'] == 'LoadAnnotations3D':
                show_pipeline.insert(i, cfg.train_pipeline[i])
            if cfg.train_pipeline[i]['type'] == 'Collect3D':
                if show_pipeline[-1]['type'] == 'Collect3D':
                    show_pipeline[-1] = cfg.train_pipeline[i]
                else:
                    show_pipeline.append(cfg.train_pipeline[i])

    train_data_cfg['pipeline'] = [x for x in show_pipeline if x['type'] not in skip_type]
    return cfg

def to_depth_mode(points, bboxes):
    """将点和边界框转换为深度坐标和深度框模式"""
    if points is not None:
        points = Coord3DMode.convert_point(points.copy(), Coord3DMode.LIDAR, Coord3DMode.DEPTH)
    if bboxes is not None:
        bboxes = Box3DMode.convert(bboxes.clone(), Box3DMode.LIDAR, Box3DMode.DEPTH)
    return points, bboxes

def show_det_data(input, out_dir, show=False):
    """可视化3D点云和3D边界框"""
    img_metas = input['img_metas']._data
    points = input['points']._data.numpy()
    gt_bboxes = input['gt_bboxes_3d']._data.tensor
    if img_metas['box_mode_3d'] != Box3DMode.DEPTH:
        points, gt_bboxes = to_depth_mode(points, gt_bboxes)
    filename = osp.splitext(osp.basename(img_metas['pts_filename']))[0]
    show_result(points, gt_bboxes.clone(), None, out_dir, filename, show=show, snapshot=True)

def show_seg_data(input, out_dir, show=False):
    """可视化3D点云和分割掩码"""
    img_metas = input['img_metas']._data
    points = input['points']._data.numpy()
    gt_seg = input['pts_semantic_mask']._data.numpy()
    filename = osp.splitext(osp.basename(img_metas['pts_filename']))[0]
    show_seg_result(points, gt_seg.copy(), None, out_dir, filename, np.array(img_metas['PALETTE']), img_metas['ignore_index'], show=show, snapshot=True)

def show_proj_bbox_img(input, out_dir, show=False, is_nus_mono=False):
    """通过投影在2D图像上可视化3D边界框"""
    gt_bboxes = input['gt_bboxes_3d']._data
    img_metas = input['img_metas']._data
    img = input['img']._data.numpy()
    img = img.transpose(1, 2, 0)
    if gt_bboxes.tensor.shape[0] == 0:
        gt_bboxes = None
    filename = Path(img_metas['filename']).name
    if isinstance(gt_bboxes, DepthInstance3DBoxes):
        show_multi_modality_result(img, gt_bboxes, None, None, out_dir, filename, box_mode='depth', img_metas=img_metas, show=show)
    elif isinstance(gt_bboxes, LiDARInstance3DBoxes):
        show_multi_modality_result(img, gt_bboxes, None, img_metas['lidar2img'], out_dir, filename, box_mode='lidar', img_metas=img_metas, show=show)
    elif isinstance(gt_bboxes, CameraInstance3DBoxes):
        show_multi_modality_result(img, gt_bboxes, None, img_metas['cam2img'], out_dir, filename, box_mode='camera', img_metas=img_metas, show=show)
    else:
        warnings.warn(f'无法识别的边界框类型 {type(gt_bboxes)}, 仅显示图像')
        show_multi_modality_result(img, None, None, None, out_dir, filename, show=show)

def main():
    args = parse_args()
    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)
    cfg = build_data_cfg(args.config, args.skip_type, args.aug, args.cfg_options)
    try:
        dataset = build_dataset(cfg.data.train, default_args=dict(filter_empty_gt=False))
    except TypeError:
        dataset = build_dataset(cfg.data.train)
    dataset_type = cfg.dataset_type
    vis_task = args.task
    progress_bar = mmcv.ProgressBar(len(dataset))
    for input in dataset:
        if vis_task in ['det', 'multi_modality-det']:
            show_det_data(input, args.output_dir, show=args.online)
        if vis_task in ['multi_modality-det', 'mono-det']:
            show_proj_bbox_img(input, args.output_dir, show=args.online, is_nus_mono=(dataset_type == 'NuScenesMonoDataset'))
        elif vis_task in ['seg']:
            show_seg_data(input, args.output_dir, show=args.online)
        progress_bar.update()

if __name__ == '__main__':
    main()
