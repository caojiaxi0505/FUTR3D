# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp
from tools.data_converter import indoor_converter as indoor
from tools.data_converter import kitti_converter as kitti
from tools.data_converter import lyft_converter as lyft_converter
from tools.data_converter import nuscenes_converter as nuscenes_converter
from tools.data_converter.create_gt_database import (GTDatabaseCreater, create_groundtruth_database)


def kitti_data_prep(root_path, info_prefix, version, out_dir, with_plane=False):
    kitti.create_kitti_info_file(root_path, info_prefix, with_plane)
    kitti.create_reduced_point_cloud(root_path, info_prefix)
    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(root_path, f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    kitti.export_2d_annotation(root_path, info_train_path)
    kitti.export_2d_annotation(root_path, info_val_path)
    kitti.export_2d_annotation(root_path, info_trainval_path)
    kitti.export_2d_annotation(root_path, info_test_path)
    create_groundtruth_database('KittiDataset', root_path, info_prefix, f'{out_dir}/{info_prefix}_infos_train.pkl', relative_path=False, mask_anno_path='instances_train.json', with_mask=(version == 'mask'))

def nuscenes_data_prep(root_path, info_prefix, version, dataset_name, out_dir, max_sweeps=10):
    nuscenes_converter.create_nuscenes_infos(root_path, info_prefix, version=version, max_sweeps=max_sweeps)
    if version == 'v1.0-test':
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
        return
    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
    nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)
    create_groundtruth_database(dataset_name, root_path, info_prefix, 'data/nuscenes/nuscenes_infos_train.pkl')

def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    lyft_converter.create_lyft_infos(root_path, info_prefix, version=version, max_sweeps=max_sweeps)

def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    indoor.create_indoor_info_file(root_path, info_prefix, out_dir, workers=workers)

def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    indoor.create_indoor_info_file(root_path, info_prefix, out_dir, workers=workers)

def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers, num_points):
    indoor.create_indoor_info_file(root_path, info_prefix, out_dir, workers=workers, num_points=num_points)

def waymo_data_prep(root_path, info_prefix, version, out_dir, workers, max_sweeps=5):
    from tools.data_converter import waymo_converter as waymo
    splits = ['training', 'validation', 'testing']
    for i, split in enumerate(splits):
        load_dir = osp.join(root_path, 'waymo_format', split)
        if split == 'validation':
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(load_dir, save_dir, prefix=str(i), workers=workers, test_mode=(split == 'testing'))
        converter.convert()
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(out_dir, info_prefix, max_sweeps=max_sweeps, workers=workers)
    GTDatabaseCreater('WaymoDataset', out_dir, info_prefix, f'{out_dir}/{info_prefix}_infos_train.pkl', relative_path=False, with_mask=False, num_worker=workers).create()


parser = argparse.ArgumentParser(description='数据转换器参数解析')
parser.add_argument('dataset', metavar='kitti', help='数据集名称')
parser.add_argument('--root-path', type=str, default='./data/kitti', help='指定数据集根目录')
parser.add_argument('--version', type=str, default='v1.0', required=False, help='指定数据集版本,kitti数据集不需要')
parser.add_argument('--max-sweeps', type=int, default=10, required=False, help='每个样本的激光雷达扫描次数')
parser.add_argument('--with-plane', action='store_true', help='是否使用kitti的平面信息')
parser.add_argument('--num-points', type=int, default=-1, help='室内数据集采样点数量')
parser.add_argument('--out-dir', type=str, default='./data/kitti', required=False, help='info pkl文件名')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument('--workers', type=int, default=6, help='使用的线程数')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            with_plane=args.with_plane)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        # train_version = f'{args.version}-trainval'
        # nuscenes_data_prep(
        #     root_path=args.root_path,
        #     info_prefix=args.extra_tag,
        #     version=train_version,
        #     dataset_name='NuScenesDataset',
        #     out_dir=args.out_dir,
        #     max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            num_points=args.num_points,
            out_dir=args.out_dir,
            workers=args.workers)
