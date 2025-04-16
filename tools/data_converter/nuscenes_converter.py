# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from mmdet3d.core.bbox import points_cam2img
from mmdet3d.datasets import NuScenesDataset

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
nus_attributes = ('cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving', 'pedestrian.standing', 'pedestrian.sitting_lying_down', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'None')

def create_nuscenes_infos(root_path, info_prefix, version='v1.0-trainval', max_sweeps=10):
    """创建nuscenes数据集的信息文件。
    根据原始数据生成pkl格式的信息文件。
    参数:
        root_path (str): 数据根目录路径
        info_prefix (str): 生成的信息文件前缀
        version (str): 数据版本,默认'v1.0-trainval'
        max_sweeps (int): 最大sweep数,默认10
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])
    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)
    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path, '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path, '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)

def get_available_scenes(nusc):
    """获取可用场景。
    从输入的nuscenes类中获取可用场景的基本信息。
    参数:
        nusc (class): NuScenes数据集类
    返回:
        available_scenes (list[dict]): 可用场景的基本信息列表
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, max_sweeps=10):
    """从原始数据生成训练/验证信息。
    参数:
        nusc: NuScenes数据集类
        train_scenes: 训练场景的基本信息列表
        val_scenes: 验证场景的基本信息列表 
        test: 是否使用测试模式,测试模式下无法访问标注,默认False
        max_sweeps: 最大sweep数,默认10
    返回:
        tuple[list[dict]]: 将保存到info文件的训练集和验证集信息
    """
    train_nusc_infos = []
    val_nusc_infos = []
    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        mmcv.check_file_exist(lidar_path)
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'radars': dict(), 
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        # 获取每帧的6个相机信息
        camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})
        radar_names = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        for radar_name in radar_names:
            radar_token = sample['data'][radar_name]
            radar_rec = nusc.get('sample_data', radar_token)
            sweeps = []
            while len(sweeps) < 5:
                if not radar_rec['prev'] == '':
                    radar_path, _, radar_intrin = nusc.get_sample_data(radar_token)
                    radar_info = obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, radar_name)
                    sweeps.append(radar_info)
                    radar_token = radar_rec['prev']
                    radar_rec = nusc.get('sample_data', radar_token)
                else:
                    radar_path, _, radar_intrin = nusc.get_sample_data(radar_token)
                    radar_info = obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, radar_name)
                    sweeps.append(radar_info)
            info['radars'].update({radar_name: sweeps})
        # 获取单个关键帧的sweeps
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # 获取标注
        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
            velocity = np.array([nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0 for anno in annotations], dtype=bool).reshape(-1)
            # 将速度从全局坐标系转换到激光雷达坐标系
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                velocity[i] = velo[:2]
            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # 将box尺寸转换为激光雷达坐标系格式(x_size, y_size, z_size对应l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array([a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array([a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)
    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type='lidar'):
    """获取从传感器到顶部激光雷达的RT矩阵信息。
    参数:
        nusc: NuScenes数据集类
        sensor_token: 对应特定传感器类型的样本数据token
        l2e_t: 从激光雷达到ego的平移矩阵,shape(1, 3)
        l2e_r_mat: 从激光雷达到ego的旋转矩阵,shape(3, 3)
        e2g_t: 从ego到全局的平移矩阵,shape(1, 3)
        e2g_r_mat: 从ego到全局的旋转矩阵,shape(3, 3)
        sensor_type: 要标定的传感器,默认'lidar'
    返回:
        sweep: 变换后的sweep信息
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # lyftdataset的路径是绝对路径
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # 相对路径
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']
    # 获取从传感器到顶部激光雷达的RT矩阵
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """从info文件和原始数据导出2D标注。
    参数:
        root_path: 原始数据根目录路径
        info_path: info文件路径
        version: 数据集版本
        mono3d: 是否导出mono3d标注,默认True
    """
    camera_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    cat2Ids = [dict(id=nus_categories.index(cat_name), name=cat_name) for cat_name in nus_categories]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(nusc, cam_info['sample_data_token'], visibilities=['', '1', '2', '3', '4'], mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(dict(
                file_name=cam_info['data_path'].split('data/nuscenes/')[-1],
                id=cam_info['sample_data_token'],
                token=info['token'],
                cam2ego_rotation=cam_info['sensor2ego_rotation'],
                cam2ego_translation=cam_info['sensor2ego_translation'],
                ego2global_rotation=info['ego2global_rotation'],
                ego2global_translation=info['ego2global_translation'],
                cam_intrinsic=cam_info['cam_intrinsic'],
                width=width,
                height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc, sample_data_token: str, visibilities: List[str], mono3d=True):
    """获取指定sample_data_token的2D标注记录。
    参数:
        sample_data_token (str): 相机关键帧的sample data token
        visibilities (list[str]): 可见性过滤器
        mono3d (bool): 是否获取带有mono3d标注的框,默认True
    返回:
        list[dict]: 属于输入sample_data_token的2D标注记录列表
    """
    sd_rec = nusc.get('sample_data', sample_data_token)
    assert sd_rec['sensor_modality'] == 'camera', 'Error: get_2d_boxes只适用于相机sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError('2D重投影仅适用于关键帧。')
    s_rec = nusc.get('sample', sd_rec['sample_token'])
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities)]
    repro_recs = []
    for ann_rec in ann_recs:
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token
        box = nusc.get_box(ann_rec['token'])
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
        final_coords = post_process_coords(corner_coords)
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()
            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]
            dim = dim.tolist()
            rot = [-box.orientation.yaw_pitch_roll[0]]
            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()
            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo
            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            if repro_rec['center2d'][2] <= 0:
                continue
            ann_token = nusc.get('sample_annotation', box.token)['attribute_tokens']
            attr_name = 'None' if len(ann_token) == 0 else nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id
        repro_recs.append(repro_rec)
    return repro_recs


def post_process_coords(corner_coords: List, imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """获取投影边界框角点的凸包与图像画布的交集,如果没有交集则返回None。
    参数:
        corner_coords (list[int]): 投影边界框的角点坐标
        imsize (tuple[int]): 图像画布大小
    返回:
        tuple [float]: 2D边界框角点凸包与图像画布的交集
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])
    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1]) 
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])
        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float, sample_data_token: str, filename: str) -> OrderedDict:
    """根据2D边界框坐标和其他信息生成一条2D标注记录。
    参数:
        ann_rec (dict): 原始3D标注记录
        x1 (float): x坐标最小值
        y1 (float): y坐标最小值 
        x2 (float): x坐标最大值
        y2 (float): y坐标最大值
        sample_data_token (str): 样本数据token
        filename (str): 对应的图像文件名

    返回:
        dict: 一条2D标注记录,包含以下字段:
            - file_name (str): 文件名
            - image_id (str): 样本数据token
            - area (float): 2D框面积
            - category_name (str): 类别名称
            - category_id (int): 类别ID
            - bbox (list[float]): 2D框左上角坐标和宽高
            - iscrowd (int): 是否为密集区域
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()
    relevant_keys = ['attribute_tokens', 'category_name', 'instance_token', 'next', 'num_lidar_pts', 'num_radar_pts', 'prev', 'sample_annotation_token', 'sample_data_token', 'visibility_token']
    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value
    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename
    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)
    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0
    return coco_rec
