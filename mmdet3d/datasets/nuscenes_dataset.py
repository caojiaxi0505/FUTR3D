# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pyquaternion
import tempfile
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp


@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    """NuScenes数据集。
    此类作为NuScenes数据集实验的API。
    请参考 `NuScenes Dataset <https://www.nuscenes.org/download>` 下载数据。
    参数:
        ann_file (str): 标注文件路径。
        pipeline (list[dict], 可选): 用于数据处理的管道。默认为None。
        data_root (str): 数据集根目录路径。
        classes (tuple[str], 可选): 数据集中使用的类别。默认为None。
        load_interval (int, 可选): 加载数据集的间隔。用于均匀采样数据集。默认为1。
        with_velocity (bool, 可选): 是否在实验中包含速度预测。默认为True。
        modality (dict, 可选): 指定用作输入的传感器数据的模态。默认为None。
        box_type_3d (str, 可选): 此数据集的3D框类型。基于box_type_3d，数据集将封装框到其原始格式，然后转换为box_type_3d。默认为'LiDAR'。可用选项包括:
            - 'LiDAR': LiDAR坐标系中的框。
            - 'Depth': 深度坐标系中的框，通常用于室内数据集。
            - 'Camera': 相机坐标系中的框。
        filter_empty_gt (bool, 可选): 是否过滤空的GT。默认为True。
        test_mode (bool, 可选): 数据集是否处于测试模式。默认为False。
        eval_version (bool, 可选): 评估的配置版本。默认为'detection_cvpr_2019'。
        use_valid_flag (bool, 可选): 是否使用info文件中的use_valid_flag键作为掩码来过滤gt_boxes和gt_names。默认为False。"""
    NameMapping = {'movable_object.barrier': 'barrier', 'vehicle.bicycle': 'bicycle', 'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus', 'vehicle.car': 'car', 'vehicle.construction': 'construction_vehicle', 'vehicle.motorcycle': 'motorcycle', 'human.pedestrian.adult': 'pedestrian', 'human.pedestrian.child': 'pedestrian', 'human.pedestrian.construction_worker': 'pedestrian', 'human.pedestrian.police_officer': 'pedestrian', 'movable_object.trafficcone': 'traffic_cone', 'vehicle.trailer': 'trailer', 'vehicle.truck': 'truck'}
    DefaultAttribute = {'car': 'vehicle.parked', 'pedestrian': 'pedestrian.moving', 'trailer': 'vehicle.parked', 'truck': 'vehicle.parked', 'bus': 'vehicle.moving', 'motorcycle': 'cycle.without_rider', 'construction_vehicle': 'vehicle.parked', 'bicycle': 'cycle.without_rider', 'barrier': '', 'traffic_cone': ''}
    AttrMapping = {'cycle.with_rider': 0, 'cycle.without_rider': 1, 'pedestrian.moving': 2, 'pedestrian.standing': 3, 'pedestrian.sitting_lying_down': 4, 'vehicle.moving': 5, 'vehicle.parked': 6, 'vehicle.stopped': 7}
    AttrMapping_rev = ['cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving', 'pedestrian.standing', 'pedestrian.sitting_lying_down', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped']
    ErrNameMapping = {'trans_err': 'mATE', 'scale_err': 'mASE', 'orient_err': 'mAOE', 'vel_err': 'mAVE', 'attr_err': 'mAAE'}
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')

    def __init__(self, ann_file, pipeline=None, data_root=None, classes=None, load_interval=1, with_velocity=True, modality=None, box_type_3d='LiDAR', filter_empty_gt=True, test_mode=False, eval_version='detection_cvpr_2019', use_valid_flag=False):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline, classes=classes, modality=modality, box_type_3d=box_type_3d, filter_empty_gt=filter_empty_gt, test_mode=test_mode)
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(use_camera=False, use_lidar=True, use_radar=False, use_map=False, use_external=False)

    def get_cat_ids(self, idx):
        """获取单个场景的类别分布。
        参数:
            idx (int): 数据信息的索引。
        返回:
            dict[list]: 对于每个类别,如果当前场景包含该类别的框,则存储包含idx的列表,否则存储空列表。"""
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])
        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """从标注文件加载标注。
        参数:
            ann_file (str): 标注文件路径。
        返回:
            list[dict]: 按时间戳排序的标注列表。"""
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """根据给定索引获取数据信息。
        参数:
            index (int): 要获取的样本数据的索引。
        返回:
            dict: 将传递给数据预处理管道的数据信息,包括以下键:
                - sample_idx (str): 样本索引
                - pts_filename (str): 点云文件名
                - sweeps (list[dict]): 扫描信息
                - timestamp (float): 样本时间戳
                - img_filename (str, 可选): 图像文件名
                - lidar2img (list[np.ndarray], 可选): 从激光雷达到不同相机的变换
                - ann_info (dict): 标注信息"""
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        if self.modality['use_radar']:
            input_dict['radar'] = info['radars']
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            camera2lidar_rts = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = cam_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = cam_info["sensor2lidar_translation"]
                camera2lidar_rts.append(camera2lidar)
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    camera2lidar=camera2lidar_rts,
                ))
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        return input_dict

    def get_ann_info(self, index):
        """根据给定索引获取标注信息。
        参数:
            index (int): 要获取标注数据的索引。
        返回:
            dict: 标注信息包含以下键:
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D真实边界框
                - gt_labels_3d (np.ndarray): 真实标签
                - gt_names (list[str]): 类别名称"""
        info = self.data_infos[index]
        if self.use_valid_flag: mask = info['valid_flag']
        else: mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES: gt_labels_3d.append(self.CLASSES.index(cat))
            else: gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d,box_dim=gt_bboxes_3d.shape[-1],origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        anns_results = dict(gt_bboxes_3d=gt_bboxes_3d,gt_labels_3d=gt_labels_3d,gt_names=gt_names_3d)
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """将结果转换为标准格式。
        参数:
            results (list[dict]): 数据集的测试结果。
            jsonfile_prefix (str): 输出json文件的前缀。可以通过修改jsonfile_prefix指定输出目录/文件名。默认: None。
        返回:
            str: 输出json文件的路径。"""
        nusc_annos = {}
        mapped_class_names = self.CLASSES
        print('开始转换检测格式...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.with_velocity)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,mapped_class_names,self.eval_detection_configs,self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in ['car','construction_vehicle','bus','truck','trailer']: attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']: attr = 'cycle.with_rider'
                    else: attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']: attr = 'pedestrian.standing'
                    elif name in ['bus']: attr = 'vehicle.stopped'
                    else: attr = NuScenesDataset.DefaultAttribute[name]
                nusc_anno = dict(sample_token=sample_token,translation=box.center.tolist(),size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),velocity=box.velocity[:2].tolist(),detection_name=name,
                    detection_score=box.score,attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {'meta': self.modality,'results': nusc_annos}
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('结果写入到', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self, result_path, logger=None, metric='bbox', result_name='pts_bbox'):
        """对单个模型进行nuScenes协议评估。
        参数:
            result_path (str): 结果文件路径。
            logger (logging.Logger | str, optional): 用于打印评估相关信息的日志器。默认: None。
            metric (str, optional): 评估使用的指标名称。默认: 'bbox'。
            result_name (str, optional): 指标前缀中的结果名称。默认: 'pts_bbox'。
        返回:
            dict: 评估详情字典。
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval
        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val'}
        nusc_eval = NuScenesEval(nusc, config=self.eval_detection_configs, result_path=result_path,
            eval_set=eval_set_map[self.version], output_dir=output_dir, verbose=False)
        nusc_eval.main(render_curves=False)
        # 记录指标
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix, self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """将结果格式化为json(用于COCO评估的标准格式)。
        参数:
            results (list[dict]): 数据集的测试结果。
            jsonfile_prefix (str): json文件的前缀,包含文件路径和文件名前缀,如"a/b/prefix"。如果未指定,将创建临时文件。默认: None。
        返回:
            tuple: 返回(result_files, tmp_dir),其中result_files是包含json文件路径的字典,
                  tmp_dir是在未指定jsonfile_prefix时创建的临时目录。
        """
        assert isinstance(results, list), 'results必须是列表'
        assert len(results) == len(self), ('结果长度与数据集长度不相等: {} != {}'.format(len(results), len(self)))
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        # 当前预测结果可能有两种格式:
        # 1. dict列表('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. dict列表('pts_bbox'或'img_bbox': dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # 这是一个临时解决方案,用于在nuScenes上评估这两种格式
        # 参考 https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = dict()
            for name in results[0]:
                print(f'\n格式化{name}的边界框')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update({name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self, results, metric='bbox', logger=None, jsonfile_prefix=None, result_names=['pts_bbox'], show=False, out_dir=None, pipeline=None):
        """按nuScenes协议进行评估。
        参数:
            results (list[dict]): 数据集的测试结果。
            metric (str | list[str], optional): 要评估的指标。默认: 'bbox'。
            logger (logging.Logger | str, optional): 用于打印评估相关信息的日志记录器。默认: None。
            jsonfile_prefix (str, optional): json文件的前缀,包含文件路径和文件名前缀,如"a/b/prefix"。如果未指定,将创建临时文件。默认: None。
            show (bool, optional): 是否可视化。默认: False。
            out_dir (str, optional): 保存可视化结果的路径。默认: None。
            pipeline (list[dict], optional): 用于显示的原始数据加载。默认: None。
        返回:
            dict[str, float]: 每个评估指标的结果。
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('评估{}的边界框'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict

    def _build_default_pipeline(self):
        """构建此数据集的默认pipeline。"""
        pipeline = [
            dict(type='LoadPointsFromFile',coord_type='LIDAR',load_dim=5,use_dim=5,file_client_args=dict(backend='disk')),
            dict(type='LoadPointsFromMultiSweeps',sweeps_num=10,file_client_args=dict(backend='disk')),
            dict(type='DefaultFormatBundle3D',class_names=self.CLASSES,with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=False, pipeline=None):
        """结果可视化。
        参数:
            results (list[dict]): 边界框结果列表。
            out_dir (str): 可视化结果的输出目录。
            show (bool): 是否在线可视化结果。默认: False。
            pipeline (list[dict], optional): 用于显示的原始数据加载。默认: None。
        """
        assert out_dir is not None, '需要out_dir,但获得None。'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,file_name, show)


def output_to_nusc_box(detection, with_velocity=True):
    """将输出转换为nuScenes中的box类。
    参数:
        detection (dict): 检测结果。
            - boxes_3d (:obj:`BaseInstance3DBoxes`): 检测边界框。
            - scores_3d (torch.Tensor): 检测分数。
            - labels_3d (torch.Tensor): 预测的框标签。
    返回:
        list[:obj:`NuScenesBox`]: 标准NuScenesBox列表。
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    nus_box_dims = box_dims[:, [1, 0, 2]]
    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        box = NuScenesBox(box_gravity_center[i],nus_box_dims[i],quat,label=labels[i],score=scores[i],velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info, boxes, classes, eval_configs, eval_version='detection_cvpr_2019'):
    """将框从自车坐标系转换到全局坐标系。
    参数:
        info (dict): 特定样本数据的信息,包含标定信息
        boxes (list[:obj:`NuScenesBox`]): 预测的NuScenes框列表
        classes (list[str]): 评估中的映射类别
        eval_configs (object): 评估配置对象 
        eval_version (str, optional): 评估版本。默认: 'detection_cvpr_2019'
    返回:
        list: 全局坐标系中标准NuScenes框列表"""
    box_list = []
    for box in boxes:
        # 将框移动到自车坐标系
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # 过滤自车坐标系中的检测
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range: continue
        # 将框移动到全局坐标系
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
