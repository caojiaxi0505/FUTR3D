# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import warnings
from .. import builder
from ..builder import DETECTORS
from .base import Base3DDetector
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from mmdet.core import multi_apply
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result, merge_aug_bboxes_3d, show_result)
from os import path as osp
from torch.nn import functional as F


@DETECTORS.register_module()
class MVXTwoStageDetector(Base3DDetector):
    """Base class of Multi-modality VoxelNet."""
    def __init__(self, pts_voxel_layer=None, pts_voxel_encoder=None, pts_middle_encoder=None,
                 pts_fusion_layer=None, img_backbone=None, pts_backbone=None, img_neck=None, 
                 pts_neck=None, pts_bbox_head=None, img_roi_head=None, img_rpn_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(MVXTwoStageDetector, self).__init__(init_cfg=init_cfg)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_fusion_layer:
            self.pts_fusion_layer = builder.build_fusion_layer(pts_fusion_layer)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_rpn_head is not None:
            self.img_rpn_head = builder.build_head(img_rpn_head)
        if img_roi_head is not None:
            self.img_roi_head = builder.build_head(img_roi_head)
        self.train_cfg = train_cfg 
        self.test_cfg = test_cfg
        if pretrained is None:
            img_pretrained = pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)  
        else:
            raise ValueError(f'pretrained应为字典类型,而不是{type(pretrained)}')
        if self.with_img_backbone and img_pretrained is not None:
            warnings.warn('警告:pretrained已弃用,请使用init_cfg')
            self.img_backbone.init_cfg = dict(type='Pretrained', checkpoint=img_pretrained)
        if self.with_img_roi_head and img_pretrained is not None:
            warnings.warn('警告:pretrained已弃用,请使用init_cfg') 
            self.img_roi_head.init_cfg = dict(type='Pretrained', checkpoint=img_pretrained)
        if self.with_pts_backbone and pts_pretrained is not None:
            warnings.warn('警告:pretrained已弃用,请使用init_cfg')
            self.pts_backbone.init_cfg = dict(type='Pretrained', checkpoint=pts_pretrained)

    @property
    def with_img_shared_head(self):
        """布尔值:检测器是否在图像分支中有共享头"""
        return hasattr(self, 'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """布尔值:检测器是否有3D检测头"""
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    @property  
    def with_img_bbox(self):
        """布尔值:检测器是否有2D图像检测头"""
        return hasattr(self, 'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """布尔值:检测器是否有2D图像骨干网络"""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """布尔值:检测器是否有3D骨干网络"""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """布尔值:检测器是否有特征融合层"""
        return hasattr(self, 'pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """布尔值:检测器在图像分支中是否有neck网络"""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """布尔值:检测器在3D检测分支中是否有neck网络"""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """布尔值:检测器在图像检测分支中是否有2D RPN"""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """布尔值:检测器在图像分支中是否有RoI头"""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """布尔值:检测器是否有体素编码器"""
        return hasattr(self, 'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """布尔值:检测器是否有中间编码器"""
        return hasattr(self, 'middle_encoder') and self.middle_encoder is not None

    def extract_img_feat(self, img, img_metas):
        """提取图像特征"""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """提取点云特征"""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors, img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas):
        """从图像和点云提取特征"""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """对点云进行动态体素化
        参数:
            points (list[torch.Tensor]): 每个样本的点云
        返回:
            tuple[torch.Tensor]: 拼接的点云、每个体素的点数和坐标
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None, gt_labels_3d=None, gt_labels=None, gt_bboxes=None, img=None, proposals=None, gt_bboxes_ignore=None):
        """前向训练函数
        参数:
            points: 每个样本的点云
            img_metas: 每个样本的元信息
            gt_bboxes_3d: 3D框真值
            gt_labels_3d: 3D框标签真值
            gt_labels: 图像2D框标签真值
            gt_bboxes: 图像2D框真值
            img: 每个样本的图像
            proposals: 用于训练Fast RCNN的预测框
            gt_bboxes_ignore: 需要忽略的图像2D框真值
        返回:
            dict: 各分支的损失
        """
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(img_feats, img_metas=img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_bboxes_ignore=gt_bboxes_ignore, proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore=None):
        """点云分支的前向函数
        参数:
            pts_feats: 点云分支特征
            gt_bboxes_3d: 每个样本的3D框真值
            gt_labels_3d: 每个样本框的标签真值
            img_metas: 样本元信息
            gt_bboxes_ignore: 需要忽略的框真值
        返回:
            dict: 各分支损失
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_img_train(self, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, proposals=None, **kwargs):
        """图像分支前向函数
        参数:
            x (list[torch.Tensor]): 多层图像特征(B, C, H, W)
            img_metas (list[dict]): 图像元信息
            gt_bboxes (list[torch.Tensor]): 每个样本的框真值
            gt_labels (list[torch.Tensor]): 框标签真值
            gt_bboxes_ignore (list[torch.Tensor], 可选): 需要忽略的框真值
            proposals (list[torch.Tensor], 可选): 每个样本的预选框
        返回:
            dict: 各分支的损失
        """
        losses = dict()
        if self.with_img_rpn:
            rpn_outs = self.img_rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas, self.train_cfg.img_rpn)
            rpn_losses = self.img_rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get('img_rpn_proposal', self.test_cfg.img_rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        if self.with_img_bbox:
            img_roi_losses = self.img_roi_head.forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, **kwargs)
            losses.update(img_roi_losses)
        return losses

    def simple_test_img(self, x, img_metas, proposals=None, rescale=False):
        """不带增强的测试"""
        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_metas, self.test_cfg.img_rpn)
        else:
            proposal_list = proposals
        return self.img_roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

    def simple_test_rpn(self, x, img_metas, rpn_test_cfg):
        """RPN测试函数"""
        rpn_outs = self.img_rpn_head(x)
        proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)
        return self.img_rpn_head.get_bboxes(*proposal_inputs)

    def simple_test_pts(self, x, img_metas, rescale=False):
        """点云分支测试函数"""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        return [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """不带增强的测试函数"""
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """带增强的测试函数"""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """提取多个样本的点云和图像特征"""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs, img_metas)
        return img_feats, pts_feats

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """点云分支增强测试函数"""
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_meta, rescale=rescale)
            bbox_list = [dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels) for bboxes, scores, labels in bbox_list]
            aug_bboxes.append(bbox_list[0])
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas, self.pts_bbox_head.test_cfg)
        return merged_bboxes

    def show_results(self, data, result, out_dir):
        """结果可视化
        参数:
            data: 输入点云和样本信息
            result: 预测结果
            out_dir: 可视化结果输出目录
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"不支持类型{type(data['points'][0])}的可视化!")
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id]['box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(f"不支持类型{type(data['img_metas'][0])}的可视化!")
            file_name = osp.split(pts_filename)[-1].split('.')[0]
            assert out_dir is not None, '需要out_dir参数!'
            inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR, Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d, Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(f'不支持转换box_mode_3d {box_mode_3d}!')
            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name, show=True)
