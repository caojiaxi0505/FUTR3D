import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import mmcv
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from matplotlib.colors import LinearSegmentedColormap
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from mmdet3d.core import Box3DMode, Coord3DMode, bbox3d2result, merge_aug_bboxes_3d, show_result
from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors import Base3DDetector, MVXTwoStageDetector
from plugin.futr3d.models.utils.grid_mask import GridMask
from torch.nn import functional as F


@DETECTORS.register_module(force=True)
class FUTR3D(MVXTwoStageDetector):
    def __init__(self,
                 use_lidar=True,
                 use_camera=False,
                 use_radar=False,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 freeze_backbone=False,
                 aux_weight=1.0,
                 init_cfg=None):
        super(FUTR3D, self).__init__(init_cfg=init_cfg)
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar
        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
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
        if radar_voxel_layer:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if radar_voxel_encoder:
            self.radar_voxel_encoder = builder.build_voxel_encoder(radar_voxel_encoder)
        if radar_middle_encoder:
            self.radar_middle_encoder = builder.build_middle_encoder(radar_middle_encoder)
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
        self.aux_weight = aux_weight
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for modules in [self.img_backbone, self.img_neck, self.pts_backbone, self.pts_middle_encoder, self.pts_neck]: 
            if modules is not None:
                modules.eval()
                for param in modules.parameters():
                    param.requires_grad = False
    
    def extract_img_feat(self, img, img_metas):
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_pts_feat(self, pts):
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    def extract_radar_feat(self, radar, img_metas):
        voxels, num_points, coors = self.radar_voxelize(radar)
        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.radar_middle_encoder(voxel_features, coors, batch_size)
        return [x]

    def extract_feat(self, points, img, radar, img_metas):
        img_feats = self.extract_img_feat(img, img_metas) if self.use_camera else None
        pts_feats = self.extract_pts_feat(points) if self.use_lidar else None
        radar_feats = self.extract_radar_feat(radar, img_metas) if self.use_radar else None
        # points = Coord3DMode.convert_point(points[0], Coord3DMode.LIDAR, Coord3DMode.DEPTH)
        # if points is not None and len(points) > 0:
        #     fig, ax = plt.subplots(figsize=(10, 10))
        #     point_cloud = points[..., :3].cpu().numpy()
        #     sc = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.1, c=point_cloud[:, 2], cmap='viridis', label='Point Cloud')
        #     plt.colorbar(sc)
        #     plt.title('Point Cloud with GT Boxes and Reference Points in BEV')
        #     plt.xlabel('X')
        #     plt.ylabel('Y')
        #     plt.xlim(-54, 54)
        #     plt.ylim(-54, 54)
        #     plt.legend()
        #     plt.savefig('绘图/point_cloud_with_gt_boxes_and_reference_points.png')
        #     plt.close()
        # self.plt_heatmap(pts_feats[0][0].unsqueeze(0))
        return (img_feats, pts_feats, radar_feats)

    def plt_heatmap(self, x, coors=None, name="default"):
        # TODO: 检查为什么要转置再左右翻转再上下翻转
        feats = x.detach()
        norm_feats = torch.norm(feats, dim=1).squeeze().cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(np.fliplr(norm_feats.T), cmap='viridis', vmin=norm_feats.min(), vmax=norm_feats.max(), origin='lower')
        plt.colorbar()
        plt.title('Norm of Feature Map')
        if coors is not None:
            plt.plot(coors[0], coors[1], 'ro', markersize=8)
        flat_indices = np.argsort(np.fliplr(norm_feats.T).flatten())[-100:]
        high_y, high_x = np.unravel_index(flat_indices, np.fliplr(norm_feats.T).shape)
        plt.scatter(high_x, high_y, color='red', s=10, marker='.', alpha=0.7)
        plt.savefig(f'绘图/{name}.png')
        if coors is not None:
            target_point = feats[:, :, coors[0], coors[1]].clone()
            target_normalized = F.normalize(target_point, p=2, dim=1)
            all_features = feats.permute(0, 2, 3, 1)
            all_normalized = F.normalize(all_features, p=2, dim=3)
            similarity_map = torch.matmul(all_normalized, target_normalized.unsqueeze(-1)).squeeze(-1)
            similarity_map_np = similarity_map[0].cpu().numpy()
            cmap = LinearSegmentedColormap.from_list('similarity', ['white', 'red'], N=256)
            plt.figure(figsize=(10, 10))
            plt.imshow(np.fliplr(similarity_map_np.T), cmap=cmap, origin='lower')
            plt.colorbar()
            plt.title(f'Similarity with Point ({coors[0]}, {coors[1]})')
            plt.plot(coors[0], coors[1], 'ko', markersize=6)
            plt.savefig(f'绘图/similarity_with_point_X{coors[0]}Y{coors[1]}.png')
            plt.close()

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
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

    @torch.no_grad()
    @force_fp32()
    def radar_voxelize(self, points):
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
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

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      radar=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        img_feats, pts_feats, radar_feats = self.extract_feat(points=points, img=img, radar=radar, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(pts_feats, img_feats, radar_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          radar_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outputs_classes, outputs_coords, aux_outs, fore_pred = self.pts_bbox_head(pts_feats, img_feats, radar_feats, img_metas)
        loss_inputs = (outputs_classes, outputs_coords, gt_bboxes_3d, gt_labels_3d, img_metas, fore_pred, pts_feats)
        losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if aux_outs is not None:
            aux_loss_inputs = [gt_bboxes_3d, gt_labels_3d, aux_outs]
            aux_losses = self.pts_bbox_head.aux_head.loss(*aux_loss_inputs)
            for k, v in aux_losses.items():
                losses[f'aux_{k}'] = v * self.aux_weight
        return losses
    
    def forward_test(self, img_metas, points=None, img=None, radar=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var))) 
        num_augs = len(points) if points is not None else len(img)
        img = [img] if img is None else img
        points = [points] if points is None else points
        radar = [radar] if radar is None else radar
        return self.simple_test(img_metas[0], points[0], img[0], radar[0], **kwargs)

    def dummy_forward(self, img_metas, img=None, radar=None, points=None):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var))) 
        num_augs = len(points) if points is not None else len(img)
        img = [img] if img is None else img
        points = [points] if points is None else points
        radar = [radar] if radar is None else radar
        img_feats, pts_feats, radar_feats = self.extract_feat(points=points, img=img, radar=radar, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        outs = self.pts_bbox_head(pts_feats, img_feats, radar_feats, img_metas)
        return outs

    def simple_test_pts(self, pts_feats, img_feats, radar_feats, img_metas, rescale=False):
        outputs_classes, outputs_coords, aux_outs, fore_pred = self.pts_bbox_head(pts_feats, img_feats, radar_feats, img_metas)
        outs = (outputs_classes, outputs_coords)
        bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img_metas, points=None, img=None, radar=None, rescale=False):
        img_feats, pts_feats, radar_feats = self.extract_feat(points=points, img=img, radar=radar, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(pts_feats, img_feats, radar_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
    
    def aug_test(self, img_metas, points=None, imgs=None, radar=None, rescale=False):
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def aug_test_pts(self, feats, img_metas, rescale=False):
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list]
            aug_bboxes.append(bbox_list[0])
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas, self.pts_bbox_head.test_cfg)
        return merged_bboxes
