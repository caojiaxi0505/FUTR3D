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
from mmdet3d.core import (
    bbox3d2result,
    Box3DMode,
    Coord3DMode,
    merge_aug_bboxes_3d,
    show_result,
)
from mmdet3d.models import builder
from mmdet3d.models.backbones.base_bev_res_backbone import BaseBEVResBackbone
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors import Base3DDetector, MVXTwoStageDetector
from mmdet3d.models.middle_encoders.lion import LION3DBackboneOneStride
from plugin.dssmss.models.utils.grid_mask import GridMask
from torch.nn import functional as F


@DETECTORS.register_module(force=True)
class FUTR3D(MVXTwoStageDetector):
    def __init__(
        self,
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
        init_cfg=None,
    ):
        super(FUTR3D, self).__init__(init_cfg=init_cfg)
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_radar = use_radar
        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
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
            self.radar_middle_encoder = builder.build_middle_encoder(
                radar_middle_encoder
            )
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
        self.pts_voxel_layer_cfg = pts_voxel_layer
        self.pts_voxel_encoder_cfg = pts_voxel_encoder
        # ---------------- 已弃用 ----------------
        use_mss = pts_bbox_head.get("use_mss", False)
        mss_num_scales = pts_bbox_head.get("mss_num_scales", 4)
        mss_fore_pred_net_in_channels = pts_bbox_head.get("out_channels", 256)
        mss_fore_pred_net_intermediate_channels = pts_bbox_head.get(
            "mss_fore_pred_net_intermediate_channels", 128
        )
        mss_fore_pred_net_out_channels = pts_bbox_head.get(
            "mss_fore_pred_net_out_channels", 1
        )
        mss_dstate = pts_bbox_head.get("mss_dstate", 4)
        # ---------------- 已弃用 ----------------
        self.use_mss = False
        if use_mss:
            self.use_mss = use_mss
            from plugin.dssmss.mamba.mss import MSSMamba

            self.mss = MSSMamba(
                d_model=mss_fore_pred_net_in_channels, d_state=mss_dstate
            )
            self.batch_norm = nn.BatchNorm2d(mss_fore_pred_net_in_channels)
        # ---------------- 已弃用 ----------------

    def _freeze_backbone(self):
        for modules in [
            self.img_backbone,
            self.img_neck,
            self.pts_backbone,
            self.pts_middle_encoder,
            self.pts_neck,
        ]:
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

    def extract_img_feat_petr(self, img, img_metas):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
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
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        type = self.pts_voxel_encoder_cfg.get("type", "HardSimpleVFE")
        # ================ pts_middle_encoder使用amp ================
        if type == "HardSimpleVFE":
            voxels, num_points, coors = self.voxelize(pts)
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0] + 1
            x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        elif type == "DynamicVFE":
            voxels, coors = self.dynamic_voxelize(pts)
            voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors, pts)
            batch_size = coors[-1, 0] + 1
            if isinstance(self.pts_middle_encoder, LION3DBackboneOneStride):
                batch_dict = dict(
                    voxel_features=voxel_features,
                    voxel_coords=feature_coors,
                    batch_size=batch_size,
                )
                x = self.pts_middle_encoder(batch_dict)
            else:
                x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        if isinstance(self.pts_backbone, BaseBEVResBackbone):
            backbone_feats = self.pts_backbone(x)['spatial_features_2d']
        else:
            backbone_feats = self.pts_backbone(x)
        # self.plt_heatmap(backbone_feats[0], coors=None, name="norm_backbone_feats_scale0")
        # self.plt_heatmap(backbone_feats[1], coors=None, name="norm_backbone_feats_scale1")
        if self.with_pts_neck:
            if isinstance(self.pts_backbone, BaseBEVResBackbone):
                x = self.pts_neck([backbone_feats])
            elif len(self.pts_neck.in_channels) == 1:
                x = self.pts_neck([backbone_feats[0]])
            else:
                x = self.pts_neck(backbone_feats)
        return x

    def extract_radar_feat(self, radar, img_metas):
        voxels, num_points, coors = self.radar_voxelize(radar)
        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.radar_middle_encoder(voxel_features, coors, batch_size)
        return [x]

    def extract_feat(self, points, img, radar, img_metas):
        img_feats = self.extract_img_feat(img, img_metas) if self.use_camera else None
        # img_feats = self.extract_img_feat_petr(img, img_metas) if self.use_camera else None
        pts_feats = self.extract_pts_feat(points) if self.use_lidar else None
        radar_feats = (self.extract_radar_feat(radar, img_metas) if self.use_radar else None)
        return (img_feats, pts_feats, radar_feats)

    def plt_heatmap(self, x, coors=None, name="default"):
        # 可视化经过backbone的BEV特征的norm值
        backbone_feat = x.detach()
        norm = torch.norm(backbone_feat, dim=1).squeeze().cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(
            np.fliplr(norm.T),
            cmap="viridis",
            vmin=norm.min(),
            vmax=norm.max(),
            origin="lower",
        )
        plt.colorbar()
        plt.title("Norm of BEV Feature Map after Backbone")
        if coors is not None:
            plt.plot(coors[0], coors[1], "ro", markersize=8)
        # 找出值最高的200个点并标记为红色
        # flipped_norm = np.fliplr(norm.T)  # 与绘图保持一致的翻转
        # 将norm数组转为一维数组并找出最大的200个值的索引
        # flat_indices = np.argsort(flipped_norm.flatten())[-200:]
        # 将一维索引转换为二维坐标
        # high_y, high_x = np.unravel_index(flat_indices, flipped_norm.shape)
        # 在图上标记这些点
        # plt.scatter(high_x, high_y, color='red', s=10, marker='.', alpha=0.7)
        plt.savefig(f"绘图/{name}.png")
        # 计算点(130,87)与其他点的相似度
        if coors is not None:
            target_point = backbone_feat[:, :, coors[0], coors[1]].clone()  # [B, C]
            target_normalized = F.normalize(target_point, p=2, dim=1)  # [B, C]
            all_features = backbone_feat.permute(0, 2, 3, 1)  # [B, H, W, C]
            all_normalized = F.normalize(all_features, p=2, dim=3)  # [B, H, W, C]
            # 计算余弦相似度
            similarity_map = torch.matmul(
                all_normalized, target_normalized.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [B, H, W]
            # 绘制相似度热力图
            similarity_map_np = (
                similarity_map[0].cpu().numpy()
            )  # 可视化batch为1的heatmap
            # 创建自定义colormap: 从白色(低相似度)到(高相似度)
            cmap = LinearSegmentedColormap.from_list(
                "similarity", ["white", "red"], N=256
            )
            plt.figure(figsize=(10, 10))
            plt.imshow(np.fliplr(similarity_map_np.T), cmap=cmap, origin="lower")
            plt.colorbar(label="Cosine Similarity")
            plt.title("Similarity with Point")
            plt.plot(coors[0], coors[1], "ko", markersize=6)  # 黑点标记目标点
            plt.savefig("绘图/similarity_with_point.png")
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
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def dynamic_voxelize(self, points):
        coors = []
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

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
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        radar=None,
        proposals=None,
        gt_bboxes_ignore=None,
    ):
        img_feats, pts_feats, radar_feats = self.extract_feat(
            points=points, img=img, radar=radar, img_metas=img_metas
        )
        # points = Coord3DMode.convert_point(points[0], Coord3DMode.LIDAR, Coord3DMode.DEPTH)
        # if points is not None and len(points) > 0:
        #     fig, ax = plt.subplots(figsize=(10, 10))
        #     point_cloud = points[..., :3].cpu().numpy()
        #     sc = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.1, c=point_cloud[:, 2], cmap='viridis', label='Point Cloud')
        #     plt.colorbar(sc)
        #     # 可视化GT框
        #     gt_bev_boxes = Box3DMode.convert(gt_bboxes_3d[0], img_metas[0]['box_mode_3d'], Box3DMode.DEPTH)
        #     gt_bev_boxes = gt_bev_boxes.bev.cpu().numpy()
        #     for box in gt_bev_boxes:
        #         center_x, center_y, width, length, angle = box
        #         corner_x = center_x - width / 2
        #         corner_y = center_y - length / 2
        #         rect = plt.Rectangle((corner_x, corner_y), width, length, edgecolor='red', facecolor='none', linewidth=1.5, label='GT Box' if 'GT Box' not in plt.gca().get_legend_handles_labels()[1] else "")
        #         rotation_transform = transforms.Affine2D().rotate_around(center_x, center_y, angle)
        #         rect.set_transform(rotation_transform + ax.transData)
        #         ax.add_patch(rect)
        #     # 绘制网格线
        #     grid_size = 45
        #     x_ticks = np.linspace(-54, 54, grid_size + 1)
        #     y_ticks = np.linspace(-54, 54, grid_size + 1)
        #     for x in x_ticks:
        #         ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
        #     for y in y_ticks:
        #         ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
        #     # 设置标题、标签及显示范围
        #     plt.title('Point Cloud with GT Boxes and Reference Points in BEV')
        #     plt.xlabel('X')
        #     plt.ylabel('Y')
        #     plt.xlim(-54, 54)
        #     plt.ylim(-54, 54)
        #     plt.legend()
        #     plt.savefig('绘图/point_cloud_with_gt_boxes_and_reference_points.png')
        #     plt.close()
        losses = dict()
        losses_pts = self.forward_pts_train(
            pts_feats,
            img_feats,
            radar_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
        )
        losses.update(losses_pts)
        return losses

    def forward_pts_train(
        self,
        pts_feats,
        img_feats,
        radar_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        outputs_classes, outputs_coords, aux_outs, fore_pred = self.pts_bbox_head(
            pts_feats, img_feats, radar_feats, img_metas
        )
        loss_inputs = (
            outputs_classes,
            outputs_coords,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            fore_pred,
            pts_feats,
        )
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
        )
        if aux_outs is not None:
            aux_loss_inputs = [gt_bboxes_3d, gt_labels_3d, aux_outs]
            aux_losses = self.pts_bbox_head.aux_head.loss(*aux_loss_inputs)
            for k, v in aux_losses.items():
                losses[f"aux_{k}"] = v * self.aux_weight
        return losses

    def forward_test(self, img_metas, points=None, img=None, radar=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        num_augs = len(points) if points is not None else len(img)
        img = [img] if img is None else img
        points = [points] if points is None else points
        radar = [radar] if radar is None else radar
        return self.simple_test(img_metas[0], points[0], img[0], radar[0], **kwargs)
        # return self.aug_test(img_metas, points, None, None, **kwargs)

    def dummy_forward(self, img_metas, img=None, radar=None, points=None):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        num_augs = len(points) if points is not None else len(img)
        img = [img] if img is None else img
        points = [points] if points is None else points
        radar = [radar] if radar is None else radar
        img_feats, pts_feats, radar_feats = self.extract_feat(
            points=points, img=img, radar=radar, img_metas=img_metas
        )
        bbox_list = [dict() for i in range(len(img_metas))]
        outs = self.pts_bbox_head(pts_feats, img_feats, radar_feats, img_metas)
        return outs

    def simple_test_pts(
        self, pts_feats, img_feats, radar_feats, img_metas, rescale=False
    ):
        outputs_classes, outputs_coords, aux_outs, fore_pred = self.pts_bbox_head(
            pts_feats, img_feats, radar_feats, img_metas
        )
        outs = (outputs_classes, outputs_coords)
        bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(
        self,
        img_metas,
        points=None,
        img=None,
        radar=None,
        rescale=False,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
    ):
        """Test function without augmentaiton."""
        img_feats, pts_feats, radar_feats = self.extract_feat(
            points=points, img=img, radar=radar, img_metas=img_metas
        )
        # 可视化4个尺度的BEV特征
        # plt.ioff()  # 关闭交互模式
        # for i, feat in enumerate(pts_feats):
        #     norm = torch.norm(feat, dim=1).squeeze().cpu().numpy()
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(np.fliplr(norm.T), cmap='viridis', vmin=norm.min(), vmax=norm.max(), origin='lower') # 这行代码没问题，请不要质疑
        #     plt.colorbar()
        #     plt.title(f'Norm of BEV Feature Map at Scale {i}')
        #     plt.savefig(f'绘图/norm_bev_feat_map_scale_{i}.png')
        #     plt.close()
        # 可视化BEV视角下的点云和GT框
        # points = Coord3DMode.convert_point(points[0], Coord3DMode.LIDAR, Coord3DMode.DEPTH)
        # if points is not None and len(points) > 0 and gt_bboxes_3d is not None:
        # fig, ax = plt.subplots(figsize=(10, 10))
        # # 可视化参考点
        # ref_points = self.pts_bbox_head.refpoint_embed.weight.sigmoid() * 2 - 1
        # ref_points = ref_points.cpu().numpy()
        # ref_points[:, 0] = ref_points[:, 0] * 54
        # ref_points[:, 1] = ref_points[:, 1] * 54
        # ax.scatter(ref_points[:, 0], ref_points[:, 1], s=1, c='red', label='Reference Points')
        # 可视化点云
        # point_cloud = points[..., :3].cpu().numpy()
        # sc = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=0.1, c=point_cloud[:, 2], cmap='viridis', label='Point Cloud')
        # plt.colorbar(sc)
        # # 可视化GT框
        # gt_bev_boxes = Box3DMode.convert(gt_bboxes_3d[0][0], img_metas[0]['box_mode_3d'], Box3DMode.DEPTH)
        # gt_bev_boxes = gt_bev_boxes.bev.cpu().numpy()
        # for box in gt_bev_boxes:
        #     center_x, center_y, width, length, angle = box
        #     corner_x = center_x - width / 2
        #     corner_y = center_y - length / 2
        #     rect = plt.Rectangle((corner_x, corner_y), width, length, edgecolor='red', facecolor='none', linewidth=1.5, label='GT Box' if 'GT Box' not in plt.gca().get_legend_handles_labels()[1] else "")
        #     rotation_transform = transforms.Affine2D().rotate_around(center_x, center_y, angle)
        #     rect.set_transform(rotation_transform + ax.transData)
        #     ax.add_patch(rect)
        # # 绘制网格线
        # grid_size = 45
        # x_ticks = np.linspace(-54, 54, grid_size + 1)
        # y_ticks = np.linspace(-54, 54, grid_size + 1)
        # for x in x_ticks:
        #     ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)
        # for y in y_ticks:
        #     ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
        # 设置标题、标签及显示范围
        # plt.title('Point Cloud with GT Boxes and Reference Points in BEV')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.xlim(-54, 54)
        # plt.ylim(-54, 54)
        # plt.legend()
        # # 保存图像
        # plt.savefig('绘图/point_cloud_with_gt_boxes_and_reference_points.png')
        # plt.close()
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            pts_feats, img_feats, radar_feats, img_metas, rescale=rescale
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        # bbox_list[0]["pts_bbox"]["gt_bbox_3d"] = gt_bboxes_3d[0][
        #     0
        # ]  # 添加GT框以便在3D可视化时使用
        return bbox_list
    
    def aug_test(self, img_metas, points=None, imgs=None, radar=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats=[]
        pts_feats=[]
        radar_feats=[]
        for i in range(len(img_metas)):
            img_feat, pts_feat, radar_feat = self.extract_feat(
                points=points[i], img=None if imgs==imgs else imgs[i], radar=radar if radar==None else radar[i], img_metas=img_metas[i]
            )
            img_feats.append(img_feat)
            pts_feats.append(pts_feat)
            radar_feats.append(radar_feat)
        
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_feats, radar_feats, img_metas, rescale)
            bbox_list["pts_bbox"] = bbox_pts
        return [bbox_list]

    def aug_test_pts(self, pts_feats, img_feats, radar_feats, img_metas, rescale=False):
        # only support aug_test for one sample
        aug_bboxes = []
        for pts_feat, img_feat, radar_feat, img_meta in zip(pts_feats, img_feats, radar_feats, img_metas):
            outputs_classes, outputs_coords, aux_outs, fore_pred = self.pts_bbox_head(
                pts_feat, img_feat, radar_feat, img_meta
            )
            outs = (outputs_classes, outputs_coords)
            bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_meta, rescale=rescale)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_results[0])
        # # 构造增强测试所需的img_metas，添加必要的键
        # aug_img_metas = []
        # for i, img_meta in enumerate(img_metas if isinstance(img_metas, list) else [img_metas]):
        #     aug_img_meta = [{
        #         'pcd_scale_factor': 1.0,
        #         'pcd_horizontal_flip': False,
        #         'pcd_vertical_flip': False
        #     }]
        #     aug_img_metas.append(aug_img_meta)
        # # 如果aug_img_metas为空，添加默认元素
        # if len(aug_img_metas) == 0:
        #     aug_img_metas = [[{
        #         'pcd_scale_factor': 1.0,
        #         'pcd_horizontal_flip': False,
        #         'pcd_vertical_flip': False
        #     }]]
        # # 确保test_cfg包含所需的属性
        # from mmcv.utils import ConfigDict
        # test_cfg = self.pts_bbox_head.test_cfg
        # if not hasattr(test_cfg, 'use_rotate_nms'):
        #     test_cfg = ConfigDict({
        #         'use_rotate_nms': True,
        #         'nms_thr': 0.1,
        #         'max_num': 500
        #     })
        merged_bboxes = merge_aug_bboxes_3d(
            aug_bboxes, img_metas, self.pts_bbox_head.test_cfg
        )
        return merged_bboxes  # 与simple_test_pts返回格式保持一致

    def show_results(self, data, result, out_dir):
        for batch_id in range(len(result)):
            if isinstance(data["points"][0], DC):
                points = data["points"][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data["points"][0], torch.Tensor):
                points = data["points"][0][batch_id]
            else:
                ValueError(
                    f"Unsupported data type {type(data['points'][0])} for visualization!"
                )
            if isinstance(data["img_metas"][0], DC):
                pts_filename = data["img_metas"][0]._data[0][batch_id]["pts_filename"]
                box_mode_3d = data["img_metas"][0]._data[0][batch_id]["box_mode_3d"]
            elif mmcv.is_list_of(data["img_metas"][0], dict):
                pts_filename = data["img_metas"][0][batch_id]["pts_filename"]
                box_mode_3d = data["img_metas"][0][batch_id]["box_mode_3d"]
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} for visualization!"
                )
            file_name = osp.split(pts_filename)[-1].split(".")[0]
            assert out_dir is not None, "Expect out_dir, got none."
            inds = result[batch_id]["pts_bbox"]["scores_3d"] > 0.1
            pred_bboxes = result[batch_id]["pts_bbox"]["boxes_3d"][inds]
            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(
                    points, Coord3DMode.LIDAR, Coord3DMode.DEPTH
                )
                pred_bboxes = Box3DMode.convert(
                    pred_bboxes, box_mode_3d, Box3DMode.DEPTH
                )
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(f"Unsupported box_mode_3d {box_mode_3d} for conversion!")
            gt_bbox_3d = result[batch_id]["pts_bbox"]["gt_bbox_3d"]
            gt_bbox_3d = Box3DMode.convert(gt_bbox_3d, box_mode_3d, Box3DMode.DEPTH)
            gt_bbox_3d = gt_bbox_3d.tensor.cpu().numpy()
            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(
                points,
                gt_bboxes=None,
                pred_bboxes=None,
                out_dir=out_dir,
                filename=file_name,
                show=True,
            )
