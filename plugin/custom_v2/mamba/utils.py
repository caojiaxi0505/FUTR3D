import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmdet.models.task_modules.assigners import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmdet.models.task_modules.builder import build_match_cost
from mmdet3d.registry import TASK_UTILS
from mmengine.structures.instance_data import InstanceData
from torch import Tensor
from typing import Tuple, List, Optional
import matplotlib.transforms as transforms

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def normalize_bbox(bboxes, pc_range=None):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range=None):
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 2:3]
    w = normalized_bboxes[..., 3:4]
    l = normalized_bboxes[..., 4:5]
    h = normalized_bboxes[..., 5:6]
    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


@TASK_UTILS.register_module(force=True)
class BBox3DL1Cost(object):
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@TASK_UTILS.register_module(force=True)
class HungarianAssigner3D(BaseAssigner):
    def __init__(
        self,
        cls_cost=dict(type="mmdet.ClassificationCost", weight=1.0),
        reg_cost=dict(type="mmdet.BBoxL1Cost", weight=1.0),
        iou_cost=dict(type="mmdet.IoUCost", weight=0.0),
        pc_range=None,
    ):
        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.iou_cost = TASK_UTILS.build(iou_cost)
        self.pc_range = pc_range

    def assign(
        self,
        bbox_pred,
        cls_pred,
        gt_bboxes,
        gt_labels,
    ):
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
        # 2. compute the weighted costs classification and bboxcost.
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        pred_instances = InstanceData(scores=cls_pred, bboxes=bbox_pred[:, :8])
        gt_instances = InstanceData(
            labels=gt_labels, bboxes=normalized_gt_bboxes[:, :8]
        )
        cls_cost = self.cls_cost(pred_instances, gt_instances)
        if bbox_pred.size(-1) > 8:
            reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
        else:
            reg_cost = self.reg_cost(bbox_pred, normalized_gt_bboxes)
        cost = cls_cost + reg_cost
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)
        # 4. assign backgrounds and foregrounds assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # 5. assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    elif pos_tensor.size(-1) == 3:
        z_embed = pos_tensor[:, :, 2] * scale
        pos_z = z_embed[:, :, None] / dim_t
        pos_z = torch.stack(
            (pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_z), dim=2)
    elif pos_tensor.size(-1) == 6:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        z_embed = pos_tensor[:, :, 4] * scale
        pos_z = z_embed[:, :, None] / dim_t
        pos_z = torch.stack(
            (pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        l_embed = pos_tensor[:, :, 5] * scale
        pos_l = l_embed[:, :, None] / dim_t
        pos_l = torch.stack(
            (pos_l[:, :, 0::2].sin(), pos_l[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h, pos_z, pos_l), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def plot_multi_scale_batch_foreground_predictions(
    multi_scale_batch_foreground_predictions: List[Tensor],
    scale_indexes: Optional[List[int]],
    batch_indexes: Optional[List[int]],
    threhold: float = 0.5,
) -> None:
    """
    Input:
        multi_scale_foreground_predictions: List[Tensor], each tensor has shape (bs, 1, hi, wi)
        scale_indexes: Optional[List[int]], to indicate which scales foreground predictions need to be showed
        batch_indexes: Optional[List[int]], to indicate which batches foreground predictions need to be showed
    """
    # for scale_index in scale_indexes:
    #     batch_foreground_predictions = multi_scale_batch_foreground_predictions[
    #         scale_index
    #     ]
    # for batch_index in batch_indexes:
    #     foreground_predictions = batch_foreground_predictions[batch_index].squeeze()
    #     foreground_predictions = foreground_predictions.detach().cpu().numpy()
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(foreground_predictions, cmap="gray", aspect="auto")
    #     plt.colorbar()
    #     plt.title(
    #         f"Foreground Predictions at scale {scale_index} and batch {batch_index}"
    #     )
    #     plt.savefig(f"fg_scale{scale_index}_batch{batch_index}.png")
    for scale_index in scale_indexes:
        batch_foreground_predictions = multi_scale_batch_foreground_predictions[
            scale_index
        ]
        for batch_index in batch_indexes:
            foreground_predictions = batch_foreground_predictions[batch_index].squeeze()
            foreground_predictions = foreground_predictions.detach().cpu().numpy()
            k = int(
                foreground_predictions.shape[0] * foreground_predictions.shape[1] * 0.1
            )
            threhold = np.partition(foreground_predictions.flatten(), -k)[-k]
            binary_foreground_predictions = (foreground_predictions > threhold).astype(
                int
            )
            plt.figure(figsize=(8, 8))
            plt.imshow(binary_foreground_predictions, cmap="gray", aspect="auto")
            plt.colorbar()
            plt.title(
                f"Foreground Predictions at scale {scale_index} and batch {batch_index} threshold {threhold}"
            )
            plt.savefig(
                f"fg_scale{scale_index}_batch{batch_index}_threshold{threhold}.png"
            )


def plot_multi_scale_batch_bev_feats(
    multi_scale_batch_bev_feats: Tuple[Tensor],
    batch_gt_bboxes: Optional[List[Tensor]],
    scale_indexes: Optional[List[int]],
    batch_indexes: Optional[List[int]],
    multi_level_batch_reference_points: Optional[List[Tensor]],
    level_indexes: Optional[List[int]],
    name: str,
) -> None:
    """
    Input:
        multi_scale_batch_bev_feats: Tuple[Tensor], each tensor has shape (bs, c, hi, wi)
        batch_gt_bboxes: Optional[List[Tensor]], each tensor has shape (num_gt, 9)
        scale_indexes: Optional[List[int]], to indicate which scales bev feats need to be showed
        batch_indexes: Optional[List[int]], to indicate which batches bev feats need to be showed
        multi_level_batch_reference_points: Optional[List[Tensor]], shape (bs, nq, 2)
        level_indexes: Optional[List[int]], to indicate which levels reference_points need to be showed
    """
    point_cloud_range = [-54, -54, -5, 54, 54, 3]
    for scale_index in scale_indexes:
        batch_bev_feats = multi_scale_batch_bev_feats[scale_index]
        for batch_index in batch_indexes:
            bev_feats = batch_bev_feats[batch_index]
            bev_feats_var = bev_feats.var(axis=0).detach().cpu().numpy()
            bev_feats_mean = bev_feats.mean(axis=0).detach().cpu().numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(bev_feats_var, cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.title(
                f"Var of Bev Feats at scale {scale_index} and batch {batch_index}"
            )
            ax = plt.gca()
            if batch_gt_bboxes is not None:
                gt_bboxes = batch_gt_bboxes[batch_index]
                gt_bboxes = gt_bboxes.detach().cpu().numpy()
                bboxes = np.array(gt_bboxes)
                x = (
                    (bboxes[:, 0] + 54)
                    / (point_cloud_range[3] - point_cloud_range[0])
                    * bev_feats_var.shape[0]
                )
                y = (
                    (bboxes[:, 1] + 54)
                    / (point_cloud_range[4] - point_cloud_range[1])
                    * bev_feats_var.shape[1]
                )
                w = (
                    bboxes[:, 3]
                    / (point_cloud_range[3] - point_cloud_range[0])
                    * bev_feats_var.shape[0]
                )
                h = (
                    bboxes[:, 4]
                    / (point_cloud_range[4] - point_cloud_range[1])
                    * bev_feats_var.shape[1]
                )
                angles = np.degrees(bboxes[:, 6])
                cx, cy = x, y
                dx = -w / 2
                dy = -h / 2
                trans = [
                    transforms.Affine2D().rotate_deg_around(cx[i], cy[i], angles[i])
                    for i in range(len(cx))
                ]
                transformed_points = np.array(
                    [
                        trans[i].transform_point((cx[i] + dx[i], cy[i] + dy[i]))
                        for i in range(len(cx))
                    ]
                )
                x0, y0 = transformed_points[:, 0], transformed_points[:, 1]
                for i in range(len(cx)):
                    rect = patches.Rectangle(
                        (x0[i], y0[i]),
                        w[i],
                        h[i],
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                        angle=angles[i],
                    )
                    ax.add_patch(rect)
            if multi_level_batch_reference_points is not None:
                for level_index in level_indexes:
                    batch_reference_points = multi_level_batch_reference_points[
                        level_index
                    ]
                    reference_points = batch_reference_points[batch_index]
                    reference_points = reference_points[:, :2].detach().cpu().numpy()
                    ref_x = (reference_points[:, 0] + 54) / 108 * bev_feats_var.shape[0]
                    ref_y = (reference_points[:, 1] + 54) / 108 * bev_feats_var.shape[1]
                    plt.scatter(
                        ref_x, ref_y, c="blue", marker="*", label="Reference Points"
                    )
                    plt.legend()
                    plt.title(
                        f"Var of Bev Feats at scale {scale_index} and batch {batch_index} with reference points at level {level_index}"
                    )
                    plt.savefig(
                        f"var_scale{scale_index}_batch{batch_index}_refLevel{level_index}.png"
                    )
                continue
            plt.savefig(f"{name}_var_scale{scale_index}_batch{batch_index}.png")
    # batch_bev_feats = multi_scale_batch_bev_feats[scale_index]  # bs, c, hi, wi
    # bev_feats = batch_bev_feats[batch_index]  # c, hi, wi
    # bev_feats = bev_feats.detach().cpu().numpy()
    # bev_feats_var = bev_feats.var(axis=0)
    # bev_feats_mean = bev_feats.mean(axis=0)
    # Calculate the threshold for the top 30% of bev_feats_var
    # threshold = np.percentile(bev_feats_var, 95)
    # Create a mask for the points above the threshold
    # foreground_mask = bev_feats_var > threshold
    # Create a new image where foreground points are one color and background points are another color
    # output_image = np.zeros_like(bev_feats_var)
    # output_image[foreground_mask] = 1  # Foreground points
    # output_image[~foreground_mask] = 0  # Background points
    # Save the new image
    # plt.figure(figsize=(8, 8))
    # plt.imshow(output_image, cmap="gray", aspect="auto")
    # plt.colorbar()
    # plt.title("Foreground and Background Points")
    # plt.savefig("foreground_background.png")
    # plt.figure(figsize=(8, 8))
    # plt.imshow(bev_feats_var, cmap="viridis", aspect="auto")
    # plt.colorbar()
    # plt.title("Heatmap")
    # ax = plt.gca()
    # if gt_bboxes_list is None:
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(bev_feats_var, cmap="viridis", aspect="auto")
    #     plt.colorbar()
    #     plt.title("Heatmap")
    #     return
    # gt_bboxes = gt_bboxes_list[batch_index]
    # gt_bboxes = gt_bboxes.detach().cpu().numpy()
    # for bbox in gt_bboxes:
    #     x, y, w, h, angle = bbox[0], bbox[1], bbox[3], bbox[4], bbox[6]
    #     x = (x + 54) / 108 * bev_feats_var.shape[1]
    #     y = (y + 54) / 108 * bev_feats_var.shape[0]
    #     w = w / 108 * bev_feats_var.shape[1]
    #     h = h / 108 * bev_feats_var.shape[0]
    #     angle = np.degrees(angle)  # Convert angle from radians to degrees
    #     cx, cy = x, y
    #     dx = -w / 2
    #     dy = -h / 2
    #     # 创建变换对象
    #     trans = transforms.Affine2D().rotate_deg_around(cx, cy, angle)
    #     # 计算旋转后的左下角坐标
    #     x0, y0 = trans.transform_point((cx + dx, cy + dy))
    #     if use_angle:
    #         rect = patches.Rectangle(
    #             (x0, y0),
    #             w,
    #             h,
    #             linewidth=1,
    #             edgecolor="r",
    #             facecolor="none",
    #             angle=angle,
    #         )
    #     else:
    #         rect = patches.Rectangle(
    #             (x0, y0), w, h, linewidth=1, edgecolor="r", facecolor="none", angle=0
    #         )
    #     ax.add_patch(rect)
    # if first_level:
    #     reference_points = reference_points[0]
    # else:
    #     reference_points = reference_points[5]
    # reference_points = reference_points[batch_index]
    # reference_points = reference_points[:,:2].detach().cpu().numpy()
    # ref_x = (reference_points[:, 0] + 54) / 108 * bev_feats_var.shape[1]
    # ref_y = (reference_points[:, 1] + 54) / 108 * bev_feats_var.shape[0]
    # plt.scatter(ref_x, ref_y, c='blue', marker='x', label='Reference Points')
    # plt.legend()


def plot_pts(pts: Tensor):
    pts = pts.detach().cpu()
    x = pts[:, 0].numpy()
    y = pts[:, 1].numpy()
    z = pts[:, 2].numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="r", marker="o")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.savefig("pts.png")

def horizen_snake_extraction(mask: torch.Tensor) -> torch.Tensor:
    # 获取mask的设备
    device = mask.device
    h, w = mask.shape
    
    # 将tensor创建在相同设备上
    tensor = torch.arange(h*w, device=device).reshape(h,w)
    
    extracted = []
    
    for row_idx in range(mask.shape[0]):
        # 确保cols在相同设备上
        cols = (mask[row_idx] != 0).nonzero()
        if cols.numel() == 0:
            continue
            
        cols = cols.view(-1)
        
        if row_idx % 2 == 1:
            cols = cols.flip(0)
            
        extracted.append(tensor[row_idx, cols])
    
    # 确保返回值也在正确的设备上    
    return torch.cat(extracted) if extracted else torch.tensor([], device=device)


def vertical_snake_extraction(mask: torch.Tensor) -> torch.Tensor:
    # 获取mask的设备
    device = mask.device
    h, w = mask.shape
    
    # 将tensor创建在相同设备上
    tensor = torch.arange(h * w, device=device).reshape(h, w)
    
    extracted = []
    
    for col_idx in range(w):
        # 确保rows在相同设备上
        rows = (mask[:, col_idx] != 0).nonzero()
        if rows.numel() == 0:
            continue
            
        # 确保rows是1维的
        rows = rows.view(-1)
        
        if col_idx % 2 == 1:
            rows = rows.flip(0)
            
        extracted.append(tensor[rows, col_idx])
    
    # 确保返回值也在正确的设备上    
    return torch.cat(extracted) if extracted else torch.tensor([], device=device)

def pad_to_same_length(indices_list: List[Tensor]) -> Tensor:
    """
    将一组索引tensor补齐到相同长度
    Args:
        indices_list: List[Tensor] 需要补齐的索引列表
    Returns:
        Tensor: 补齐后的索引张量 shape: (batch_size, max_length)
    """
    # 找出最大长度
    import ipdb; ipdb.set_trace()
    max_length = max(indices.numel() for indices in indices_list)
    
    # 补齐每个tensor
    padded_indices = []
    for indices in indices_list:
        if indices.numel() < max_length:
            # 计算需要补充的长度
            pad_length = max_length - indices.numel()
            # 在尾部补充0
            padded = torch.cat([indices, torch.zeros(pad_length, dtype=indices.dtype, device=indices.device)])
        else:
            padded = indices
        padded_indices.append(padded)
    
    # 堆叠所有补齐后的tensor
    return torch.stack(padded_indices)

def get_sorted_indices(foreground_predictions: List[Tensor]):
    """
    获取多尺度前景预测的水平和垂直蛇形索引
    Args:
        foreground_predictions: List[Tensor] 每个tensor形状为(bs,1,h,w)
    Returns:
        horizen_indices: List[Tensor] 每个tensor形状为(bs, num_valid_points)
        vertical_indices: List[Tensor] 每个tensor形状为(bs, num_valid_points)
    """
    horizen_indices = []
    vertical_indices = []
    
    for pred in foreground_predictions:
        bs = pred.shape[0]
        h_batch_indices = []
        v_batch_indices = []
        
        for b in range(bs):
            mask = pred[b, 0] > 0.5  # shape: (h, w)
            h_idx = horizen_snake_extraction(mask)
            v_idx = vertical_snake_extraction(mask)
            h_batch_indices.append(h_idx)
            v_batch_indices.append(v_idx)
        
        horizen_indices.append(h_batch_indices)
        vertical_indices.append(v_batch_indices)
    
    return horizen_indices, vertical_indices

def extract_features_by_indices(features: List[Tensor], horizen_indices: List[List[Tensor]], vertical_indices: List[List[Tensor]]):
    bs = features[0].shape[0]
    c = features[0].shape[1]
    device = features[0].device
    h_batch_features = []
    v_batch_features = []
    for batch_idx in range(bs):
        curr_h_features = []
        curr_v_features = []
        for feat, h_indices_scale, v_indices_scale in zip(features, horizen_indices, vertical_indices):
            curr_feat = feat[batch_idx].view(c, -1)
            h_idx = h_indices_scale[batch_idx]
            v_idx = v_indices_scale[batch_idx]
            if h_idx.numel() > 0:
                curr_h_features.append(curr_feat[:, h_idx].t())
            if v_idx.numel() > 0:
                curr_v_features.append(curr_feat[:, v_idx].t())
        if curr_h_features:
            h_batch_features.append(torch.cat(curr_h_features, dim=0))
        else:
            h_batch_features.append(torch.zeros((0, c), device=device))
        if curr_v_features:
            v_batch_features.append(torch.cat(curr_v_features, dim=0))
        else:
            v_batch_features.append(torch.zeros((0, c), device=device))
    max_h_len = max(feat.size(0) for feat in h_batch_features)
    max_v_len = max(feat.size(0) for feat in v_batch_features)
    padded_h_features = []
    padded_v_features = []
    for h_feat, v_feat in zip(h_batch_features, v_batch_features):
        if h_feat.size(0) < max_h_len:
            padding = torch.zeros((max_h_len - h_feat.size(0), c), device=device)
            padded_h_features.append(torch.cat([h_feat, padding], dim=0))
        else:
            padded_h_features.append(h_feat)
        if v_feat.size(0) < max_v_len:
            padding = torch.zeros((max_v_len - v_feat.size(0), c), device=device)
            padded_v_features.append(torch.cat([v_feat, padding], dim=0))
        else:
            padded_v_features.append(v_feat)
    return torch.stack(padded_h_features), torch.stack(padded_v_features)

def concat_multi_scale_indices(indices_list: List[Tensor], batch_idx: int) -> Tensor:
    """
    拼接指定batch的所有尺度索引
    Args:
        indices_list: List[Tensor] 每个元素是一个尺度的索引tensor，shape为(bs, n)
        batch_idx: int 要处理的batch索引
    Returns:
        Tensor: 拼接后的该batch的所有尺度索引
    """
    return torch.cat([indices[batch_idx, :((indices[batch_idx] != 0).sum())] 
                     for indices in indices_list])

# def extract_features_by_indices(features: List[Tensor], horizen_indices: List[Tensor], vertical_indices: List[Tensor]):
#     """
#     根据水平和垂直蛇形索引提取多尺度特征图中的元素，并将同一batch的不同尺度特征拼接
#     Args:
#         features: List[Tensor] 多尺度特征图列表,每个tensor形状为(bs,c,h,w)
#         horizen_indices: List[Tensor] 每个tensor形状为(bs, num_valid_points)
#         vertical_indices: List[Tensor] 每个tensor形状为(bs, num_valid_points)
#     Returns:
#         h_features: List[Tensor] 水平蛇形排列提取的特征
#         v_features: List[Tensor] 垂直蛇形排列提取的特征
#     """
#     bs = features[0].shape[0]
#     c = features[0].shape[1]
    
#     h_batch_features = [[] for _ in range(bs)]
#     v_batch_features = [[] for _ in range(bs)]
    
#     for feat, h_idx, v_idx in zip(features, horizen_indices, vertical_indices):
#         for b in range(bs):
#             curr_feat = feat[b].view(c, -1)  # shape: (c, h*w)
            
#             h_feat = curr_feat[:, h_idx[b]]
#             h_batch_features[b].append(h_feat)
            
#             v_feat = curr_feat[:, v_idx[b]]
#             v_batch_features[b].append(v_feat)
    
#     # 对每个batch拼接不同尺度的特征
#     h_features = []
#     v_features = []
    
#     for b in range(bs):
#         if h_batch_features[b]:
#             h_features.append(torch.cat(h_batch_features[b], dim=1))
#         else:
#             h_features.append(torch.tensor([]))
            
#         if v_batch_features[b]:
#             v_features.append(torch.cat(v_batch_features[b], dim=1))
#         else:
#             v_features.append(torch.tensor([]))
    
#     return h_features, v_features

def insert_features_back(features: Tensor, indices: Tensor, h: int, w: int) -> Tensor:
    """
    将提取的特征插回到原始位置
    Args:
        features: Tensor, shape (c, n) 提取的特征
        indices: Tensor, shape (n,) 特征对应的索引
        h: int, 原始高度
        w: int, 原始宽度
    Returns:
        Tensor: shape (c, h, w) 特征插回原位后的张量
    """
    c = features.shape[0]
    # 创建输出张量，初始化为0
    output = torch.zeros((c, h * w), device=features.device)
    
    # 将特征插回到对应位置
    output[:, indices] = features
    
    # 重塑为原始形状
    output = output.reshape(c, h, w)
    
    return output


# def remap_vertical_to_horizontal(v_features: torch.Tensor, v_indices: list, h_indices: list) -> torch.Tensor:
#     """
#     将垂直方向提取的特征重新映射到水平方向的顺序
#     Args:
#         v_features: Tensor, shape (b, l, c) 垂直方向提取的特征 (batch, length, channels)
#         v_indices: List[Tensor], 每个元素shape为(b, n)的垂直方向索引列表
#         h_indices: List[Tensor], 每个元素shape为(b, m)的水平方向索引列表
#     Returns:
#         Tensor: shape (b, m, c) 重新映射后的特征 (batch, new_length, channels)
#     """
#     b, l, c = v_features.shape
#     device = v_features.device
    
#     remapped_features = []
#     for batch_idx in range(b):
#         # 获取当前batch在每个尺度的索引并合并
#         curr_v_indices = torch.cat([indices[batch_idx] for indices in v_indices], dim=-1)  # (N,)
#         curr_h_indices = torch.cat([indices[batch_idx] for indices in h_indices], dim=-1)  # (M,)
        
#         # 获取当前batch的特征
#         curr_features = v_features[batch_idx]  # (L, C)
        
#         # 创建临时数组存储当前batch的垂直特征完整映射
#         max_idx = max(curr_v_indices.max().item(), curr_h_indices.max().item())
#         temp_full = torch.zeros((max_idx + 1, c), device=device)
        
#         # 使用索引直接赋值
#         temp_full.index_copy_(0, curr_v_indices.long(), curr_features)
        
#         # 按照水平索引提取特征
#         remapped = temp_full[curr_h_indices.long()]
#         remapped_features.append(remapped)
    
#     return torch.stack(remapped_features)

def remap_vertical_to_horizontal(v_features: torch.Tensor, v_indices: List[List[Tensor]], h_indices: List[List[Tensor]]) -> torch.Tensor:
    """
    将垂直方向提取的特征重新映射到水平方向的顺序
    Args:
        v_features: Tensor, shape (b, l, c) 垂直方向提取的特征 (batch, length, channels)
        v_indices: List[List[Tensor]], 每个尺度包含b个tensor的列表，每个tensor为该batch在该尺度的索引
        h_indices: List[List[Tensor]], 每个尺度包含b个tensor的列表，每个tensor为该batch在该尺度的索引
    Returns:
        Tensor: shape (b, m, c) 重新映射后的特征 (batch, new_length, channels)
    """
    b, l, c = v_features.shape
    device = v_features.device
    remapped_features = []
    for batch_idx in range(b):
        # 计算每个尺度的有效特征数量
        valid_nums = [scale_indices[batch_idx].shape[0] for scale_indices in v_indices]
        total_valid = sum(valid_nums)
        
        # 根据有效数量获取索引
        curr_v_indices = []
        curr_h_indices = []
        curr_start = 0
        
        for v_scale_indices, h_scale_indices, valid_num in zip(v_indices, h_indices, valid_nums):
            if valid_num > 0:
                curr_v_indices.append(v_scale_indices[batch_idx])
                curr_h_indices.append(h_scale_indices[batch_idx])
                curr_start += valid_num
                
        curr_v_indices = torch.cat(curr_v_indices) if curr_v_indices else torch.tensor([], device=device)
        curr_h_indices = torch.cat(curr_h_indices) if curr_h_indices else torch.tensor([], device=device)
        
        # 获取当前batch的特征，只取有效长度部分
        curr_features = v_features[batch_idx, :total_valid]  # (total_valid, C)
        
        if curr_features.shape[0] > 0:
            # 创建临时数组存储当前batch的垂直特征完整映射
            max_idx = max(curr_v_indices.max().item(), curr_h_indices.max().item())
            temp_full = torch.zeros((max_idx + 1, c), device=device)
            
            # 使用索引直接赋值
            temp_full.index_copy_(0, curr_v_indices.long(), curr_features)
            
            # 按照水平索引提取特征
            remapped = temp_full[curr_h_indices.long()]
        else:
            remapped = torch.zeros((0, c), device=device)
            
        remapped_features.append(remapped)
    
    # 找出最大长度进行padding
    max_len = max(feat.size(0) for feat in remapped_features)
    padded_features = []
    for feat in remapped_features:
        if feat.size(0) < max_len:
            padding = torch.zeros((max_len - feat.size(0), c), device=device)
            padded_features.append(torch.cat([feat, padding], dim=0))
        else:
            padded_features.append(feat)
    
    return torch.stack(padded_features)


def insert_features_back_batched(features: torch.Tensor, h_indices: List[List[Tensor]], 
                               heights: List[int], widths: List[int]) -> List[torch.Tensor]:
    """
    将特征插回原始空间位置，处理batch维度
    Args:
        features: Tensor, shape (b, L, c) batch中每个样本的特征，部分batch可能包含填充
        h_indices: List[List[Tensor]] 每个尺度包含b个tensor的列表，每个tensor为该batch在该尺度的索引
        heights: List[int] 每个尺度特征图的高度
        widths: List[int] 每个尺度特征图的宽度
    Returns:
        List[Tensor]: 每个tensor形状为(b, c, h, w)的多尺度特征图列表
    """
    b, _, c = features.shape
    device = features.device
    
    output_features = []
    
    # 处理每个尺度
    for scale_idx, (scale_indices, h, w) in enumerate(zip(h_indices, heights, widths)):
        scale_output = []
        
        # 处理每个batch
        for batch_idx in range(b):
            # 创建当前batch的输出tensor
            curr_output = torch.zeros((c, h * w), device=device)
            
            # 获取当前batch在当前尺度的索引
            curr_indices = scale_indices[batch_idx]
            if curr_indices.numel() > 0:
                # 计算当前batch在features中的起始位置
                start_idx = sum(indices[batch_idx].shape[0] for indices in h_indices[:scale_idx])
                end_idx = start_idx + curr_indices.shape[0]
                
                # 获取当前batch的特征
                curr_features = features[batch_idx, start_idx:end_idx]  # (n, c)
                
                # 将特征插回到对应位置
                curr_output[:, curr_indices.long()] = curr_features.t()
            
            # 重塑为原始形状
            curr_output = curr_output.reshape(c, h, w)
            scale_output.append(curr_output)
        
        # 堆叠当前尺度的所有batch
        output_features.append(torch.stack(scale_output))
    
    return output_features






# 测试代码
def test_feature_extraction():
    print("\n=== 测试特征提取函数 ===")
    
    # 为了便于验证，使用固定值而不是随机值
    features = [
        torch.tensor([
            # batch 1
            [[1, 2],
             [3, 4]], 
            # batch 2
            [[5, 6],
             [7, 8]]
        ]).float().unsqueeze(1),  # (2,1,2,2)
        
        torch.tensor([
            # batch 1
            [[10, 11, 12],
             [13, 14, 15],
             [16, 17, 18]],
            # batch 2
            [[20, 21, 22],
             [23, 24, 25],
             [26, 27, 28]]
        ]).float().unsqueeze(1)  # (2,1,3,3)
    ]
    
    masks = [
        torch.tensor([
            [[1, 0],
             [1, 1]],
            [[1, 1],
             [0, 1]]
        ]).unsqueeze(1),  # (2,1,2,2)
        
        torch.tensor([
            [[1, 1, 1],
             [0, 1, 0],
             [1, 1, 1]],
            [[1, 0, 1],
             [1, 1, 1],
             [0, 1, 0]]
        ]).unsqueeze(1)  # (2,1,3,3)
    ]
    
    print("\n输入特征图:")
    for i, feat in enumerate(features):
        print(f"\n尺度{i+1}:")
        for b in range(feat.shape[0]):
            print(f"Batch {b}:\n", feat[b, 0])
    
    print("\n输入mask:")
    for i, mask in enumerate(masks):
        print(f"\n尺度{i+1}:")
        for b in range(mask.shape[0]):
            print(f"Batch {b}:\n", mask[b, 0])
    
    # 获取索引
    horizen_indices, vertical_indices = get_sorted_indices(masks)
    
    print("\n提取的索引:")
    for i, (h_idx, v_idx) in enumerate(zip(horizen_indices, vertical_indices)):
        print(f"\n尺度{i+1}:")
        print("水平索引:")
        for b in range(h_idx.shape[0]):
            print(f"Batch {b}:", h_idx[b])
        print("\n垂直索引:")
        for b in range(v_idx.shape[0]):
            print(f"Batch {b}:", v_idx[b])
    
    # 提取特征
    h_features, v_features = extract_features_by_indices(features, horizen_indices, vertical_indices)
    
    print("\n提取的特征:")
    print("\n水平方向特征:")
    for i, feat in enumerate(h_features):
        print(f"\nBatch {i}:")
        if feat.numel() > 0:
            print(feat)
        else:
            print("空tensor")
    
    print("\n垂直方向特征:")
    for i, feat in enumerate(v_features):
        print(f"\nBatch {i}:")
        if feat.numel() > 0:
            print(feat)
        else:
            print("空tensor")

    # 验证提取结果的正确性
    print("\n验证结果:")
    print("1. 检查每个batch的特征数量是否正确")
    print("2. 检查特征值是否与输入对应")
    print("3. 检查蛇形提取顺序是否正确")
    
    # 举例验证第一个batch的水平特征
    if h_features[0].numel() > 0:
        print("\n第一个batch水平特征示例验证:")
        print("特征值:", h_features[0])
        print("应对应原始特征图中的位置")

def test_feature_insertion():
    print("\n=== 测试特征插回 ===")
    
    # 创建测试用例
    # 第一个尺度
    features1 = torch.tensor([
        [[1, 2],
         [3, 4]]
    ]).float()  # (1,2,2)
    
    mask1 = torch.tensor([[1, 0],
                         [1, 1]])
    
    # 第二个尺度
    features2 = torch.tensor([
        [[10, 11, 12],
         [13, 14, 15],
         [16, 17, 18]]
    ]).float()  # (1,3,3)
    
    mask2 = torch.tensor([[1, 1, 1],
                         [0, 1, 0],
                         [1, 1, 1]])
    
    # 获取索引
    h_idx1 = horizen_snake_extraction(mask1)
    h_idx2 = horizen_snake_extraction(mask2)
    
    print("第一个尺度:")
    print("原始特征图:\n", features1[0])
    print("mask:\n", mask1)
    print("提取的索引:", h_idx1)
    
    # 提取特征
    feat1 = features1[0].reshape(1, -1)[:, h_idx1]
    print("提取的特征:", feat1)
    
    # 插回特征
    inserted1 = insert_features_back(feat1, h_idx1, h=2, w=2)
    print("插回后的特征图:\n", inserted1)
    
    print("\n第二个尺度:")
    print("原始特征图:\n", features2[0])
    print("mask:\n", mask2)
    print("提取的索引:", h_idx2)
    
    # 提取特征
    feat2 = features2[0].reshape(1, -1)[:, h_idx2]
    print("提取的特征:", feat2)
    
    # 插回特征
    inserted2 = insert_features_back(feat2, h_idx2, h=3, w=3)
    print("插回后的特征图:\n", inserted2)

def test_vertical_to_horizontal_mapping():
    print("\n=== 测试垂直特征到水平顺序的映射 ===")
    
    # 创建测试用例
    # 原始特征图 (1 channel for simplicity)
    features = torch.tensor([
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    ]).float()  # (1,3,3)
    
    # mask
    mask = torch.tensor([
        [1, 1, 1],
        [0, 1, 0],
        [1, 1, 1]
    ])
    
    print("原始特征图:\n", features[0])
    print("mask:\n", mask)
    
    # 获取水平和垂直索引
    h_indices = horizen_snake_extraction(mask)
    v_indices = vertical_snake_extraction(mask)
    
    print("\n水平索引:", h_indices)
    print("垂直索引:", v_indices)
    
    # 提取特征
    feat_flat = features[0].reshape(1, -1)
    h_features = feat_flat[:, h_indices]
    v_features = feat_flat[:, v_indices]
    
    print("\n水平方向提取的特征:", h_features)
    print("垂直方向提取的特征:", v_features)
    
    # 将垂直特征重映射到水平顺序
    remapped_features = remap_vertical_to_horizontal(v_features, v_indices, h_indices)
    print("\n重映射后的特征:", remapped_features)
    
    # 验证重映射后特征的形状是否与水平特征相同
    print("\n形状验证:")
    print("水平特征形状:", h_features.shape)
    print("重映射特征形状:", remapped_features.shape)
    
    # 将特征插回原位验证
    h_inserted = insert_features_back(h_features, h_indices, h=3, w=3)
    v_inserted = insert_features_back(v_features, v_indices, h=3, w=3)
    remapped_inserted = insert_features_back(remapped_features, h_indices, h=3, w=3)
    
    print("\n插回后的特征图对比:")
    print("水平特征插回:\n", h_inserted)
    print("垂直特征插回:\n", v_inserted)
    print("重映射特征插回:\n", remapped_inserted)

if __name__ == "__main__":
    test_vertical_to_horizontal_mapping()

