import math
import torch
from torch import Tensor
from typing import List, Optional, Tuple


def generate_gt_foreground_v1(multi_scale_batch_bev_feats, gt_bboxes, bev_scales):
    bs = len(gt_bboxes)  # Batch size

    # Extract BEV scales
    bev_x_min, bev_y_min, bev_x_max, bev_y_max = bev_scales
    bev_width = bev_x_max - bev_x_min
    bev_height = bev_y_max - bev_y_min

    # Initialize the output list
    gt_foreground = []

    # Loop through multi-scale BEV features
    for bev_feat in multi_scale_batch_bev_feats:
        _, _, feat_h, feat_w = bev_feat.shape
        scale_x = feat_w / bev_width
        scale_y = feat_h / bev_height

        # Initialize foreground mask for this scale
        foreground_mask = torch.zeros((bs, 1, feat_h, feat_w), device=bev_feat.device)

        for b_idx in range(bs):
            for bbox in gt_bboxes[b_idx]:
                x_center, y_center, width, height, yaw = bbox

                # Create a rotation matrix for yaw
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)

                # Calculate the four corners of the rotated bounding box
                corners = [
                    [
                        x_center + cos_yaw * (width / 2) - sin_yaw * (height / 2),
                        y_center + sin_yaw * (width / 2) + cos_yaw * (height / 2),
                    ],
                    [
                        x_center - cos_yaw * (width / 2) - sin_yaw * (height / 2),
                        y_center - sin_yaw * (width / 2) + cos_yaw * (height / 2),
                    ],
                    [
                        x_center - cos_yaw * (width / 2) + sin_yaw * (height / 2),
                        y_center - sin_yaw * (width / 2) - cos_yaw * (height / 2),
                    ],
                    [
                        x_center + cos_yaw * (width / 2) + sin_yaw * (height / 2),
                        y_center + sin_yaw * (width / 2) - cos_yaw * (height / 2),
                    ],
                ]

                # Clip corners to BEV boundaries
                clipped_corners = [
                    [
                        max(bev_x_min, min(bev_x_max, corner[0])),
                        max(bev_y_min, min(bev_y_max, corner[1])),
                    ]
                    for corner in corners
                ]

                # Determine grid boundaries of the rotated bounding box
                x_coords = [corner[0] for corner in clipped_corners]
                y_coords = [corner[1] for corner in clipped_corners]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Map to grid indices
                x1_idx = int((x_min - bev_x_min) * scale_x)
                x2_idx = int((x_max - bev_x_min) * scale_x)
                y1_idx = int((y_min - bev_y_min) * scale_y)
                y2_idx = int((y_max - bev_y_min) * scale_y)

                # Vectorized polygon rasterization approach for rotated bounding boxes
                grid_x, grid_y = torch.meshgrid(
                    torch.arange(x1_idx, x2_idx + 1, device=bev_feat.device),
                    torch.arange(y1_idx, y2_idx + 1, device=bev_feat.device),
                    indexing="ij",
                )
                grid_x = bev_x_min + grid_x.float() / scale_x
                grid_y = bev_y_min + grid_y.float() / scale_y

                # Compute relative positions to bbox center
                rel_x = cos_yaw * (grid_x - x_center) + sin_yaw * (grid_y - y_center)
                rel_y = -sin_yaw * (grid_x - x_center) + cos_yaw * (grid_y - y_center)

                # Check if points are within the rotated rectangle
                inside_mask = (
                    (rel_x >= -width / 2)
                    & (rel_x <= width / 2)
                    & (rel_y >= -height / 2)
                    & (rel_y <= height / 2)
                )

                # Ensure valid range for mask application
                valid_height = min(y2_idx + 1, feat_h) - max(y1_idx, 0)
                valid_width = min(x2_idx + 1, feat_w) - max(x1_idx, 0)

                foreground_mask[
                    b_idx,
                    0,
                    max(y1_idx, 0) : max(y1_idx, 0) + valid_height,
                    max(x1_idx, 0) : max(x1_idx, 0) + valid_width,
                ] = inside_mask.T[:valid_height, :valid_width]

        # Append the mask for the current scale
        gt_foreground.append(foreground_mask)

    return gt_foreground


def generate_gt_foreground_v2(multi_scale_batch_bev_feats, gt_bboxes, bev_scales):
    bs = len(gt_bboxes)  # Batch size

    # Extract BEV scales
    bev_x_min, bev_y_min, bev_x_max, bev_y_max = bev_scales
    bev_width = bev_x_max - bev_x_min
    bev_height = bev_y_max - bev_y_min

    # Initialize the output list
    gt_foreground = []

    # Loop through multi-scale BEV features
    for bev_feat in multi_scale_batch_bev_feats:
        _, _, feat_h, feat_w = bev_feat.shape
        scale_x = feat_w / bev_width
        scale_y = feat_h / bev_height

        # Initialize foreground mask for this scale
        foreground_mask = torch.zeros((bs, 1, feat_h, feat_w), device=bev_feat.device)

        for b_idx in range(bs):
            bboxes = gt_bboxes[b_idx]
            if bboxes.shape[0] == 0:
                continue

            # Extract bbox parameters
            x_centers, y_centers, widths, heights, yaws = bboxes.T

            # Precompute rotation matrices
            cos_yaws = torch.cos(yaws)
            sin_yaws = torch.sin(yaws)

            # Compute corner points for all bboxes
            corners = torch.stack(
                [
                    torch.stack(
                        [
                            x_centers
                            + cos_yaws * (widths / 2)
                            - sin_yaws * (heights / 2),
                            y_centers
                            + sin_yaws * (widths / 2)
                            + cos_yaws * (heights / 2),
                        ],
                        dim=-1,
                    ),
                    torch.stack(
                        [
                            x_centers
                            - cos_yaws * (widths / 2)
                            - sin_yaws * (heights / 2),
                            y_centers
                            - sin_yaws * (widths / 2)
                            + cos_yaws * (heights / 2),
                        ],
                        dim=-1,
                    ),
                    torch.stack(
                        [
                            x_centers
                            - cos_yaws * (widths / 2)
                            + sin_yaws * (heights / 2),
                            y_centers
                            - sin_yaws * (widths / 2)
                            - cos_yaws * (heights / 2),
                        ],
                        dim=-1,
                    ),
                    torch.stack(
                        [
                            x_centers
                            + cos_yaws * (widths / 2)
                            + sin_yaws * (heights / 2),
                            y_centers
                            + sin_yaws * (widths / 2)
                            - cos_yaws * (heights / 2),
                        ],
                        dim=-1,
                    ),
                ],
                dim=1,
            )  # Shape: (num_bboxes, 4, 2)

            # Clip corners to BEV boundaries
            corners[..., 0] = torch.clamp(corners[..., 0], bev_x_min, bev_x_max)
            corners[..., 1] = torch.clamp(corners[..., 1], bev_y_min, bev_y_max)

            # Compute grid boundaries
            x_min = torch.min(corners[..., 0], dim=1)[0]
            x_max = torch.max(corners[..., 0], dim=1)[0]
            y_min = torch.min(corners[..., 1], dim=1)[0]
            y_max = torch.max(corners[..., 1], dim=1)[0]

            x1_idx = ((x_min - bev_x_min) * scale_x).long()
            x2_idx = ((x_max - bev_x_min) * scale_x).long()
            y1_idx = ((y_min - bev_y_min) * scale_y).long()
            y2_idx = ((y_max - bev_y_min) * scale_y).long()

            # Batch process all bboxes
            for bbox_idx in range(bboxes.shape[0]):
                grid_x, grid_y = torch.meshgrid(
                    torch.arange(
                        x1_idx[bbox_idx], x2_idx[bbox_idx] + 1, device=bev_feat.device
                    ),
                    torch.arange(
                        y1_idx[bbox_idx], y2_idx[bbox_idx] + 1, device=bev_feat.device
                    ),
                    indexing="ij",
                )
                grid_x = bev_x_min + grid_x.float() / scale_x
                grid_y = bev_y_min + grid_y.float() / scale_y

                # Compute relative positions to bbox center
                rel_x = cos_yaws[bbox_idx] * (grid_x - x_centers[bbox_idx]) + sin_yaws[
                    bbox_idx
                ] * (grid_y - y_centers[bbox_idx])
                rel_y = -sin_yaws[bbox_idx] * (grid_x - x_centers[bbox_idx]) + cos_yaws[
                    bbox_idx
                ] * (grid_y - y_centers[bbox_idx])

                # Check if points are within the rotated rectangle
                inside_mask = (
                    (rel_x >= -widths[bbox_idx] / 2)
                    & (rel_x <= widths[bbox_idx] / 2)
                    & (rel_y >= -heights[bbox_idx] / 2)
                    & (rel_y <= heights[bbox_idx] / 2)
                )

                # Ensure valid range for mask application
                valid_height = min(y2_idx[bbox_idx] + 1, feat_h) - max(
                    y1_idx[bbox_idx], 0
                )
                valid_width = min(x2_idx[bbox_idx] + 1, feat_w) - max(
                    x1_idx[bbox_idx], 0
                )

                foreground_mask[
                    b_idx,
                    0,
                    max(y1_idx[bbox_idx], 0) : max(y1_idx[bbox_idx], 0) + valid_height,
                    max(x1_idx[bbox_idx], 0) : max(x1_idx[bbox_idx], 0) + valid_width,
                ] = inside_mask.T[:valid_height, :valid_width]

        # Append the mask for the current scale
        gt_foreground.append(foreground_mask)

    return gt_foreground


def generate_gt_foreground(
    multi_scale_pts_feats: Tuple[Tensor],
    gt_bboxes_list: List[Tensor],
    bev_scales: List[int],
):
    """
    multi_scale_pts_feats: Tuple of tensors, each tensor has shape (bs, h, w, c)
    gt_bboxes_list: List of tensors, each tensor has shape (num_bboxes, 5)
    bev_scales: bev_x_min, bev_y_min, bev_x_max, bev_y_max
    """
    bs = len(gt_bboxes_list)
    bev_x_min, bev_y_min, bev_x_max, bev_y_max = bev_scales
    bev_width = bev_x_max - bev_x_min
    bev_height = bev_y_max - bev_y_min
    # Initialize the output list
    gt_foreground = []
    for pts_feats in multi_scale_pts_feats:
        # TODO: h, w顺序存疑
        _, _, w, h = pts_feats.shape
        scale_x = w / bev_width
        scale_y = h / bev_height
        foreground_mask = torch.zeros(
            (bs, 1, w, h), device=pts_feats.device, dtype=torch.bool
        )
        for batch_idx in range(bs):
            gt_bboxes = gt_bboxes_list[batch_idx]
            if gt_bboxes.shape[0] == 0:
                continue
            x_centers, y_centers, widths, heights, yaws = gt_bboxes.T
            # Precompute rotation matrices
            cos_yaws = torch.cos(yaws)
            sin_yaws = torch.sin(yaws)
            # Compute corner points for all bboxes
            corners = torch.stack(
                [
                    torch.stack(
                        [
                            x_centers
                            + cos_yaws * (widths / 2)
                            - sin_yaws * (heights / 2),
                            y_centers
                            + sin_yaws * (widths / 2)
                            + cos_yaws * (heights / 2),
                        ],
                        dim=-1,
                    ),
                    torch.stack(
                        [
                            x_centers
                            - cos_yaws * (widths / 2)
                            - sin_yaws * (heights / 2),
                            y_centers
                            - sin_yaws * (widths / 2)
                            + cos_yaws * (heights / 2),
                        ],
                        dim=-1,
                    ),
                    torch.stack(
                        [
                            x_centers
                            - cos_yaws * (widths / 2)
                            + sin_yaws * (heights / 2),
                            y_centers
                            - sin_yaws * (widths / 2)
                            - cos_yaws * (heights / 2),
                        ],
                        dim=-1,
                    ),
                    torch.stack(
                        [
                            x_centers
                            + cos_yaws * (widths / 2)
                            + sin_yaws * (heights / 2),
                            y_centers
                            + sin_yaws * (widths / 2)
                            - cos_yaws * (heights / 2),
                        ],
                        dim=-1,
                    ),
                ],
                dim=1,
            )
            corners[..., 0] = torch.clamp(corners[..., 0], bev_x_min, bev_x_max)
            corners[..., 1] = torch.clamp(corners[..., 1], bev_y_min, bev_y_max)
            x_min = torch.min(corners[..., 0], dim=1)[0]
            x_max = torch.max(corners[..., 0], dim=1)[0]
            y_min = torch.min(corners[..., 1], dim=1)[0]
            y_max = torch.max(corners[..., 1], dim=1)[0]
            x1_idx = ((x_min - bev_x_min) * scale_x).long()
            x2_idx = ((x_max - bev_x_min) * scale_x).long()
            y1_idx = ((y_min - bev_y_min) * scale_y).long()
            y2_idx = ((y_max - bev_y_min) * scale_y).long()
            # Create global grid
            global_grid_x, global_grid_y = torch.meshgrid(
                torch.arange(0, w, device=pts_feats.device),
                torch.arange(0, h, device=pts_feats.device),
                indexing="xy",
            )
            global_grid_x = bev_x_min + global_grid_x.float() / scale_x  # -54~54
            global_grid_y = bev_y_min + global_grid_y.float() / scale_y
            # Precompute bbox mask for all points in the grid
            # TODO: 没看懂
            for bbox_idx in range(gt_bboxes.shape[0]):
                grid_rel_x = cos_yaws[bbox_idx] * (
                    global_grid_x - x_centers[bbox_idx]
                ) + sin_yaws[bbox_idx] * (global_grid_y - y_centers[bbox_idx])
                grid_rel_y = -sin_yaws[bbox_idx] * (
                    global_grid_x - x_centers[bbox_idx]
                ) + cos_yaws[bbox_idx] * (global_grid_y - y_centers[bbox_idx])
                bbox_mask = (
                    (grid_rel_x >= -widths[bbox_idx] / 2)
                    & (grid_rel_x <= widths[bbox_idx] / 2)
                    & (grid_rel_y >= -heights[bbox_idx] / 2)
                    & (grid_rel_y <= heights[bbox_idx] / 2)
                )
                # Combine mask into the foreground_mask
                foreground_mask[batch_idx, 0] = (
                    foreground_mask[batch_idx, 0] | bbox_mask
                )
        # Append the mask for the current scale
        gt_foreground.append(foreground_mask)
    return gt_foreground
