import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST


@MATCH_COST.register_module(force=True)
class BBox3DL1Cost(object):
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight
