import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.models.dense_heads.detr_head import DETRHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS
from plugin.dssmss.core.bbox.utils import denormalize_bbox, normalize_bbox
# from plugin.dssmss.mamba.dss import DSS
from plugin.dssmss.mamba.dss_0511 import DSS
# from plugin.dssmss.mamba.dss_0514 import DSS
from plugin.dssmss.mamba.mss import ForePredNet, MSSMamba, generate_foregt
from transformers.activations import ACT2FN
from timm.layers import DropPath

@HEADS.register_module(force=True)
class FUTR3DHead(DETRHead):
    def __init__(
        self,
        *args,
        use_dab=True,
        # ---- DSS参数 --------------------------------
        use_dss=True,
        use_hybrid=False,
        hybrid=None,
        dss_drop_prob=0.5,
        dss_mamba_version="DSSMamba_Pico",
        dss_num_layers=6,
        dss_use_morton=False,
        dss_use_conv=False,
        dss_use_xy=False,
        dss_use_rope=False,
        dss_rope_fraction=1.0,
        dss_rope_base=10000.0,
        dss_rope_max_seq_len=900,
        # ---- 弃用的参数 --------------------------------
        dss_mamba_prenorm=False,
        dss_mamba_cfg=dict(d_model=256, d_state=128, expand=1),
        dss_rope=False,
        dss_deepseek_format=False,
        use_mss=False,
        mss_num_scales=4,
        mss_fore_pred_net_in_channels=256,
        mss_fore_pred_net_intermediate_channels=16,
        mss_fore_pred_net_out_channels=1,
        mss_dstate=8,
        # ---- 其他参数 --------------------------------
        num_patterns=0,
        anchor_size=2,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        code_weights=None,
        pc_range=None,
        use_aux=False,
        aux_head=None,
        **kwargs,
    ):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        self.code_weights = code_weights
        self.code_size = len(code_weights)
        self.pc_range = pc_range
        self.use_dab = use_dab
        self.anchor_size = anchor_size
        self.num_patterns = num_patterns
        self.use_aux = use_aux
        super(FUTR3DHead, self).__init__(*args, transformer=transformer, **kwargs)
        if self.use_aux:
            train_cfg = kwargs["train_cfg"]
            aux_head.update(train_cfg=train_cfg)
            test_cfg = kwargs["test_cfg"]
            aux_head.update(test_cfg=test_cfg)
            self.aux_head = builder.build_head(aux_head)
        self.use_dss = False
        if use_dss:
            self.use_dss = use_dss
            if use_hybrid:
                for lid in hybrid:
                    attention_layer = self.transformer.decoder.layers[lid].attentions[0]
                    self.transformer.decoder.layers[lid].attentions[0] = DSS(
                        d_model=256,
                        drop_prob=dss_drop_prob,
                        mamba_version=dss_mamba_version,
                        num_layers=dss_num_layers,
                        use_morton=dss_use_morton,
                        use_conv=dss_use_conv,
                        use_xy=dss_use_xy,
                        use_rope=dss_use_rope,
                        rope_fraction=dss_rope_fraction,
                        rope_base=dss_rope_base,
                        rope_max_seq_len=dss_rope_max_seq_len
                    )
            else:
                for lid in range(len(self.transformer.decoder.layers)):
                    attention_layer = self.transformer.decoder.layers[lid].attentions[0]
                    self.transformer.decoder.layers[lid].attentions[0] = nn.ModuleList([
                        attention_layer,
                        nn.LayerNorm(256),
                        # DropPath(0.3),
                        DSS(
                            d_model=256,
                            drop_prob=dss_drop_prob,
                            mamba_version=dss_mamba_version,
                            num_layers=dss_num_layers,
                            use_morton=dss_use_morton,
                            use_conv=dss_use_conv,
                            use_xy=dss_use_xy,
                            use_rope=dss_use_rope,
                            rope_fraction=dss_rope_fraction,
                            rope_base=dss_rope_base,
                            rope_max_seq_len=dss_rope_max_seq_len
                        )
                    ])
                        
        # ---------------- 已弃用 ----------------
        self.use_mss = False
        if use_mss:
            pass
            # self.use_mss = use_mss
            # self.ForePredNet = nn.ModuleList([ForePredNet(mss_fore_pred_net_in_channels, mss_fore_pred_net_intermediate_channels, mss_fore_pred_net_out_channels) for _ in range(mss_num_scales)])
            # self.mss = MSSMamba(d_model=mss_fore_pred_net_in_channels, d_state=mss_dstate)
            # self.batch_norm = nn.ModuleList([nn.BatchNorm2d(mss_fore_pred_net_in_channels) for _ in range(mss_num_scales)])
            # # self.fore_pred_criterion = nn.BCELoss()
            # self.fore_pred_criterion = FocalLoss(alpha=0.25, gamma=2.0)

    def _init_layers(self):
        # fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        fc_cls = MLP(input_dims=self.embed_dims, intermediate_size=self.embed_dims * 4, num_classes=self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            if not self.use_dab:
                self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            else:
                self.tgt_embed = nn.Embedding(self.num_query, self.embed_dims)
                self.refpoint_embed = nn.Embedding(self.num_query, self.anchor_size)

    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            # bias_init = bias_init_with_prob(0.01)
            # for m in self.cls_branches:
            #     nn.init.constant_(m.bias, bias_init)
            bias_init = bias_init_with_prob(0.01)
            for m_head in self.cls_branches:
                final_project_layer = m_head.down_proj
                if hasattr(final_project_layer, 'bias') and final_project_layer.bias is not None:
                    nn.init.constant_(final_project_layer.bias, bias_init)
                else:
                    print(f"Warning: Final projection layer in {type(m_head)} does not have a bias or is not found as expected.")
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def get_mask_pos_enc(self, mlvl_feats):
        if mlvl_feats is not None:
            batch_size = mlvl_feats[0].size(0)
            input_img_h, input_img_w = mlvl_feats[0].shape[-2:]
            img_masks = mlvl_feats[0].new_zeros((batch_size, input_img_h, input_img_w))
            mlvl_masks = []
            mlvl_positional_encodings = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(img_masks[None], size=feat.shape[-2:])
                    .to(torch.bool)
                    .squeeze(0)
                )
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[-1])
                )
        else:
            mlvl_masks = None
            mlvl_positional_encodings = None
        return mlvl_masks, mlvl_positional_encodings

    def forward(self, mlvl_pts_feats, mlvl_img_feats, radar_feats, img_metas):
        if self.use_aux:
            aux_feat = F.interpolate(mlvl_pts_feats[1], mlvl_pts_feats[0].shape[-2:])
            aux_feat = torch.cat((mlvl_pts_feats[0], aux_feat), dim=1)
            aux_outs = self.aux_head([aux_feat])
        else:
            aux_outs = None
        mlvl_masks, mlvl_positional_encodings = self.get_mask_pos_enc(mlvl_pts_feats)
        mlvl_rad_masks, mlvl_rad_positional_encodings = self.get_mask_pos_enc(
            radar_feats
        )
        query_embeds = None
        if not self.use_dab:
            query_embeds = self.query_embedding.weight
        else:
            if self.num_patterns == 0:
                tgt_all_embed = tgt_embed = self.tgt_embed.weight
                refanchor = self.refpoint_embed.weight
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
            else:
                assert NotImplementedError
        fore_pred = None
        # ---------------- 已弃用 ----------------
        if self.use_mss:
            # fore_pred = [pred_net(feat) for pred_net, feat in zip(self.ForePredNet, mlvl_pts_feats)]
            # self.plt_heatmap(mlvl_pts_feats[0], None, "增强前0")
            # self.plt_heatmap(mlvl_pts_feats[1], None, "增强前1")
            # self.plt_heatmap(mlvl_pts_feats[2], None, "增强前2")
            # self.plt_heatmap(mlvl_pts_feats[3], None, "增强前3")
            # enhance_pts_feats = self.mss(mlvl_pts_feats, fore_pred)
            # self.plt_heatmap(enhance_pts_feats[0], None, "增强")
            # mlvl_pts_feats = [self.batch_norm[i](enhance_pts_feats[i] + feat) for i, feat in enumerate(mlvl_pts_feats)]
            # self.plt_heatmap(mlvl_pts_feats[0], None, "增强后")
            pass
        # ---------------- 已弃用 ----------------
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord = (
            self.transformer(
                mlvl_pts_feats,
                mlvl_img_feats,
                radar_feats,
                mlvl_masks,
                query_embeds,
                mlvl_positional_encodings,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                mlvl_rad_masks=mlvl_rad_masks,
                mlvl_rad_pos_embeds=mlvl_rad_positional_encodings,
            )
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            if reference.shape[-1] == 3:
                tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        # ---------------- 已弃用 ----------------
        if self.use_mss:
            return outputs_classes, outputs_coords, aux_outs, fore_pred
        # ---------------- 已弃用 ----------------
        return outputs_classes, outputs_coords, aux_outs, None

    # TODO: 之后要删除这部分
    def plt_heatmap(self, x, coors=None, name="default"):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap

        # TODO: 检查为什么要转置再左右翻转再上下翻转
        feats = x.detach()
        norm_feats = torch.norm(feats, dim=1).squeeze().cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(
            np.fliplr(norm_feats.T),
            cmap="viridis",
            vmin=norm_feats.min(),
            vmax=norm_feats.max(),
            origin="lower",
        )
        plt.colorbar()
        plt.title("Norm of Feature Map")
        if coors is not None:
            plt.plot(coors[0], coors[1], "ro", markersize=8)
        # flat_indices = np.argsort(np.fliplr(norm_feats.T).flatten())[-100:]
        # high_y, high_x = np.unravel_index(flat_indices, np.fliplr(norm_feats.T).shape)
        # plt.scatter(high_x, high_y, color='red', s=10, marker='.', alpha=0.7)
        plt.savefig(f"绘图/{name}.png")
        if coors is not None:
            target_point = feats[:, :, coors[0], coors[1]].clone()
            target_normalized = F.normalize(target_point, p=2, dim=1)
            all_features = feats.permute(0, 2, 3, 1)
            all_normalized = F.normalize(all_features, p=2, dim=3)
            similarity_map = torch.matmul(
                all_normalized, target_normalized.unsqueeze(-1)
            ).squeeze(-1)
            similarity_map_np = similarity_map[0].cpu().numpy()
            cmap = LinearSegmentedColormap.from_list(
                "similarity", ["white", "red"], N=256
            )
            plt.figure(figsize=(10, 10))
            plt.imshow(np.fliplr(similarity_map_np.T), cmap=cmap, origin="lower")
            plt.colorbar()
            plt.title(f"Similarity with Point ({coors[0]}, {coors[1]})")
            plt.plot(coors[0], coors[1], "ko", markersize=6)
            plt.savefig(f"绘图/similarity_with_point_X{coors[0]}Y{coors[1]}.png")
            plt.close()

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def loss(
        self,
        all_cls_scores,
        all_bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        fore_pred=None,
        pts_feats=None,
        gt_bboxes_ignore=None,
    ):
        assert (
            gt_bboxes_ignore is None
        ), f"{self.__class__.__name__} only supports for gt_bboxes_ignore setting to None."
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(
                device
            )
            for gt_bboxes in gt_bboxes_list
        ]
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            img_metas_list,
            all_gt_bboxes_ignore_list,
        )
        loss_dict = dict()
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            num_dec_layer += 1
        # if fore_pred is not None:
        #     dims = torch.tensor([0, 1, 3, 4, 6], device=gt_bboxes_list[0].device)
        #     foreground_gt_bboxes_list = [torch.index_select(gt_bboxes, 1, dims) for gt_bboxes in gt_bboxes_list]
        #     gt_foreground = generate_foregt(
        #         multi_scale_batch_bev_feats=pts_feats,
        #         gt_bboxes=foreground_gt_bboxes_list,
        #         bev_scales=[-54, -54, 54, 54])
        #     loss_foreground_predict_0 = self.fore_pred_criterion(fore_pred[0], gt_foreground[0].float())
        #     loss_foreground_predict_1 = self.fore_pred_criterion(fore_pred[1], gt_foreground[1].float())
        #     loss_foreground_predict_2 = self.fore_pred_criterion(fore_pred[2], gt_foreground[2].float())
        #     loss_foreground_predict_3 = self.fore_pred_criterion(fore_pred[3], gt_foreground[3].float())
        #     loss_dict[f"fore_pred_0"] = loss_foreground_predict_0
        #     loss_dict[f"fore_pred_1"] = loss_foreground_predict_1
        #     loss_dict[f"fore_pred_2"] = loss_foreground_predict_2
        #     loss_dict[f"fore_pred_3"] = loss_foreground_predict_3
        return loss_dict

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        code_weights = torch.tensor(
            self.code_weights, requires_grad=False, device=bbox_weights.device
        )
        bbox_weights = bbox_weights * code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, : self.code_size],
            normalized_bbox_targets[isnotnan, : self.code_size],
            bbox_weights[isnotnan, : self.code_size],
            avg_factor=num_total_pos,
        )
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        gt_bboxes,
        gt_labels,
        img_meta,
        gt_bboxes_ignore=None,
    ):
        num_bboxes = bbox_pred.size(0)
        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore
        )
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        bbox_targets = torch.zeros_like(bbox_pred)[..., : self.code_size - 1]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def get_bboxes(self, all_cls_scores, all_bbox_preds, img_metas, rescale=False):
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            preds = self._get_bboxes_single(cls_score, bbox_pred)
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[img_id]["box_type_3d"](bboxes, self.code_size - 1)
            scores = preds["scores"]
            labels = preds["labels"]
            result_list.append([bboxes, scores, labels])
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_preds):
        max_num = self.test_cfg["max_num"]
        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels
        if self.test_cfg["score_threshold"] > 0:
            thresh_mask = final_scores > self.test_cfg["score_threshold"]
        if self.test_cfg["post_center_range"] is not None:
            self.test_cfg["post_center_range"] = torch.tensor(
                self.test_cfg["post_center_range"], device=scores.device
            )
            mask = (
                final_box_preds[..., :3] >= self.test_cfg["post_center_range"][:3]
            ).all(1)
            mask &= (
                final_box_preds[..., :3] <= self.test_cfg["post_center_range"][3:]
            ).all(1)
            if self.test_cfg["score_threshold"] > 0:
                mask &= thresh_mask
            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {"bboxes": boxes3d, "scores": scores, "labels": labels}
        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only support post_center_range is not None for now!"
            )
        return predictions_dict


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        eps = 1e-7
        predictions = torch.clamp(predictions, min=eps, max=1 - eps)
        ce_loss = -(
            targets * torch.log(predictions)
            + (1 - targets) * torch.log(1 - predictions)
        )
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_weight * focal_weight * ce_loss
        return loss.mean()


class MLP(nn.Module):
    def __init__(self, input_dims, intermediate_size, num_classes): # 参数名调整以更清晰
        super().__init__()
        self.gate_proj = nn.Linear(input_dims, intermediate_size, bias=False)
        self.up_proj = nn.Linear(input_dims, intermediate_size, bias=False)
        self.act_fn = ACT2FN['silu']
        self.down_proj = nn.Linear(intermediate_size, num_classes, bias=True) # 输出维度改为num_classes

    def forward(self, x):
        # x 的维度是 self.embed_dims (即这里的 input_dims)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))