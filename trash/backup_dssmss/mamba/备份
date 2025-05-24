    def extract_features_by_indices(self, features, morton_H_indices, morton_V_indices):
        """批量提取特征。
        
        Args:
            features: 多尺度特征列表 [tensor(B,C,H,W), ...]
            morton_H_indices: 水平morton索引列表 [[tensor, ...], ...]
            morton_V_indices: 垂直morton索引列表 [[tensor, ...], ...]
            
        Returns:
            tuple: (水平特征tensor(B,L,C), 垂直特征tensor(B,L,C))
        """
        bs, c, device = features[0].shape[0], features[0].shape[1], features[0].device
        
        # 预分配存储空间
        h_features_list = []
        v_features_list = []
        
        # 批量处理每个尺度
        for feat, h_indices_scale, v_indices_scale in zip(features, morton_H_indices, morton_V_indices):
            # 重塑特征以便索引 (B,C,H*W)
            feat_flat = feat.view(bs, c, -1)
            
            # 构建批量索引
            h_batch_indices = [indices for indices in h_indices_scale]
            v_batch_indices = [indices for indices in v_indices_scale]
            
            # 批量提取特征
            if any(idx.numel() > 0 for idx in h_batch_indices):
                h_max_len = max(idx.numel() for idx in h_batch_indices)
                h_padded_indices = torch.stack([
                    F.pad(idx, (0, h_max_len - idx.numel()), value=-1)
                    for idx in h_batch_indices
                ])
                h_mask = h_padded_indices != -1
                h_padded_indices = torch.where(h_mask, h_padded_indices, 0)
                
                # 使用高级索引一次性提取所有特征
                h_features = torch.gather(feat_flat, 2, 
                    h_padded_indices.unsqueeze(1).expand(-1, c, -1))
                h_features = torch.where(h_mask.unsqueeze(1), h_features, 
                    torch.zeros_like(h_features))
                h_features_list.append(h_features.transpose(1, 2))  # (B,L,C)
            
            if any(idx.numel() > 0 for idx in v_batch_indices):
                v_max_len = max(idx.numel() for idx in v_batch_indices)
                v_padded_indices = torch.stack([
                    F.pad(idx, (0, v_max_len - idx.numel()), value=-1)
                    for idx in v_batch_indices
                ])
                v_mask = v_padded_indices != -1
                v_padded_indices = torch.where(v_mask, v_padded_indices, 0)
                
                v_features = torch.gather(feat_flat, 2,
                    v_padded_indices.unsqueeze(1).expand(-1, c, -1))
                v_features = torch.where(v_mask.unsqueeze(1), v_features,
                    torch.zeros_like(v_features))
                v_features_list.append(v_features.transpose(1, 2))  # (B,L,C)
        
        # 合并所有尺度的特征
        h_features = torch.cat(h_features_list, dim=1) if h_features_list else \
            torch.zeros((bs, 0, c), device=device)
        v_features = torch.cat(v_features_list, dim=1) if v_features_list else \
            torch.zeros((bs, 0, c), device=device)
            
        return h_features, v_features

def remap_vertical_to_horizontal(v_features: torch.Tensor, v_indices, h_indices):
    """基于morton编码将垂直方向特征高效重映射到水平方向顺序。
    
    使用批量操作和torch.scatter_替代循环和字典操作，提高性能。
    
    Args:
        v_features: 垂直方向特征 (B,L,C)  
        v_indices: 垂直方向morton索引列表 [[tensor,...],...]
        h_indices: 水平方向morton索引列表 [[tensor,...],...]
        
    Returns:
        tensor: 重映射后的特征 (B,L,C)
    """
    b, l, c = v_features.shape
    device = v_features.device
    
    # 计算每个batch中最大的特征序列长度
    max_len = max(sum(h_idx.numel() for h_idx in h_indices_batch)
                 for h_indices_batch in zip(*h_indices))
    
    # 预分配输出张量
    output = torch.zeros((b, max_len, c), device=device)
    
    # 为每个batch处理重映射
    for batch_idx in range(b):
        curr_out_idx = 0
        
        for scale_idx in range(len(v_indices)):
            v_idx = v_indices[scale_idx][batch_idx]
            h_idx = h_indices[scale_idx][batch_idx]
            
            if v_idx.numel() > 0:
                # 获取当前尺度的特征
                start_idx = sum(indices[batch_idx].shape[0] 
                              for indices in v_indices[:scale_idx])
                curr_features = v_features[batch_idx, 
                                        start_idx:start_idx + v_idx.shape[0]]
                
                # 计算排序索引
                _, sort_idx = h_idx.sort()
                
                # 使用排序索引重排特征
                sorted_features = curr_features[sort_idx]
                
                # 写入输出张量
                out_end_idx = curr_out_idx + sorted_features.shape[0]
                output[batch_idx, curr_out_idx:out_end_idx] = sorted_features
                curr_out_idx = out_end_idx
    
    return output


def insert_features_back_batched(features: torch.Tensor, h_indices, heights, widths):
    b, _, c = features.shape
    device = features.device
    output_features = []
    for scale_idx, (scale_indices, h, w) in enumerate(zip(h_indices, heights, widths)):
        scale_output = []
        for batch_idx in range(b):
            curr_output = torch.zeros((c, h * w), device=device)
            curr_indices = scale_indices[batch_idx]
            if curr_indices.numel() > 0:
                start_idx = sum(indices[batch_idx].shape[0] for indices in h_indices[:scale_idx])
                end_idx = start_idx + curr_indices.shape[0]
                curr_features = features[batch_idx, start_idx:end_idx]
                curr_output[:, curr_indices.long()] = curr_features.t()
            curr_output = curr_output.reshape(c, h, w)
            scale_output.append(curr_output)
        output_features.append(torch.stack(scale_output))
    return output_features


def generate_foregt(multi_scale_batch_bev_feats, gt_bboxes, bev_scales):
    bs = len(gt_bboxes)
    bev_x_min, bev_y_min, bev_x_max, bev_y_max = bev_scales
    bev_width = bev_x_max - bev_x_min
    bev_height = bev_y_max - bev_y_min
    gt_foreground = []
    for bev_feat in multi_scale_batch_bev_feats:
        _, _, feat_h, feat_w = bev_feat.shape
        scale_x = feat_w / bev_width
        scale_y = feat_h / bev_height
        foreground_mask = torch.zeros((bs, 1, feat_h, feat_w), device=bev_feat.device)
        for b_idx in range(bs):
            bboxes = gt_bboxes[b_idx]
            if bboxes.shape[0] == 0:
                continue
            x_centers, y_centers, widths, heights, yaws = bboxes.T
            cos_yaws = torch.cos(yaws)
            sin_yaws = torch.sin(yaws)
            corners = torch.stack([
                torch.stack([
                    x_centers + cos_yaws * (widths / 2) - sin_yaws * (heights / 2),
                    y_centers + sin_yaws * (widths / 2) + cos_yaws * (heights / 2)], dim=-1),
                torch.stack([
                    x_centers - cos_yaws * (widths / 2) - sin_yaws * (heights / 2),
                    y_centers - sin_yaws * (widths / 2) + cos_yaws * (heights / 2)], dim=-1),
                torch.stack([
                    x_centers - cos_yaws * (widths / 2) + sin_yaws * (heights / 2),
                    y_centers - sin_yaws * (widths / 2) - cos_yaws * (heights / 2)], dim=-1,
                ),
                torch.stack([
                    x_centers + cos_yaws * (widths / 2) + sin_yaws * (heights / 2),
                    y_centers + sin_yaws * (widths / 2) - cos_yaws * (heights / 2)], dim=-1)], dim=1)   # Shape: (num_bboxes, 4, 2)
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
            for bbox_idx in range(bboxes.shape[0]):
                grid_x, grid_y = torch.meshgrid(
                    torch.arange(x1_idx[bbox_idx], x2_idx[bbox_idx] + 1, device=bev_feat.device),
                    torch.arange(y1_idx[bbox_idx], y2_idx[bbox_idx] + 1, device=bev_feat.device),
                    indexing="ij")
                grid_x = bev_x_min + grid_x.float() / scale_x
                grid_y = bev_y_min + grid_y.float() / scale_y
                rel_x = cos_yaws[bbox_idx] * (grid_x - x_centers[bbox_idx]) + sin_yaws[bbox_idx] * (grid_y - y_centers[bbox_idx])
                rel_y = -sin_yaws[bbox_idx] * (grid_x - x_centers[bbox_idx]) + cos_yaws[bbox_idx] * (grid_y - y_centers[bbox_idx])
                inside_mask = ((rel_x >= -widths[bbox_idx] / 2) & (rel_x <= widths[bbox_idx] / 2) & (rel_y >= -heights[bbox_idx] / 2) & (rel_y <= heights[bbox_idx] / 2))
                valid_height = min(y2_idx[bbox_idx] + 1, feat_h) - max(y1_idx[bbox_idx], 0)
                valid_width = min(x2_idx[bbox_idx] + 1, feat_w) - max(x1_idx[bbox_idx], 0)
                foreground_mask[b_idx, 0, max(y1_idx[bbox_idx], 0) : max(y1_idx[bbox_idx], 0) + valid_height, max(x1_idx[bbox_idx], 0) : max(x1_idx[bbox_idx], 0) + valid_width] = inside_mask.T[:valid_height, :valid_width]
        gt_foreground.append(foreground_mask)
    return gt_foreground


    def extract_features_by_indices(self, features, morton_H_indices, morton_V_indices):
        """高效批量提取特征。
        优化内存使用并减少中间步骤，直接使用torch原生操作进行批量特征提取。
        Args:
            features: 多尺度特征列表 [tensor(B,C,H,W), ...]
            morton_H_indices: 水平morton索引列表 [[tensor, ...], ...]
            morton_V_indices: 垂直morton索引列表 [[tensor, ...], ...]
        Returns:
            tuple: (水平特征tensor(B,L,C), 垂直特征tensor(B,L,C))
        """
        bs, c, device = features[0].shape[0], features[0].shape[1], features[0].device
        # 计算每个batch的总特征长度
        h_total_lens = torch.zeros(bs, dtype=torch.long, device=device)
        v_total_lens = torch.zeros(bs, dtype=torch.long, device=device)
        for h_indices_scale, v_indices_scale in zip(morton_H_indices, morton_V_indices):
            for b in range(bs):
                h_total_lens[b] += h_indices_scale[b].numel()
                v_total_lens[b] += v_indices_scale[b].numel()
        # 预分配最终输出张量
        max_h_len = h_total_lens.max().item()
        max_v_len = v_total_lens.max().item()
        h_features = torch.zeros((bs, max_h_len, c), device=device)
        v_features = torch.zeros((bs, max_v_len, c), device=device)
        # 追踪每个batch的当前写入位置
        h_curr_pos = torch.zeros(bs, dtype=torch.long, device=device)
        v_curr_pos = torch.zeros(bs, dtype=torch.long, device=device)
        # 批量处理特征
        for feat, h_indices_scale, v_indices_scale in zip(features, morton_H_indices, morton_V_indices):
            feat_flat = feat.view(bs, c, -1)  # (B,C,H*W)
            # 处理水平特征
            for b in range(bs):
                h_idx = h_indices_scale[b]
                if h_idx.numel() > 0:
                    curr_pos = h_curr_pos[b]
                    next_pos = curr_pos + h_idx.numel()
                    # 直接索引赋值，避免额外的内存分配
                    h_features[b, curr_pos:next_pos] = feat_flat[b, :, h_idx].t()
                    h_curr_pos[b] = next_pos
            # 处理垂直特征
            for b in range(bs):
                v_idx = v_indices_scale[b]
                if v_idx.numel() > 0:
                    curr_pos = v_curr_pos[b]
                    next_pos = curr_pos + v_idx.numel()
                    v_features[b, curr_pos:next_pos] = feat_flat[b, :, v_idx].t()
                    v_curr_pos[b] = next_pos
        return h_features, v_features