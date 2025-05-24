import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(bev_feat, gt_bboxes, mask, scale_idx=0, bev_scales=(0.0, 0.0, 32.0, 32.0)):
    """可视化BEV特征、GT边界框和生成的前景掩码
    Args:
        bev_feat: BEV特征图 (batch, channel, H, W)
        gt_bboxes: GT边界框列表 [tensor([x,y,w,h,yaw]), ...]
        mask: 生成的前景掩码 (batch, 1, H, W)
        scale_idx: 特征图尺度索引
        bev_scales: BEV坐标范围 (x_min, y_min, x_max, y_max)
    """
    plt.figure(figsize=(15, 5))
    
    # 获取特征图尺寸和BEV范围
    _, _, feat_h, feat_w = bev_feat.shape
    x_min, y_min, x_max, y_max = bev_scales
    
    # 计算坐标转换比例
    scale_x = feat_w / (x_max - x_min)
    scale_y = feat_h / (y_max - y_min)
    
    # 可视化BEV特征
    plt.subplot(1, 3, 1)
    plt.imshow(bev_feat[0, 0].cpu().numpy(), cmap='gray', 
              extent=[x_min, x_max, y_max, y_min])
    plt.title(f'BEV Feature (Scale {scale_idx})')
    plt.colorbar()
    
    # 可视化GT边界框
    plt.subplot(1, 3, 2)
    plt.gca().set_aspect('equal')
    for bbox in gt_bboxes[0]:
        if bbox.shape[0] == 0:
            continue
            
        x, y, w, h, yaw = bbox.cpu().numpy()
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # 计算边界框角点(实际坐标)
        corners = np.array([
            [x + cos_yaw*w/2 - sin_yaw*h/2, y + sin_yaw*w/2 + cos_yaw*h/2],
            [x - cos_yaw*w/2 - sin_yaw*h/2, y - sin_yaw*w/2 + cos_yaw*h/2],
            [x - cos_yaw*w/2 + sin_yaw*h/2, y - sin_yaw*w/2 - cos_yaw*h/2],
            [x + cos_yaw*w/2 + sin_yaw*h/2, y + sin_yaw*w/2 - cos_yaw*h/2]
        ])
        plt.plot(corners[[0,1,2,3,0], 0], corners[[0,1,2,3,0], 1], 
                'r-', linewidth=2)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('GT Bounding Boxes')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    # 可视化前景掩码
    plt.subplot(1, 3, 3)
    plt.imshow(mask[0, 0].cpu().numpy(), cmap='jet', origin='lower')
    plt.title('Generated Foreground Mask')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
