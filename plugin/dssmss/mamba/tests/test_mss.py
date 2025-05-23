import torch
import pytest
from ..mss import remap_vertical_to_horizontal, MSSMamba, generate_foregt

class TestMSS:
    """MSS模块测试类"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_remap_vertical_to_horizontal_single_scale(self, device):
        """测试单尺度、单batch的特征重映射"""
        v_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).to(device)
        v_indices = [[torch.tensor([10, 20, 30]).to(device)]]
        h_indices = [[torch.tensor([20, 10, 30]).to(device)]]
        
        output = remap_vertical_to_horizontal(v_features, v_indices, h_indices)
        expected = torch.tensor([[[3.0, 4.0], [1.0, 2.0], [5.0, 6.0]]]).to(device)
        
        assert torch.allclose(output, expected), "单尺度特征重映射失败"
    
    def test_remap_vertical_to_horizontal_multi_scale(self, device):
        """测试多尺度、多batch的特征重映射"""
        v_features = torch.tensor([
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]
        ]).to(device)
        
        v_indices = [
            [torch.tensor([100, 200]).to(device), torch.tensor([400, 500]).to(device)],
            [torch.tensor([300]).to(device), torch.tensor([600]).to(device)]
        ]
        h_indices = [
            [torch.tensor([200, 100]).to(device), torch.tensor([500, 400]).to(device)],
            [torch.tensor([300]).to(device), torch.tensor([600]).to(device)]
        ]
        
        output = remap_vertical_to_horizontal(v_features, v_indices, h_indices)
        expected = torch.tensor([
            [[2.0, 2.0], [1.0, 1.0], [3.0, 3.0]],
            [[5.0, 5.0], [4.0, 4.0], [6.0, 6.0]]
        ]).to(device)
        
        assert torch.allclose(output, expected), "多尺度特征重映射失败"
    
    def test_remap_vertical_to_horizontal_empty(self, device):
        """测试空输入情况"""
        v_features = torch.tensor([[[1.0, 2.0]]]).to(device)
        v_indices = [[torch.tensor([]).to(device)]]
        h_indices = [[torch.tensor([]).to(device)]]
        
        output = remap_vertical_to_horizontal(v_features, v_indices, h_indices)
        expected = torch.zeros((1, 0, 2)).to(device)
        
        assert torch.allclose(output, expected), "空输入处理失败"
    
    def test_mssmamba_basic(self, device):
        """测试MSSMamba的基本功能"""
        model = MSSMamba(d_model=256, device=device)
        pts_feats = [torch.randn(2, 256, 32, 32).to(device) for _ in range(3)]
        forepred = [torch.rand(2, 1, 32, 32).to(device) for _ in range(3)]
        
        output = model(pts_feats, forepred)
        
        assert len(output) == 3, "输出特征数量错误"
        for feat in output:
            assert feat.shape == (2, 256, 32, 32), "输出特征形状错误"
    
    def test_mssmamba_empty_input(self, device):
        """测试MSSMamba的空输入处理"""
        model = MSSMamba(d_model=256, device=device)
        empty_pts = [torch.randn(2, 256, 32, 32).to(device) for _ in range(3)]
        empty_pred = [torch.zeros(2, 1, 32, 32).to(device) for _ in range(3)]
        
        output = model(empty_pts, empty_pred)
        
        for feat in output:
            assert torch.allclose(feat, torch.zeros_like(feat), atol=1e-6)
    
    def test_mssmamba_multi_resolution(self, device):
        """测试MSSMamba处理多分辨率输入"""
        model = MSSMamba(d_model=256, device=device)
        pts_feats = [
            torch.randn(2, 256, 64, 64).to(device),
            torch.randn(2, 256, 32, 32).to(device),
            torch.randn(2, 256, 16, 16).to(device)
        ]
        forepred = [
            torch.rand(2, 1, 64, 64).to(device),
            torch.rand(2, 1, 32, 32).to(device),
            torch.rand(2, 1, 16, 16).to(device)
        ]
        
        output = model(pts_feats, forepred)
        
        assert len(output) == 3, "多分辨率输出数量错误"
        assert output[0].shape == (2, 256, 64, 64), "第一尺度输出形状错误"
        assert output[1].shape == (2, 256, 32, 32), "第二尺度输出形状错误"
        assert output[2].shape == (2, 256, 16, 16), "第三尺度输出形状错误"

    @pytest.mark.parametrize("visualize", [False])
    def test_generate_foregt_basic(self, device, visualize):
        """测试基本前景生成"""
        bev_feats = [torch.randn(2, 256, 32, 32).to(device) for _ in range(3)]
        gt_bboxes = [
            torch.tensor([[16.0, 16.0, 4.0, 2.0, 0.0]]).to(device),
            torch.tensor([]).to(device)
        ]
        bev_scales = [-54, -54, 54, 54]
        
        masks = generate_foregt(bev_feats, gt_bboxes, bev_scales)
        
        assert len(masks) == 3, "输出掩码数量错误"
        for mask in masks:
            assert mask.shape == (2, 1, 32, 32), "掩码形状错误"
            assert mask[0].sum() > 0, "第一个batch应有前景"
            assert mask[1].sum() == 0, "第二个batch应全为背景"

    def test_generate_foregt_rotated(self, device):
        """测试旋转框的前景生成"""
        bev_feats = [torch.randn(1, 256, 64, 64).to(device)]
        gt_bboxes = [
            torch.tensor([[32.0, 32.0, 8.0, 4.0, 0.785]]).to(device)  # 45度
        ]
        bev_scales = [-54, -54, 54, 54]
        
        masks = generate_foregt(bev_feats, gt_bboxes, bev_scales)
        
        assert masks[0].shape == (1, 1, 64, 64)
        assert masks[0][0, 0, 30:34, 30:34].sum() > 0, "旋转框中心区域应有前景"
        assert masks[0][0, 0, 28, 36] == 0, "旋转框角落应为背景"
        assert masks[0][0, 0, 36, 28] == 0, "旋转框角落应为背景"

    def test_generate_foregt_multi_scale(self, device):
        """测试多尺度前景生成"""
        bev_feats = [
            torch.randn(1, 256, 64, 64).to(device),
            torch.randn(1, 256, 32, 32).to(device),
            torch.randn(1, 256, 16, 16).to(device)
        ]
        gt_bboxes = [
            torch.tensor([[16.0, 16.0, 4.0, 4.0, 0.0]]).to(device)
        ]
        bev_scales = [-54, -54, 54, 54]
        
        masks = generate_foregt(bev_feats, gt_bboxes, bev_scales)
        
        assert len(masks) == 3, "输出掩码数量错误"
        assert masks[0].shape == (1, 1, 64, 64), "第一尺度形状错误"
        assert masks[1].shape == (1, 1, 32, 32), "第二尺度形状错误"
        assert masks[2].shape == (1, 1, 16, 16), "第三尺度形状错误"

if __name__ == "__main__":
    pytest.main([__file__])
