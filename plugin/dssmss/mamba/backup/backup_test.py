def test_remap_vertical_to_horizontal():
    """测试remap_vertical_to_horizontal函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 测试用例1: 单尺度、单batch
    v_features1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).to(device)
    v_indices1 = [[torch.tensor([10, 20, 30]).to(device)]]  # 垂直索引
    h_indices1 = [[torch.tensor([20, 10, 30]).to(device)]]  # 水平索引(顺序不同)
    output1 = remap_vertical_to_horizontal(v_features1, v_indices1, h_indices1)
    expected1 = torch.tensor([[[3.0, 4.0], [1.0, 2.0], [5.0, 6.0]]]).to(device)
    assert torch.allclose(output1, expected1), "测试用例1失败"
    # 测试用例2: 多尺度、多batch
    v_features2 = torch.tensor([
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]]]).to(device)  # (2, 4, 2)
    # 尺度0和尺度1的索引
    v_indices2 = [
        [torch.tensor([100, 200]).to(device), torch.tensor([400, 500]).to(device)],
        [torch.tensor([300]).to(device), torch.tensor([600]).to(device)]]
    h_indices2 = [
        [torch.tensor([200, 100]).to(device), torch.tensor([500, 400]).to(device)],
        [torch.tensor([300]).to(device), torch.tensor([600]).to(device)]]
    output2 = remap_vertical_to_horizontal(v_features2, v_indices2, h_indices2)
    expected2 = torch.tensor([
        [[2.0, 2.0], [1.0, 1.0], [3.0, 3.0]],
        [[6.0, 6.0], [5.0, 5.0], [7.0, 7.0]]]).to(device)
    assert torch.allclose(output2, expected2), "测试用例2失败"
    # 测试用例3: 空索引情况
    v_features3 = torch.tensor([[[1.0, 2.0]]]).to(device)
    v_indices3 = [[torch.tensor([]).to(device)]]
    h_indices3 = [[torch.tensor([]).to(device)]]
    output3 = remap_vertical_to_horizontal(v_features3, v_indices3, h_indices3)
    expected3 = torch.zeros((1, 0, 2)).to(device)
    assert torch.allclose(output3, expected3), "测试用例3失败"
    # 测试用例4: 不同batch长度不同
    v_features4 = torch.tensor([
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]]]).to(device)
    # 尺度0和尺度1的索引
    v_indices4 = [
        [torch.tensor([100, 200]).to(device), torch.tensor([400]).to(device)],
        [torch.tensor([300]).to(device), torch.tensor([]).to(device)]]
    h_indices4 = [
        [torch.tensor([200, 100]).to(device), torch.tensor([400]).to(device)],
        [torch.tensor([300]).to(device), torch.tensor([]).to(device)]]
    output4 = remap_vertical_to_horizontal(v_features4, v_indices4, h_indices4)
    expected4 = torch.tensor([
        [[2.0, 2.0], [1.0, 1.0], [3.0, 3.0]],
        [[5.0, 5.0], [0.0, 0.0], [0.0, 0.0]]]).to(device)
    assert torch.allclose(output4, expected4), "测试用例4失败"
    print("所有测试用例通过!")

def test_MSSMamba():
    """测试MSSMamba类"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 测试用例1: 基本功能测试
    model = MSSMamba(d_model=256, device=device)
    pts_feats = [torch.randn(2, 256, 32, 32).to(device) for _ in range(3)]  # 3个尺度的特征
    forepred = [torch.rand(2, 1, 32, 32).to(device) for _ in range(3)]  # 3个尺度的前景预测
    output = model(pts_feats, forepred)
    assert len(output) == 3, "输出特征尺度数量错误"
    for feat in output:
        assert feat.shape == (2, 256, 32, 32), "输出特征形状错误"
    # 测试用例2: 空输入测试
    empty_pts = [torch.randn(2, 256, 32, 32).to(device) for _ in range(3)]
    empty_pred = [torch.zeros(2, 1, 32, 32).to(device) for _ in range(3)]
    empty_output = model(empty_pts, empty_pred)
    for feat in empty_output:
        assert torch.allclose(feat, torch.zeros_like(feat), atol=1e-6), "空输入输出应为零"
    # 测试用例3: 不同分辨率测试
    multi_res_pts = [
        torch.randn(2, 256, 64, 64).to(device),
        torch.randn(2, 256, 32, 32).to(device),
        torch.randn(2, 256, 16, 16).to(device)]
    multi_res_pred = [
        torch.rand(2, 1, 64, 64).to(device),
        torch.rand(2, 1, 32, 32).to(device),
        torch.rand(2, 1, 16, 16).to(device)]
    multi_res_output = model(multi_res_pts, multi_res_pred)
    assert len(multi_res_output) == 3, "多分辨率输出数量错误"
    assert multi_res_output[0].shape == (2, 256, 64, 64), "第一尺度输出形状错误"
    assert multi_res_output[1].shape == (2, 256, 32, 32), "第二尺度输出形状错误"
    assert multi_res_output[2].shape == (2, 256, 16, 16), "第三尺度输出形状错误"
    print("MSSMamba测试用例全部通过!")
    # 自定义测试用例
    model = MSSMamba(d_model=64, device=device)
    pts_feats = [torch.tensor([[[1.0,4.0],[2.0,3.0]]]).cuda().unsqueeze(1).repeat(1,64,1,1)]
    forepred = [torch.tensor([[[0.4,0.6],[0.6,0.6]]]).cuda().unsqueeze(1)]
    output = model(pts_feats, forepred)
    print(output)