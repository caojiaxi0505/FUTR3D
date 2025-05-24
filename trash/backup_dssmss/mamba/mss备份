import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import ssd_selective_scan


class ForePredNet(nn.Module):
    def __init__(self, in_channels=256, intermediate_channels=16, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self._init_weights()

    def forward(self, x):
        return self.net(x)

    def _init_weights(self):
        for m in self.modules():
            # 初始化conv2d
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            # 初始化BatchNorm2d
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class MSSMamba(nn.Module):
    """MSS (Multi-Scale Selective) Mamba模块。
    实现了基于Mamba架构的多尺度选择性扫描。

    Args:
        d_model (int): 输入特征维度. 默认: 256
        d_state (int): 状态空间维度. 默认: 128
        expand (int): 特征扩展倍数. 默认: 2
        headdim (int): 注意力头维度. 默认: 64
        ngroups (int): 分组数量. 默认: 1
        A_init_range (tuple): A矩阵初始化范围. 默认: (1, 16)
        D_has_hdim (bool): D矩阵是否有头维度. 默认: False
        dt_min (float): 时间步长最小值. 默认: 0.001
        dt_max (float): 时间步长最大值. 默认: 0.1
        dt_init_floor (float): 时间步长初始化下限. 默认: 1e-4
        dt_limit (tuple): 时间步长限制范围. 默认: (0.0, inf)
        bias (bool): 是否使用偏置. 默认: False
        chunk_size (int): 分块大小. 默认: 256
        device (torch.device): 设备. 默认: None
        dtype (torch.dtype): 数据类型. 默认: None
    """
    def __init__(
        self,
        d_model=256,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        chunk_size=256,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = (self.expand * self.d_model)
        self.headdim = headdim
        self.d_ssm = self.d_inner
        self.ngroups = ngroups
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        assert self.d_ssm % self.headdim == 0
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj_H = nn.Linear(self.d_model, d_in_proj*2, bias=bias, **factory_kwargs)
        self.in_proj_V = nn.Linear(self.d_model, d_in_proj*2, bias=bias, **factory_kwargs)
        self.act = nn.SiLU()
        dt_HF = torch.clamp(torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)), min=dt_init_floor)
        dt_HB = torch.clamp(torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)), min=dt_init_floor)
        dt_VF = torch.clamp(torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)), min=dt_init_floor)
        dt_VB = torch.clamp(torch.exp(torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)), min=dt_init_floor)
        inv_dt_HF = dt_HF + torch.log(-torch.expm1(-dt_HF))
        inv_dt_HB = dt_HB + torch.log(-torch.expm1(-dt_HB))
        inv_dt_VF = dt_VF + torch.log(-torch.expm1(-dt_VF))
        inv_dt_VB = dt_VB + torch.log(-torch.expm1(-dt_VB))
        self.dt_bias_HF = nn.Parameter(inv_dt_HF)
        self.dt_bias_HB = nn.Parameter(inv_dt_HB)
        self.dt_bias_VF = nn.Parameter(inv_dt_VF)
        self.dt_bias_VB = nn.Parameter(inv_dt_VB)
        self.dt_bias_HF._no_weight_decay = True
        self.dt_bias_HB._no_weight_decay = True
        self.dt_bias_VF._no_weight_decay = True
        self.dt_bias_VB._no_weight_decay = True
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A_HF = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_HB = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_VF = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_VB = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log_HF = torch.log(A_HF).to(dtype=dtype)
        A_log_HB = torch.log(A_HB).to(dtype=dtype)
        A_log_VF = torch.log(A_VF).to(dtype=dtype)
        A_log_VB = torch.log(A_VB).to(dtype=dtype)
        self.A_log_HF = nn.Parameter(A_log_HF)
        self.A_log_HB = nn.Parameter(A_log_HB)
        self.A_log_VF = nn.Parameter(A_log_VF)
        self.A_log_VB = nn.Parameter(A_log_VB)
        self.A_log_HF._no_weight_decay = True
        self.A_log_HB._no_weight_decay = True
        self.A_log_VF._no_weight_decay = True
        self.A_log_VB._no_weight_decay = True
        self.D_HF = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_HB = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_VF = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_VB = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_HF._no_weight_decay = True
        self.D_HB._no_weight_decay = True
        self.D_VF._no_weight_decay = True
        self.D_VB._no_weight_decay = True
        self.out_proj_HF = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_HB = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_VF = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_VB = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Linear(self.d_model*4, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, pts_feats: list, forepred: list) -> list:
        """前向传播。

        Args:
            pts_feats (list[Tensor]): 点特征列表 [Tensor(B,C,H,W),...]
            forepred (list[Tensor]): 前景预测列表 [Tensor(B,1,H,W),...]

        Returns:
            list[Tensor]: 处理后的特征列表 [Tensor(B,C,H,W),...]
        
        Raises:
            ValueError: pts_feats和forepred的长度不匹配时
            RuntimeError: 特征处理出错时
        """
        # 输入验证
        if len(pts_feats) != len(forepred):
            raise ValueError("pts_feats和forepred的长度必须相同")
        morton_H_indices, morton_V_indices = self.get_indices(forepred)
        H_feats, V_feats = self.extract_features_by_indices(pts_feats, morton_H_indices, morton_V_indices)
        heights = []
        widths = []
        for i in range(len(pts_feats)):
            heights.append(pts_feats[i].shape[2])
            widths.append(pts_feats[i].shape[3])
        if H_feats.shape[1] == 0:
            out = []
            for i in range(len(pts_feats)):
                out.append(torch.zeros_like(pts_feats[i]))
            return out
        zxbcdt_HF, zxbcdt_HB = self.in_proj_H(H_feats).chunk(2, dim=2)
        zxbcdt_VF, zxbcdt_VB = self.in_proj_V(V_feats).chunk(2, dim=2)
        zxbcdt_HB = zxbcdt_HB.flip(dims=[1])
        zxbcdt_VB = zxbcdt_VB.flip(dims=[1])
        A_HF = -torch.exp(self.A_log_HF.float())
        A_HB = -torch.exp(self.A_log_HB.float())
        A_VF = -torch.exp(self.A_log_VF.float())
        A_VB = -torch.exp(self.A_log_VB.float())
        dt_limit_kwargs = ({} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit))

        dim = self.nheads * self.headdim

        z_HF, xBC_HF, dt_HF = torch.split(zxbcdt_HF, [dim, dim + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        xBC_HF = self.act(xBC_HF)
        x_HF, B_HF, C_HF = torch.split(xBC_HF, [dim, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x_HF = rearrange(x_HF, "b l (h p) -> b l h p", h=self.nheads)
        B_HF = rearrange(B_HF, "b l (g n) -> b l g n", g=self.ngroups)
        C_HF = rearrange(C_HF, "b l (g n) -> b l g n", g=self.ngroups)
        z_HF = rearrange(z_HF, "b l (h p) -> b l h p", h=self.nheads)
        out_HF = ssd_selective_scan(x_HF, dt_HF.to(x_HF.dtype), A_HF, B_HF, C_HF, D=self.D_HF.float(), z=z_HF, dt_bias=self.dt_bias_HF, dt_softplus=True, dt_limit=(0.0, float("inf")))
        out_HF = rearrange(out_HF, "b s h p -> b s (h p)")
        out_HF = self.out_proj_HB(out_HF)

        z_HB, xBC_HB, dt_HB = torch.split(zxbcdt_HB, [dim, dim + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        xBC_HB = self.act(xBC_HB)
        x_HB, B_HB, C_HB = torch.split(xBC_HB, [dim, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x_HB = rearrange(x_HB, "b l (h p) -> b l h p", h=self.nheads)
        B_HB = rearrange(B_HB, "b l (g n) -> b l g n", g=self.ngroups)
        C_HB = rearrange(C_HB, "b l (g n) -> b l g n", g=self.ngroups)
        z_HB = rearrange(z_HB, "b l (h p) -> b l h p", h=self.nheads)
        out_HB = ssd_selective_scan(x_HB, dt_HB.to(x_HB.dtype), A_HB, B_HB, C_HB, D=self.D_HB.float(), z=z_HB, dt_bias=self.dt_bias_HB, dt_softplus=True, dt_limit=(0.0, float("inf")))
        out_HB = rearrange(out_HB, "b s h p -> b s (h p)")
        out_HB = self.out_proj_HB(out_HB)
        out_HB = out_HB.flip(dims=[1])

        z_VF, xBC_VF, dt_VF = torch.split(zxbcdt_VF, [dim, dim + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        xBC_VF = self.act(xBC_VF)
        x_VF, B_VF, C_VF = torch.split(xBC_VF, [dim, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x_VF = rearrange(x_VF, "b l (h p) -> b l h p", h=self.nheads)
        B_VF = rearrange(B_VF, "b l (g n) -> b l g n", g=self.ngroups)
        C_VF = rearrange(C_VF, "b l (g n) -> b l g n", g=self.ngroups)
        z_VF = rearrange(z_VF, "b l (h p) -> b l h p", h=self.nheads)
        out_VF = ssd_selective_scan(x_VF, dt_VF.to(x_VF.dtype), A_VF, B_VF, C_VF, D=self.D_VF.float(), z=z_VF, dt_bias=self.dt_bias_VF, dt_softplus=True, dt_limit=(0.0, float("inf")))
        out_VF = rearrange(out_VF, "b s h p -> b s (h p)")
        out_VF = self.out_proj_VF(out_VF)

        z_VB, xBC_VB, dt_VB = torch.split(zxbcdt_VB, [dim, dim + 2 * self.ngroups * self.d_state, self.nheads], dim=-1)
        xBC_VB = self.act(xBC_VB)
        x_VB, B_VB, C_VB = torch.split(xBC_VB, [dim, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x_VB = rearrange(x_VB, "b l (h p) -> b l h p", h=self.nheads)
        B_VB = rearrange(B_VB, "b l (g n) -> b l g n", g=self.ngroups)
        C_VB = rearrange(C_VB, "b l (g n) -> b l g n", g=self.ngroups)
        z_VB = rearrange(z_VB, "b l (h p) -> b l h p", h=self.nheads)
        out_VB = ssd_selective_scan(x_VB, dt_VB.to(x_VB.dtype), A_VB, B_VB, C_VB, D=self.D_VB.float(), z=z_VB, dt_bias=self.dt_bias_VB, dt_softplus=True, dt_limit=(0.0, float("inf")))
        out_VB = rearrange(out_VB, "b s h p -> b s (h p)")
        out_VB = self.out_proj_VB(out_VB)
        out_VB = out_VB.flip(dims=[1])

        out = self.out_act(torch.cat([out_HF, out_HB, remap_vertical_to_horizontal(out_VF, morton_V_indices, morton_H_indices), remap_vertical_to_horizontal(out_VB, morton_V_indices, morton_H_indices)], dim=-1))
        out = self.out_proj(out)
        out = insert_features_back_batched(out, morton_H_indices, heights, widths)
        return out
    
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
        # 预分配最终输出张量 - 使用empty代替zeros以减少内存初始化开销
        max_h_len = h_total_lens.max().item()
        max_v_len = v_total_lens.max().item()
        h_features = torch.empty((bs, max_h_len, c), device=device).zero_()
        v_features = torch.empty((bs, max_v_len, c), device=device).zero_()
        # 追踪每个batch的当前写入位置 - 同样使用empty优化
        h_curr_pos = torch.empty(bs, dtype=torch.long, device=device).zero_()
        v_curr_pos = torch.empty(bs, dtype=torch.long, device=device).zero_()
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

    def get_indices(self, forepred: list) -> tuple:
        """从前景预测中提取morton编码索引。

        Args:
            forepred (list[Tensor]): 前景预测列表 [Tensor(B,1,H,W),...]

        Returns:
            tuple: (水平morton索引列表, 垂直morton索引列表)
        
        Raises:
            ValueError: 输入格式无效时
            RuntimeError: 处理失败时
        """
        if not forepred or not isinstance(forepred[0], torch.Tensor):
            raise ValueError("无效的前景预测格式")

        morton_indices_list_1 = []
        morton_indices_list_2 = []
        batch_size = forepred[0].shape[0]

        for pred_mask in forepred:  # 不同尺度
            if pred_mask.ndim != 4 or pred_mask.shape[1] != 1:
                raise ValueError(f"无效的前景预测形状: {pred_mask.shape}")
            morton_indices_batch_1 = []
            morton_indices_batch_2 = []
            for b in range(batch_size):
                mask = pred_mask[b, 0] > 0.5  # 所有大于0.5的位置都是前景点
                idx_1, idx_2 = self.morton_code_extraction(mask)
                morton_indices_batch_1.append(idx_1)
                morton_indices_batch_2.append(idx_2)
            morton_indices_list_1.append(morton_indices_batch_1)
            morton_indices_list_2.append(morton_indices_batch_2)
        return morton_indices_list_1, morton_indices_list_2

    def morton_code_extraction(self, mask: torch.Tensor) -> tuple:
        """提取Morton编码。

        Args:
            mask (Tensor): 二值掩码 (H,W)

        Returns:
            tuple: (水平morton索引, 垂直morton索引)
        """
        device = mask.device
        h, w = mask.shape
        row_indices, col_indices = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')    # row_indices[2,3]的值为2，表示该位置的行索引为2，col_indices[2,3]的值为3，表示该位置的列索引为3
        row_indices = row_indices.flatten()
        col_indices = col_indices.flatten()
        valid_indices = mask.flatten() != 0
        row_indices = row_indices[valid_indices]
        col_indices = col_indices[valid_indices]
        morton_codes_1 = self.interleave_bits(col_indices, row_indices)
        morton_codes_2 = self.interleave_bits_x_last(col_indices, row_indices)
        sorted_indices_1 = torch.argsort(morton_codes_1)
        sorted_indices_2 = torch.argsort(morton_codes_2)
        linear_indices_1 = row_indices[sorted_indices_1] * w + col_indices[sorted_indices_1]
        linear_indices_2 = row_indices[sorted_indices_2] * w + col_indices[sorted_indices_2]
        return linear_indices_1, linear_indices_2

    def interleave_bits(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """交错位运算，用于morton编码生成。

        Args:
            x (Tensor): x坐标
            y (Tensor): y坐标

        Returns:
            Tensor: 交错后的编码
        """
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        y = (y | (y << 8)) & 0x00FF00FF
        y = (y | (y << 4)) & 0x0F0F0F0F
        y = (y | (y << 2)) & 0x33333333
        y = (y | (y << 1)) & 0x55555555
        z = (x << 1) | y
        return z

    def interleave_bits_x_last(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """交错位运算的变体，x位在最后。

        Args:
            x (Tensor): x坐标
            y (Tensor): y坐标

        Returns:
            Tensor: 交错后的编码
        """
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        y = (y | (y << 8)) & 0x00FF00FF
        y = (y | (y << 4)) & 0x0F0F0F0F
        y = (y | (y << 2)) & 0x33333333
        y = (y | (y << 1)) & 0x55555555
        z = (y << 1) | x
        return z
    
def remap_vertical_to_horizontal(v_features: torch.Tensor, v_indices: list, h_indices: list) -> torch.Tensor:
    """基于morton编码将垂直方向特征高效重映射到水平方向顺序。
    使用torch.scatter_进行高效的特征重排，避免循环和临时变量。
    Args:
        v_features: 垂直方向特征 (B,L,C)  
        v_indices: 垂直方向morton索引列表 [[tensor,...],...]
        h_indices: 水平方向morton索引列表 [[tensor,...],...]
    Returns:
        tensor: 重映射后的特征 (B,L,C)
    """
    b, _, c = v_features.shape
    device = v_features.device
    # 计算每个batch的目标长度和起始位置
    batch_lengths = []
    v_start_indices = []
    h_start_indices = []
    for batch_idx in range(b):
        # 计算当前batch的目标长度
        curr_length = sum(h_indices[i][batch_idx].numel() for i in range(len(h_indices)))
        batch_lengths.append(curr_length)
        # 计算v_indices的起始位置
        v_starts = [0]
        for i in range(len(v_indices)-1):
            v_starts.append(v_starts[-1] + v_indices[i][batch_idx].numel())
        v_start_indices.append(v_starts)
        # 计算h_indices的起始位置
        h_starts = [0]
        for i in range(len(h_indices)-1):
            h_starts.append(h_starts[-1] + h_indices[i][batch_idx].numel())
        h_start_indices.append(h_starts)
    max_length = max(batch_lengths)
    output = torch.zeros((b, max_length, c), device=device)
    # 使用scatter进行高效重映射
    for batch_idx in range(b):
        if batch_lengths[batch_idx] == 0:
            continue
        # 创建完整的索引映射
        src_indices = []
        dst_indices = []
        for scale_idx in range(len(v_indices)):
            v_idx = v_indices[scale_idx][batch_idx]
            h_idx = h_indices[scale_idx][batch_idx]
            if v_idx.numel() > 0:
                # 获取源特征的起始位置
                v_start = v_start_indices[batch_idx][scale_idx]
                src_pos = torch.arange(v_idx.numel(), device=device) + v_start
                # 获取目标位置（基于h_indices的顺序）
                h_start = h_start_indices[batch_idx][scale_idx]
                _, sort_indices = h_idx.sort()
                dst_pos = torch.arange(h_idx.numel(), device=device) + h_start
                src_indices.append(src_pos)
                dst_indices.append(dst_pos[sort_indices])
        if src_indices:  # 确保有需要重映射的特征
            src_indices = torch.cat(src_indices)
            dst_indices = torch.cat(dst_indices)
            # 一次性完成特征重映射
            output[batch_idx].scatter_(
                0,
                dst_indices.unsqueeze(-1).expand(-1, c),
                v_features[batch_idx, src_indices])
    return output

def insert_features_back_batched(features: torch.Tensor, h_indices: list, heights: list, widths: list) -> list:
    """将特征批量插回原始形状。

    Args:
        features (Tensor): 特征张量 (B,L,C)
        h_indices (list): morton索引列表 [[tensor,...],...]
        heights (list[int]): 各尺度高度列表
        widths (list[int]): 各尺度宽度列表

    Returns:
        list: 恢复形状后的特征列表 [tensor(B,C,H,W),...]
    
    Raises:
        ValueError: 输入参数不匹配时
        RuntimeError: 内存分配或计算失败时
    """
    # 输入验证
    if len(heights) != len(widths):
        raise ValueError("heights和widths长度必须相同")
    if not all(isinstance(h, int) and h > 0 for h in heights):
        raise ValueError("heights必须全为正整数")
    if not all(isinstance(w, int) and w > 0 for w in widths):
        raise ValueError("widths必须全为正整数")
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

def generate_foregt(multi_scale_batch_bev_feats: list, gt_bboxes: list, bev_scales: tuple) -> list:
    """高效地生成前景掩码。
    使用批量操作和并行计算提高性能，减少循环和内存使用。
    Args:
        multi_scale_batch_bev_feats: 多尺度BEV特征列表 [tensor(B,C,H,W),...]
        gt_bboxes: GT边界框列表 [tensor(N,5),...]，每个边界框包含(x,y,w,h,yaw)
        bev_scales: BEV范围参数 (x_min, y_min, x_max, y_max)
    Returns:
        list: 多尺度前景掩码列表 [tensor(B,1,H,W),...]
    """
    bs = len(gt_bboxes)
    bev_x_min, bev_y_min, bev_x_max, bev_y_max = bev_scales
    bev_width = bev_x_max - bev_x_min
    bev_height = bev_y_max - bev_y_min
    device = multi_scale_batch_bev_feats[0].device
    # 预计算旋转变换矩阵
    def get_rotation_matrices(yaws: torch.Tensor) -> torch.Tensor:
        """计算旋转变换矩阵。

        Args:
            yaws (Tensor): 旋转角度 (N,)

        Returns:
            Tensor: 旋转矩阵 (N,2,2)
        """
        cos_yaws = torch.cos(yaws)
        sin_yaws = torch.sin(yaws)
        R = torch.stack([
            torch.stack([cos_yaws, -sin_yaws], dim=-1),
            torch.stack([sin_yaws, cos_yaws], dim=-1)], dim=-2)
        return R
    gt_foreground = []
    for bev_feat in multi_scale_batch_bev_feats:
        _, _, feat_h, feat_w = bev_feat.shape
        scale_x = feat_w / bev_width
        scale_y = feat_h / bev_height
        # 预分配结果张量
        foreground_mask = torch.zeros((bs, 1, feat_h, feat_w), device=device)
        for b_idx in range(bs):
            bboxes = gt_bboxes[b_idx]
            if bboxes.shape[0] == 0:
                continue
            centers = bboxes[:, :2]  # (N, 2)
            dims = bboxes[:, 2:4]    # (N, 2)
            yaws = bboxes[:, 4]      # (N,)
            # 批量计算旋转矩阵
            R = get_rotation_matrices(yaws)  # (N, 2, 2)
            # 批量计算边界框的四个角点
            half_dims = dims / 2
            corners_local = torch.stack([
                torch.stack([half_dims[:, 0], half_dims[:, 1]], dim=-1),
                torch.stack([-half_dims[:, 0], half_dims[:, 1]], dim=-1),
                torch.stack([-half_dims[:, 0], -half_dims[:, 1]], dim=-1),
                torch.stack([half_dims[:, 0], -half_dims[:, 1]], dim=-1)
            ], dim=1)  # (N, 4, 2)
            # 应用旋转变换
            corners_rotated = torch.matmul(corners_local, R.transpose(-1, -2))  # (N, 4, 2)
            # 平移到全局坐标
            corners = corners_rotated + centers.unsqueeze(1)  # (N, 4, 2)
            # 裁剪到BEV范围
            corners = torch.stack([
                corners[..., 0].clamp(bev_x_min, bev_x_max),
                corners[..., 1].clamp(bev_y_min, bev_y_max)
            ], dim=-1)
            # 计算边界框的边界
            box_mins = corners.min(dim=1)[0]  # (N, 2)
            box_maxs = corners.max(dim=1)[0]  # (N, 2)
            # 转换到特征图索引
            box_mins_idx = ((box_mins - torch.tensor([bev_x_min, bev_y_min], device=device)) * torch.tensor([scale_x, scale_y], device=device)).long()
            box_maxs_idx = ((box_maxs - torch.tensor([bev_x_min, bev_y_min], device=device)) * torch.tensor([scale_x, scale_y], device=device)).long()
            # 为每个框生成掩码
            for bbox_idx in range(bboxes.shape[0]):
                # 生成网格点
                x_range = torch.arange(
                    max(0, box_mins_idx[bbox_idx, 0]), 
                    min(feat_w, box_maxs_idx[bbox_idx, 0] + 1),
                    device=device)
                y_range = torch.arange(
                    max(0, box_mins_idx[bbox_idx, 1]), 
                    min(feat_h, box_maxs_idx[bbox_idx, 1] + 1),
                    device=device)
                if x_range.numel() == 0 or y_range.numel() == 0:
                    continue
                grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing="ij")
                # 转换回实际坐标
                grid_points = torch.stack([
                    bev_x_min + grid_x.float() / scale_x,
                    bev_y_min + grid_y.float() / scale_y], dim=-1)  # (H, W, 2)
                # 计算相对坐标
                rel_points = grid_points - centers[bbox_idx]  # (H, W, 2)
                # 应用逆旋转
                rel_points_rotated = torch.matmul(
                    rel_points.view(-1, 2),
                    R[bbox_idx].t()).view(grid_x.shape + (2,))  # (H, W, 2)
                # 检查点是否在框内
                inside_mask = (
                    (rel_points_rotated[..., 0].abs() <= dims[bbox_idx, 0] / 2) &
                    (rel_points_rotated[..., 1].abs() <= dims[bbox_idx, 1] / 2))
                # 更新前景掩码
                valid_mask = inside_mask.T
                foreground_mask[b_idx, 0, y_range[0]:y_range[-1]+1, x_range[0]:x_range[-1]+1] = torch.logical_or(
                    foreground_mask[b_idx, 0, y_range[0]:y_range[-1]+1, x_range[0]:x_range[-1]+1],
                    valid_mask.float())
        gt_foreground.append(foreground_mask)
    return gt_foreground

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

if __name__ == "__main__":
    pass
