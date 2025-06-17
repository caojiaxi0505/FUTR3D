import torch
import torch.nn as nn
from einops import repeat

# RoPE 标准辅助函数
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将输入张量的最后一个维度分成两半，交换它们，并将新前半部分的元素取反。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class GridRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,                # 特征维度, 必须为偶数
        pt_height: int = 224,    # 预训练时的高度
        pt_width: int = 224,     # 预训练时的宽度
        ft_height: int = None,   # 微调时的高度 (默认为 pt_height)
        ft_width: int = None,    # 微调时的宽度 (默认为 pt_width)
        custom_freqs = None,     # 可选的自定义基础频率 (形状: dim // 2)
        max_freq: float = 10.0   # 用于生成基础频率
    ):
        super().__init__()
        self.dim = dim
        if dim % 2 != 0:
            raise ValueError("特征维度 (dim) 必须为偶数以应用旋转位置编码 (RoPE)。")

        ft_h = ft_height if ft_height is not None else pt_height
        ft_w = ft_width if ft_width is not None else pt_width

        if custom_freqs is not None:
            base_freqs = custom_freqs
        else:
            base_freqs = torch.linspace(1.0, max_freq / 2, dim // 2, dtype=torch.float32) * torch.pi
        
        t_h = torch.arange(ft_h, dtype=torch.float32) / ft_h * pt_height
        t_w = torch.arange(ft_w, dtype=torch.float32) / ft_w * pt_width

        freqs_h = torch.einsum('h, f -> h f', t_h, base_freqs)
        freqs_w = torch.einsum('w, f -> w f', t_w, base_freqs)
        
        phase_angles_2d = freqs_h[:, None, :] + freqs_w[None, :, :]
        phase_angles_2d_repeated = repeat(phase_angles_2d, 'h w n -> h w (n r)', r=2)
        
        self.register_buffer("freqs_cos", phase_angles_2d_repeated.cos().unsqueeze(0), persistent=False)
        self.register_buffer("freqs_sin", phase_angles_2d_repeated.sin().unsqueeze(0), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        original_shape = t.shape
        input_was_permuted = False

        # self.freqs_cos 的形状是 (1, H_rope, W_rope, C_rope)
        # self.dim 是 C_rope
        h_rope = self.freqs_cos.shape[1]
        w_rope = self.freqs_cos.shape[2]
        
        # 检查输入是否为4D张量
        if t.ndim == 4:
            # 检查是否为 (BS, C, H, W) 格式
            # 条件: 第1维是通道维 (self.dim)，第2维是高度 (h_rope)，第3维是宽度 (w_rope)
            if original_shape[1] == self.dim and \
               original_shape[2] == h_rope and \
               original_shape[3] == w_rope:
                t = t.permute(0, 2, 3, 1)  # 转换为 (BS, H, W, C)
                input_was_permuted = True
            # 如果不是 (BS, C, H, W)，则假定它应该是 (BS, H, W, C) 或无效格式
            # 后续的形状检查会处理无效格式的情况
        
        # 此时，t 应该已经是 (BS, H, W, C) 格式
        # 进行严格的形状和维度匹配检查
        if not (t.ndim == 4 and \
                t.shape[1] == h_rope and \
                t.shape[2] == w_rope and \
                t.shape[3] == self.dim):
            
            processed_shape_str = f"(处理后形状: {t.shape})" if input_was_permuted else ""
            err_msg = (
                f"输入张量形状 {original_shape} {processed_shape_str} 与 RoPE 模块不兼容。\n"
                f"模块期望输入为 (BS, H, W, C) 格式，其中 H={h_rope}, W={w_rope}, C={self.dim} (在内部转换后)。\n"
                f"或者直接输入 (BS, C, H, W) 格式，其中 C={self.dim}, H={h_rope}, W={w_rope}。"
            )
            raise ValueError(err_msg)
            
        # 应用 RoPE
        output = t * self.freqs_cos + rotate_half(t) * self.freqs_sin

        # 如果输入被转换过，则转换回原始格式 (BS, C, H, W)
        if input_was_permuted:
            output = output.permute(0, 3, 1, 2)
            
        return output