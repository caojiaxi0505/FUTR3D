"""
这个代码实现了2D的RoPE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VisionRoPEMixed(nn.Module):
    def __init__(self, dim, max_position=14, base=100):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        self.freq_x = nn.Parameter(torch.randn(dim // 4))
        self.freq_y = nn.Parameter(torch.randn(dim // 4))
        with torch.no_grad():
            inv_freq = 1.0 / (base ** (torch.arange(0, dim//4, 2).float() / (dim//4)))
            self.freq_x.copy_(inv_freq)
            self.freq_y.copy_(inv_freq)
        self._build_cache(max_position)
    
    def _build_cache(self, seq_len):
        pos_x = torch.arange(seq_len).float()
        pos_y = torch.arange(seq_len).float()
        freqs_x = torch.einsum('i,j->ij', pos_x, self.freq_x)
        freqs_y = torch.einsum('i,j->ij', pos_y, self.freq_y)
        freqs_mixed = freqs_x.unsqueeze(1) + freqs_y.unsqueeze(0)
        emb = torch.cat([freqs_mixed, freqs_mixed], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
    
    def apply_mixed_rope(self, x, pos_x, pos_y):
        # 输入x的形状应该为batch, heads, height, width, dim
        # 输入pos_x的形状应该为[height]
        # 输入pos_y的形状应该为[width]
        cos = self.cos_cache[pos_x][:, pos_y]  # [H, W, d/2]
        sin = self.sin_cache[pos_x][:, pos_y]  # [H, W, d/2]
        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, H, W, d/2]
        sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, H, W, d/2]
        x1 = x[..., :self.dim//2]
        x2 = x[..., self.dim//2:]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        return torch.cat([rx1, rx2], dim=-1)
    
    def forward(self, x, pos_x=None, pos_y=None):
        """
        Args:
            x: 输入的形状应该为 [batch, heads, height, width, dim]
            pos_x: x position indices [height]
            pos_y: y position indices [width]
        """
        current_max = max(
            x.size(2) if pos_x is None else len(pos_x),
            x.size(3) if pos_y is None else len(pos_y)
        )
        if current_max > self.max_position:
            self._build_cache(current_max)
            self.max_position = current_max
        if pos_x is None:
            pos_x = torch.arange(x.size(2), device=x.device)
        if pos_y is None:
            pos_y = torch.arange(x.size(3), device=x.device)
        return self.apply_mixed_rope(x, pos_x, pos_y)
