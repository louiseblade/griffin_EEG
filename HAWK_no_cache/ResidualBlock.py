import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Optional, Union, Dict, List
from HAWK_no_cache.RecurrentBlock import RecurrentBlock
# from HAWK_no_cache.MQA_block import LocalAttentionBlock
from HAWK_no_cache.MLP_block import MLPBlock
from HAWK_no_cache.noncausal_MQA_block import LocalAttentionBlock

class TemporalBlockType:
    RECURRENT = "RECURRENT"
    ATTENTION = "ATTENTION"
def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")

class ResidualBlock(nn.Module):
    def __init__(
        self,
        width: int,
        mlp_expanded_width: int,
        num_heads: int,
        attention_window_size: int,
        temporal_block_type: str,
        lru_width: Optional[int] = None,
        conv1d_temporal_width: int = 4,
        final_w_init_variance_scale=1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.width = width
        self.mlp_expanded_width = mlp_expanded_width
        self.num_heads = num_heads
        self.attention_window_size = attention_window_size
        self.temporal_block_type = temporal_block_type
        self.lru_width = lru_width
        self.conv1d_temporal_width = conv1d_temporal_width
        self.final_w_init_variance_scale = final_w_init_variance_scale

        # self.temporal_pre_norm = RMSNorm(width, device=device, dtype=dtype)
        self.temporal_pre_norm = nn.BatchNorm1d(width)
        if temporal_block_type == TemporalBlockType.RECURRENT:
            self.recurrent_block = RecurrentBlock(
                width=width,
                num_heads=num_heads,
                lru_width=lru_width,
                conv1d_temporal_width=conv1d_temporal_width,
                final_w_init_variance_scale=final_w_init_variance_scale,
                device=device,
                dtype=dtype,
            )
        else:
            self.attention_block = LocalAttentionBlock(
                width=width,
                num_heads=num_heads,
                window_size=attention_window_size,
                final_w_init_variance_scale=final_w_init_variance_scale,
                device=device,
                dtype=dtype,
            )

        # self.channel_pre_norm = RMSNorm(width, device=device, dtype=dtype)
        self.channel_pre_norm = nn.BatchNorm1d(width)
        self.mlp_block = MLPBlock(
            width=width,
            expanded_width=mlp_expanded_width,
            final_w_init_variance_scale=final_w_init_variance_scale,
            device=device,
            dtype=dtype,
        )

    def reset_parameters(self):
        self.temporal_pre_norm.reset_parameters()
        if self.temporal_block_type == TemporalBlockType.RECURRENT:
            self.recurrent_block.reset_parameters()
        else:
            self.attention_block.reset_parameters()
        self.channel_pre_norm.reset_parameters()
        self.mlp_block.reset_parameters()

    @property
    def temporal_block(self) -> nn.Module:
        if self.temporal_block_type == TemporalBlockType.RECURRENT:
            return self.recurrent_block
        return self.attention_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_x = x
        x_normed = self.temporal_pre_norm(raw_x.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
        # x_normed = self.temporal_pre_norm(raw_x)
        if self.temporal_block_type == TemporalBlockType.RECURRENT:
            out = self.recurrent_block(x_normed)
        else:
            out = self.attention_block(x_normed)  # no tuple returned

        x_res = out + raw_x

        x_normed = self.channel_pre_norm(x_res.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
        # x_normed = self.channel_pre_norm(x_res)
        x_mlp = self.mlp_block(x_normed)
        x_out = x_mlp + x_res

        return x_out
class RMSNorm(nn.Module):
    """RMS Norm."""
    def __init__(self, width: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.width = width
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x**2, dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        return normed * (1 + self.scale.view([1]*(x.ndim-1) + [-1]))

if __name__ == '__main__':
    batch_size = 230
    seq_len = 1125
    width = 22
    num_heads = 2
    x = torch.ones(batch_size, seq_len, width)
    block = ResidualBlock(width, 2*width, num_heads, attention_window_size=2, lru_width=64, temporal_block_type=TemporalBlockType.RECURRENT)

    out = block(x)
