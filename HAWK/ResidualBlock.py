import torch
import torch.nn as nn
from typing import Optional
from HAWK.RecurrentBlock import RecurrentBlock, RecurrentBlockCache
from HAWK.MQA_block import LocalAttentionBlock, AttentionBlockCache
from HAWK.MLP_block import MLPBlock
from typing import Union

ResidualBlockCache = Union[RecurrentBlockCache, AttentionBlockCache]
class TemporalBlockType:
    RECURRENT = "RECURRENT"
    ATTENTION = "ATTENTION"
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
        dropout: float = 0.0,
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
        self.dropout = nn.Dropout(dropout)
        self.temporal_pre_norm = RMSNorm(width, device=device, dtype=dtype)

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

        self.channel_pre_norm = RMSNorm(width, device=device, dtype=dtype)
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

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor = None, cache: Optional[ResidualBlockCache] = None, return_cache: bool = True):
        raw_x = x
        x_normed = self.temporal_pre_norm(raw_x)
        if segment_pos is None:
            segment_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)

        if self.temporal_block_type == TemporalBlockType.RECURRENT:
            out, new_cache = self.recurrent_block(x_normed, segment_pos, cache, return_cache=return_cache)
        else:
            out, new_cache = self.attention_block(x_normed, segment_pos, cache, return_cache=return_cache)

        # optional dropout
        out = self.dropout(out)

        x_res = out + raw_x

        x_normed = self.channel_pre_norm(x_res)
        x_mlp = self.mlp_block(x_normed)

        # optional dropout
        x_mlp = self.dropout(x_mlp)

        x_out = x_mlp + x_res

        return x_out, new_cache

    @classmethod
    def init_cache(
        cls,
        batch_size: int,
        width: int,
        num_heads: int,
        attention_window_size: int,
        temporal_block_type: str,
        dtype,
        lru_width: Optional[int] = None,
        conv1d_temporal_width: int = 4,
        device=None,
    ):
        if temporal_block_type == TemporalBlockType.RECURRENT:
            return RecurrentBlock.init_cache(batch_size, lru_width or width, dtype, conv1d_temporal_width, device=device)
        else:
            heads_dim = width // num_heads
            return LocalAttentionBlock.init_cache(batch_size, attention_window_size, heads_dim, dtype, device=device)

if __name__ == '__main__':
    batch_size = 230
    seq_len = 1125
    width = 22
    num_heads = 2
    x = torch.ones(batch_size, seq_len, width)
    block = ResidualBlock(width, 2*width, num_heads, attention_window_size=2, lru_width=64, temporal_block_type=TemporalBlockType.RECURRENT)
    out, cache = block(x)
