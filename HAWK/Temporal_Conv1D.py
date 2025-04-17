import math
import torch
from torch import nn
from typing import Optional

class Conv1D(nn.Module):
    """Manual 1D convolution with a small 'temporal_width' kernel."""
    def __init__(self, width: int, temporal_width: int, w_init_variance_scale=0.01, device=None, dtype=None):
        super().__init__()
        self.width = width
        self.temporal_width = temporal_width
        self.w_init_variance_scale = w_init_variance_scale
        self.w = nn.Parameter(torch.empty(temporal_width, width, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(self.w_init_variance_scale / self.temporal_width)
        nn.init.normal_(self.w, mean=0.0, std=std)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor, cache: Optional[torch.Tensor] = None, return_cache: bool = True):
        batch_size, seq_len, width = x.shape
        if cache is not None:
            x_full = torch.cat([cache.to(x.dtype), x], dim=1)
            prompt_len = self.temporal_width - 1
        else:
            x_full = x
            prompt_len = 0

        out_len = seq_len
        conv_out = torch.zeros_like(x)

        for shift in range(self.temporal_width):
            start_idx = max(prompt_len - shift, 0)
            end_idx = prompt_len + out_len - shift
            if start_idx >= end_idx:
                continue

            x_window = x_full[:, start_idx:end_idx]
            # (masking across doc boundaries is omitted for brevity)

            actual_window_len = x_window.shape[1]
            pad_len = out_len - actual_window_len
            x_window_padded = torch.cat([
                torch.zeros(batch_size, pad_len, width, device=x.device, dtype=x.dtype),
                x_window
            ], dim=1)

            w_slice = self.w[self.temporal_width - shift - 1]
            conv_out += x_window_padded * w_slice[None, None, :]

        conv_out += self.b[None, None, :]

        if not return_cache:
            return conv_out, None

        new_cache = torch.cat([x_full[:, 1 - self.temporal_width:]], dim=1).to(
            cache.dtype if cache is not None else x.dtype
        )
        needed = self.temporal_width - 1 - new_cache.shape[1]
        if needed > 0:
            new_cache = torch.cat([
                torch.zeros(batch_size, needed, width, device=new_cache.device, dtype=new_cache.dtype),
                new_cache
            ], dim=1)

        return conv_out, new_cache

    @classmethod
    def init_cache(cls, batch_size: int, width: int, temporal_width: int, device=None, dtype=None):
        return torch.zeros((batch_size, temporal_width - 1, width), device=device, dtype=dtype)

if __name__ == '__main__':
    # Example usage
    batch_size = 32
    seq_len = 16
    width = 32
    temporal_width = 2
    x = torch.randn(batch_size, seq_len, width)

    segment_pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    conv1d = Conv1D(width, temporal_width)
    cache = Conv1D.init_cache(batch_size, width, temporal_width)

    out, new_cache = conv1d(x, segment_pos, cache)

