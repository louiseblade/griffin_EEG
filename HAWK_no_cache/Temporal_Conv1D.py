import math
import torch
import torch.nn as nn

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

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, width = x.shape

        conv_out = torch.zeros_like(x)
        # Slide over x itself
        for shift in range(self.temporal_width):
            start_idx = 0
            end_idx = seq_len - shift
            if end_idx <= 0:
                continue
            x_window = x[:, start_idx:end_idx]  # no appended old states

            pad_len = seq_len - x_window.shape[1]
            x_window_padded = torch.cat([
                torch.zeros(batch_size, pad_len, width, device=x.device, dtype=x.dtype),
                x_window
            ], dim=1)

            w_slice = self.w[self.temporal_width - shift - 1]
            conv_out += x_window_padded * w_slice[None, None, :]

        conv_out += self.b[None, None, :]

        # Always return just (output, None)
        return conv_out, None
