import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from HAWK_no_cache.RG_LRU_stable import RGLRU
from HAWK_no_cache.Temporal_Conv1D import Conv1D

def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")

class ChannelAttention1D(nn.Module):
    def __init__(self, num_channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction),
            nn.ReLU(),
            nn.Linear(num_channels // reduction, num_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, T, C]
        x = x.permute(0, 2, 1)  # [B, C, T]
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)  # [B, C]
        y = self.fc(y).view(b, c, 1)  # [B, C, 1]
        return (x * y).permute(0, 2, 1)  # Back to [B, T, C]


class CBAM1D(nn.Module):
    def __init__(self, num_channels, reduction=4, kernel_size=7):
        super().__init__()
        # Channel attention
        self.channel_att = ChannelAttention1D(num_channels, reduction)

        # Temporal attention
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // reduction, kernel_size,
                      padding=kernel_size // 2, groups=1),
            nn.ReLU(),
            nn.Conv1d(num_channels // reduction, num_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, T, C]
        # Channel attention
        x = self.channel_att(x)  # [B, T, C]

        # Temporal attention
        x_temp = x.permute(0, 2, 1)  # [B, C, T]
        att = self.temporal_conv(x_temp)  # [B, C, T]
        return (x_temp * att).permute(0, 2, 1)  # [B, T, C]
class RecurrentBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        lru_width: Optional[int] = None,
        conv1d_temporal_width: int = 4,
        final_w_init_variance_scale=1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.lru_width = lru_width or width
        self.conv1d_temporal_width = conv1d_temporal_width
        self.final_w_init_variance_scale = final_w_init_variance_scale

        self.linear_y = nn.Linear(self.width, self.lru_width, device=device, dtype=dtype)
        self.linear_x = nn.Linear(self.width, self.lru_width, device=device, dtype=dtype)
        self.linear_out = nn.Linear(self.lru_width, self.width, device=device, dtype=dtype)

        self.conv_1d = Conv1D(self.lru_width, self.conv1d_temporal_width, device=device, dtype=dtype)
        self.rg_lru = RGLRU(self.lru_width, self.num_heads, device=device, dtype=dtype)
        # self.se = ChannelAttention1D(num_channels=22)  # Match input channels
        # self.cbam = CBAM1D(num_channels=22)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_y.weight, mean=0.0, std=math.sqrt(1.0/self.width))
        nn.init.zeros_(self.linear_y.bias)
        nn.init.normal_(self.linear_x.weight, mean=0.0, std=math.sqrt(1.0/self.width))
        nn.init.zeros_(self.linear_x.bias)

        std = math.sqrt(self.final_w_init_variance_scale / self.lru_width)
        nn.init.normal_(self.linear_out.weight, mean=0.0, std=std)
        nn.init.zeros_(self.linear_out.bias)

        self.conv_1d.reset_parameters()
        self.rg_lru.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.cbam(x)  # Apply CBAM first
        y = gelu(self.linear_y(x))
        x_proj = self.linear_x(x)

        # Add SE to original input
        # x_proj  = self.se(x_proj )  # [B, 1125, 22] -> [B, 1125, 22]

        x_conv, _ = self.conv_1d(x_proj)
        x_lru = self.rg_lru(x_conv)

        out = x_lru * y
        out = self.linear_out(out)
        return out