import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import NamedTuple, Optional, Tuple
from HAWK.RGLRU import RGLRU
from HAWK.Temporal_Conv1D import Conv1D
def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")

class RecurrentBlockCache:
    def __init__(self, rg_lru_state: torch.Tensor, conv1d_state: torch.Tensor):
        self.rg_lru_state = rg_lru_state
        self.conv1d_state = conv1d_state

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

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor, cache: Optional[RecurrentBlockCache] = None, return_cache=True):
        y = gelu(self.linear_y(x))
        x_proj = self.linear_x(x)

        if cache is not None:
            x_conv, conv1d_state = self.conv_1d(x_proj, segment_pos, cache.conv1d_state, return_cache=return_cache)
            x_lru, rg_lru_state = self.rg_lru(x_conv, segment_pos, cache.rg_lru_state, return_cache=return_cache)
        else:
            x_conv, conv1d_state = self.conv_1d(x_proj, segment_pos, None, return_cache=return_cache)
            x_lru, rg_lru_state = self.rg_lru(x_conv, segment_pos, None, return_cache=return_cache)

        out = x_lru * y
        out = self.linear_out(out)

        if return_cache:
            new_cache = RecurrentBlockCache(rg_lru_state, conv1d_state)
            return out, new_cache
        return out, None

    @classmethod
    def init_cache(cls, batch_size: int, lru_width: int, dtype, conv1d_temporal_width=4, device=None):
        return RecurrentBlockCache(
            rg_lru_state=RGLRU.init_cache(batch_size, lru_width, device=device),
            conv1d_state=Conv1D.init_cache(batch_size, lru_width, conv1d_temporal_width, device=device, dtype=dtype),
        )

if __name__ == "__main__":
    batch_size = 32
    seq_len = 16
    width = 32
    num_heads = 2

    x = torch.randn(batch_size, seq_len, width)
    segment_pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    model = RecurrentBlock(width, num_heads)
    cache = RecurrentBlock.init_cache(batch_size, model.lru_width, dtype=x.dtype, device=x.device)
    y, new_cache = model(x, segment_pos, cache)
    print(y.shape, new_cache.rg_lru_state.shape, new_cache.conv1d_state.shape)