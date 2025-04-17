import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Optional

class BlockDiagonalLinear(nn.Module):
    """Block-diagonal linear transformation with stable initialization."""

    def __init__(self, width: int, num_blocks: int,
                 w_init_variance_scale=1.0,
                 device=None, dtype=None, init_bias=0.0):
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.block_width = width // num_blocks
        self.w_init_variance_scale = w_init_variance_scale
        self.init_bias = init_bias

        # Weight tensor: (num_blocks, block_width, block_width)
        self.w = nn.Parameter(
            torch.empty(num_blocks, self.block_width, self.block_width,
                        device=device, dtype=dtype))

        # Bias tensor: (num_blocks, block_width)
        self.b = nn.Parameter(
            torch.empty(num_blocks, self.block_width,
                        device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        # 1. WEIGHTS: Orthogonal initialization per block
        for block_idx in range(self.num_blocks):
            nn.init.orthogonal_(self.w[block_idx], gain=math.sqrt(self.w_init_variance_scale))

        # 2. BIAS: Small controlled initialization
        nn.init.normal_(self.b, mean=self.init_bias, std=0.02)  # Not zero!

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [..., width]
        # Rearrange to [..., num_blocks, block_width]
        x_blocks = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

        # Einstein summation: [..., h i] @ [h i j] = [..., h j]
        y = torch.einsum("...hi,hij->...hj", x_blocks, self.w)

        # Add bias and rearrange back
        y = y + self.b.unsqueeze(0).unsqueeze(0)  # Add to penultimate dimension
        return einops.rearrange(y, "... h j -> ... (h j)")

def closed_form_rnn_scan(
        x: torch.Tensor,  # [B, T, D]
        a: torch.Tensor,  # [B, T, D]
        h0: Optional[torch.Tensor] = None,
        eps=1e-7
):
    B, T, D = x.shape
    h0 = x.new_zeros(B, D) if h0 is None else h0

    # 1. Cumulative product with safety
    a_cumprod = torch.cumprod(a, dim=1)
    p = torch.cat([torch.ones(B, 1, D, device=x.device), a_cumprod], dim=1)

    # 2. Avoid division by zero (add epsilon to numerator and denominator)
    p_prefix = p[:, :-1, :].clamp(min=eps)
    w = (x + eps) / p_prefix  # Stabilized division

    # 3. Cumulative sum with carry-over
    S = torch.cumsum(w, dim=1)
    alpha = p_prefix * h0.unsqueeze(1)
    h = p_prefix * S + alpha

    return h, h[:, -1, :]

class RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit with numerical safeguards."""

    def __init__(self, width: int, num_heads: int,
                 w_init_variance_scale=1.0,
                 min_decay=0.001,  # New: Minimum decay rate
                 eps=1e-5,  # New: Safety margin for divisions
                 device=None, dtype=None):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.min_decay = min_decay
        self.eps = eps

        # Learnable decay parameter (initialized for stability)
        self.a_param = nn.Parameter(torch.full((width,), -2.0, device=device, dtype=dtype))

        # Gating projections
        self.input_gate = BlockDiagonalLinear(width, num_heads, w_init_variance_scale, device, dtype)
        self.a_gate = BlockDiagonalLinear(width, num_heads, w_init_variance_scale, device, dtype)

        self.reset_parameters()

    def reset_parameters(self):
        # Gate weights
        nn.init.xavier_uniform_(self.input_gate.w)
        nn.init.xavier_uniform_(self.a_gate.w)

        # Bias for "forget" behavior
        nn.init.constant_(self.a_gate.b, 1.0)  # Encourage initial decay

        # init constant
        nn.init.constant_(self.a_param, -3.0)
        nn.init.constant_(self.input_gate.b, 1.0)
    def forward(self, x):
        # --- Gating Mechanisms ---

        gate_x = torch.sigmoid(self.input_gate(x))  # [B, T, D]
        gate_a = torch.sigmoid(self.a_gate(x))  # [B, T, D]

        # --- Decay Rate with Constraints ---
        log_a = -gate_a * (F.softplus(self.a_param) + self.min_decay)
        a = torch.exp(log_a).clamp(max=1.0 - self.eps)  # Prevent a=1

        # --- Stabilized Input Normalization ---
        gated_x = x * gate_x
        a_sq = a ** 2
        multiplier = torch.sqrt((1 - a_sq).clamp(min=self.eps))  # Avoid sqrt(0)
        normed_x = gated_x * multiplier

        # --- Closed-Form Scan with Safety ---
        h, _ = closed_form_rnn_scan(normed_x, a)

        return h

