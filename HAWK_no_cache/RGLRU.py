import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Optional

class BlockDiagonalLinear(nn.Module):
    """Block-diagonal linear transformation."""
    def __init__(self, width: int, num_blocks: int, w_init_variance_scale=1.0, device=None, dtype=None):
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.block_width = width // num_blocks
        self.w_init_variance_scale = w_init_variance_scale

        self.w = nn.Parameter(torch.empty(num_blocks, self.block_width, self.block_width, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.empty(num_blocks, self.block_width, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(self.w_init_variance_scale / self.block_width)
        nn.init.normal_(self.w, mean=0.0, std=std)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x => [batch, time, width], then group blocks
        x_blocks = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)
        y = torch.einsum("... h i, h i j -> ... h j", x_blocks, self.w) + self.b
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)

###############################################################################
#                         RG-LRU (Recurrent) Layers                           #
###############################################################################

def rnn_param_init(tensor: torch.Tensor, min_rad=0.9, max_rad=0.999, eps=1e-8) -> torch.Tensor:
    """Uniformly initializes `A` on a ring, similar to the original code."""
    with torch.no_grad():
        tensor.uniform_(min_rad**2 + eps, max_rad**2 + eps)
        tensor.log_().mul_(0.5)
        tensor.neg_().exp_().sub_(1.0).log_()
    return tensor

def closed_form_rnn_scan(
        x: torch.Tensor,  # shape [B, T, D]
        a: torch.Tensor,  # shape [B, T, D]
        h0: Optional[torch.Tensor] = None,
):
    """
    Vectorized version of the diagonal RNN: h_{t+1} = a_t * h_t + x_{t},
    with one reset at t=0 per sequence (i.e., h_0 = 0 or h0 if given).

    Returns:
      y: [B, T, D] – the hidden state at every time-step
      last_h: [B, D] – the final hidden state (y[:, -1, :])
    """
    B, T, D = x.shape

    # If you never pass in a real cache, define h0 as zeros
    if h0 is None:
        h0 = x.new_zeros(B, D)

    # 1) Prefix product p, shape [B, T+1, D], with p[:, 0] = 1
    #    p[:, t+1] = p[:, t] * a[:, t]
    p = torch.ones(B, T + 1, D, device=x.device, dtype=x.dtype)
    p[:, 1:] = torch.cumprod(a, dim=1)  # cumulative product along time

    # 2) w = x / p[:, :-1], shape [B, T, D]
    #    We'll call the chunk of p up to T as "p_prefix"

    p_prefix = p[:, :-1, :]  # shape [B, T, D]

    p_prefix = torch.clamp(p_prefix, min=1e-7)
    w = x / p_prefix # the problem is here
    # 3) S = cumsum(w, dim=1) => shape [B, T, D]
    S = torch.cumsum(w, dim=1)

    # 4) Incorporate h0.  The closed-form for h(t) is:
    #      h(t) = p[t]* S[t]  +  (p[t] * h0)    [since h0 is added to every step, scaled by product of a's]
    #    alpha = p[:, :-1] * h0[:, None, :]
    alpha = p_prefix * h0.unsqueeze(1)

    # So the full sequence of hidden states:
    h = p_prefix * S + alpha  # shape [B, T, D]

    # final hidden state is h[:, -1, :]
    last_h = h[:, -1, :]

    return h, last_h

class SqrtBoundDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clamp(4.0 * x, min=1e-6)
        return grad_output / torch.sqrt(clipped_x_times_4)

class RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit."""

    def __init__(self, width: int, num_heads: int, w_init_variance_scale=1.0, device=None, dtype=None):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.a_param = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        self.input_gate = BlockDiagonalLinear(width, num_heads, w_init_variance_scale, device=device, dtype=dtype)
        self.a_gate = BlockDiagonalLinear(width, num_heads, w_init_variance_scale, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_gate.reset_parameters()
        self.a_gate.reset_parameters()
        rnn_param_init(self.a_param)

    def forward(self, x):

        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))
        log_a = -8.0 * gate_a * F.softplus(self.a_param)
        a = torch.exp(log_a)
        a2 = torch.exp(2 * log_a)

        gated_x = x * gate_x
        multiplier = SqrtBoundDerivative.apply(
            torch.clamp(1 - a2, min=1e-7)  # <--- ensures no negative inside sqrt
        )

        normed_x = gated_x * multiplier

        y, last_h = closed_form_rnn_scan(
            normed_x,  # replaces x
            a,  # replaces a
            None
        )

        return y

class RGLRU_stable(nn.Module):
    """Real-Gated Linear Recurrent Unit with improved stability."""

    def __init__(self, width: int, num_heads: int, w_init_variance_scale=1.0, device=None, dtype=None):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.a_param = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        self.input_gate = BlockDiagonalLinear(width, num_heads, w_init_variance_scale, device=device, dtype=dtype)
        self.a_gate = BlockDiagonalLinear(width, num_heads, w_init_variance_scale, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.input_gate.w)
        nn.init.xavier_uniform_(self.a_gate.w)
        nn.init.normal_(self.a_param, mean=0.0, std=0.01)  # Small initialization for stability

    def forward(self, x):
        # Compute gates with GELU to avoid saturation
        gate_x = torch.sigmoid(self.input_gate(x))  # Retain sigmoid for gating
        gate_a = torch.sigmoid(self.a_gate(x))  # Retain sigmoid for gating

        # Compute decay rate with learnable scaling
        log_a = -gate_a * F.softplus(self.a_param)  # Remove hardcoded scaling
        a = torch.exp(log_a)
        a2 = torch.exp(2 * log_a)

        # Apply input gating
        gated_x = x * gate_x

        # Compute multiplier with smooth gradient
        multiplier = torch.sqrt(torch.clamp(1 - a2, min=1e-7))  # Use standard sqrt with clamping

        # Normalize input
        normed_x = gated_x * multiplier

        # Compute hidden states
        y, last_h = closed_form_rnn_scan(normed_x, a, None)

        return y


