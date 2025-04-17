import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import math

_MAX_SQRT_GRADIENT = 1000.0

class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim, dim * mult)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * mult, dim)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = F.gelu(x1)
        x1 = self.dropout(x1)
        x2 = self.linear1(x)
        x = x1 * x2
        x = self.linear2(x)
        return x
class BlockDiagonalLinear(nn.Module):
    def __init__(self, width, num_blocks=None, w_init_variance_scale=1.0):
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks if num_blocks is not None else width  # Set num_blocks to width if None
        self.block_width = self.width // self.num_blocks

        # Recalculate num_blocks if sequence length isn't divisible by num_blocks
        if self.width % self.num_blocks != 0:
            self.num_blocks = math.gcd(self.width, self.num_blocks)
            self.block_width = self.width // self.num_blocks

        self.w = nn.Parameter(torch.empty([self.num_blocks, self.block_width, self.block_width]))
        self.b = nn.Parameter(torch.empty([self.num_blocks, self.block_width]))
        self.w_init_variance_scale = w_init_variance_scale
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(self.w_init_variance_scale / self.block_width)
        nn.init.normal_(self.w, mean=0.0, std=std)
        nn.init.zeros_(self.b)

    def forward(self, x):

        batch_size, seq_len, _ = x.shape
        # Adjust h to be the greatest common divisor of sequence length and num_blocks if there's a mismatch

        if seq_len % self.num_blocks != 0:
            self.num_blocks = math.gcd(seq_len, self.num_blocks)

        x = rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)  # Adjusted to ensure compatibility
        y = torch.einsum("... h i, h i j -> ... h j", x, self.w) + self.b
        return rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)

class SqrtBoundDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT ** 2))
        return grad_output / torch.sqrt(clipped_x_times_4)

class RGLRU(nn.Module):
    def __init__(self, width, num_heads, w_init_variance_scale=1.0):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.a_param = nn.Parameter(torch.empty([self.width]))
        self.input_gate = BlockDiagonalLinear(width=self.width, num_blocks=self.num_heads,
                                              w_init_variance_scale=w_init_variance_scale)
        self.a_gate = BlockDiagonalLinear(width=self.width, num_blocks=self.num_heads,
                                          w_init_variance_scale=w_init_variance_scale)
        self.reset_parameters()

    def reset_parameters(self):

        self.input_gate.reset_parameters()
        self.a_gate.reset_parameters()
        self.a_param_init(self.a_param)

    def a_param_init(self, w):
        with torch.no_grad():
            w.uniform_(0.9, 0.999)
            # w.uniform_(0.9 ** 2 + 1e-8, 0.999 ** 2 + 1e-8)
            # w.log_().mul_(0.5)
            # w.neg_().exp_().sub_(1.0).log_()

    def forward(self, x, prev_h=None):

        batch_size, sequence_length, _ = x.shape
        if prev_h is None:
            prev_h = torch.zeros(batch_size, self.width, dtype=torch.float32, device=x.device)

        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))

        log_a = -8.0 * gate_a * F.softplus(self.a_param.clamp(min=-10.0, max=10.0))

        a = torch.exp(log_a).clamp(max=1.0)

        a_square = torch.exp(2 * log_a).clamp(max=1.0)

        gated_x = x * gate_x
        multiplier = SqrtBoundDerivative.apply(1 - a_square)
        multiplier = torch.clamp(multiplier, min=0.0)
        normalized_x = gated_x * multiplier

        y = torch.zeros_like(x)
        list_hidden = []

        for t in range(sequence_length):
            prev_h = a[:, t] * prev_h + normalized_x[:, t]
            list_hidden.append(prev_h)
            y[:, t] = prev_h

        print(len(list_hidden))
        return y

if __name__ == '__main__':
    x = torch.randn(1, 1125, 22)  # (batch_size, sequence_length, input_size)

    model = RGLRU(width=22, num_heads=1125)
    output = model(x)


