import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class MLPBlock(nn.Module):
    def __init__(self, width: int, expanded_width: int, final_w_init_variance_scale=1.0, device=None, dtype=None):
        super().__init__()
        self.width = width
        self.expanded_width = expanded_width
        self.final_w_init_variance_scale = final_w_init_variance_scale

        self.ffw_up = Einsum(
            w_shape=(2, self.width, self.expanded_width),
            b_shape=(2, 1, 1, self.expanded_width),
            eqn="...td,cdD->c...tD",
            device=device,
            dtype=dtype,
        )
        self.ffw_down = nn.Linear(self.expanded_width, self.width, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        self.ffw_up.reset_parameters()
        std = math.sqrt(self.final_w_init_variance_scale / self.expanded_width)
        nn.init.normal_(self.ffw_down.weight, mean=0.0, std=std)
        nn.init.zeros_(self.ffw_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ffw_up(x)
        gate_value = gelu(out[0])
        activation = gate_value * out[1]
        return self.ffw_down(activation)


class Einsum(nn.Module):
    """Parameter-multiplication using an einsum pattern."""

    def __init__(self, w_shape, b_shape, eqn, w_init_variance_scale=1.0, device=None, dtype=None):
        super().__init__()
        self.eqn = eqn
        self.w = nn.Parameter(torch.empty(w_shape, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.empty(b_shape, device=device, dtype=dtype))
        self.w_init_variance_scale = w_init_variance_scale
        self.reset_parameters()

    def reset_parameters(self):
        in_dim = self.w.shape[1]
        std = math.sqrt(self.w_init_variance_scale / in_dim)
        nn.init.normal_(self.w, mean=0.0, std=std)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum(self.eqn, x, self.w) + self.b

if __name__ == '__main__':
    # Test MLPBlock
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    mlp_block = MLPBlock(width=32, expanded_width=64, final_w_init_variance_scale=1.0, device=device, dtype=dtype)
    x = torch.randn(32, 16, 32, device=device, dtype=dtype)
    y = mlp_block(x)
    print(y.shape)

