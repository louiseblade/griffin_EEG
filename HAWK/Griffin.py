import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Optional, Union, Dict, List


###############################################################################
#                               Config & Enums                                #
###############################################################################

class TemporalBlockType:
    RECURRENT = "RECURRENT"
    ATTENTION = "ATTENTION"

class GriffinConfig:
    """Minimal config for the Griffin model."""
    def __init__(
        self,
        width: int,
        num_layers: int,
        num_heads: int,
        block_types: List[str],
        mlp_expanded_width: int,
        attention_window_size: int,
        lru_width: Optional[int] = None,
        logits_soft_cap: Optional[float] = None,
        embeddings_scale_by_sqrt_dim: bool = True,
        # Optional: number of classes if you're doing classification on EEG:
        num_classes: Optional[int] = None,
    ):
        self.width = width
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.block_types = block_types
        self.mlp_expanded_width = mlp_expanded_width
        self.attention_window_size = attention_window_size
        self.lru_width = lru_width
        self.logits_soft_cap = logits_soft_cap
        self.embeddings_scale_by_sqrt_dim = embeddings_scale_by_sqrt_dim
        self.num_classes = num_classes  # if classification is desired

###############################################################################
#                              Auxiliary Functions                            #
###############################################################################

def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")

_MIN_LOGITS_VALUE = -1e30
_MAX_WAVELENGTH = 10_000

###############################################################################
#                                 Core Layers                                 #
###############################################################################

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

def rnn_scan(
    x: torch.Tensor,
    a: torch.Tensor,
    reset: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
):
    """Manually unrolled RNN with a diagonal recurrence."""
    seq_len = x.shape[1]
    batch_size, width = x.shape[0], x.shape[2]

    if h0 is None:
        h_t = torch.zeros(batch_size, width, dtype=x.dtype, device=x.device)
    else:
        h_t = h0

    outputs = []
    for t in range(seq_len):
        a_t = a[:, t] * (~reset[:, t]).unsqueeze(-1)
        h_t = a_t * h_t + x[:, t]
        outputs.append(h_t.unsqueeze(1))

    return torch.cat(outputs, dim=1), h_t

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

    def forward(self, x, segment_pos):
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))
        log_a = -8.0 * gate_a * F.softplus(self.a_param)
        a = torch.exp(log_a)
        a2 = torch.exp(2 * log_a)

        gated_x = x * gate_x
        multiplier = SqrtBoundDerivative.apply(1 - a2)
        reset = (segment_pos == 0)

        normed_x = gated_x * multiplier
        y, _ = rnn_scan(normed_x, a, reset, h0=None)
        return y, None

###############################################################################
#                            1D Temporal Convolution                          #
###############################################################################

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


###############################################################################
#                       Local (Windowed) Attention Layer                      #
###############################################################################

def apply_rope(inputs: torch.Tensor, positions: torch.Tensor, max_wavelength=_MAX_WAVELENGTH) -> torch.Tensor:
    """Applies rotary embeddings (RoPE)."""
    first, second = torch.chunk(inputs, 2, dim=-1)
    half1, half2 = torch.chunk(first, 2, dim=-1)

    batch, seq_len = positions.shape
    head_dim = half1.shape[-1]

    freq = torch.arange(head_dim, device=inputs.device)
    freq_exponents = 2 * freq / (2 * head_dim)
    timescale = max_wavelength ** freq_exponents
    inv_frequencies = 1.0 / timescale

    pos = positions.unsqueeze(-1)
    sinusoids = pos * inv_frequencies
    sin = torch.sin(sinusoids)
    cos = torch.cos(sinusoids)

    out1 = half1 * cos - half2 * sin
    out2 = half2 * cos + half1 * sin

    return torch.cat([out1, out2, second], dim=-1)

def compute_forward_pass_mask(batch_size, seq_len, window_size, device):
    arange = torch.arange(seq_len, device=device)
    q_pos = arange.unsqueeze(-1)
    k_pos = arange.unsqueeze(0)
    causal_mask = (q_pos >= k_pos)
    window_mask = (q_pos <= k_pos + window_size)
    mask = causal_mask & window_mask
    mask = mask[None, None, :, :]
    mask = mask.repeat(batch_size, 1, 1, 1)
    return mask

class LocalAttentionBlock(nn.Module):
    """Local MHA block."""
    def __init__(
        self,
        width: int,
        num_heads: int,
        window_size: int,
        final_w_init_variance_scale=1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.window_size = window_size
        self.final_w_init_variance_scale = final_w_init_variance_scale

        self.proj_q = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        self.proj_k = nn.Linear(width, width//num_heads, bias=False, device=device, dtype=dtype)
        self.proj_v = nn.Linear(width, width//num_heads, bias=False, device=device, dtype=dtype)
        self.proj_final = nn.Linear(width, width, bias=True, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        for w in (self.proj_q.weight, self.proj_k.weight, self.proj_v.weight):
            nn.init.normal_(w, mean=0.0, std=math.sqrt(1.0/self.width))
        std = math.sqrt(self.final_w_init_variance_scale / self.width)
        nn.init.normal_(self.proj_final.weight, mean=0.0, std=std)
        nn.init.zeros_(self.proj_final.bias)

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        queries = self.proj_q(x)
        keys = self.proj_k(x)
        values = self.proj_v(x)

        queries = einops.rearrange(queries, "b t (n h) -> b t n h", n=self.num_heads)
        keys = einops.rearrange(keys, "b t (h1 h2) -> b t 1 (h1 h2)", h1=1)
        values = einops.rearrange(values, "b t (h1 h2) -> b t 1 (h1 h2)", h1=1)

        attn_mask = compute_forward_pass_mask(b, t, self.window_size, x.device)
        logits = einops.einsum(queries, keys, "b t n h, b s n h -> b n t s")
        scale = 1.0 / math.sqrt((self.width // self.num_heads) * 1.0)
        logits = logits * scale

        masked_logits = torch.where(attn_mask, logits, torch.tensor(_MIN_LOGITS_VALUE, device=x.device))
        probs = F.softmax(masked_logits.float(), dim=-1).type_as(logits)
        out = einops.einsum(probs, values, "b n t s, b s n h -> b t n h")

        out = einops.rearrange(out, "b t n h -> b t (n h)")
        out = self.proj_final(out)
        return out


###############################################################################
#                           Recurrent or Attention Block                       #
###############################################################################

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

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor) -> torch.Tensor:
        y = gelu(self.linear_y(x))
        x_proj = self.linear_x(x)

        x_conv, _ = self.conv_1d(x_proj)
        x_lru, _ = self.rg_lru(x_conv, segment_pos)

        out = x_lru * y
        out = self.linear_out(out)
        return out

###############################################################################
#                                 MLP Block                                   #
###############################################################################

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

###############################################################################
#                              Residual Block                                 #
###############################################################################F

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

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor) -> torch.Tensor:
        raw_x = x
        x_normed = self.temporal_pre_norm(raw_x)

        if self.temporal_block_type == TemporalBlockType.RECURRENT:
            out = self.recurrent_block(x_normed, segment_pos)
        else:
            out = self.attention_block(x_normed, segment_pos)  # no tuple returned

        x_res = out + raw_x

        x_normed = self.channel_pre_norm(x_res)
        x_mlp = self.mlp_block(x_normed)
        x_out = x_mlp + x_res
        return x_out


###############################################################################
#                        Griffin for EEG (continuous data)                    #
###############################################################################

class GriffinEEG(nn.Module):
    def __init__(
            self,
            config: GriffinConfig,
            gradient_checkpointing: bool = True,
            in_channels: int = 22,
    ):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing

        # 1) Project EEG from channels -> model width
        # Make sure in_features=YOUR_CHANNEL_DIM (e.g. 22).
        self.eeg_projection = nn.Linear(
            in_features=in_channels,
            out_features=config.width,
            bias=True,
        )

        # 2) Residual blocks
        self.blocks = nn.ModuleList()
        for block_type in self.config.block_types:
            block = ResidualBlock(
                width=self.config.width,
                mlp_expanded_width=self.config.mlp_expanded_width,
                num_heads=self.config.num_heads,
                attention_window_size=self.config.attention_window_size,
                temporal_block_type=block_type,
                lru_width=self.config.lru_width,
                final_w_init_variance_scale=2.0 / self.config.num_layers
            )
            self.blocks.append(block)

        self.final_norm = RMSNorm(width=self.config.width)

        # (Optional) classification head
        if self.config.num_classes:
            self.classifier_head = nn.Linear(config.width, config.num_classes)
        else:
            self.classifier_head = None

        self.reset_parameters()
    def reset_parameters(self):
        nn.init.normal_(self.eeg_projection.weight, mean=0.0, std=math.sqrt(1.0 / self.config.width))
        if self.eeg_projection.bias is not None:
            nn.init.zeros_(self.eeg_projection.bias)

        for block in self.blocks:
            block.reset_parameters()

        self.final_norm.reset_parameters()
        if self.classifier_head is not None:
            nn.init.normal_(self.classifier_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.classifier_head.bias)

    def forward(
            self,
            x_eeg: torch.Tensor,
            segment_pos: torch.Tensor,
            return_logits: bool = True
    ):
        # 1) project from channels->width
        x = self.eeg_projection(x_eeg)

        for block in self.blocks:
            x = block(x, segment_pos)

        # 3) final norm
        x = self.final_norm(x)

        # classification head?
        if not return_logits:
            return x

        if self.classifier_head is not None:
            x_pooled = x.mean(dim=1)
            logits = self.classifier_head(x_pooled)
        else:
            logits = x
        return logits

###############################################################################
# Example usage for EEG data
###############################################################################
if __name__ == "__main__":
    # Suppose we have batch_size=2, time_steps=1125, channels=22
    # We'll do a minimal demonstration with smaller dims:
    batch_size = 32
    time_steps = 1125   # just for a short demo
    channels = 22  # pretend EEG channels
    eeg_data = torch.randn(batch_size, time_steps, channels)
    cfg = GriffinConfig(
        width=22,
        num_layers=2,
        num_heads=2,
        block_types=[TemporalBlockType.ATTENTION, TemporalBlockType.RECURRENT],
        mlp_expanded_width=16,
        attention_window_size=4,
        lru_width=None,
        logits_soft_cap=None,
        embeddings_scale_by_sqrt_dim=False,  # not used, no embedder
        num_classes=4,  # for classification e.g. 4 classes
    )

    # Build model
    model = GriffinEEG(cfg, gradient_checkpointing=False)

    # segment_pos can be all zeros if it's truly continuous / no restarts
    segment_pos = torch.zeros(batch_size, time_steps, dtype=torch.long)

    # Forward pass
    logits = model(eeg_data, segment_pos)
    print("EEG-based logits shape:", logits.shape)  # => [batch_size, num_classes] = [2, 4]
