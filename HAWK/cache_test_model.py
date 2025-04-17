import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Optional, Union, Dict, List

def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")

class novaNet(nn.Module):
    def __init__(self, lru_layer, mlp_mult, num_classes=4, n_windows=5, o_factor=0.99, from_logit=True):
        super(novaNet, self).__init__()
        self.conv_block = ConvBlock(F1=16, kernLength=64, poolSize=7, D=2, in_chans=22,
                                    dropout=0.3)
        self.n_windows = n_windows
        self.overlap_factor = o_factor
        self.from_logits = from_logit

        self.recurrent_list = nn.ModuleList([ResidualBlock(width=22, mlp_expanded_width=22 * mlp_mult,
                                                           num_heads=2, attention_window_size=32,
                                                           temporal_block_type=TemporalBlockType.RECURRENT,
                                                           lru_width=lru_layer,
                                                           conv1d_temporal_width=4, final_w_init_variance_scale=1.0,
                                                           dropout=0.3) for _ in range(n_windows)])

        self.slide_out_list = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        segment_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)
        x, _ = self.recurrent_list[0](x, segment_pos, None, return_cache=False)

        x = x.permute(0, 2, 1).contiguous()

        x = self.conv_block(x)[..., -1]

        # feed forward
        fc = self.slide_out_list(x)

        if self.from_logits == False:
            return F.softmax(fc, dim=-1)
        else:
            return fc

class ConvBlock(nn.Module):
    def __init__(self, F1=16, kernLength=64, poolSize=7, D=2, in_chans=22,
                 dropout=0.3):
        super(ConvBlock, self).__init__()

        F2 = F1 * D

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)

        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise convolutional layer
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (in_chans, 1), groups=F1, padding="valid", bias=False)

        self.bn2 = nn.BatchNorm2d(F1 * D)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(F2, F2, (1, 16), padding='same', bias=False)

        self.bn3 = nn.BatchNorm2d(F2)

        # Pooling and dropout layers
        self.avgpool1 = nn.AvgPool2d((1, 8))
        self.avgpool2 = nn.AvgPool2d((1, poolSize))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)

        # First block
        x = self.conv1(x)
        x = self.bn1(x)

        # Second block
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)

        x = self.avgpool1(x)
        x = self.dropout(x)

        # Third block
        x = self.conv2(x)
        x = self.bn3(x)

        x = F.elu(x)
        x = self.avgpool2(x)
        x = self.dropout(x)

        return x[:, :, -1, :]


class TemporalBlockType:
    RECURRENT = "RECURRENT"
    ATTENTION = "ATTENTION"


class RecurrentBlockCache:
    def __init__(self, rg_lru_state: torch.Tensor, conv1d_state: torch.Tensor):
        self.rg_lru_state = rg_lru_state
        self.conv1d_state = conv1d_state


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

        self.recurrent_block = RecurrentBlock(
            width=width,
            num_heads=num_heads,
            lru_width=lru_width,
            conv1d_temporal_width=conv1d_temporal_width,
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
        self.recurrent_block.reset_parameters()

        self.channel_pre_norm.reset_parameters()
        self.mlp_block.reset_parameters()

    @property
    def temporal_block(self) -> nn.Module:
        return self.recurrent_block

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor = None, cache: Optional[RecurrentBlockCache] = None,
                return_cache: bool = True):
        raw_x = x
        x_normed = self.temporal_pre_norm(raw_x)
        if segment_pos is None:
            segment_pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)

        out, new_cache = self.recurrent_block(x_normed, segment_pos, cache, return_cache=return_cache)

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
        return RecurrentBlock.init_cache(batch_size, lru_width or width, dtype, conv1d_temporal_width, device=device)


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
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        return normed * (1 + self.scale.view([1] * (x.ndim - 1) + [-1]))


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
        nn.init.normal_(self.linear_y.weight, mean=0.0, std=math.sqrt(1.0 / self.width))
        nn.init.zeros_(self.linear_y.bias)
        nn.init.normal_(self.linear_x.weight, mean=0.0, std=math.sqrt(1.0 / self.width))
        nn.init.zeros_(self.linear_x.bias)

        std = math.sqrt(self.final_w_init_variance_scale / self.lru_width)
        nn.init.normal_(self.linear_out.weight, mean=0.0, std=std)
        nn.init.zeros_(self.linear_out.bias)

        self.conv_1d.reset_parameters()
        self.rg_lru.reset_parameters()

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor, cache: Optional[RecurrentBlockCache] = None,
                return_cache=True):
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

    def forward(self, x: torch.Tensor, segment_pos: torch.Tensor, cache: Optional[torch.Tensor] = None,
                return_cache: bool = True):
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


def rnn_param_init(tensor: torch.Tensor, min_rad=0.9, max_rad=0.999, eps=1e-8) -> torch.Tensor:
    with torch.no_grad():
        tensor.uniform_(min_rad ** 2 + eps, max_rad ** 2 + eps)
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


def closed_form_rnn_scan(
        x: torch.Tensor,  # shape [B, T, D]
        a: torch.Tensor,  # shape [B, T, D]
        reset: torch.Tensor,  # shape [B, T], but only True at t=0
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

    def forward(self, x, segment_pos, cache=None, return_cache=True):
        reset = segment_pos == 0
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
            reset,  # shape [B, T], True at t=0 only
            # if you truly never want to keep a 'cache' from previous calls, pass None or zeros:
            None
        )

        if return_cache:
            return y, last_h
        return y, None
    @classmethod
    def init_cache(cls, batch_size: int, width: int, device=None):
        return torch.zeros(batch_size, width, dtype=torch.float32, device=device)


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
    import logging
    from logging.handlers import RotatingFileHandler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Create a rotating file handler
    log_file = 'training.log'
    rotating_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )

    rotating_handler.setFormatter(log_formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[rotating_handler, console_handler]
    )

    x = torch.randn(230, 22, 1125).to(device)
    model = novaNet(22, 2).to(device)

    #label
    y = torch.randint(0, 4, (230,)).to(device)

    #train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    for i in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        logging.info(f'Epoch {i} Loss: {loss.item()}')
