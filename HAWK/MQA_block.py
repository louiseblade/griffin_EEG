import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Optional

_MIN_LOGITS_VALUE = -1e30
_MAX_WAVELENGTH = 10_000


class AttentionBlockCache:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor, num_tokens: torch.Tensor):
        self.keys = keys
        self.values = values
        self.num_tokens = num_tokens


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
        self.proj_k = nn.Linear(width, width // num_heads, bias=False, device=device, dtype=dtype)
        self.proj_v = nn.Linear(width, width // num_heads, bias=False, device=device, dtype=dtype)
        self.proj_final = nn.Linear(width, width, bias=True, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        for w in (self.proj_q.weight, self.proj_k.weight, self.proj_v.weight):
            nn.init.normal_(w, mean=0.0, std=math.sqrt(1.0 / self.width))
        std = math.sqrt(self.final_w_init_variance_scale / self.width)
        nn.init.normal_(self.proj_final.weight, mean=0.0, std=std)
        nn.init.zeros_(self.proj_final.bias)

    def forward(
            self,
            x: torch.Tensor,
            segment_pos: torch.Tensor,
            cache: Optional[AttentionBlockCache] = None,
            return_cache: bool = True,
    ):
        b, t, _ = x.shape
        queries = self.proj_q(x)
        keys = self.proj_k(x)
        values = self.proj_v(x)

        queries = einops.rearrange(queries, "b t (n h) -> b t n h", n=self.num_heads)
        keys = einops.rearrange(keys, "b t (h1 h2) -> b t 1 (h1 h2)", h1=1)
        values = einops.rearrange(values, "b t (h1 h2) -> b t 1 (h1 h2)", h1=1)

        # queries = apply_rope(queries, segment_pos)
        # keys = apply_rope(keys, segment_pos)

        if cache is not None:
            all_keys = torch.cat([cache.keys, keys], dim=1)
            all_values = torch.cat([cache.values, values], dim=1)
            s = all_keys.shape[1]
            attn_mask = torch.ones(b, 1, t, s, dtype=torch.bool, device=x.device)

            if return_cache:
                if s > self.window_size:
                    new_keys = all_keys[:, -self.window_size:]
                    new_values = all_values[:, -self.window_size:]
                    new_num_tokens = cache.num_tokens + 1
                else:
                    new_keys = all_keys
                    new_values = all_values
                    new_num_tokens = cache.num_tokens + 1
                new_cache = AttentionBlockCache(new_keys, new_values, new_num_tokens)
            else:
                new_cache = None

            logits = einops.einsum(queries, all_keys, "b t n h, b s n h -> b n t s")
            scale = 1.0 / math.sqrt((self.width // self.num_heads) * 1.0)
            logits = logits * scale

            masked_logits = torch.where(attn_mask, logits, torch.tensor(_MIN_LOGITS_VALUE, device=x.device))
            probs = F.softmax(masked_logits.float(), dim=-1).type_as(masked_logits)
            out = einops.einsum(probs, all_values, "b n t s, b s n h -> b t n h")

        else:
            attn_mask = compute_forward_pass_mask(b, t, self.window_size, x.device)
            logits = einops.einsum(queries, keys, "b t n h, b s n h -> b n t s")
            scale = 1.0 / math.sqrt((self.width // self.num_heads) * 1.0)
            logits = logits * scale

            masked_logits = torch.where(attn_mask, logits, torch.tensor(_MIN_LOGITS_VALUE, device=x.device))
            probs = F.softmax(masked_logits.float(), dim=-1).type_as(logits)
            out = einops.einsum(probs, values, "b n t s, b s n h -> b t n h")

            if return_cache:
                s = min(self.window_size, t)
                new_keys = keys[:, -s:]
                new_values = values[:, -s:]
                new_num_tokens = torch.tensor([t] * b, dtype=torch.int32, device=x.device)
                new_cache = AttentionBlockCache(new_keys, new_values, new_num_tokens)
            else:
                new_cache = None

        out = einops.rearrange(out, "b t n h -> b t (n h)")
        out = self.proj_final(out)
        return out, new_cache

    @classmethod
    def init_cache(cls, batch_size: int, window_size: int, heads_dim: int, dtype, device=None):
        shape = (batch_size, window_size, 1, heads_dim)
        keys = torch.zeros(shape, device=device, dtype=dtype)
        values = torch.zeros(shape, device=device, dtype=dtype)
        num_tokens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        return AttentionBlockCache(keys, values, num_tokens)


if __name__ == '__main__':
    batch_size = 32
    seq_len = 16
    width = 32
    num_heads = 16
    window_size = 2

    x = torch.ones(batch_size, seq_len, width)
    segment_pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    block = LocalAttentionBlock(width, num_heads, window_size)
    out, cache = block(x, segment_pos)
    print(out.shape, cache.keys.shape, cache.values.shape, cache.num_tokens.shape)

