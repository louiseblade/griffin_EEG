import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

_MIN_LOGITS_VALUE = -1e30

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

def compute_forward_pass_mask(batch_size, seq_len, window_size, device):
    # Each position can attend to neighbors within +/- window_size steps
    arange = torch.arange(seq_len, device=device)
    q_pos = arange.unsqueeze(-1)  # [seq_len, 1]
    k_pos = arange.unsqueeze(0)   # [1, seq_len]

    diff = torch.abs(q_pos - k_pos)
    window_mask = diff <= window_size  # shape [seq_len, seq_len]

    # Expand and repeat for batch dimension
    mask = window_mask[None, None, :, :]
    mask = mask.repeat(batch_size, 1, 1, 1)
    return mask
