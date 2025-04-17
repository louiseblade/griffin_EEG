import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads=8, dropout=0.5):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)

        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        # if mask is not None:
        #     fill_value = torch.finfo(torch.float32).min
        #     energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

if __name__ == '__main__':
    model = MultiHeadAttention(emb_size=512, num_heads=8, dropout=0.1)
    x = torch.rand(1, 100, 512)
    out = model(x)
    print(out.shape)