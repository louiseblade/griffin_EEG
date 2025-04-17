import torch
import torch.nn as nn
import torch.nn.functional as F

from Griffin_Module.RGLRU import RGLRU
from einops.layers.torch import Rearrange
from Griffin_Module.Attention import MultiHeadAttention

torch.autograd.set_detect_anomaly(True)

N_WINDOWS = 5
DIM = 20
DEPTHS = 1
DROPOUT_RATE = 0.25
MLP_MULT = 3
LEARNING_RATE = 0.001
LRU_LAYER = 22
subject_list = [3]
validation = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.g = nn.Parameter(torch.ones(dim))
        self.scale = torch.sqrt(torch.tensor(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        # Compute the RMS norm
        norm = x.norm(dim=-1, keepdim=True) + self.eps  # Add epsilon to avoid division by zero
        return x / norm * self.scale * self.g

class GriffinResidualBlock(nn.Module):
    def __init__(self, dim, mlp_mult, lru_layer, dropout):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        self.mlp = FeedForward(dim, mlp_mult, dropout)
        self.lru = RGLRU(dim, lru_layer)

        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

        self.depthwise = nn.Conv1d(self.dim, self.dim, kernel_size=9, padding="same")
        self.pointwise = nn.Conv1d(self.dim, self.dim, kernel_size=1)

        self.merge_linear = nn.Linear(dim, dim)
    def forward(self, x, prev_h=None):
        skip = x

        x = self.norm(x)
        linear_1, linear_2 = self.linear_1(x), self.linear_2(x)
        linear_1 = self.depthwise(linear_1.permute(0, 2, 1).contiguous())
        linear_1 = self.pointwise(linear_1).permute(0, 2, 1).contiguous()

        linear_1 = self.lru(linear_1, prev_h)
        print(linear_1.shape)
        linear_2 = F.gelu(linear_2)

        x = linear_1 * linear_2

        x += skip

        skip_2 = x
        x = self.norm(x)
        x = self.mlp(x)

        x += skip_2
        return x
class Griffin(nn.Module):
    def __init__(self, dim, depth, lru_layer, mlp_mult, dropout):
        super(Griffin, self).__init__()
        self.conv1D = nn.Conv1d(22, 22, kernel_size=9, padding="same")


        # for input shape of (batch, 1, 1125, 22)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(25, 1), padding='valid'),
            nn.Conv2d(25, 50, kernel_size=(1, 22), padding='valid'),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(75, 1), stride=(25, 1)),
            nn.Dropout(0.5),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

        self.MHA = MultiHeadAttention(50, num_heads=10, dropout=0.35)

        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=(5, 1), padding='valid'),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(6, 1), stride=(2, 1)),
            nn.Dropout(0.1)
        )
        self.norm = RMSNorm(dim)
        self.griffin_layers = nn.ModuleList([GriffinResidualBlock(dim, mlp_mult, lru_layer, dropout) for _ in range(depth)])
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            # nn.Linear(2100, 256),
            # nn.ELU(),
            # nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Dropout(0.3),
        )
        self.final = nn.Linear(16, 4)

    def forward(self, x):
        # residual block for griffin
        for layer in self.griffin_layers:
            x = layer(x)[0] + x

        x = torch.unsqueeze(x, 1)
        x = self.flatten(x)
        x = self.fc(x)

        x = self.final(x)
        x = F.softmax(x, dim=-1)
        return x

if __name__ == '__main__':

    dim = DIM
    depth = DEPTHS
    mlp_mult = MLP_MULT
    dropout = DROPOUT_RATE
    lru_layer = LRU_LAYER

    model = Griffin(dim=1125, depth=depth, lru_layer=lru_layer, mlp_mult=mlp_mult, dropout=dropout).to(device)
    x = torch.randn(32, 22, 1125).to(device)

    output = model(x)
    # Set print options to display more elements of the tensor
    torch.set_printoptions(threshold=500)
    bool = output > 0

