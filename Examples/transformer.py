import torch
import torch.nn as nn
import torch.nn.functional as F
from multiheadattention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask = mask)
        x = x + self.ff(self.ln2(x))
        return x


embed_dim = 32
seq_len = 10
batch_size = 4

x = torch.randn(batch_size, seq_len, embed_dim)

attn2 = MultiHeadAttention(embed_dim, num_heads= 2)
out2 = attn2(x)
print("attn2:", out2.shape)


attn4 = MultiHeadAttention(embed_dim, num_heads= 4)
out4 = attn4(x)
print("attn4:", out4.shape)

block1 = TransformerBlock(embed_dim, num_heads=4, ff_dim=64)
block2 = TransformerBlock(embed_dim, num_heads=4, ff_dim=64)

out = block1(x)
out = block2(out)
print("output shape after 2 transformer blocks:", out.shape)