import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len 
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.head.weight = self.token_embedding.weight

    def forward(self, idx, mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T,device=idx.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x, mask=mask) if mask is not None else self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    