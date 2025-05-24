import torch
import torch.nn as nn
import torch.nn.functional as F

embedding_dim = 16
seq_len = 5

x = torch.randn(1, seq_len, embedding_dim)

to_q = nn.Linear(embedding_dim, embedding_dim)
to_k = nn.Linear(embedding_dim, embedding_dim)
to_v = nn.Linear(embedding_dim, embedding_dim)

Q = to_q(x)
K = to_k(x)
V = to_v(x)

scores = torch.matmul(Q, K.transpose(-2, -1)) / (embedding_dim ** 0.5)

attn_weights = F.softmax(scores, dim=1)

out = torch.matmul(attn_weights, V)

print(out.shape)
