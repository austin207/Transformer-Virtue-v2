import torch
import torch.nn as nn

vocab_size = 27
embedding_dim = 16
max_seq_len = 32

token_embedding = nn.Embedding(vocab_size, embedding_dim)
pos_embedding = nn.Embedding(max_seq_len, embedding_dim)

input_ids = torch.tensor([[7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3]])
batch_size, seq_len = input_ids.shape

tok_emb = token_embedding(input_ids)

positions = torch.arange(seq_len).unsqueeze(0)
pos_emb = pos_embedding(positions)

x = tok_emb + pos_emb

print(positions)