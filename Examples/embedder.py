import torch
import torch.nn as nn
from tokeniser import vocab_size, encoded

embedding_dim = 16
embedding = nn.Embedding(vocab_size, embedding_dim)

input_ids = torch.tensor([encoded])
embedded = embedding(input_ids)

print(embedded.shape)