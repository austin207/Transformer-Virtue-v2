import torch

vocab = list("abcdefghijklmnopqrstuvwxyz ")
vocab_size = len(vocab)
#mapping string to index and Back
stoi = {ch: i for i, ch in  enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}

text = "hello world"
encoded = [stoi[ch] for ch in text]

print(encoded)