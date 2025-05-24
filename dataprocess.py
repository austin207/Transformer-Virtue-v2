from Tokenizer import tokenizer
#read and store data
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()
#print("sample:", text[:3000])

#lowercasing the text
text = text.lower()
#print("sample:", text[:3000])

#building Vocbulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

#print(vocab_size)
#print(chars)

#converting words to numbers so the model can understand
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

#print(stoi)
#print(itos)


# Defining encoding and decoding functions 
def encode(s):
    return[stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

import torch

data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
#print("encoded sample:", data[:20])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size, block_size):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x,y 

