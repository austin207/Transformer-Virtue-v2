import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from tokenizers import Tokenizer
from miniGPT import MiniGPT

# --- 1. Load the Word-Level Tokenizer ---
tokenizer = Tokenizer.from_file("wordlevel.json")
vocab_size = tokenizer.get_vocab_size()

# --- 2. Load and Encode the Dataset ---
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()
data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)

# --- 3. Train/Val Split ---
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --- 4. Batching Function ---
def get_batch(split, batch_size, block_size):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x, y

# --- 5. Model Setup ---
model = MiniGPT(
    vocab_size=vocab_size,
    embed_dim=128,        # You can tune this up or down for your hardware
    num_heads=4,
    ff_dim=512,
    num_layers=4,
    max_seq_len=128       # Should be >= block_size
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 8
block_size = 128
max_iters = 20000
eval_interval = 100
checkpoint_interval = 500

# --- 6. Generate Function (Word-level) ---
def generate(model, start_text, max_new_tokens=20, temperature=1.0):
    model.eval()
    idx = torch.tensor([tokenizer.encode(start_text).ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        if idx.shape[1] >= model.max_seq_len:
            break
        logits = model(idx)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].tolist())

# --- 7. Resume from Checkpoint (Optional) ---
checkpoint_path = "model_checkpoint_step2000.pt"
start_step = 1
start_time = time.time()
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step'] + 1
    elapsed_time_previous = checkpoint.get('elapsed_time', 0)
    start_time = time.time() - elapsed_time_previous
    print(f"Resumed training from step {start_step}")

# --- 8. Training Loop ---
for step in range(start_step, max_iters + 1):
    model.train()
    xb, yb = get_batch('train', batch_size, block_size)
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging and validation
    if step % eval_interval == 0:
        elapsed = time.time() - start_time
        steps_done = step - start_step + 1
        steps_left = max_iters - step
        time_per_step = elapsed / max(1, steps_done)
        eta = steps_left * time_per_step

        model.eval()
        with torch.no_grad():
            xb, yb = get_batch('val', batch_size, block_size)
            val_logits = model(xb)
            val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), yb.view(-1))
            print(f"Step: {step} | train loss: {loss.item():.4f} | val loss: {val_loss.item():.4f} | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

            sample_text = "Once upon a time"
            generated_text = generate(model, sample_text, max_new_tokens=20, temperature=0.8)
            print(f"Sample Text generation after step@{step}: {generated_text}")

    # Checkpointing
    if step % checkpoint_interval == 0 or step == max_iters:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'elapsed_time': time.time() - start_time
        }
        checkpoint_path = f"model_checkpoint_step{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at step {step} to {checkpoint_path}")
