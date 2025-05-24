import torch
import time
from tokenizers import Tokenizer
from miniGPT import MiniGPT

# --- 1. Load tokenizer and model ---
tokenizer = Tokenizer.from_file("wordlevel.json")
vocab_size = tokenizer.get_vocab_size()

# Set model parameters to match your trained model
model = MiniGPT(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=4,
    ff_dim=512,
    num_layers=4,
    max_seq_len=128
)
checkpoint_path = "model_checkpoint_step20000.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- 2. Show model parameter count ---
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# --- 3. Sampling helpers ---

def top_k_logits(logits, k):
    """Keep only top-k tokens with highest probability."""
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    logits[logits < min_values] = -float('Inf')
    return logits

def top_p_logits(logits, p=0.9):
    """Keep the smallest set of tokens with cumulative probability >= p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    for batch in range(logits.size(0)):
        remove_ids = sorted_indices[batch][sorted_indices_to_remove[batch]]
        logits[batch, remove_ids] = -float('Inf')
    
    return logits

# --- 4. Streaming generation function ---
def generate_stream(
    model, tokenizer, prompt, 
    max_new_tokens=50, 
    temperature=1.0, 
    top_k=None, 
    top_p=None,
    repetition_penalty=2.0
):
    idx = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
    generated = []
    start_time = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if idx.shape[1] >= model.max_seq_len:
                break

            logits = model(idx)
            logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            for token_id in set(generated):
                logits[0, token_id] /= repetition_penalty

            # Apply Top-K and/or Top-P filtering
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            if top_p is not None:
                logits = top_p_logits(logits, top_p)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            generated.append(next_id.item())
            print(tokenizer.decode([next_id.item()]), end=' ', flush=True)

    elapsed = time.time() - start_time
    tps = len(generated) / elapsed if elapsed > 0 else 0
    print(f"\n[Generated {len(generated)} tokens in {elapsed:.2f} seconds | {tps:.2f} tokens/sec]")
    return idx

# --- 5. Main input loop ---
while True:
    prompt = input("\nEnter your prompt (or type 'exit' to quit): ")
    if prompt.lower() == 'exit':
        break

    print("\nStreaming output:")
    generate_stream(
        model, tokenizer, prompt, 
        max_new_tokens=90,
        temperature=2.0,
        top_k=100,
        top_p=0.9,
        repetition_penalty=1.8
    )
