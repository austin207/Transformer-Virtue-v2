# MiniGPT: Lightweight Transformer for Text Generation

**MiniGPT** is a minimal yet powerful implementation of a GPT-style language model built for learning, customization, and real-time inference. It includes a complete training pipeline, modular architecture, and support for modern text generation strategies like streaming, top-k, top-p, temperature sampling, and repetition penalty.

---

##  Features

*  Word-level tokenization (via Hugging Face `tokenizers`)
*  Customizable transformer architecture
*  Full training pipeline with checkpointing
*  Real-time streaming inference with:

  * Temperature Sampling
  * Top-K Filtering
  * Top-P (Nucleus) Sampling
  * Repetition Penalty
*  Modular code: easy to read, extend, and debug
*  Hugging Face Hub integration

---

##  Project Structure

```
MiniGPT/
├── miniGPT.py               # Core Transformer model
├── transformer.py           # Transformer block logic
├── multiheadattention.py    # Multi-head self-attention
├── Tokenizer.py             # Word-level tokenizer wrapper
├── dataprocess.py           # Text cleaning and preprocessing
├── training.py              # Training pipeline with checkpointing
├── inference.py             # CLI interface for streaming generation
├── wordlevel.json           # Pre-trained tokenizer
├── alphabetical_dataset.txt # Sample training corpus
├── requirements.txt         # Dependencies
└── README.md                # This documentation
```

---

##  Model Architecture

The model is based on the Transformer decoder stack and includes:

* Positional and token embeddings
* Multi-head self-attention
* Feed-forward network (FFN)
* Residual connections + LayerNorm

**Default Configuration:**

| Parameter          | Value |
| ------------------ | ----- |
| Embedding Dim      | 128   |
| Transformer Layers | 4     |
| Attention Heads    | 4     |
| FFN Dim            | 512   |
| Max Seq Length     | 128   |

---

##  Setup and Installation

```bash
git clone https://github.com/austin207/Transformer-Virtue-v2.git
cd Transformer-Virtue-v2
pip install -r requirements.txt
```

> Requires: Python 3.7+, PyTorch, `tokenizers`, and `huggingface_hub` for optional uploads.

---

##  Training

To train the model on your own dataset:

```bash
python training.py
```

### Training Features:

* Automatic checkpointing every N steps
* Training logs with loss metrics
* Sample generation during training
* Supports custom datasets (plain text)

---

##  Inference & Streaming Text Generation

```bash
python inference.py
```

You will be prompted to enter a prompt in the terminal:

```
Enter your prompt (or type 'exit' to quit): Once upon a time
Streaming output:
Once upon a time, in a realm beyond stars...
```

### Parameters:

* `max_new_tokens`: Length of generated output
* `temperature`: Sampling randomness (0.7–1.2 typical)
* `top_k`: Top-k filtering (limits tokens)
* `top_p`: Nucleus sampling (focus on top cumulative prob.)
* `repetition_penalty`: Reduces loops and redundancy

---

##  Example Usage in Python

```python
from miniGPT import MiniGPT
from tokenizers import Tokenizer
from inference import generate_stream

tokenizer = Tokenizer.from_file("wordlevel.json")

model = MiniGPT(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=128,
    num_heads=4,
    ff_dim=512,
    num_layers=4,
    max_seq_len=128
)

model.load_state_dict(torch.load("model_checkpoint_step20000.pt")["model_state_dict"])
model.eval()

prompt = "The sky cracked open"
generate_stream(model, tokenizer, prompt, max_new_tokens=60, temperature=1.0, top_k=50, top_p=0.9)
```

---

##  Upload to Hugging Face Hub

Install tools:

```bash
pip install huggingface_hub hf_transfer
huggingface-cli login
```

Upload with fast transfer:

```bash
hf_transfer upload ./Transformer-MiniGPT --repo-id your-username/Transformer-MiniGPT
```

Or standard CLI:

```bash
huggingface-cli upload your-username/Transformer-MiniGPT ./Transformer-MiniGPT/
```

View my repo at: [Transformer-MiniGPT](https://huggingface.co/Austin207/Transformer-MiniGPT/tree/main)

---

##  License

This project is licensed under the [MIT License](LICENSE).

---

##  Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/foo`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/foo`)
5. Open a pull request

---

##  Acknowledgments

Inspired by:

* OpenAI GPT architecture
* Karpathy’s nanoGPT
* Hugging Face Transformers

