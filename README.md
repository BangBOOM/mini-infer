# mini-infer

A minimal LLM inference implementation for learning and understanding transformer architectures. Currently supports Qwen3 models.

## What This Does

This repo implements a from-scratch inference engine for transformer-based language models without relying on heavy frameworks like `transformers`. It demonstrates:

- **Custom tokenizer** using `tokenizers` + `jinja2` for chat templates (~5MB vs ~500MB)
- **Manual attention implementation** with KV caching and RoPE (Rotary Position Embedding)
- **Efficient inference** with prefill + decode phase

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- A Qwen3 model (e.g., `Qwen3-0.6B`)

## Setup

```bash
# Clone the repo
git clone https://github.com/BangBOOM/mini-infer.git
cd mini-infer

# Install dependencies
uv sync
```

## Run

Update the model path in `main.py`:
```python
base_path = "/path/to/your/Qwen3-0.6B/"
```

Then run:
```bash
uv run python main.py
```

## Example Output

```
['<|im_start|>user\nlist all prime numbers within 100<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n']
tensor([[151644, 872, 198, ...]])
Done Prefill
 prime numbers between 1 and 100 are:

**2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
Done!!!
```

## Project Structure

```
mini-infer/
├── main.py                 # Entry point
├── tokenizer_wrapper.py    # Lightweight tokenizer
├── models/
│   └── qwen3dense.py       # Qwen3 model implementation
└── CLAUDE.md               # Architecture notes
```

## License

MIT
