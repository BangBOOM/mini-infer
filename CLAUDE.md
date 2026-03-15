# CLAUDE.md

this is mini-infer, it implements a minimal llm inference to learn and understand different arch

## Basic Rules

+ think before act
+ do small change and try to make it easy to understand

## How to run the project

`uv run python main.py`

## Tokenizer

We use a minimal tokenizer implementation (`tokenizer_wrapper.py`) instead of `transformers.AutoTokenizer`:

- **Library**: `tokenizers` (Rust-based, fast) + `jinja2` (chat templates)
- **Why**: Much lighter (~5MB vs ~500MB) and faster than full transformers
- **Usage**: 
  ```python
  from tokenizer_wrapper import ChatTokenizer
  tokenizer = ChatTokenizer("/path/to/model/")
  
  # Apply chat template
  prompt = tokenizer.apply_chat_template(
      [{"role": "user", "content": "Hello"}],
      add_generation_prompt=True
  )
  
  # Encode/decode
  ids = tokenizer.encode("Hello")
  text = tokenizer.decode(ids)
  ```

The tokenizer loads `tokenizer.json` and `tokenizer_config.json` from the model directory.

## Model Architecture

### RoPE (Rotary Position Embedding)

RoPE is applied via `apply_rope(x, cos, sin)` function in `models/qwen3dense.py`:

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embedding to input tensor.
    
    Args:
        x: Input tensor of shape (num_tokens, num_heads, head_dim)
        cos: Cosine values of shape (num_tokens, 1, head_dim//2)
        sin: Sine values of shape (num_tokens, 1, head_dim//2)
    
    Returns:
        Tensor with RoPE applied, same shape as input
    """
```

Used in attention:
```python
q = apply_rope(q.view(num_tokens, -1, head_dim), cos, sin).view(batch_size, seqlen, num_attention_heads, head_dim)
k = apply_rope(k.view(num_tokens, -1, head_dim), cos, sin).view(batch_size, seqlen, num_key_value_heads, head_dim)
```
