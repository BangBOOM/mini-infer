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
