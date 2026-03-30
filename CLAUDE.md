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

### Qwen3.5 Dense (text + vision)

**Files**: `models/qwen35dense.py`, `models/utils.py`

#### Config Classes (`models/qwen35dense.py`)

- `RopeConfig` — mRoPE params: `rope_theta`, `partial_rotary_factor`, `mrope_section`, `mrope_interleaved`
- `VisionConfig` — ViT encoder params: `hidden_size=768`, `num_heads=12`, `patch_size=16`, `spatial_merge_size=2`, `temporal_patch_size=2`, `out_hidden_size=1024`
- `TextConfig` — LLM params: `hidden_size=1024`, `head_dim=256`, 24 hybrid layers, plus linear attention params (`linear_key/value_head_dim`, `linear_conv_kernel_dim`)
- `Qwen3_5Config` — top-level: `text_config`, `vision_config`, token IDs (`image_token_id`, `vision_start/end_token_id`)

#### Text Model Classes

| Class | Description |
|-------|-------------|
| `FullAtention` | Standard multi-head attention with output gate (`q_proj` splits into q+gate), RMS norm on q/k, GQA support |
| `LinearAttention` | Gated DeltaNet — conv1d → qkv split → l2norm(q,k) → recurrent gated delta rule (Python for-loop) → RMS norm + SiLU gating |
| `MLP` | SwiGLU: gate_proj/up_proj → SiLU → down_proj |
| `Layer` | input_layernorm → attention → post_attention_layernorm → MLP (dispatches `full_attention` or `linear_attention` based on `layer_types`) |
| `Qwen3_5TextModel` | embed_tokens → 24 layers → model_norm → logits |

#### Vision Model Classes

| Class | Description |
|-------|-------------|
| `VisionAttention` | Standard MHA with vision RoPE (`rotate_half` formulation) and block-diagonal causal mask via `cu_seqlens` |
| `VisionMLP` | fc1 → GELU(approximate="tanh") → fc2 |
| `VisionBlock` | norm1 → attn → norm2 → mlp (uses `nn.LayerNorm`, not RMS) |
| `VisionPatchMerger` | LayerNorm(768) → fc1(3072) → GELU → fc2(1024), merges spatial tokens to text hidden dim |
| `VisionRotaryEmbedding` | Simple inv_freq with theta=10000, dim=head_dim//2=32 |
| `Qwen3_5VisionModel` | patch_embed(Conv3d) → pos_embed(bilinear interpolation) → vision RoPE → 12 blocks → merger |

### RoPE (Rotary Position Embedding)

Three distinct RoPE implementations in `models/utils.py`:

**1. Text mRoPE** (`compute_mrope_cos_sin`) — used by FullAttention
- 3D position IDs `(T, H, W)`, for text-only all channels are identical
- `inv_freq` computed with `rotary_dim = head_dim * partial_rotary_factor` (not full head_dim!)
- `mrope_section = [11, 11, 10]` → 32 freq components interleaved across T/H/W channels
- Output: cos, sin of shape `(B, T, rotary_dim//2)`

**2. Vision RoPE** (`apply_rotary_pos_emb_vision`) — used by VisionAttention
- 2D (H/W) positions, theta=10000
- Uses `rotate_half` formulation: `[x1, x2] → [-x2, x1]`

**3. Generic apply_rope** — shared by text model
- Supports partial RoPE (only first `rope_dim` dims rotated, rest passed through)
- cos/sin shape `(T, 1, rope_dim//2)`, applies `(x1*cos - x2*sin, x2*cos + x1*sin)`

### Hybrid Attention

The model alternates between `full_attention` and `linear_attention` layers (defined per-layer in `layer_types`). Cache structures differ:
- Full: KV cache `(kv_heads, max_pos, head_dim)`
- Linear: recurrent state `(1, v_heads, k_dim, v_dim)` + conv state `(1, conv_dim, kernel_dim)`

### Multimodal Pipeline

1. **Image processing** (`process_image` in utils.py): resize → normalize → temporal duplicate → patch extraction
2. **Vision encode**: patches → Conv3d patch embed → pos embed → 12 ViT blocks → merger → `(N, 1024)` embeddings
3. **Prompt**: `<|vision_start|><|image_pad|>*N<|vision_end|>...text...<|im_start|>assistant\n`
4. **Token replacement**: image_pad embeddings replaced with vision embeddings in hidden_states
5. **3D position IDs**: text tokens get sequential `(t, t, t)`, vision tokens get `(t, t+h, t+w)` based on spatial grid

### Inference Scripts

- `main_qwen35.py` — text-only inference
- `main_qwen35_image.py` — multimodal (vision + text) inference
