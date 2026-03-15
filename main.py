import json
import os
from dataclasses import fields

import torch

from models.qwen3dense import Qwen3, Qwen3Config
from tokenizer_wrapper import ChatTokenizer

base_path = "/Users/bangboom/Documents/models/Qwen3-0.6B/"
config_path = os.path.join(base_path, "config.json")
model_path = os.path.join(base_path, "model.safetensors")

# Use our custom tokenizer instead of AutoTokenizer
tokenizer = ChatTokenizer(base_path)

with open(config_path, "r", encoding="utf-8") as f:
    qwen3_config = Qwen3Config(
        **{k: v for k, v in json.load(f).items() if k in [field.name for field in fields(Qwen3Config)]}
    )
torch.set_default_device("cpu")
torch.set_default_dtype(torch.bfloat16)
model = Qwen3(qwen3_config)
model.load_weight(model_path)

# Create KV cache
kv_cache = torch.zeros(qwen3_config.num_hidden_layers, 2, qwen3_config.num_key_value_heads, qwen3_config.max_position_embeddings, qwen3_config.head_dim)

# Create RoPE cos_sin cache
inv_freq = 1.0 / (qwen3_config.rope_theta ** (torch.arange(0, qwen3_config.head_dim, 2, dtype=torch.float) / qwen3_config.head_dim))
t = torch.arange(qwen3_config.max_position_embeddings, dtype=torch.float)
freqs = torch.einsum("i,j -> ij", t, inv_freq)
cos = freqs.cos()
sin = freqs.sin()
cos_sin_cache = torch.cat((cos, sin), dim=-1)



model = model.eval()
generated_token = ""
prompts = ["list all prime numbers within 100"]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    for prompt in prompts
]
print(prompts)
# prefill
input_ids = tokenizer(prompts, return_tensors="pt")["input_ids"]
print(input_ids)
positions = torch.tensor(list(range(input_ids.shape[-1])), dtype=torch.int)
predict_tokens = model(input_ids, positions, kv_cache, cos_sin_cache, is_prefill=True)
generated_token = tokenizer.decode(predict_tokens)
res = ""
res += generated_token
position_id = positions[-1]
print("Done Prefill")
while position_id < 100:
    position_id += 1
    output = [generated_token]
    input_ids = tokenizer(output, return_tensors="pt")["input_ids"]
    positions = torch.tensor([position_id], dtype=torch.int)
    predict_tokens = model(input_ids, positions, kv_cache, cos_sin_cache)
    generated_token = tokenizer.decode(predict_tokens)
    res += generated_token
    if predict_tokens == tokenizer.eos_token_id:
        break
    print(generated_token, end="", flush=True)

print("\nDone!!!")
