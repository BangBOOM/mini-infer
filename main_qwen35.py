import json
import os
from dataclasses import fields

import torch

from models.qwen35dense import Qwen3_5, Qwen3_5Config
from tokenizer_wrapper import ChatTokenizer

base_path = "/Users/bangboom/Documents/models/Qwen3.5-0.8B"
config_path = os.path.join(base_path, "config.json")
model_path = os.path.join(base_path, "model.safetensors-00001-of-00001.safetensors")

# Use our custom tokenizer instead of AutoTokenizer
tokenizer = ChatTokenizer(base_path)

with open(config_path, "r", encoding="utf-8") as f:
    qwen3_config = Qwen3_5Config(
        **{k: v for k, v in json.load(f)["text_config"].items() if k in [field.name for field in fields(Qwen3_5Config)]}
    )
torch.set_default_device("cpu")
torch.set_default_dtype(torch.bfloat16)
model = Qwen3_5(qwen3_config)
model.load_weight(model_path)

# Create RoPE cos_sin cache
rotary_dim = int(qwen3_config.head_dim * qwen3_config.rope_parameters["partial_rotary_factor"])
inv_freq = 1.0 / (qwen3_config.rope_parameters["rope_theta"] ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
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
predict_tokens = model(input_ids, positions, cos_sin_cache, is_prefill=True)
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
    predict_tokens = model(input_ids, positions, cos_sin_cache, is_prefill=False)
    generated_token = tokenizer.decode(predict_tokens)
    res += generated_token
    if predict_tokens == tokenizer.eos_token_id:
        break
    print(generated_token, end="", flush=True)

print("\nDone!!!")
