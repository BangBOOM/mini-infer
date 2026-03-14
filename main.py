import json
from dataclasses import fields

import torch
from transformers import AutoTokenizer

from models.qwen3dense import Qwen3, Qwen3Config

base_path = "/Users/bangboom/Documents/models/Qwen3.5-0.8B/"
config_path = base_path + "config.json"
# model_path = base_path + "model.safetensors"
model_path = base_path + "model.safetensors-00001-of-00001.safetensors"
tokenizer = AutoTokenizer.from_pretrained(base_path)
with open(config_path, "r", encoding="utf-8") as f:
    qwen3_config = Qwen3(**{k:v for k,v in json.load(f)["text_config"].items() if k in [field.name for field in fields(Qwen3Config)]})
torch.set_default_device("cpu")
torch.set_default_dtype(torch.bfloat16)
model = Qwen3(qwen3_config)
model.load_weight(model_path)
model = model.eval()
generated_token = ""
prompts = ["list all prime numbers within 100"]
prompts = [
    tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for prompt in prompts
]
print(prompts)
# prefill
input_ids = tokenizer(prompts, return_tensors="pt")["input_ids"]
print(input_ids)
positions = torch.tensor(list(range(input_ids.shape[-1])), dtype=torch. int)
predict_tokens = model(input_ids, positions, is_prefill=True)
generated_token = tokenizer.decode(predict_tokens)
res = ""
res += generated_token
position_id = positions [-1]
print("Done Prefill")
while position_id < 100:
    position_id += 1
    output = [generated_token]
    input_ids = tokenizer(output, return_tensors="pt") ["input_ids"]
    positions = torch.tensor([position_id], dtype=torch.int)
    predict_tokens = model(input_ids, positions)
    generated_token = tokenizer.decode(predict_tokens)
    res += generated_token
    if predict_tokens == tokenizer.eos_token_id:
        break
    print(generated_token, end="=", flush=True)

print("\nDone!!!")
