import json
import os

import torch
from PIL import Image

from models.qwen35dense import Qwen3_5Config, Qwen3_5TextModel, Qwen3_5VisionModel
from models.utils import process_image
from tokenizer_wrapper import ChatTokenizer

torch.manual_seed(123)
base_path = "/home/bangboom/models/Qwen3.5-0.8B"
config_path = os.path.join(base_path, "config.json")
model_path = os.path.join(base_path, "model.safetensors-00001-of-00001.safetensors")

with open(config_path, "r", encoding="utf-8") as f:
    config = Qwen3_5Config(**json.load(f))

tokenizer = ChatTokenizer(base_path)
torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

# Load vision model
vision_model = Qwen3_5VisionModel(config.vision_config)
vision_model.load_weight(model_path)
vision_model = vision_model.cuda().eval()

# Load text model
text_model = Qwen3_5TextModel(config.text_config)
text_model.load_weight(model_path)
text_model = text_model.cuda().eval()


# Create Hybrid Cache
hybrid_cache = []
conv_dim = (
    2 * config.text_config.linear_num_key_heads * config.text_config.linear_key_head_dim
    + config.text_config.linear_num_value_heads * config.text_config.linear_value_head_dim
)
for attention_type in config.text_config.layer_types:
    if attention_type == "full_attention":
        hybrid_cache.append(
            {
                "key": torch.zeros(config.text_config.num_key_value_heads, config.text_config.max_position_embeddings, config.text_config.head_dim).cuda(),
                "value": torch.zeros(config.text_config.num_key_value_heads, config.text_config.max_position_embeddings, config.text_config.head_dim).cuda(),
            }
        )
    else:
        hybrid_cache.append(
            {
                "recurrent_state": torch.zeros(1, config.text_config.linear_num_value_heads, config.text_config.linear_key_head_dim, config.text_config.linear_value_head_dim).cuda(),
                "conv_state": torch.zeros(1, conv_dim, config.text_config.linear_conv_kernel_dim).cuda(),
            }
        )


# --- Image processing ---
image_path = "/home/bangboom/code/tiny-qwen/test/data/test-img-1.jpg"
image = Image.open(image_path).convert("RGB")

flatten_patches, grid_t, grid_h, grid_w = process_image(
    image,
    patch_size=config.vision_config.patch_size,
    spatial_merge_size=config.vision_config.spatial_merge_size,
    temporal_patch_size=config.vision_config.temporal_patch_size,
    max_pixels=16777216,
    min_pixels=65536,
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
)
print(f"Image patches: {flatten_patches.shape}, grid: t={grid_t} h={grid_h} w={grid_w}")

# Run vision encoder
pixels = torch.tensor(flatten_patches, dtype=torch.bfloat16).cuda()
d_image = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long).cuda()
with torch.no_grad():
    vision_embed = vision_model(pixels, d_image)  # (N_vision, 1024)
print(f"Vision embed: {vision_embed.shape}")

# Number of merged vision tokens (= number of <|image_pad|> tokens in input)
num_vision_tokens = vision_embed.shape[0]
print(f"Num vision tokens: {num_vision_tokens}")

# --- Build multimodal prompt ---
# Format: <|im_start|>user\n<|vision_start|><|image_pad|>*N<|vision_end|>\nDescribe this image...<|im_end|>\n<|im_start|>assistant\n
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"

prompt_text = f"{VISION_START}{IMAGE_PAD * num_vision_tokens}{VISION_END}\nDescribe this image accurately in 2-3 sentences."
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True, enable_thinking=False
)
print(f"Prompt length: {len(prompt)}")

# Tokenize
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
print(f"Input ids shape: {input_ids.shape}")

# Verify vision token count
image_pad_token_id = config.image_token_id
vision_mask = (input_ids == image_pad_token_id)[0]  # (seq_len,)
num_pads = vision_mask.sum().item()
print(f"Image pad tokens in input: {num_pads}")
assert num_pads == num_vision_tokens, f"Token mismatch: {num_pads} pads vs {num_vision_tokens} vision tokens"

# --- Compute 3D position_ids ---
seq_len = input_ids.shape[-1]
position_ids = torch.zeros(3, 1, seq_len, dtype=torch.long).cuda()

text_pos = 0
vision_idx = 0
i = 0
while i < seq_len:
    if input_ids[0, i].item() == image_pad_token_id:
        # Vision block: assign 3D position IDs
        spatial_merge_size = config.vision_config.spatial_merge_size
        h_img = grid_h // spatial_merge_size
        w_img = grid_w // spatial_merge_size
        for offset in range(h_img * w_img):
            h_pos = offset // w_img
            w_pos = offset % w_img
            position_ids[0, 0, i + offset] = text_pos
            position_ids[1, 0, i + offset] = text_pos + h_pos
            position_ids[2, 0, i + offset] = text_pos + w_pos
        i += h_img * w_img
        text_pos += 1
    else:
        position_ids[:, 0, i] = text_pos
        text_pos += 1
        i += 1

# --- Prefill ---
vision_mask_2d = vision_mask.unsqueeze(0)  # (1, seq_len)
predict_tokens = text_model(
    input_ids.cuda(), position_ids, hybrid_cache,
    is_prefill=True, vision_embed=vision_embed, vision_mask=vision_mask_2d,
)
generated_token = tokenizer.decode(predict_tokens)
res = generated_token
text_pos += 1
print("Done Prefill")
print(generated_token, end="", flush=True)

# --- Decode ---
while text_pos < 300:
    output = [generated_token]
    new_ids = tokenizer(output, return_tensors="pt")["input_ids"]
    # For decode, all 3 channels use the text position
    position_ids_decode = torch.full((3, 1, 1), text_pos, dtype=torch.long).cuda()
    predict_tokens = text_model(
        new_ids.cuda(), position_ids_decode, hybrid_cache, is_prefill=False,
    )
    generated_token = tokenizer.decode(predict_tokens)
    res += generated_token
    if predict_tokens == tokenizer.eos_token_id:
        break
    text_pos += 1
    print(generated_token, end="", flush=True)

print("\nDone!!!")
