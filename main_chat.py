import argparse
import json
import os

import torch
from PIL import Image

from models.qwen35dense import Qwen3_5Config, Qwen3_5TextModel, Qwen3_5VisionModel
from models.utils import process_image
from tokenizer_wrapper import ChatTokenizer

BASE_PATH = "/home/bangboom/models/Qwen3.5-0.8B"
MAX_NEW_TOKENS = 512
IMAGE_PAD = "<|image_pad|>"


def load_model(config):
    model_path = os.path.join(BASE_PATH, "model.safetensors-00001-of-00001.safetensors")
    tokenizer = ChatTokenizer(BASE_PATH)

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    vision_model = Qwen3_5VisionModel(config.vision_config)
    vision_model.load_weight(model_path)
    vision_model = vision_model.cuda().eval()

    text_model = Qwen3_5TextModel(config.text_config)
    text_model.load_weight(model_path)
    text_model = text_model.cuda().eval()

    # Hybrid cache
    hybrid_cache = build_cache(config)

    return tokenizer, vision_model, text_model, hybrid_cache


def build_cache(config):
    tc = config.text_config
    conv_dim = 2 * tc.linear_num_key_heads * tc.linear_key_head_dim + tc.linear_num_value_heads * tc.linear_value_head_dim
    cache = []
    for attn_type in tc.layer_types:
        if attn_type == "full_attention":
            cache.append({
                "key": torch.zeros(tc.num_key_value_heads, tc.max_position_embeddings, tc.head_dim).cuda(),
                "value": torch.zeros(tc.num_key_value_heads, tc.max_position_embeddings, tc.head_dim).cuda(),
            })
        else:
            cache.append({
                "recurrent_state": torch.zeros(1, tc.linear_num_value_heads, tc.linear_key_head_dim, tc.linear_value_head_dim).cuda(),
                "conv_state": torch.zeros(1, conv_dim, tc.linear_conv_kernel_dim).cuda(),
            })
    return cache


def encode_image(image_path, config, vision_model):
    image = Image.open(image_path).convert("RGB")
    flatten_patches, grid_t, grid_h, grid_w = process_image(
        image,
        patch_size=config.vision_config.patch_size,
        spatial_merge_size=config.vision_config.spatial_merge_size,
        temporal_patch_size=config.vision_config.temporal_patch_size,
        max_pixels=16777216, min_pixels=65536,
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
    )
    pixels = torch.tensor(flatten_patches, dtype=torch.bfloat16).cuda()
    d_image = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long).cuda()
    with torch.no_grad():
        vision_embed = vision_model(pixels, d_image)
    return vision_embed, grid_h, grid_w


def build_position_ids(input_ids, config, grid_h, grid_w):
    seq_len = input_ids.shape[-1]
    position_ids = torch.zeros(3, 1, seq_len, dtype=torch.long).cuda()
    image_pad_token_id = config.image_token_id
    spatial_merge_size = config.vision_config.spatial_merge_size
    h_img = grid_h // spatial_merge_size
    w_img = grid_w // spatial_merge_size
    num_vision = h_img * w_img

    text_pos = 0
    i = 0
    while i < seq_len:
        if input_ids[0, i].item() == image_pad_token_id:
            for offset in range(num_vision):
                h_pos = offset // w_img
                w_pos = offset % w_img
                position_ids[0, 0, i + offset] = text_pos
                position_ids[1, 0, i + offset] = text_pos + h_pos
                position_ids[2, 0, i + offset] = text_pos + w_pos
            i += num_vision
            text_pos += 1
        else:
            position_ids[:, 0, i] = text_pos
            text_pos += 1
            i += 1

    return position_ids, text_pos


@torch.no_grad()
def generate(tokenizer, text_model, vision_model, hybrid_cache, config, user_text, image_path=None):
    vision_embed = None
    grid_h = grid_w = 0

    if image_path:
        print(f"[Loading image: {image_path}]")
        vision_embed, grid_h, grid_w = encode_image(image_path, config, vision_model)
        num_vision = vision_embed.shape[0]
        prompt_content = f"<|vision_start|>{IMAGE_PAD * num_vision}<|vision_end|>\n{user_text}"
    else:
        prompt_content = user_text

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    # Vision mask
    vision_mask = None
    if vision_embed is not None:
        vision_mask = (input_ids == config.image_token_id)

    # Position IDs
    if vision_embed is not None:
        position_ids, text_pos = build_position_ids(input_ids, config, grid_h, grid_w)
    else:
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, 1, -1).cuda()
        text_pos = seq_len - 1

    # Prefill
    predict_tokens = text_model(
        input_ids.cuda(), position_ids, hybrid_cache,
        is_prefill=True, vision_embed=vision_embed, vision_mask=vision_mask,
    )
    generated_token = tokenizer.decode(predict_tokens)
    print(generated_token, end="", flush=True)
    text_pos += 1

    # Decode
    for _ in range(MAX_NEW_TOKENS):
        if predict_tokens == tokenizer.eos_token_id:
            break
        new_ids = tokenizer([generated_token], return_tensors="pt")["input_ids"]
        position_ids_decode = torch.full((3, 1, 1), text_pos, dtype=torch.long).cuda()
        predict_tokens = text_model(new_ids.cuda(), position_ids_decode, hybrid_cache, is_prefill=False)
        generated_token = tokenizer.decode(predict_tokens)
        print(generated_token, end="", flush=True)
        text_pos += 1

    print()


def main():
    parser = argparse.ArgumentParser(description="Mini-infer chat: text and image inference")
    parser.add_argument("--text", type=str, required=True, help="User prompt text")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    args = parser.parse_args()

    config_path = os.path.join(BASE_PATH, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = Qwen3_5Config(**json.load(f))

    print("[Loading model...]")
    tokenizer, vision_model, text_model, hybrid_cache = load_model(config)
    print("[Model loaded]")

    generate(tokenizer, text_model, vision_model, hybrid_cache, config, args.text, args.image)


if __name__ == "__main__":
    main()
