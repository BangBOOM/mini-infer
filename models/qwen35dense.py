import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from safetensors import safe_open
from tqdm import tqdm

from models.utils import add_rms_norm, apply_rope, apply_rotary_pos_emb_vision, compute_mrope_cos_sin, l2norm, rms_norm

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger("Qwen3_5")


class RopeConfig(BaseModel):
    mrope_interleaved: bool
    mrope_section: list[int]
    rope_type: str
    rope_theta: int
    partial_rotary_factor: float


class VisionConfig(BaseModel):
    deepstack_visual_indexes: list[int]
    depth: int
    hidden_act: str
    hidden_size: int
    in_channels: int
    initializer_range: float
    intermediate_size: int
    model_type: str
    num_heads: int
    num_position_embeddings: int
    out_hidden_size: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int


class TextConfig(BaseModel):
    head_dim: int
    hidden_size: int
    hidden_act: str
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    attn_output_gate: bool
    vocab_size: int

    rms_norm_eps: float
    rope_parameters: RopeConfig

    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_value_head_dim: int

    layer_types: list[str]


class Qwen3_5Config(BaseModel):
    image_token_id: int
    model_type: str
    text_config: TextConfig
    tie_word_embeddings: bool
    video_token_id: int
    vision_config: VisionConfig
    vision_end_token_id: int
    vision_start_token_id: int


class FullAtention(nn.Module):
    def __init__(self, layer_idx: int, config: TextConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.q_proj = nn.Parameter(torch.empty(config.num_attention_heads * config.head_dim * 2, config.hidden_size))
        self.k_proj = nn.Parameter(torch.empty(config.num_key_value_heads * config.head_dim, config.hidden_size))
        self.v_proj = nn.Parameter(torch.empty(config.num_key_value_heads * config.head_dim, config.hidden_size))
        self.o_proj = nn.Parameter(torch.empty(config.hidden_size, config.num_attention_heads * config.head_dim))

        self.q_norm = nn.Parameter(torch.empty(config.head_dim))
        self.k_norm = nn.Parameter(torch.empty(config.head_dim))

    def load_weight(self, f):
        self.q_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.q_proj.weight"))
        self.k_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.k_proj.weight"))
        self.v_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.v_proj.weight"))
        self.o_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.o_proj.weight"))

        self.q_norm.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.q_norm.weight"))
        self.k_norm.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.k_norm.weight"))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: dict[str, torch.Tensor],
        is_prefill: bool,
    ):
        assert self.config.attn_output_gate, "attn_output_gate must be True for Qwen3.5 dense attention"
        num_attention_heads = self.config.num_attention_heads
        head_dim = self.config.head_dim
        num_key_value_heads = self.config.num_key_value_heads
        num_tokens = position.size(0)
        batch_size, seqlen, _ = hidden_states.shape
        q_gate = torch.einsum("bsh,oh->bso", hidden_states, self.q_proj.data)
        k = torch.einsum("bsh,oh->bso", hidden_states, self.k_proj.data)
        v = torch.einsum("bsh,oh->bso", hidden_states, self.v_proj.data)

        q_gate = q_gate.view(batch_size, seqlen, num_attention_heads, head_dim * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = k.view(batch_size, seqlen, num_key_value_heads, head_dim)
        v = v.view(batch_size, seqlen, num_key_value_heads, head_dim)

        rms_norm_eps = self.config.rms_norm_eps
        q = rms_norm(q, self.q_norm.data + 1.0, rms_norm_eps)
        k = rms_norm(k, self.k_norm.data + 1.0, rms_norm_eps)

        # Apply RoPE to q and k
        q = apply_rope(q.view(num_tokens, -1, head_dim), cos, sin).view(
            batch_size, seqlen, num_attention_heads, head_dim
        )
        k = apply_rope(k.view(num_tokens, -1, head_dim), cos, sin).view(
            batch_size, seqlen, num_key_value_heads, head_dim
        )

        # batch, head_cnt, seq_len, head_dim
        gate = gate.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        cache["key"][:, position, :] = k
        cache["value"][:, position, :] = v

        k = cache["key"][:, : position[-1] + 1, :]
        v = cache["value"][:, : position[-1] + 1, :]

        o = F.scaled_dot_product_attention(q, k, v, is_causal=is_prefill, enable_gqa=True)
        o = o * torch.sigmoid(gate)

        hidden_states = torch.einsum("bhsd,ohd->bso", o, self.o_proj.data.view(-1, num_attention_heads, head_dim))
        return hidden_states


class LinearAttention(nn.Module):
    def __init__(self, layer_idx: int, config: TextConfig):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        self.value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        self.activation = config.hidden_act
        self.conv_dim = self.key_dim * 2 + self.value_dim
        # Initialize A_log same as reference: uniform(0, 16) then log
        A = torch.empty(config.linear_num_value_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.conv1d = nn.Parameter(torch.empty(self.conv_dim, 1, config.linear_conv_kernel_dim))
        self.dt_biase = nn.Parameter(torch.ones(config.linear_num_value_heads))
        self.in_proj_a = nn.Parameter(torch.empty(config.linear_num_value_heads, config.hidden_size))
        self.in_proj_b = nn.Parameter(torch.empty(config.linear_num_value_heads, config.hidden_size))
        self.in_proj_qkv = nn.Parameter(
            torch.empty(sum([self.key_dim, self.key_dim, self.value_dim]), config.hidden_size)
        )
        self.in_proj_z = nn.Parameter(torch.empty(self.value_dim, config.hidden_size))
        self.norm = nn.Parameter(torch.empty(config.linear_value_head_dim, dtype=torch.float32))
        self.out_proj = nn.Parameter(torch.empty(config.hidden_size, self.value_dim))

    def load_weight(self, f):
        self.A_log.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.A_log"))
        self.conv1d.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.conv1d.weight"))
        self.dt_biase.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.dt_bias"))
        self.in_proj_a.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.in_proj_a.weight")
        )
        self.in_proj_b.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.in_proj_b.weight")
        )
        self.in_proj_qkv.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.in_proj_qkv.weight")
        )
        self.in_proj_z.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.in_proj_z.weight")
        )
        self.norm.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.norm.weight"))
        self.out_proj.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.linear_attn.out_proj.weight")
        )

    def forward(self, hidden_states: torch.Tensor, cache: dict[str, torch.Tensor], is_prefill: bool):
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "currently only support batch size == 1"
        mixed_qkv = torch.einsum("bsh, oh->bso", hidden_states, self.in_proj_qkv.data)

        """
        输入 T=5:
        原始: [x1, x2, x3, x4, x5]
        padding后: [0, 0, 0, x1, x2, x3, x4, x5]  ← 左边补3个0
                                 ↑
        conv输出长度 = 5 + 3 = 8
        [:, :, :5] 截断 → 取前5个，丢掉右边3个
        """
        mixed_qkv = mixed_qkv.transpose(1, 2)
        if is_prefill:
            cache["conv_state"].copy_(F.pad(mixed_qkv, (self.config.linear_conv_kernel_dim - mixed_qkv.shape[-1], 0)))
            mixed_qkv = F.silu(
                F.conv1d(
                    input=mixed_qkv,
                    weight=self.conv1d.data,
                    bias=None,
                    stride=1,
                    padding=self.config.linear_conv_kernel_dim - 1,
                    groups=self.conv_dim,
                )[:, :, :seq_len]
            )  # need to save the recent
        else:
            conv_state = cache["conv_state"]
            state_len = conv_state.shape[-1]
            hidden_states_new = torch.cat([conv_state, mixed_qkv], dim=-1)
            cache["conv_state"].copy_(hidden_states_new[:, :, -state_len:])
            mixed_qkv = F.silu(
                F.conv1d(
                    input=hidden_states_new,
                    weight=self.conv1d.data,
                    bias=None,
                    stride=1,
                    padding=0,
                    groups=self.conv_dim,
                )[:, :, -seq_len:]
            )
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.config.linear_key_head_dim)
        key = key.reshape(batch_size, seq_len, -1, self.config.linear_key_head_dim)
        value = value.reshape(batch_size, seq_len, -1, self.config.linear_value_head_dim)

        a = torch.einsum("bsh, oh->bso", hidden_states, self.in_proj_a.data)
        b = torch.einsum("bsh, oh->bso", hidden_states, self.in_proj_b.data)
        # Use float32 for A_log and computation (same as reference)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_biase.float())
        beta = b.sigmoid()

        z = torch.einsum("bsh, oh->bso", hidden_states, self.in_proj_z.data)
        z = z.reshape(batch_size, seq_len, -1, self.config.linear_value_head_dim)

        # ratio = self.config.linear_num_value_heads // self.config.linear_num_key_heads
        # if ratio > 1: 0.7B value and key share the same attention heads count
        #     query = query.repeat_interleave(ratio, dim=2)
        #     key = key.repeat_interleave(ratio, dim=2)

        # qk l2norm
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

        # recurrent gated delta rule
        # batch, seq, head_cnt, head_dim --> batch, head_cnt, seq, head_dim
        query = query.transpose(1, 2).contiguous().to(torch.float32)
        key = key.transpose(1, 2).contiguous().to(torch.float32)
        value = value.transpose(1, 2).contiguous().to(torch.float32)
        beta = beta.transpose(1, 2).contiguous().to(torch.float32)
        g = g.transpose(1, 2).contiguous().to(torch.float32)

        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        core_attn_out = torch.zeros_like(value)

        if is_prefill:
            cache["recurrent_state"].zero_()

        last_recurrent_state = cache["recurrent_state"]
        for i in range(seq_len):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            last_recurrent_state = last_recurrent_state * g_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

        cache["recurrent_state"].copy_(last_recurrent_state)
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(torch.bfloat16)

        core_attn_out = core_attn_out.reshape(-1, self.config.linear_value_head_dim)
        z = z.reshape(-1, self.config.linear_value_head_dim)

        core_attn_out = core_attn_out.to(torch.float32)
        variance = core_attn_out.pow(2).mean(-1, keepdim=True)
        core_attn_out = core_attn_out * torch.rsqrt(variance + 1e-6)
        core_attn_out = self.norm.data * core_attn_out.to(torch.bfloat16)
        core_attn_out = core_attn_out * F.silu(z.to(torch.float32))
        core_attn_out = core_attn_out.to(torch.bfloat16)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        hidden_states = torch.einsum("bsh, oh->bso", core_attn_out, self.out_proj.data)
        return hidden_states


class MLP(nn.Module):
    def __init__(self, layer_idx: int, config: TextConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.mlp_gate_proj = nn.Parameter(torch.empty(config.intermediate_size, config.hidden_size))
        self.mlp_up_proj = nn.Parameter(torch.empty(config.intermediate_size, config.hidden_size))
        self.mlp_down_proj = nn.Parameter(torch.empty(config.hidden_size, config.intermediate_size))

    def load_weight(self, f):
        self.mlp_gate_proj.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.mlp.gate_proj.weight")
        )
        self.mlp_up_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.mlp.up_proj.weight"))
        self.mlp_down_proj.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.mlp.down_proj.weight")
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states_gate = torch.einsum("bsh,oh->bso", hidden_states, self.mlp_gate_proj.data)
        hidden_states_up = torch.einsum("bsh,oh->bso", hidden_states, self.mlp_up_proj.data)
        hidden_states = F.silu(hidden_states_gate) * hidden_states_up
        hidden_states = torch.einsum("bsh, oh->bso", hidden_states, self.mlp_down_proj.data)
        return hidden_states


class Layer(nn.Module):
    def __init__(self, layer_idx: int, config: TextConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.config = config
        self.input_layernorm = nn.Parameter(torch.empty(config.hidden_size))
        self.post_attention_layernorm = nn.Parameter(torch.empty(config.hidden_size))

        if self.layer_type == "full_attention":
            self.self_attn = FullAtention(layer_idx, config)
        elif self.layer_type == "linear_attention":
            self.linear_attn = LinearAttention(layer_idx, config)
        self.mlp = MLP(layer_idx, config)

    def load_weight(self, f):
        self.input_layernorm.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.input_layernorm.weight")
        )
        self.post_attention_layernorm.data.copy_(
            f.get_tensor(f"model.language_model.layers.{self.layer_idx}.post_attention_layernorm.weight")
        )

        if self.layer_type == "full_attention":
            self.self_attn.load_weight(f)
        elif self.layer_type == "linear_attention":
            self.linear_attn.load_weight(f)
        self.mlp.load_weight(f)

    def forward(
        self,
        hidden_state: torch.Tensor,
        position: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: dict[str, torch.Tensor],
        is_prefill: bool,
    ):
        hidden_state, residual = add_rms_norm(
            hidden_state, residual, self.input_layernorm.data + 1.0, self.config.rms_norm_eps
        )
        if self.layer_type == "full_attention":
            hidden_state = self.self_attn(hidden_state, position, cos, sin, cache, is_prefill)
        elif self.layer_type == "linear_attention":
            hidden_state = self.linear_attn(hidden_state, cache, is_prefill)
        hidden_state, residual = add_rms_norm(
            hidden_state, residual, self.post_attention_layernorm.data + 1.0, self.config.rms_norm_eps
        )
        hidden_state = self.mlp(hidden_state)
        return hidden_state, residual


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, _freeze=True
        )
        self.layers = nn.ModuleList([Layer(layer_idx, config) for layer_idx in range(config.num_hidden_layers)])
        self.model_norm = nn.Parameter(torch.empty(config.hidden_size))

        # mRoPE inverse frequencies — use rotary_dim (partial), not full head_dim
        rotary_dim = int(config.head_dim * config.rope_parameters.partial_rotary_factor)
        rope_theta = config.rope_parameters.rope_theta
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = config.rope_parameters.mrope_section

    def load_weight(self, path):
        # currently only support single file
        logger.info("Model Loading...")
        with safe_open(path, "pt", "cpu") as f:
            self.embed_tokens.weight.copy_(f.get_tensor("model.language_model.embed_tokens.weight"))
            for i in tqdm(range(self.config.num_hidden_layers)):
                self.layers[i].load_weight(f)
            self.model_norm.data.copy_(f.get_tensor("model.language_model.norm.weight"))

        logger.info("Model Loaded")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hybrid_cache: list[dict[str, torch.Tensor]],
        is_prefill=False,
        vision_embed: torch.Tensor = None,
        vision_mask: torch.Tensor = None,
    ):
        rms_norm_eps = self.config.rms_norm_eps
        hidden_states = self.embed_tokens(input_ids)

        # Replace image pad tokens with vision embeddings
        if vision_embed is not None and vision_mask is not None:
            hidden_states[vision_mask] = vision_embed

        residual = None

        batch_size, seqlen, _ = hidden_states.shape
        assert batch_size == 1, "Currently only support singual request"

        # 1D positions for KV cache indexing (from text position channel)
        positions = position_ids[0, 0]  # (T,)

        # mRoPE cos/sin
        cos, sin = compute_mrope_cos_sin(position_ids, self.inv_freq, self.mrope_section)
        # cos, sin: (B, T, head_dim//2) → reshape for apply_rope: (T, 1, head_dim//2)
        cos = cos[0].unsqueeze(-2)  # (T, 1, head_dim//2)
        sin = sin[0].unsqueeze(-2)  # (T, 1, head_dim//2)

        for layer_idx in range(self.config.num_hidden_layers):
            hidden_states, residual = self.layers[layer_idx](
                hidden_states, positions, residual, cos, sin, hybrid_cache[layer_idx], is_prefill
            )

        hidden_states, _ = add_rms_norm(hidden_states, residual, self.model_norm.data + 1.0, rms_norm_eps)
        hidden_states = (hidden_states[:, -1, :]).squeeze(1)
        logits = torch.einsum("bh,vh->bv", hidden_states, self.embed_tokens.weight)
        return logits.argmax(dim=-1)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True)

    def load_weight(self, f, block_idx: int):
        self.qkv.weight.data.copy_(f.get_tensor(f"model.visual.blocks.{block_idx}.attn.qkv.weight"))
        self.qkv.bias.data.copy_(f.get_tensor(f"model.visual.blocks.{block_idx}.attn.qkv.bias"))
        self.proj.weight.data.copy_(f.get_tensor(f"model.visual.blocks.{block_idx}.attn.proj.weight"))
        self.proj.bias.data.copy_(f.get_tensor(f"model.visual.blocks.{block_idx}.attn.proj.bias"))

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor):
        seq_len = x.shape[0]
        q, k, v = (
            self.qkv(x)
            .reshape(seq_len, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # Block-diagonal attention mask from cu_seqlens
        attn_mask = torch.full(
            [1, seq_len, seq_len],
            torch.finfo(q.dtype).min,
            device=q.device,
            dtype=q.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attn_mask[
                ...,
                cu_seqlens[i - 1] : cu_seqlens[i],
                cu_seqlens[i - 1] : cu_seqlens[i],
            ] = 0

        # (n_heads, seq_len, head_dim) → (seq_len, n_heads, head_dim)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, -1)
        return self.proj(attn_output)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=True)

    def load_weight(self, f, block_idx: int):
        self.linear_fc1.weight.data.copy_(f.get_tensor(f"model.visual.blocks.{block_idx}.mlp.linear_fc1.weight"))
        self.linear_fc1.bias.data.copy_(f.get_tensor(f"model.visual.blocks.{block_idx}.mlp.linear_fc1.bias"))
        self.linear_fc2.weight.data.copy_(f.get_tensor(f"model.visual.blocks.{block_idx}.mlp.linear_fc2.weight"))
        self.linear_fc2.bias.data.copy_(f.get_tensor(f"model.visual.blocks.{block_idx}.mlp.linear_fc2.bias"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class VisionBlock(nn.Module):
    def __init__(self, block_idx: int, config: VisionConfig):
        super().__init__()
        self.block_idx = block_idx
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def load_weight(self, f):
        self.norm1.weight.data.copy_(f.get_tensor(f"model.visual.blocks.{self.block_idx}.norm1.weight"))
        self.norm1.bias.data.copy_(f.get_tensor(f"model.visual.blocks.{self.block_idx}.norm1.bias"))
        self.norm2.weight.data.copy_(f.get_tensor(f"model.visual.blocks.{self.block_idx}.norm2.weight"))
        self.norm2.bias.data.copy_(f.get_tensor(f"model.visual.blocks.{self.block_idx}.norm2.bias"))
        self.attn.load_weight(f, self.block_idx)
        self.mlp.load_weight(f, self.block_idx)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor):
        x = x + self.attn(self.norm1(x), cu_seqlens, rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x

class VisionPatchMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size, bias=True)

    def load_weight(self, f):
        self.norm.weight.data.copy_(f.get_tensor("model.visual.merger.norm.weight"))
        self.norm.bias.data.copy_(f.get_tensor("model.visual.merger.norm.bias"))
        self.linear_fc1.weight.data.copy_(f.get_tensor("model.visual.merger.linear_fc1.weight"))
        self.linear_fc1.bias.data.copy_(f.get_tensor("model.visual.merger.linear_fc1.bias"))
        self.linear_fc2.weight.data.copy_(f.get_tensor("model.visual.merger.linear_fc2.weight"))
        self.linear_fc2.bias.data.copy_(f.get_tensor("model.visual.merger.linear_fc2.bias"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, hidden_size) where N = grid_t * grid_h * grid_w
        x = self.norm(x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3_5VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        kernel_size = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.patch_embed = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([VisionBlock(i, config) for i in range(config.depth)])
        self.merger = VisionPatchMerger(config)

    def load_weight(self, path):
        logger.info("Vision Model Loading...")
        with safe_open(path, "pt", "cpu") as f:
            self.patch_embed.weight.data.copy_(f.get_tensor("model.visual.patch_embed.proj.weight"))
            self.patch_embed.bias.data.copy_(f.get_tensor("model.visual.patch_embed.proj.bias"))
            self.pos_embed.weight.data.copy_(f.get_tensor("model.visual.pos_embed.weight"))
            for block in self.blocks:
                block.load_weight(f)
            self.merger.load_weight(f)
        logger.info("Vision Model Loaded")

    def fast_pos_embed_interpolate(self, d_image: torch.Tensor) -> torch.Tensor:
        """Interpolate learned position embeddings to match image dimensions."""
        grid_ts, grid_hs, grid_ws = d_image[:, 0], d_image[:, 1], d_image[:, 2]
        device = d_image.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        merge_size = self.spatial_merge_size
        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return torch.cat(patch_pos_embeds_permute, dim=0)

    def rot_pos_emb(self, d_image: torch.Tensor) -> torch.Tensor:
        """Generate 2D RoPE frequencies for spatial positions after merge."""
        pos_ids = []
        sms = self.spatial_merge_size

        for t, h, w in d_image:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.view(h // sms, sms, w // sms, sms).transpose(1, 2).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.view(h // sms, sms, w // sms, sms).transpose(1, 2).flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = d_image[:, 1:].max()

        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    @torch.no_grad()
    def forward(self, pixels: torch.Tensor, d_image: torch.Tensor) -> torch.Tensor:
        # pixels: (N, C*T*H*W), each row is a patch (C, temporal_patch_size, patch_size, patch_size)
        x = pixels.view(-1, self.config.in_channels, self.config.temporal_patch_size, self.config.patch_size, self.config.patch_size)
        hidden_states = self.patch_embed(x).view(-1, self.config.hidden_size)

        # Add learnable position embeddings
        pos_embeds = self.fast_pos_embed_interpolate(d_image)
        hidden_states = hidden_states + pos_embeds

        # Generate RoPE for 2D spatial positions
        rotary_pos_emb = self.rot_pos_emb(d_image)

        # Build cumulative sequence lengths for block-diagonal attention mask
        cu_seqlens = torch.repeat_interleave(d_image[:, 1] * d_image[:, 2], d_image[:, 0])
        cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens, rotary_pos_emb)

        return self.merger(hidden_states)

