import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm

from models.utils import add_rms_norm, apply_rope, l2norm, rms_norm

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger("Qwen3_5")


@dataclass
class RopeConfig:
    mrope_interleaved: bool
    mrope_section: list[int]
    rope_type: str
    rope_theta: int
    partial_rotary_factor: float


@dataclass
class Qwen3_5Config:
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
    # rope_theta: int
    rms_norm_eps: float
    rope_parameters: RopeConfig

    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_value_head_dim: int

    layer_types: list[str]


class FullAtention(nn.Module):
    def __init__(self, layer_idx: int, config: Qwen3_5Config):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.q_proj = nn.Parameter(
            torch.empty(
                config.num_attention_heads * config.head_dim * (1 + config.attn_output_gate), config.hidden_size
            )
        )
        self.k_proj = nn.Parameter(torch.empty(config.num_key_value_heads * config.head_dim, config.hidden_size))
        self.v_proj = nn.Parameter(torch.empty(config.num_key_value_heads * config.head_dim, config.hidden_size))
        self.o_proj = nn.Parameter(torch.empty(config.hidden_size, config.num_attention_heads * config.head_dim))

        self.q_norm = nn.Parameter(torch.empty(config.head_dim))
        self.k_norm = nn.Parameter(torch.empty(config.head_dim))

        self.register_buffer(
            "kv_cache", torch.zeros(2, config.num_key_value_heads, config.max_position_embeddings, config.head_dim)
        )

    def load_weight(self, f):
        self.q_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.q_proj.weight"))
        self.k_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.k_proj.weight"))
        self.v_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.v_proj.weight"))
        self.o_proj.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.o_proj.weight"))

        self.q_norm.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.q_norm.weight"))
        self.k_norm.data.copy_(f.get_tensor(f"model.language_model.layers.{self.layer_idx}.self_attn.k_norm.weight"))

    def forward(self, hidden_states: torch.Tensor, position: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_prefill:bool):
        assert self.config.attn_output_gate, "attn_output_gate must be True for Qwen3.5 dense attention"
        num_attention_heads = self.config.num_attention_heads
        head_dim = self.config.head_dim
        num_key_value_heads = self.config.num_key_value_heads
        num_tokens = position.size(0)
        batch_size, seqlen, _ = hidden_states.shape
        q_gate = torch.einsum('bsh,oh->bso', hidden_states, self.q_proj.data)
        k = torch.einsum('bsh,oh->bso', hidden_states, self.k_proj.data)
        v = torch.einsum('bsh,oh->bso', hidden_states, self.v_proj.data)

        q, gate = torch.chunk(q_gate, 2, dim=-1)
        gate = gate.view(batch_size, seqlen, num_attention_heads, head_dim)
        q = q.view(batch_size, seqlen, num_attention_heads, head_dim)
        k = k.view(batch_size, seqlen, num_key_value_heads, head_dim)
        v = v.view(batch_size, seqlen, num_key_value_heads, head_dim)

        rms_norm_eps = self.config.rms_norm_eps
        q = rms_norm(q, self.q_norm.data, rms_norm_eps)
        k = rms_norm(k, self.k_norm.data, rms_norm_eps)

        # Apply RoPE to q and k
        q = apply_rope(q.view(num_tokens, -1, head_dim), cos, sin).view(batch_size, seqlen, num_attention_heads, head_dim)
        k = apply_rope(k.view(num_tokens, -1, head_dim), cos, sin).view(batch_size, seqlen, num_key_value_heads, head_dim)

        # batch, head_cnt, seq_len, head_dim
        gate = gate.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        self.kv_cache[0][:, position, :] = k
        self.kv_cache[1][:, position, :] = v

        k = self.kv_cache[0][:, :position[-1]+1, :]
        v = self.kv_cache[1][:, :position[-1]+1, :]

        o = F.scaled_dot_product_attention(q, k, v, is_causal=is_prefill, enable_gqa=True)
        o = o * torch.sigmoid(gate)

        hidden_states = torch.einsum('bhsd,ohd->bso', o, self.o_proj.data.view(-1, num_attention_heads, head_dim))

        return hidden_states



class LinearAttention(nn.Module):
    def __init__(self, layer_idx: int, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        self.value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        self.activation = config.hidden_act
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.A_log = nn.Parameter(torch.empty(config.linear_num_value_heads, dtype=torch.float32))
        self.conv1d = nn.Parameter(torch.empty(self.conv_dim, 1, config.linear_conv_kernel_dim))
        self.dt_biase = nn.Parameter(torch.ones(config.linear_num_value_heads))
        self.in_proj_a = nn.Parameter(torch.empty(config.linear_num_value_heads, config.hidden_size))
        self.in_proj_b = nn.Parameter(torch.empty(config.linear_num_value_heads, config.hidden_size))
        self.in_proj_qkv = nn.Parameter(torch.empty(sum([self.key_dim, self.key_dim, self.value_dim]), config.hidden_size))
        self.in_proj_z = nn.Parameter(torch.empty(self.value_dim, config.hidden_size))
        self.norm = nn.Parameter(torch.empty(config.linear_value_head_dim, dtype=torch.float32))
        self.out_proj = nn.Parameter(torch.empty(config.hidden_size, self.value_dim))

        self.register_buffer(
            "recurrent_state", torch.zeros(1, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim)
        )
        self.register_buffer(
            "conv_state", torch.zeros(1, self.conv_dim, config.linear_conv_kernel_dim)
        )

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

    def forward(self, hidden_states: torch.Tensor, is_prefill:bool):
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "currently only support batch size == 1"
        mixed_qkv = torch.einsum('bsh, oh->bso', hidden_states, self.in_proj_qkv.data)

        '''
        输入 T=5:
        原始: [x1, x2, x3, x4, x5]
        padding后: [0, 0, 0, x1, x2, x3, x4, x5]  ← 左边补3个0
                                 ↑
        conv输出长度 = 5 + 3 = 8
        [:, :, :5] 截断 → 取前5个，丢掉右边3个
        '''
        mixed_qkv = mixed_qkv.transpose(1, 2)
        if is_prefill:
            self.conv_state = F.pad(mixed_qkv, (self.config.linear_conv_kernel_dim - mixed_qkv.shape[-1], 0))
            mixed_qkv = F.silu(
                F.conv1d(
                    input=mixed_qkv,
                    weight=self.conv1d.data, bias=None, stride=1,
                    padding=self.config.linear_conv_kernel_dim-1,
                    groups=self.conv_dim
                )[:, :, :seq_len]
            )   # need to save the recent
        else:
            conv_state = self.conv_state
            state_len = conv_state.shape[-1]
            hidden_states_new = torch.cat([conv_state, mixed_qkv], dim=-1)
            self.conv_state.copy_(hidden_states_new[:, :, -state_len:])
            mixed_qkv = F.silu(
                F.conv1d(
                    input=hidden_states_new,
                    weight=self.conv1d.data, bias=None, stride=1,
                    padding=0,
                    groups=self.conv_dim
                )[:, :, -seq_len:]
            )
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.config.linear_key_head_dim)
        key = key.reshape(batch_size, seq_len, -1, self.config.linear_key_head_dim)
        value = value.reshape(batch_size, seq_len, -1, self.config.linear_value_head_dim)

        a = torch.einsum('bsh, oh->bso', hidden_states, self.in_proj_a.data)
        b = torch.einsum('bsh, oh->bso', hidden_states, self.in_proj_b.data)
        g = -self.A_log.data.exp() * F.softplus(a.float() + self.dt_biase.data)
        beta = b.sigmoid()

        z = torch.einsum('bsh, oh->bso', hidden_states, self.in_proj_z.data)
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
            self.recurrent_state.zero_()

        last_recurrent_state = self.recurrent_state
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

        self.recurrent_state.copy_(last_recurrent_state)
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

        hidden_states = torch.einsum('bsh, oh->bso', core_attn_out, self.out_proj.data)
        return hidden_states


class MLP(nn.Module):
    def __init__(self, layer_idx: int, config: Qwen3_5Config):
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
    def __init__(self, layer_idx: int, config: Qwen3_5Config):
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

    def forward(self, hidden_state: torch.Tensor, position: torch.Tensor, residual: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, is_prefill:bool):
        hidden_state, residual = add_rms_norm(
            hidden_state, residual, self.input_layernorm.data, self.config.rms_norm_eps
        )
        if self.layer_type == "full_attention":
            hidden_state = self.self_attn(hidden_state, position, cos, sin, is_prefill)
        elif self.layer_type == "linear_attention":
            hidden_state = self.linear_attn(hidden_state, is_prefill)
        hidden_state, residual = add_rms_norm(
            hidden_state, residual, self.post_attention_layernorm.data, self.config.rms_norm_eps
        )
        hidden_state = self.mlp(hidden_state)
        return hidden_state, residual


class Qwen3_5(nn.Module):
    def __init__(self, config: Qwen3_5Config, device="cpu"):
        super().__init__()
        torch.set_default_device(device)
        self.config = config
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, _freeze=True
        )
        self.layers = nn.ModuleList([Layer(layer_idx, config) for layer_idx in range(config.num_hidden_layers)])
        self.model_norm = nn.Parameter(torch.empty(config.hidden_size))

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
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, is_prefill=False):
        rms_norm_eps = self.config.rms_norm_eps
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        batch_size, seqlen, _ = hidden_states.shape
        assert batch_size == 1, "Currently only support singual request"
        # get cos and sin
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        for layer_idx in range(self.config.num_hidden_layers):
            hidden_states, residual = self.layers[layer_idx](hidden_states, positions, residual, cos, sin, is_prefill)

        hidden_states, _ = add_rms_norm(hidden_states, residual, self.model_norm.data, rms_norm_eps)
        hidden_states = (hidden_states[:, -1, :]).squeeze(1)
        logits = torch.einsum("bh,vh->bv", hidden_states, self.embed_tokens.weight)

        return logits.argmax(dim=-1)
