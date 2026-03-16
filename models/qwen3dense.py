import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from safetensors import safe_open
from tqdm import tqdm

from models.utils import add_rms_norm, apply_rope, rms_norm

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Qwen3")


class Qwen3Config(BaseModel):
    head_dim: int
    hidden_size: int
    hidden_act: str
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    vocab_size: int
    rope_theta: int
    rms_norm_eps: float


class Qwen3(nn.Module):
    def __init__(self, config:Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, _freeze=True)
        self.input_layernorm = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size)) for _ in range(config.num_hidden_layers)])

        self.q_projs = nn.ParameterList([nn.Parameter(torch.empty(config.num_attention_heads*config.head_dim, config.hidden_size)) for _ in range(config.num_hidden_layers)])
        self.k_projs = nn.ParameterList([nn.Parameter(torch.empty(config.num_key_value_heads*config.head_dim, config.hidden_size)) for _ in range(config.num_hidden_layers)])
        self.v_projs = nn.ParameterList([nn.Parameter(torch.empty(config.num_key_value_heads*config.head_dim, config.hidden_size)) for _ in range(config.num_hidden_layers)])
        self.o_projs = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size, config.num_attention_heads*config.head_dim)) for _ in range(config.num_hidden_layers)])

        self.q_norm = nn.ParameterList([nn.Parameter(torch.empty(config.head_dim)) for _ in range(config.num_hidden_layers)])
        self.k_norm = nn.ParameterList([nn.Parameter(torch.empty(config.head_dim)) for _ in range(config.num_hidden_layers)])

        self.post_attention_norm = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size)) for _ in range(config.num_hidden_layers)])

        self.mlp_gate_proj = nn.ParameterList([nn.Parameter(torch.empty(config.intermediate_size, config.hidden_size)) for _ in range(config.num_hidden_layers)])
        self.mlp_up_proj = nn.ParameterList([nn.Parameter(torch.empty(config.intermediate_size, config.hidden_size)) for _ in range(config.num_hidden_layers)])
        self.mlp_down_proj = nn.ParameterList([nn.Parameter(torch.empty(config.hidden_size, config.intermediate_size)) for _ in range(config.num_hidden_layers)])

        self.model_norm = nn.Parameter(torch.empty(config.hidden_size))
        self.lm_head = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))


    def load_weight(self, path):
        # currently only support single file
        logger.info("Model Loading...")
        with safe_open(path, "pt", "cpu") as f:
            self.embed_tokens.weight.copy_(f.get_tensor("model.embed_tokens.weight"))
            self.lm_head.data.copy_(f.get_tensor("lm_head.weight"))
            for i in tqdm(range(self.config.num_hidden_layers)):
                self.input_layernorm[i].data.copy_(f.get_tensor(f"model.layers.{i}.input_layernorm.weight"))

                self.q_projs[i].data.copy_(f.get_tensor(f"model.layers.{i}.self_attn.q_proj.weight"))
                self.k_projs[i].data.copy_(f.get_tensor(f"model.layers.{i}.self_attn.k_proj.weight"))
                self.v_projs[i].data.copy_(f.get_tensor(f"model.layers.{i}.self_attn.v_proj.weight"))
                self.o_projs[i].data.copy_(f.get_tensor(f"model.layers.{i}.self_attn.o_proj.weight"))

                self.q_norm[i].data.copy_(f.get_tensor(f"model.layers.{i}.self_attn.q_norm.weight"))
                self.k_norm[i].data.copy_(f.get_tensor(f"model.layers.{i}.self_attn.k_norm.weight"))

                self.post_attention_norm[i].data.copy_(f.get_tensor(f"model.layers.{i}.post_attention_layernorm.weight"))

                self.mlp_gate_proj[i].data.copy_(f.get_tensor(f"model.layers.{i}.mlp.gate_proj.weight"))
                self.mlp_up_proj[i].data.copy_(f.get_tensor(f"model.layers.{i}.mlp.up_proj.weight"))
                self.mlp_down_proj[i].data.copy_(f.get_tensor(f"model.layers.{i}.mlp.down_proj.weight"))

            self.model_norm.data.copy_(f.get_tensor("model.norm.weight"))
            self.lm_head.data.copy_(f.get_tensor("lm_head.weight"))

        logger.info("Model Loaded")

    @torch.no_grad()
    def forward(self, input_ids:torch.Tensor, positions:torch.Tensor, kv_cache:torch.Tensor, cos_sin_cache:torch.Tensor, is_prefill=False):
        rms_norm_eps = self.config.rms_norm_eps
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        batch_size, seqlen, _ = hidden_states.shape
        assert batch_size == 1, "Currently only support singual request"
        head_dim = self.config.head_dim
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads

        # Apply RoPE
        num_tokens = positions.size(0)
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        for i in range(self.config.num_hidden_layers):
            # Input Norm
            hidden_states, residual = add_rms_norm(hidden_states, residual, self.input_layernorm[i].data, rms_norm_eps)

            # Attention
            q = torch.einsum('bsh,oh->bso', hidden_states, self.q_projs[i].data)
            k = torch.einsum('bsh,oh->bso', hidden_states, self.k_projs[i].data)
            v = torch.einsum('bsh,oh->bso', hidden_states, self.v_projs[i].data)
            q = q.view(batch_size, seqlen, num_attention_heads, head_dim)
            k = k.view(batch_size, seqlen, num_key_value_heads, head_dim)
            v = v.view(batch_size, seqlen, num_key_value_heads, head_dim)

            q = rms_norm(q, self.q_norm[i].data, rms_norm_eps)
            k = rms_norm(k, self.k_norm[i].data, rms_norm_eps)

            # Apply RoPE to q and k
            q = apply_rope(q.view(num_tokens, -1, head_dim), cos, sin).view(batch_size, seqlen, num_attention_heads, head_dim)
            k = apply_rope(k.view(num_tokens, -1, head_dim), cos, sin).view(batch_size, seqlen, num_key_value_heads, head_dim)

            # batch, head_cnt, seq_len, head_dim
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            kv_cache[i][0][:, positions, :] = k
            kv_cache[i][1][:, positions, :] = v

            k = kv_cache[i][0][:, :positions[-1]+1, :]
            v = kv_cache[i][1][:, :positions[-1]+1, :]

            o = F.scaled_dot_product_attention(q, k, v, is_causal=is_prefill, enable_gqa=True)

            hidden_states = torch.einsum('bhsd,ohd->bso', o, self.o_projs[i].data.view(-1, num_attention_heads, head_dim))

            # Post Attention Norm
            hidden_states, residual = add_rms_norm(hidden_states, residual, self.post_attention_norm[i].data, rms_norm_eps)

            # MLP
            hidden_states_gate = torch.einsum('bsh,oh->bso', hidden_states, self.mlp_gate_proj[i].data)
            hidden_states_up = torch.einsum('bsh,oh->bso', hidden_states, self.mlp_up_proj[i].data)
            hidden_states = F.silu(hidden_states_gate) * hidden_states_up
            hidden_states = torch.einsum('bsh, oh->bso', hidden_states, self.mlp_down_proj[i].data)

        hidden_states, _ = add_rms_norm(hidden_states, residual, self.model_norm.data, rms_norm_eps)

        # Compute Logits
        hidden_states = (hidden_states[:, -1, :]).squeeze(1)
        logits = torch.einsum('bh,vh->bv', hidden_states, self.lm_head.data)

        # greedy sample
        return logits.argmax(dim=-1)
