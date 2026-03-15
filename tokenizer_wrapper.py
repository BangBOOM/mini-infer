"""Minimal tokenizer wrapper using tokenizers + jinja2 for chat templates."""

import json
import os
from typing import List, Union

from jinja2 import Template
from tokenizers import Tokenizer


class ChatTokenizer:
    """Wrapper around tokenizers.Tokenizer with jinja2 chat template support."""

    def __init__(self, model_path: str):
        """Initialize from a HuggingFace model directory.
        
        Args:
            model_path: Path to the model directory containing tokenizer.json and tokenizer_config.json
        """
        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        
        # Load tokenizer config for special tokens and chat template
        config_path = os.path.join(model_path, "tokenizer_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        # Get special tokens
        self.eos_token = self.config.get("eos_token", "<|endoftext|>")
        self.eos_token_id = self.config.get("eos_token_id")
        if self.eos_token_id is None:
            # Try to encode eos_token to get its id
            self.eos_token_id = self.tokenizer.encode(self.eos_token).ids[0]
        
        # Load chat template from config
        chat_template = self.config.get("chat_template")
        if chat_template:
            self.template = Template(chat_template)
        else:
            self.template = None

    def apply_chat_template(
        self,
        messages: List[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
    ) -> str:
        """Apply chat template to messages using jinja2.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tokenize: If True, return token ids (not implemented, returns string)
            add_generation_prompt: Whether to add the assistant start prompt
            enable_thinking: Whether to enable thinking mode (model-specific)
        
        Returns:
            Formatted prompt string
        """
        if self.template is None:
            raise ValueError("No chat template found in tokenizer config")
        
        rendered = self.template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        return rendered

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: Union[int, List[int], "torch.Tensor"], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text.
        
        Args:
            token_ids: Single token id, list of token ids, or torch tensor
            skip_special_tokens: Whether to skip special tokens in output
        """
        # Handle torch tensor
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, texts: Union[str, List[str]], return_tensors: str = None):
        """Encode text(s) for model input.
        
        Args:
            texts: Single string or list of strings to encode
            return_tensors: If "pt", return PyTorch tensors
        
        Returns:
            Dict with "input_ids" key containing encoded texts
        """
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode all texts
        encoded = [self.encode(text) for text in texts]
        
        # Convert to tensor (pad to same length)
        max_len = max(len(ids) for ids in encoded)
        padded = []
        for ids in encoded:
            # Pad with 0 (typically pad token id)
            padded.append(ids + [0] * (max_len - len(ids)))
        
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(padded)}
        return {"input_ids": padded}
