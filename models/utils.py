"""Shared utility functions for model implementations."""

import torch


def add_rms_norm(x, residual, weight, eps) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RMS normalization with residual connection.
    
    Args:
        x: Input tensor
        residual: Residual tensor (None for first layer)
        weight: RMS norm weight parameter
        eps: Small constant for numerical stability
    
    Returns:
        Tuple of (normalized output, residual for next layer)
    """
    orig_dtype = x.dtype
    if residual is not None:
        x += residual
    residual = x
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x, residual


def rms_norm(x, weight, eps) -> torch.Tensor:
    """Apply RMS normalization without residual connection.
    
    Args:
        x: Input tensor
        weight: RMS norm weight parameter
        eps: Small constant for numerical stability
    
    Returns:
        Normalized output tensor
    """
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    x = x.to(orig_dtype).mul_(weight)
    return x


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to input tensor.
    
    Args:
        x: Input tensor of shape (num_tokens, num_heads, head_dim)
        cos: Cosine values of shape (num_tokens, 1, head_dim//2)
        sin: Sine values of shape (num_tokens, 1, head_dim//2)
    
    Returns:
        Tensor with RoPE applied, same shape as input
    """
    orig_shape = x.shape
    head_dim = x.shape[-1]
    x = x.view(-1, orig_shape[-2], head_dim)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype).view(orig_shape)


def l2norm(x, dim=-1, eps=1e-6):
    """L2 normalization.
    
    This function is intended to align with the l2norm implementation in the FLA library.
    
    Args:
        x: Input tensor
        dim: Dimension to normalize along
        eps: Small constant for numerical stability
    
    Returns:
        L2 normalized tensor
    """
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm
