"""Shared utility functions for model implementations."""

import numpy as np
import torch
from PIL import Image


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

    Supports partial RoPE where only the first rope_dim dimensions are rotated.

    Args:
        x: Input tensor of shape (num_tokens, num_heads, head_dim)
        cos: Cosine values of shape (num_tokens, 1, rope_dim//2)
        sin: Sine values of shape (num_tokens, 1, rope_dim//2)

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    orig_shape = x.shape
    head_dim = x.shape[-1]
    rope_dim = cos.shape[-1] * 2  # cos has shape (..., rope_dim//2)

    x = x.view(-1, orig_shape[-2], head_dim)

    # Split into rope part and non-rope part
    if rope_dim < head_dim:
        # Partial RoPE: only apply to first rope_dim dimensions
        x_rope, x_pass = x[..., :rope_dim], x[..., rope_dim:]
    else:
        # Full RoPE: apply to all dimensions
        x_rope, x_pass = x, None

    # Apply RoPE to rope part
    x1, x2 = torch.chunk(x_rope.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    x_rope = torch.cat((y1, y2), dim=-1).to(x.dtype)

    # Concatenate rope part with non-rope part (if any)
    if x_pass is not None:
        out = torch.cat((x_rope, x_pass), dim=-1)
    else:
        out = x_rope

    return out.view(orig_shape)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half: [x1, x2] -> [-x2, x1]. Used in vision model RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE for vision model.

    Args:
        tensor: (batch, seq_len, n_heads, head_dim)
        freqs: (seq_len, head_dim // 2)
    """
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = freqs.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


def compute_mrope_cos_sin(position_ids: torch.Tensor, inv_freq: torch.Tensor, mrope_section: list[int]):
    """Compute mRoPE cos/sin for multimodal inputs.

    Args:
        position_ids: (3, B, T) - T/H/W position channels
        inv_freq: (head_dim//2,) - inverse frequencies computed from rope_theta
        mrope_section: [int, int, int] - number of freq components for T/H/W

    Returns:
        cos, sin of shape (B, T, head_dim//2)
    """
    # freqs: (3, B, T, head_dim//2)
    inv_freq_exp = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
    pos_exp = position_ids[:, :, None, :].float()
    freqs = (inv_freq_exp @ pos_exp).transpose(2, 3)

    # Interleave T/H/W: start with T, place H at offset 1, W at offset 2 (stride 3)
    freqs_out = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_out[..., idx] = freqs[dim][..., idx]

    cos = freqs_out.cos()
    sin = freqs_out.sin()
    return cos, sin


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


def process_image(
    image: Image.Image,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int,
    max_pixels: int,
    min_pixels: int,
    image_mean,
    image_std
):
    '''
    +---------------+     +---------------+
    | FRAME 1       |     | FRAME 2       |
    | (Blue)        |     | (Green)       |
    | +-----+-----+ |     | +-----+-----+ |
    | | P₁  | P₂  | |     | | P₁  | P₂  | |
    | +-----+-----+ |     | +-----+-----+ |
    | | P₃  | P₄  | |     | | P₃  | P₄  | |
    | +-----+-----+ |     | +-----+-----+ |
    +---------------+     +---------------+
        |                     |
        +----------+----------+
                    |
                    V
    [P₁,P₁] [P₂,P₂] [P₃,P₃] [P₄,P₄]
    (1st)  (2nd)  (3rd)  (4th) <- Interleaved Frame Patches
                                         (Temporal grouping)
     then during the Vision Tower these 8 patch will finally merged into one token
     merger temporal (frame1 and frame2) first then merge spectial
    '''

    image_np = np.array(image, dtype=np.float32)
    height, width = image_np.shape[:2]

    factor = spatial_merge_size * patch_size

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = max(factor, int(np.floor(height / beta / factor) * factor))
        w_bar = max(factor, int(np.floor(width / beta / factor) * factor))
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = int(np.ceil(height * beta / factor) * factor)
        w_bar = int(np.ceil(width * beta / factor) * factor)

    image_resized = image.resize((w_bar, h_bar), resample=Image.BICUBIC)
    image_np_resized = np.array(image_resized, dtype=np.float32)

    # Normalize
    image_np_resized = image_np_resized / 255.0
    image_np_resized = (image_np_resized - image_mean) / image_std

    # Convert to channels-first and add batch dimension
    image_np_resized = np.transpose(image_np_resized, (2, 0, 1))
    image_np_resized = image_np_resized[np.newaxis, ...]

    # duplicate temporal dimension
    image_np_resized = np.tile(image_np_resized, (temporal_patch_size, 1, 1, 1))

    # Extract patches
    batch_size, channels, height, width = image_np_resized.shape
    grid_t = batch_size // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size

    patches = image_np_resized.reshape(
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // spatial_merge_size,
        spatial_merge_size,
        patch_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
        patch_size,
    )

    # ── transpose：把"位置索引"提到前面，"像素数据"移到后面 ──
    # 原轴：0=gt  1=Tp  2=C  3=gh//m  4=m  5=sp  6=gw//m  7=m  8=sp
    # 新轴：0=gt  1=gh//m  2=gw//m  3=m  4=m  5=C  6=Tp  7=sp  8=sp
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1 , 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size
    ).astype(np.float32)
    return flatten_patches, grid_t, grid_h, grid_w
