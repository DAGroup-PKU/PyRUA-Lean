"""Stand-in for flash_attn when flash-attn cannot be built for the installed torch (DataFarm, MIT).

flash_attn_func on [B, S, H, D] through torch scaled_dot_product_attention (its own flash / cuDNN kernels on sm_90).
Only the forward SolarRenderer inference uses: no dropout, optional causal, optional GQA. Put the parent directory on
PYTHONPATH and set SOLAR_ATTN_BACKEND=native with solarrenderer-attention-env.patch applied; see ../README.md.
"""
import torch.nn.functional as F

__version__ = "0.0-sdpa-shim"


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, deterministic=False, **_):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0,
                                         is_causal=causal, scale=softmax_scale, enable_gqa=q.shape[2] != k.shape[2])
    return out.transpose(1, 2)
