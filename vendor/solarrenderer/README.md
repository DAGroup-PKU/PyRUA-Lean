# SolarRenderer deployment helpers

The SolarRenderer route (`src/datafarm/render/solar/`) runs a
[SolarRenderer](https://github.com/Enderfga/SolarRenderer) checkout at the commit pinned in
`configs/external-assets.json`. SolarRenderer is Apache-2.0. It is not included here. With flash-attn installed, the
checkout needs no change.

SolarRenderer imports `flash_attn`. Where flash-attn cannot be built for the installed torch (for example torch 2.11 +
cu128 on L20X), two optional pieces let it run on torch SDPA instead:

| File | What it does |
|---|---|
| `flash_attn_sdpa/flash_attn/__init__.py` | A `flash_attn_func` that calls `torch.nn.functional.scaled_dot_product_attention`. It covers only the forward pass SolarRenderer inference uses. Put `flash_attn_sdpa/` on `PYTHONPATH`. |
| `solarrenderer-attention-env.patch` | A one-line change to `src/solarwm/backends/minimax_h3/optional.py`. The environment variable `SOLAR_ATTN_BACKEND` overrides the transformer attention backend; set it to `native`. |

```bash
cd "$SOLAR_REPO" && git apply /path/to/datafarm/vendor/solarrenderer/solarrenderer-attention-env.patch
export PYTHONPATH=/path/to/datafarm/vendor/solarrenderer/flash_attn_sdpa SOLAR_ATTN_BACKEND=native
```

A 447-frame window on an L20X (140 GiB) reserved 129 GiB at its peak and ran at 2.2 s per frame, the same speed as on a
B200 with flash-attn. With flash-attn installed, neither piece is needed.
