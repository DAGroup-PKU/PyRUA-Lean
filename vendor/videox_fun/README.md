# VideoX-Fun for the H3 control hybrid

The control-hybrid worker (`src/datafarm/render/h3/h3_control_worker.py`) imports VideoX-Fun's
`MiniMaxH3ControlPipeline` with local changes (Ref2VA + ControlNet-Union hybrid, xfuser fixes, soft-HED control
tool). Reproduce that checkout and point the `H3_VIDEOX_FUN` binding at it:

```bash
git clone https://github.com/aigc-apps/VideoX-Fun.git videox-fun && cd videox-fun
git checkout -b datafarm-h3-control 6f3fb60dad9b6a60ff6f962e62cffa11cafb084b
git apply <datafarm>/vendor/videox_fun/videox-fun-datafarm.patch
mkdir -p models/annotators && cp <ControlNetHED.pth> models/annotators/
export H3_VIDEOX_FUN=$PWD
```

`videox-fun-datafarm.patch` is the single diff from upstream to the checkout the hybrid uses.
`config/minimax_h3/minimax_h3_control.yaml` is upstream and unchanged.

The HED tool in the patched checkout, `<checkout>/tools/prepare_hed_control.py`, also needs the ControlNet v1.1 annotator
package (`annotator/hed/__init__.py` with `ControlNetHED_Apache2`, and `annotator/util.py`; Apache-2.0). Set
`H3_HED_ANNOTATOR_REPO` to the directory that contains `<annotator-repo>/tools/controlnet/annotator/`; the tool exits with a clear
message when it is unset.

Licences: VideoX-Fun Apache-2.0; MiniMax-H3 weights and outputs are
governed by the MiniMax-H3 LICENSE (see `docs/reference/rendering.md`).
