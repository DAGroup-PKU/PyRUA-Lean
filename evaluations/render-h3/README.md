# H3 render evaluator (EVAL-v1)

Evaluator assets for the H3 control-hybrid renderer (`docs/reference/rendering.md`). They measure candidates and
are protected from automatic modification (`evaluations/` is in `src/datafarm/evolution/controller.py` PROTECTED).

- `h3eval.py` — automatic scorer (`realpool`, `score`, `pack`, `table`): motion/structure correlation against the
  white carrier, KID/nn-cos/Lab against a real-camera pool, jumps/flips/freezes. Renderer-agnostic (H3 and
  Seedance are scored by the same code). Pass `--dino` or set `H3_EVAL_DINO` (DINOv2-base directory); there is
  no built-in default.
- `reviewer-instructions-v1.md` — instructions for the independent blind reviewer (acceptance authority).
- `prompts/` — producer-side seed-screen prompt (`seed-screen.txt`; a screen is NOT acceptance) and the
  blind-review agent prompt. They assume the dual-Piper T-layout rig (head top, wrists bottom).

Changing a scoring rule or the reviewer instructions is an evaluator revision (new version + independent
calibration), never part of a renderer candidate. Pixel metrics do not replace the blind review: they miss
content errors (cables, extra objects, white sim-coloured arms) that the VLM screen and reviewer catch.
