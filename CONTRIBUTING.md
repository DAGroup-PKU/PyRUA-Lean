# Contributing

Issues and pull requests are welcome. A few rules keep results comparable with the paper:

- **Frozen parts.** The robots' primitives, perception, knowledge files, the actor's prompt and
  the benchmark's scoring are frozen: `tests/test_paper_prompts.py` fails when the prompt
  changes. A change there is a new experiment, not a fix; open an issue first.
- **No silent fallbacks.** A missing path, key or verdict raises; it never turns into a quiet
  default that makes a run look normal.
- **Offline first.** `bash tools/test_offline.sh` must pass (no simulator, GPU or model), and
  `python -m ruff check` must be clean. Add a test with fake episodes or fake robots for new
  behavior; say in the pull request what you ran in simulation, if anything.
- **Nothing private.** Configs name the variables that hold keys, never keys; machine paths go
  into `paths.env` (git-ignored). `python tools/check_release.py` must report `clean`.
- **Style.** Follow the surrounding code: docstrings that say what a function decides and why,
  line length 100, `ruff format`.
