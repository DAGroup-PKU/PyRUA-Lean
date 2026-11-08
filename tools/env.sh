#!/usr/bin/env bash
# Source this before running anything: `source tools/env.sh`.
# Sets the repository paths, reads paths.env (copied from paths.env.example; CE_PATHS_ENV names
# another file) and keeps every writable cache inside the repository (.cache/).  It starts
# nothing and sets no API key.

if [ -n "${BASH_VERSION:-}" ]; then
  env_sh="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  eval 'env_sh="${(%):-%x}"'
else
  echo "tools/env.sh: source it from bash or zsh" >&2
  return 1
fi
CE_ROOT="$(cd "$(dirname "$env_sh")/.." && pwd)"
export CE_ROOT
unset env_sh
paths_env="${CE_PATHS_ENV:-$CE_ROOT/paths.env}"
if [ -f "$paths_env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$paths_env"
  set +a
else
  echo "tools/env.sh: no $paths_env (copy paths.env.example to paths.env and fill it in)" >&2
fi
unset paths_env

export PYTHONPATH="$CE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export ROBODOJO_ROOT="${ROBODOJO_SOURCE_ROOT:-}"
export ROBODOJO_XPOLICYLAB_ROOT="${ROBODOJO_SOURCE_ROOT:+$ROBODOJO_SOURCE_ROOT/XPolicyLab}"
export ROBODOJO_ASSETS_ROOT="${ROBODOJO_SOURCE_ROOT:-}"
export PYRUALEAN_RESULT_DIR="$CE_ROOT/runs/manual"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CE_ROOT/.cache/xdg}"
export HF_HOME="${HF_HOME:-$CE_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$CE_ROOT/.cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CE_ROOT/.cache/torch_extensions}"
# LIBERO-PRO writes its default config.yaml here
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$CE_ROOT/.cache/libero_config}"
mkdir -p "$LIBERO_CONFIG_PATH"
unset PYTHONHOME ROBODOJO_XPL_SITE
