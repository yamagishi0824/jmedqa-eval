#!/bin/bash
# scripts/common.sh — shared environment for the run scripts (sourced, not executed).
# Site-specific values go into scripts/env.sh (git-ignored); see scripts/env.example.sh.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

if [[ -f "${REPO_ROOT}/scripts/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/scripts/env.sh"
fi

# ── Environment modules (optional, HPC clusters) ─────────────────────────────
if [[ -n "${JMEDQA_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  for _mod in ${JMEDQA_MODULES}; do
    module load "$_mod"
  done
fi

# ── Cache roots ──────────────────────────────────────────────────────────────
CACHE_ROOT="${JMEDQA_CACHE_ROOT:-${REPO_ROOT}/.cache}"
mkdir -p "${CACHE_ROOT}"/{hf,torch,vllm,triton,tmp}
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export TORCH_HOME="${CACHE_ROOT}/torch"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="${CACHE_ROOT}/tmp"
# HF_TOKEN is only needed for gated repositories and must come from the environment
# (e.g. `huggingface-cli login` or scripts/env.sh).

# ── vLLM runtime ─────────────────────────────────────────────────────────────
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

# ── Python: the project venv (scripts/setup_env.sh) if present, otherwise `uv run` ──
if [[ -z "${PY:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PY="${REPO_ROOT}/.venv/bin/python"
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"   # ninja etc. for JIT-compiled kernels
  elif command -v uv >/dev/null 2>&1; then
    PY="uv run python"
  else
    echo "[ERROR] no .venv and no uv in PATH — run scripts/setup_env.sh or install uv" >&2
    exit 1
  fi
fi

INPUT_CSV="${INPUT_CSV:-data/jmedqa.csv}"
