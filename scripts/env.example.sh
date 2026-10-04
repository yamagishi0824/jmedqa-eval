#!/bin/bash
# Site-specific settings for the run scripts.
#
# Usage:
#   cp scripts/env.example.sh scripts/env.sh
#   # edit scripts/env.sh for your cluster / workstation
#
# scripts/env.sh is git-ignored and is sourced automatically (via scripts/common.sh). Every variable is optional.

# ── Environment modules (HPC clusters) ───────────────────────────────────────
# Space-separated module list. Ignored if `module` is unavailable.
# export JMEDQA_MODULES="cuda/12.8"

# ── Cache root ───────────────────────────────────────────────────────────────
# Model / compiler caches. Point this at large, fast storage. Defaults to <repo>/.cache.
# export JMEDQA_CACHE_ROOT="/path/to/large/storage/jmedqa-cache"
# Reuse an existing Hugging Face cache instead (HF_HOME takes precedence over JMEDQA_CACHE_ROOT/hf):
# export HF_HOME="/path/to/huggingface"

# ── Dataset images (vision evaluation, IMAGES=1) ─────────────────────────────
# Directory containing the dataset's images/ folder. Defaults to data/hf_jmedqa.
# export JMEDQA_IMAGE_ROOT="/path/to/JMedQA"

# ── Hugging Face credentials ─────────────────────────────────────────────────
# Required only for gated repositories (e.g. meta-llama). Prefer `huggingface-cli login` or an
# external secret store; never commit a token to the repository.
# export HF_TOKEN="$(cat ~/.secrets/hf_token)"

# ── Outbound proxy (only if your nodes need one) ─────────────────────────────
# export HTTP_PROXY="http://proxy.example.internal:8080"
# export HTTPS_PROXY="$HTTP_PROXY"
# export http_proxy="$HTTP_PROXY"
# export https_proxy="$HTTP_PROXY"

# ── Python ───────────────────────────────────────────────────────────────────
# By default the scripts use <repo>/.venv/bin/python (scripts/setup_env.sh), otherwise `uv run python`.
# export PY="/path/to/python"

# ── Overrides ────────────────────────────────────────────────────────────────
# export TP_OVERRIDE=4        # tensor-parallel size for every run (default: the `tp` column of configs/models.tsv)
