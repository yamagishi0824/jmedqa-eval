#!/bin/bash
# Site-specific settings for the evaluation scripts.
#
# Usage:
#   cp scripts/env.example.sh scripts/env.sh
#   # edit scripts/env.sh for your cluster / workstation
#
# scripts/env.sh is git-ignored and is sourced automatically by the run scripts.
# Every variable below is optional.

# ── Environment modules (HPC clusters) ───────────────────────────────────────
# Space-separated module list. Ignored if `module` is unavailable.
# export JMEDQA_MODULES="cuda/12.8"

# ── Cache root ───────────────────────────────────────────────────────────────
# Model / dataset / compiler caches. Point this at fast shared storage when the
# home directory is small. Defaults to <repo>/.cache.
# export JMEDQA_CACHE_ROOT="/path/to/large/storage/jmedqa-cache"

# ── Hugging Face credentials ─────────────────────────────────────────────────
# Required only for gated repositories. Prefer `huggingface-cli login` or an
# external secret store; never commit a token to the repository.
# export HF_TOKEN="$(cat ~/.secrets/hf_token)"

# ── Outbound proxy (only if your nodes need one) ─────────────────────────────
# export HTTP_PROXY="http://proxy.example.internal:8080"
# export HTTPS_PROXY="$HTTP_PROXY"
# export http_proxy="$HTTP_PROXY"
# export https_proxy="$HTTP_PROXY"

# ── Extra PATH entries (e.g. a locally installed uv) ─────────────────────────
# export PATH="/path/to/uv/bin:${PATH}"

# ── Inference defaults (override the script defaults) ────────────────────────
# export TP=8                 # tensor parallel size for the evaluated model
# export MAX_TOKENS=8192      # generation budget
# export EXTRACTOR_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# export EXTRACTOR_TP=8
