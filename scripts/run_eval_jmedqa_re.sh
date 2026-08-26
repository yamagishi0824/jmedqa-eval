#!/bin/bash
#SBATCH --job-name=jmedqa_eval_re
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --output=logs_slurm/%j-%x.out
#SBATCH --error=logs_slurm/%j-%x.out

# =============================================================================
# run_eval_jmedqa_re.sh — JMedQA inference + regex answer extraction
#
#   vLLM generation + regex extraction in a single pass
#     -> outputs_jmedqa_re_question_variants/{model}_{temp}/jmedqa_pred_full.jsonl
#     -> outputs_jmedqa_re_question_variants/{model}_{temp}/jmedqa_pred_light.csv
#     -> thinking models write to .../{model}_{temp}_think/
#
# Slurm usage (cluster-specific options are passed at submit time):
#   sbatch --partition=<partition> --account=<account> scripts/run_eval_jmedqa_re.sh
#
# Direct usage on a GPU machine:
#   bash scripts/run_eval_jmedqa_re.sh
#
# Site-specific settings (modules, caches, proxy, HF token, TP) belong in
# scripts/env.sh — see scripts/env.example.sh.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/env.sh"
fi

# ── Environment modules (optional, HPC clusters) ─────────────────────────────
if [[ -n "${JMEDQA_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  module purge
  for _mod in ${JMEDQA_MODULES}; do
    module load "$_mod"
  done
fi

# ── Prefer the NVIDIA libraries shipped inside the venv ──────────────────────
# NOTE: a hard-coded python3.x path breaks across environments, so the tree is
# searched instead; an empty result is tolerated under `set -euo pipefail`.
NVIDIA_LIB_PATHS=""
if [[ -d "${REPO_ROOT}/.venv/lib" ]]; then
  NVIDIA_LIB_PATHS="$(find "${REPO_ROOT}/.venv/lib" -type d -path "*/site-packages/nvidia/*/lib" 2>/dev/null | paste -sd: - || true)"
fi
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}:${LD_LIBRARY_PATH:-}"

echo "[INFO] LD_LIBRARY_PATH includes:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -E 'nvidia|cuda|nccl|cudnn' || true

# ── Cache roots ──────────────────────────────────────────────────────────────
CACHE_ROOT="${JMEDQA_CACHE_ROOT:-${REPO_ROOT}/.cache}"
mkdir -p \
  "${CACHE_ROOT}/hf" \
  "${CACHE_ROOT}/torch" \
  "${CACHE_ROOT}/vllm" \
  "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/tmp"

export HF_HOME="${CACHE_ROOT}/hf"
export TORCH_HOME="${CACHE_ROOT}/torch"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="${CACHE_ROOT}/tmp"

# HF_TOKEN is only needed for gated repositories and must come from the
# environment (e.g. `huggingface-cli login` or scripts/env.sh).

# ── vLLM runtime ─────────────────────────────────────────────────────────────
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

# ── uv resolution (Slurm runs a non-login shell) ─────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found in PATH=${PATH}" >&2
  echo "[ERROR] Install uv (https://docs.astral.sh/uv/) or extend PATH in scripts/env.sh." >&2
  exit 1
fi
UV_CMD="$(command -v uv)"
echo "[INFO] Using uv at: ${UV_CMD}"

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_CSV="${INPUT_CSV:-data/jmedqa.csv}"
OUTPUTS_DIR="${OUTPUTS_DIR:-outputs_jmedqa_re_question_variants}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# ── Inference settings ───────────────────────────────────────────────────────
TP="${TP:-8}"
MAX_TOKENS="${MAX_TOKENS:-8192}"

# Sampling temperatures to sweep.
TEMPS=(0.2)
# TEMPS=(0.0 0.1 0.2 0.3 0.5 0.7 0.9 1.0)

# ── Model list ───────────────────────────────────────────────────────────────
# Format: "name|model_path|max_model_len|system_prompt_preset|trust_remote_code|enable_thinking"
# `model_path` accepts a Hugging Face repo ID or a local checkpoint directory.
MODELS=(
  "SIP-jmed-llm-3-13b-OP-32k-R0.1|SIP-med-LLM/SIP-jmed-llm-3-13b-OP-32k-R0.1|32768|sip|true|false"
  "SIP-jmed-llm-3-8x13b-OP-32k-R0.1|SIP-med-LLM/SIP-jmed-llm-3-8x13b-OP-32k-R0.1|32768|sip|true|false"
  "SIP-jmed-llm-3-8x13b-AC-32k-instruct|SIP-med-LLM/SIP-jmed-llm-3-8x13b-AC-32k-instruct|32768|sip|true|false"
  "SIP-jmed-llm-3-8x13b-OP-4k-base|SIP-med-LLM/SIP-jmed-llm-3-8x13b-OP-4k-base|4096|sip|true|false"
  "SIP-jmed-llm-3-13b-OP-4k-base|SIP-med-LLM/SIP-jmed-llm-3-13b-OP-4k-base|4096|sip|true|false"
  "SIP-jmed-llm-2-8x13b-OP-instruct|SIP-med-LLM/SIP-jmed-llm-2-8x13b-OP-instruct|8192|sip|true|false"
)

mkdir -p "$OUTPUTS_DIR" "$RESULTS_DIR" logs_slurm

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "[ERROR] missing input csv: $INPUT_CSV" >&2
  echo "[ERROR] place the JMedQA CSV at that path (see README)." >&2
  exit 1
fi

echo ""
echo "========================================"
echo "[Step 1] inference (JMedQA, regex extraction)"
echo "========================================"

set +e
for MODEL_DEF in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_NAME MODEL_PATH MAX_MODEL_LEN SYS_PRESET TRUST_REMOTE_CODE ENABLE_THINKING <<< "$MODEL_DEF"

  TRUST_FLAG=""
  [[ "$TRUST_REMOTE_CODE" == "true" ]] && TRUST_FLAG="--trust-remote-code"

  THINKING_FLAG="--disable-thinking"
  THINK_SUFFIX=""
  if [[ "$ENABLE_THINKING" == "true" ]]; then
    THINKING_FLAG="--enable-thinking"
    THINK_SUFFIX="_think"
  fi

  for TEMPERATURE in "${TEMPS[@]}"; do
    TEMP_TAG="${TEMPERATURE}${THINK_SUFFIX}"
    OUTDIR="${OUTPUTS_DIR}/${MODEL_NAME}_${TEMP_TAG}"
    FULL_FILE="${OUTDIR}/jmedqa_pred_full.jsonl"
    LIGHT_FILE="${OUTDIR}/jmedqa_pred_light.csv"

    mkdir -p "$OUTDIR"

    if [[ -f "$FULL_FILE" && -f "$LIGHT_FILE" ]]; then
      echo "[SKIP] already inferred: $FULL_FILE / $LIGHT_FILE"
      continue
    fi

    echo "[RUN] model=${MODEL_NAME} temp=${TEMPERATURE} thinking=${ENABLE_THINKING} outdir=${OUTDIR}"

    "${UV_CMD}" run python src/infer_jmedqa_re.py \
      --model                    "$MODEL_PATH"    \
      --input-csv                "$INPUT_CSV"     \
      --out-full                 "$FULL_FILE"     \
      --out-light                "$LIGHT_FILE"    \
      --question-variants        "both"           \
      --tp                       "$TP"            \
      --max-len                  "$MAX_MODEL_LEN" \
      --max-tokens               "$MAX_TOKENS"    \
      --temperature              "$TEMPERATURE"   \
      --system-prompt-preset     "$SYS_PRESET"    \
      --keep-think               \
      $THINKING_FLAG             \
      $TRUST_FLAG

    RC=$?
    if [[ $RC -ne 0 ]]; then
      echo "[WARN] inference FAILED: model=${MODEL_NAME} temp=${TEMPERATURE} thinking=${ENABLE_THINKING} (rc=${RC})"
    fi
  done
done
set -e

echo ""
echo "========================================"
echo "[DONE] JMedQA regex inference finished"
echo "========================================"
