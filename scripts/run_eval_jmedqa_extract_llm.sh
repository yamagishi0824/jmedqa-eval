#!/bin/bash
#SBATCH --job-name=jmedqa_eval_extract
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --output=logs_slurm/%j-%x.out
#SBATCH --error=logs_slurm/%j-%x.out

# =============================================================================
# run_eval_jmedqa_extract_llm.sh — JMedQA inference + extractor-LLM pipeline
#
#   Step 1) free-form generation by the evaluated model
#             -> {outdir}/jmedqa_stage1_full.jsonl
#   Step 2) an extractor LLM reads only that output and extracts the answer
#             -> {outdir}/jmedqa_pred_full.jsonl
#             -> {outdir}/jmedqa_pred_light.csv
#   outdir = outputs_jmedqa_extract_llm_question_variants/{model}_{temp[_think]}
#
# Slurm usage (cluster-specific options are passed at submit time):
#   sbatch --partition=<partition> --account=<account> scripts/run_eval_jmedqa_extract_llm.sh
#
# Direct usage on a GPU machine:
#   bash scripts/run_eval_jmedqa_extract_llm.sh
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
OUTPUTS_DIR="${OUTPUTS_DIR:-outputs_jmedqa_extract_llm_question_variants}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# ── Inference settings ───────────────────────────────────────────────────────
TP="${TP:-8}"
MAX_TOKENS="${MAX_TOKENS:-8192}"

# Answer-extraction model. It receives only the evaluated model's output —
# never the question or the gold answer. Use a smaller model if 8 GPUs are
# not available.
EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507}"
EXTRACTOR_TP="${EXTRACTOR_TP:-8}"
EXTRACTOR_MAX_LEN="${EXTRACTOR_MAX_LEN:-32768}"
EXTRACTOR_MAX_TOKENS="${EXTRACTOR_MAX_TOKENS:-256}"

# Sampling temperatures to sweep.
TEMPS=(0.0)
# TEMPS=(0.0 0.1 0.2 0.3 0.5 0.7 0.9 1.0)

# ── Model list ───────────────────────────────────────────────────────────────
# Format: "name|model_path|max_model_len|system_prompt_preset|trust_remote_code|enable_thinking"
# `model_path` accepts a Hugging Face repo ID or a local checkpoint directory.
# Presets: sip (Japanese instruction), gemma, medical.
MODELS=(
  # ── Qwen3 Instruct / Thinking 2507 ─────────────────────────────────
  "Qwen3-4B-Instruct-2507|Qwen/Qwen3-4B-Instruct-2507|32768|medical|true|false"
  "Qwen3-4B-Thinking-2507|Qwen/Qwen3-4B-Thinking-2507|32768|medical|true|true"
  "Qwen3-30B-A3B-Instruct-2507|Qwen/Qwen3-30B-A3B-Instruct-2507|32768|medical|true|false"
  "Qwen3-30B-A3B-Thinking-2507|Qwen/Qwen3-30B-A3B-Thinking-2507|32768|medical|true|true"
  "Qwen3-235B-A22B-Instruct-2507|Qwen/Qwen3-235B-A22B-Instruct-2507|32768|medical|true|false"
  "Qwen3-235B-A22B-Thinking-2507|Qwen/Qwen3-235B-A22B-Thinking-2507|32768|medical|true|true"

  # ── Swallow / Preferred medical models ────────────────────────────
  "Qwen3-Swallow-32B-SFT-v0.2|tokyotech-llm/Qwen3-Swallow-32B-SFT-v0.2|32768|medical|true|true"
  "Qwen3-Swallow-32B-RL-v0.2|tokyotech-llm/Qwen3-Swallow-32B-RL-v0.2|32768|medical|true|true"
  "Qwen3-Swallow-30B-A3B-SFT-v0.2|tokyotech-llm/Qwen3-Swallow-30B-A3B-SFT-v0.2|32768|medical|true|true"
  "Qwen3-Swallow-30B-A3B-RL-v0.2|tokyotech-llm/Qwen3-Swallow-30B-A3B-RL-v0.2|32768|medical|true|true"
  "Qwen3-Swallow-8B-SFT-v0.2|tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2|4096|medical|true|true"
  "Qwen3-Swallow-8B-RL-v0.2|tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2|4096|medical|true|true"
  "GPT-OSS-Swallow-120B-SFT-v0.1|tokyotech-llm/GPT-OSS-Swallow-120B-SFT-v0.1|32768|medical|true|false"
  "GPT-OSS-Swallow-20B-SFT-v0.1|tokyotech-llm/GPT-OSS-Swallow-20B-SFT-v0.1|32768|medical|true|false"
  "GPT-OSS-Swallow-120B-RL-v0.1|tokyotech-llm/GPT-OSS-Swallow-120B-RL-v0.1|32768|medical|true|false"
  "GPT-OSS-Swallow-20B-RL-v0.1|tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1|32768|medical|true|false"
  "Preferred-MedLLM-Qwen-72B|pfnet/Preferred-MedLLM-Qwen-72B|4096|medical|true|true"
  "Preferred-MedRECT-32B|pfnet/Preferred-MedRECT-32B|4096|medical|true|true"
  # "Llama3-Preferred-MedSwallow-70B|pfnet/Llama3-Preferred-MedSwallow-70B|4096|medical|true|false"

  # ── gpt-oss ───────────────────────────────────────────────────────
  "gpt-oss-120b|openai/gpt-oss-120b|32768|medical|true|false"
  "gpt-oss-20b|openai/gpt-oss-20b|32768|medical|true|false"

  # ── Llama ─────────────────────────────────────────────────────────
  "Llama-4-Scout-17B-16E-Instruct|meta-llama/Llama-4-Scout-17B-16E-Instruct|32768|medical|true|false"
  "Llama-4-Maverick-17B-128E-Instruct|meta-llama/Llama-4-Maverick-17B-128E-Instruct|32768|medical|true|false"
  "Llama-3.3-70B-Instruct|meta-llama/Llama-3.3-70B-Instruct|8192|medical|true|false"

  # ── SIP-jmed ──────────────────────────────────────────────────────
  "SIP-jmed-llm-3-13b-OP-32k-R0.1|SIP-med-LLM/SIP-jmed-llm-3-13b-OP-32k-R0.1|32768|sip|true|false"
  "SIP-jmed-llm-3-8x13b-OP-32k-R0.1|SIP-med-LLM/SIP-jmed-llm-3-8x13b-OP-32k-R0.1|32768|sip|true|false"
  "SIP-jmed-llm-3-8x13b-AC-32k-instruct|SIP-med-LLM/SIP-jmed-llm-3-8x13b-AC-32k-instruct|32768|sip|true|false"
  "SIP-jmed-llm-3-8x13b-OP-4k-base|SIP-med-LLM/SIP-jmed-llm-3-8x13b-OP-4k-base|4096|sip|true|false"
  "SIP-jmed-llm-3-13b-OP-4k-base|SIP-med-LLM/SIP-jmed-llm-3-13b-OP-4k-base|4096|sip|true|false"
  "SIP-jmed-llm-2-8x13b-OP-instruct|SIP-med-LLM/SIP-jmed-llm-2-8x13b-OP-instruct|8192|sip|true|false"

  # ── llm-jp ────────────────────────────────────────────────────────
  "llm-jp-3.1-8x13b-instruct4|llm-jp/llm-jp-3.1-8x13b-instruct4|4096|sip|true|false"
  "llm-jp-3.1-13b-instruct4|llm-jp/llm-jp-3.1-13b-instruct4|4096|sip|true|false"

  # ── Gemma / MedGemma ──────────────────────────────────────────────
  # "gemma-3-27b-it|google/gemma-3-27b-it|8192|gemma|true|false"
  # "gemma-3-12b-it|google/gemma-3-12b-it|8192|gemma|true|false"
  # "gemma-3-4b-it|google/gemma-3-4b-it|8192|gemma|true|false"
  # "medgemma-27b-it|google/medgemma-27b-it|8192|gemma|true|false"
  # "medgemma-4b-it|google/medgemma-4b-it|8192|gemma|true|false"

  # ── Qwen2.5 Instruct ──────────────────────────────────────────────
  # "Qwen2.5-7B-Instruct|Qwen/Qwen2.5-7B-Instruct|32768|medical|true|false"
  # "Qwen2.5-32B-Instruct|Qwen/Qwen2.5-32B-Instruct|32768|medical|true|false"
  # "Qwen2.5-72B-Instruct|Qwen/Qwen2.5-72B-Instruct|32768|medical|true|false"
)

mkdir -p "$OUTPUTS_DIR" "$RESULTS_DIR" logs_slurm

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "[ERROR] missing input csv: $INPUT_CSV" >&2
  echo "[ERROR] place the JMedQA CSV at that path (see README)." >&2
  exit 1
fi

echo ""
echo "========================================"
echo "[Step 1] stage1 free-form generation"
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
    STAGE1_FULL_FILE="${OUTDIR}/jmedqa_stage1_full.jsonl"
    STAGE1_LIGHT_FILE="${OUTDIR}/jmedqa_stage1_light.csv"

    mkdir -p "$OUTDIR"

    if [[ -f "$STAGE1_FULL_FILE" ]]; then
      echo "[SKIP] stage1 already done: $STAGE1_FULL_FILE"
      continue
    fi

    echo "[RUN] stage1 model=${MODEL_NAME} temp=${TEMPERATURE} thinking=${ENABLE_THINKING} outdir=${OUTDIR}"

    "${UV_CMD}" run python src/infer_jmedqa_extract_llm.py \
      --mode                     "stage1"                \
      --model                    "$MODEL_PATH"           \
      --extractor-model          "$EXTRACTOR_MODEL"      \
      --input-csv                "$INPUT_CSV"            \
      --out-full                 "$STAGE1_FULL_FILE"     \
      --out-light                "$STAGE1_LIGHT_FILE"    \
      --question-variants        "both"                  \
      --tp                       "$TP"                   \
      --extractor-tp             "$EXTRACTOR_TP"         \
      --max-len                  "$MAX_MODEL_LEN"        \
      --extractor-max-len        "$EXTRACTOR_MAX_LEN"    \
      --max-tokens               "$MAX_TOKENS"           \
      --extractor-max-tokens     "$EXTRACTOR_MAX_TOKENS" \
      --temperature              "$TEMPERATURE"          \
      --extractor-temperature    0.0                     \
      --extractor-top-p          1.0                     \
      --system-prompt-preset     "$SYS_PRESET"           \
      --keep-think               \
      $THINKING_FLAG             \
      $TRUST_FLAG

    RC=$?
    if [[ $RC -ne 0 ]]; then
      echo "[WARN] stage1 FAILED: model=${MODEL_NAME} temp=${TEMPERATURE} thinking=${ENABLE_THINKING} (rc=${RC})"
    fi
  done
done
set -e

echo ""
echo "========================================"
echo "[Step 2] answer extraction (extractor LLM)"
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
    STAGE1_FULL_FILE="${OUTDIR}/jmedqa_stage1_full.jsonl"
    FULL_FILE="${OUTDIR}/jmedqa_pred_full.jsonl"
    LIGHT_FILE="${OUTDIR}/jmedqa_pred_light.csv"

    mkdir -p "$OUTDIR"

    if [[ ! -f "$STAGE1_FULL_FILE" ]]; then
      echo "[SKIP] stage1 missing: $STAGE1_FULL_FILE"
      continue
    fi
    if [[ -f "$FULL_FILE" && -f "$LIGHT_FILE" ]]; then
      echo "[SKIP] extraction already done: $FULL_FILE / $LIGHT_FILE"
      continue
    fi

    echo "[RUN] extract model=${MODEL_NAME} temp=${TEMPERATURE} extractor=${EXTRACTOR_MODEL} outdir=${OUTDIR}"

    "${UV_CMD}" run python src/infer_jmedqa_extract_llm.py \
      --mode                     "extract"               \
      --model                    "$MODEL_PATH"           \
      --extractor-model          "$EXTRACTOR_MODEL"      \
      --input-csv                "$INPUT_CSV"            \
      --stage1-full-in           "$STAGE1_FULL_FILE"     \
      --out-full                 "$FULL_FILE"            \
      --out-light                "$LIGHT_FILE"           \
      --tp                       "$TP"                   \
      --extractor-tp             "$EXTRACTOR_TP"         \
      --max-len                  "$MAX_MODEL_LEN"        \
      --extractor-max-len        "$EXTRACTOR_MAX_LEN"    \
      --max-tokens               "$MAX_TOKENS"           \
      --extractor-max-tokens     "$EXTRACTOR_MAX_TOKENS" \
      --temperature              "$TEMPERATURE"          \
      --extractor-temperature    0.0                     \
      --extractor-top-p          1.0                     \
      --system-prompt-preset     "$SYS_PRESET"           \
      --keep-think               \
      $THINKING_FLAG             \
      $TRUST_FLAG

    RC=$?
    if [[ $RC -ne 0 ]]; then
      echo "[WARN] extract FAILED: model=${MODEL_NAME} temp=${TEMPERATURE} thinking=${ENABLE_THINKING} (rc=${RC})"
    fi
  done
done
set -e

echo ""
echo "========================================"
echo "[DONE] JMedQA extractor-LLM pipeline finished"
echo "========================================"
