#!/bin/bash
# =============================================================================
# run_eval.sh — stage1 for the runs listed in configs/models.tsv
#
#   Each run: free-form generation with the model's official settings + rule-based answer extraction
#     -> outputs/<greedy|official>/<name>_<profile>_<mode>/jmedqa_stage1_{full.jsonl,full_reasoning.jsonl,light.csv}
#   Runs whose output already exists are skipped.
#
# Selection (environment variables, all optional):
#   ONLY=a,b             run names
#   GROUP=<tag>          rows whose `groups` column contains the tag
#   SAMPLING=greedy|official
#   IMAGES=1             vision evaluation (rows with vision=1; image-referencing questions with their images)
#                        -> outputs/vision_<sampling>/...   (needs the dataset images in $JMEDQA_IMAGE_ROOT)
#   LIMIT=N              first N questions only (smoke test) -> outputs/test/ or outputs/vision_test/
#
# Usage (on a GPU machine):
#   ONLY=gemma-4-E4B-it bash scripts/run_eval.sh
#
# Site-specific settings (modules, caches, proxy, HF token) belong in scripts/env.sh — see scripts/env.example.sh.
# =============================================================================
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/scripts/common.sh"

IMAGE_ROOT="${JMEDQA_IMAGE_ROOT:-data/hf_jmedqa}"
LIMIT="${LIMIT:-0}"; IMAGES="${IMAGES:-0}"

run_isolated() {  # run one evaluation in its own process group; afterwards kill leftovers and wait for GPU memory
  setsid "$@" &
  local pid=$! rc used
  wait "$pid"; rc=$?
  kill -9 -- "-$pid" 2>/dev/null   # failed vLLM workers can otherwise keep holding the GPUs
  if command -v nvidia-smi >/dev/null 2>&1; then
    for _ in $(seq 1 60); do
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"
      [[ "$used" =~ ^[0-9]+$ ]] || break   # e.g. "[N/A]" on unified-memory GPUs
      [[ "$used" -lt 2000 ]] && break
      sleep 5
    done
  fi
  return "$rc"
}

PLAN_ARGS=()
[[ -n "${ONLY:-}" ]] && PLAN_ARGS+=(--only "$ONLY")
[[ -n "${GROUP:-}" ]] && PLAN_ARGS+=(--group "$GROUP")
[[ -n "${SAMPLING:-}" ]] && PLAN_ARGS+=(--sampling "$SAMPLING")
[[ "$IMAGES" == "1" ]] && PLAN_ARGS+=(--images)
mapfile -t PLAN < <($PY src/registry.py plan "${PLAN_ARGS[@]}") || exit 1
echo "[INFO] host=$(hostname) only=${ONLY:-} group=${GROUP:-} sampling=${SAMPLING:-auto} images=${IMAGES} limit=${LIMIT} runs=${#PLAN[@]}"
[[ ${#PLAN[@]} -eq 0 ]] && { echo "[WARN] nothing to run"; exit 0; }

for DEF in "${PLAN[@]}"; do
  IFS=$'\t' read -r NAME MODEL PROFILE MODE TP SMP OUTDIR KWARGS <<< "$DEF"
  TP="${TP_OVERRIDE:-$TP}"
  if [[ "$LIMIT" -gt 0 ]]; then
    OUTDIR="outputs/$([[ "$IMAGES" == "1" ]] && echo vision_)test/$(basename "$OUTDIR")"
  fi
  S1_FULL="${OUTDIR}/jmedqa_stage1_full.jsonl"
  [[ -f "$S1_FULL" ]] && { echo "[SKIP] done: $OUTDIR"; continue; }
  mkdir -p "$OUTDIR"
  ARGS=(--model "$MODEL" --profile "$PROFILE" --mode "$MODE" --sampling "$SMP" --input-csv "$INPUT_CSV"
        --out-full "$S1_FULL" --out-light "${OUTDIR}/jmedqa_stage1_light.csv" --question-variants both
        --tp "$TP" --limit "$LIMIT" --trust-remote-code)
  [[ "$KWARGS" != "-" ]] && ARGS+=(--chat-template-kwargs "$KWARGS")
  [[ "$IMAGES" == "1" ]] && ARGS+=(--with-images "$IMAGE_ROOT")
  echo "[RUN] name=${NAME} profile=${PROFILE} mode=${MODE} sampling=${SMP} tp=${TP} kwargs=${KWARGS} $(date '+%F %T')"
  run_isolated $PY src/infer_jmedqa.py "${ARGS[@]}" || { echo "[WARN] FAILED name=${NAME} rc=$?"; rm -f "$S1_FULL"; }
done
echo "[DONE] run_eval $(date '+%F %T')"
