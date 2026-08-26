#!/bin/bash
#SBATCH --job-name=jmedqa_calc
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs_slurm/%j-%x.out
#SBATCH --error=logs_slurm/%j-%x.out

# =============================================================================
# run_calc_jmedqa.sh — aggregate prediction CSVs into a summary table
#
# Usage:
#   bash scripts/run_calc_jmedqa.sh [INPUT_DIR] [OUTPUT_FILE]
#
# Examples:
#   bash scripts/run_calc_jmedqa.sh outputs_jmedqa_re_question_variants results/jmedqa_summary_re.csv
#   bash scripts/run_calc_jmedqa.sh outputs_jmedqa_extract_llm_question_variants results/jmedqa_summary_extract_llm.csv
#
# This step is CPU-only and runs fine on a laptop.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/env.sh"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found in PATH=${PATH}" >&2
  echo "[ERROR] Install uv (https://docs.astral.sh/uv/) or extend PATH in scripts/env.sh." >&2
  exit 1
fi
UV_CMD="$(command -v uv)"

INPUT_DIR="${1:-outputs_jmedqa_re_question_variants}"
OUTPUT_FILE="${2:-results/jmedqa_summary.csv}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

"${UV_CMD}" run python src/calc_jmedqa.py \
  --input_dir "$INPUT_DIR" \
  --output_file "$OUTPUT_FILE" \
  --pattern "jmedqa_pred_light.csv" \
  --recursive

printf "\n[OK] summary: %s\n" "$OUTPUT_FILE"
