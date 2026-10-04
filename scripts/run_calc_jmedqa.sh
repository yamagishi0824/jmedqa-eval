#!/bin/bash
# =============================================================================
# run_calc_jmedqa.sh — detailed accuracy tables (by year / section / clinical area / variant ...) for every run
#
#   bash scripts/run_calc_jmedqa.sh [INPUT_DIR] [OUTPUT_FILE] [PATTERN]
#   defaults: outputs  results/jmedqa_detail.csv  jmedqa_stage1_light.csv
#
# CPU-only. For the headline tables (original / no_image / with images) use src/summarize.py.
# =============================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

INPUT_DIR="${1:-outputs}"
OUTPUT_FILE="${2:-results/jmedqa_detail.csv}"
PATTERN="${3:-jmedqa_stage1_light.csv}"
mkdir -p "$(dirname "$OUTPUT_FILE")"
$PY src/calc_jmedqa.py --input_dir "$INPUT_DIR" --output_file "$OUTPUT_FILE" --pattern "$PATTERN" --recursive
printf "\n[OK] summary: %s\n" "$OUTPUT_FILE"
