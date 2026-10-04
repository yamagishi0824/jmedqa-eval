#!/bin/bash
# =============================================================================
# setup_env.sh — create <repo>/.venv with vLLM (run on a GPU node).
#
#   The PyPI vLLM / torch wheels target CUDA 13. On drivers that only support CUDA 12.x, install the
#   +cu129 wheel from the vLLM GitHub release instead (default below); it runs on recent CUDA 12 drivers
#   through minor-version compatibility.
#
#   VLLM_VERSION (default 0.30.0), CUDA_TAG (default cu129; set CUDA_TAG=pypi to use the PyPI wheels)
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ -f scripts/env.sh ]] && source scripts/env.sh
command -v uv >/dev/null 2>&1 || { echo "[ERROR] uv not found (https://docs.astral.sh/uv/)" >&2; exit 1; }

VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
CUDA_TAG="${CUDA_TAG:-cu129}"
PY="${REPO_ROOT}/.venv/bin/python"

nvidia-smi -i 0 --query-gpu=name,driver_version --format=csv,noheader || true
uv venv .venv --python 3.12 --seed
if [[ "$CUDA_TAG" == "pypi" ]]; then
  uv pip install --python "$PY" "vllm==${VLLM_VERSION}"
else
  WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}%2B${CUDA_TAG}-cp38-abi3-manylinux_2_28_x86_64.whl"
  uv pip install --python "$PY" "vllm @ ${WHEEL}" --torch-backend="${CUDA_TAG}"
  # vLLM pulls torchcodec from PyPI (a CUDA 13 build); replace it with the matching CUDA build.
  uv pip install --python "$PY" --reinstall --no-deps torchcodec --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi
uv pip install --python "$PY" pandas tqdm pillow hf_transfer openai-harmony

"$PY" - <<'PY'
import torch, transformers, vllm
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available(),
      "devices:", torch.cuda.device_count())
print("transformers:", transformers.__version__, "vllm:", vllm.__version__)
PY
echo "[DONE] setup_env"
