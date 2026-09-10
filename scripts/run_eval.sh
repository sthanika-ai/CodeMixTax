#!/usr/bin/env bash
# Activate the venv, run an evaluation, and tee the full execution log to logs/.
#
#   scripts/run_eval.sh --models gemma3-4b-it --datasets mmlu_hineng --backend vllm
#   GPU=1 scripts/run_eval.sh --models gemma3-4b-it --datasets sa,mmlu --backend vllm
#   LOG=logs/my-run.log scripts/run_eval.sh --models echo-debug --limit 20
#
# Every argument is passed straight through to `cmb-indic run`.
#
# Why a shell-level `tee` rather than Python logging: vLLM spawns an EngineCore
# subprocess that writes to fd 1/2 directly, so an in-process log handler would
# silently miss the most diagnostic output there is (CUDA graph capture, KV-cache
# sizing, OOM messages). Redirecting at the shell captures the whole process tree.

set -euo pipefail
cd "$(dirname "$0")/.."

# --- activate the virtualenv --------------------------------------------------
if [[ ! -f .venv/bin/activate ]]; then
  echo "error: .venv not found. Run 'make install' first." >&2
  exit 1
fi
# VENV lets a model that needs a different transformers run through the same
# pipeline. The Param models declare architectures vLLM 0.10.1 cannot load and their
# remote code needs transformers 4.52.x, while vLLM itself is pinned to 4.55.2 --
# so they get .venv-hf and everything else keeps the main venv.
# shellcheck disable=SC1091
source "${VENV:-.venv}/bin/activate"
echo "venv active: $(command -v python)" >&2

# --- log destination ----------------------------------------------------------
mkdir -p logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG:-logs/eval-${STAMP}.log}"

# --- GPU selection ------------------------------------------------------------
# Default to GPU 1: on this host GPU 0 is usually occupied by other users.
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1   # so the log streams live instead of in blocks

{
  echo "=========================================================="
  echo "started_utc      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "command          : cmb-indic run $*"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "python           : $(python -V 2>&1)"
  echo "versions         : $(python -c 'import torch,vllm,transformers;print(f"torch {torch.__version__} | vllm {vllm.__version__} | transformers {transformers.__version__}")' 2>/dev/null || echo 'n/a')"
  echo "git              : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
  echo "gpu before       :"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv 2>/dev/null | sed 's/^/                   /'
  echo "=========================================================="
} | tee "${LOG}"

START=$(date +%s)
set +e
cmb-indic run "$@" 2>&1 | tee -a "${LOG}"
STATUS=${PIPESTATUS[0]}
set -e
END=$(date +%s)

{
  echo "=========================================================="
  echo "exit_status      : ${STATUS}"
  echo "elapsed_seconds  : $((END - START))"
  echo "finished_utc     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "gpu after        :"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv 2>/dev/null | sed 's/^/                   /'
  echo "=========================================================="
} | tee -a "${LOG}"

echo "log written to: ${LOG}" >&2
exit "${STATUS}"
