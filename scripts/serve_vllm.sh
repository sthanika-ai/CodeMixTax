#!/usr/bin/env bash
# Serve one model with vLLM on an OpenAI-compatible endpoint.
#
#   scripts/serve_vllm.sh google/gemma-3-12b-it
#   scripts/serve_vllm.sh RedHatAI/gemma-3-12b-it-quantized.w4a16 --port 8001
#   TP=4 scripts/serve_vllm.sh meta-llama/Llama-4-Scout-17B-16E-Instruct
#
# Then point the harness at it:
#   CMB_BASE_URL=http://localhost:8000/v1 cmb-indic run --models gemma3-12b-it
#
# Any extra arguments are passed straight through to `vllm serve`.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,12p' "$0" >&2
  exit 2
fi

MODEL="$1"; shift

PORT="${PORT:-8000}"
TP="${TP:-1}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_LEN="${MAX_LEN:-8192}"

# `vllm serve` advertises the model under its full repo id, which is exactly what
# the model cards put in served_model_name -- so no extra wiring is needed.
echo "Serving ${MODEL} on port ${PORT} (tp=${TP}, max_model_len=${MAX_LEN})" >&2
echo "Health check: curl -s http://localhost:${PORT}/v1/models | jq" >&2

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len "${MAX_LEN}" \
  --disable-log-requests \
  "$@"
