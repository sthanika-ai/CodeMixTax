#!/usr/bin/env bash
# Serve each model in turn with vLLM, evaluate it, shut it down, move on.
#
#   scripts/run_suite.sh gemma3-12b-it qwen3-8b llama3.1-8b-it
#   RUN_ID=paper-v1 scripts/run_suite.sh $(cmb-indic models | awk 'NR>2 {print $1}')
#
# Why a shell loop rather than one Python process: serving each model in a separate
# process is the only reliable way to fully reclaim GPU memory between large models.
# Every model writes into the same RUN_ID, so one report covers the whole sweep.
#
# Alternative for a single model: use `--backend vllm` to load it in-process and
# skip the server entirely.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model-config> [<model-config> ...]" >&2
  echo "available: $(cmb-indic models 2>/dev/null | awk 'NR>2 {printf "%s ", $1}')" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${PORT:-8000}"
DATASETS="${DATASETS:-all}"
SHOTS="${SHOTS:-0}"
LIMIT_ARG=""
[[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit ${LIMIT}"

export CMB_BASE_URL="http://localhost:${PORT}/v1"
export CMB_API_KEY="${CMB_API_KEY:-EMPTY}"

mkdir -p logs
echo "run_id=${RUN_ID}  models=$*  datasets=${DATASETS}" >&2

for CONFIG in "$@"; do
  HF_ID="$(python -c "
from cmb_indic.config import load_model_config, load_defaults
print(load_model_config('${CONFIG}', defaults=load_defaults()).hf_id)")"

  echo "=== ${CONFIG}  (${HF_ID}) ===" >&2

  LOG="logs/vllm-${CONFIG}.log"
  scripts/serve_vllm.sh "${HF_ID}" --port "${PORT}" >"${LOG}" 2>&1 &
  SERVER_PID=$!
  # Make sure the server dies with us, however we exit.
  trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT INT TERM

  # Wait for readiness rather than sleeping a fixed amount: large models can take
  # several minutes to load, small ones a few seconds.
  echo -n "  waiting for server" >&2
  for _ in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo " ready" >&2
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo " FAILED -- see ${LOG}" >&2
      tail -20 "${LOG}" >&2
      exit 1
    fi
    echo -n "." >&2
    sleep 5
  done

  cmb-indic run \
    --models "${CONFIG}" \
    --datasets "${DATASETS}" \
    --shots "${SHOTS}" \
    --run-id "${RUN_ID}" \
    --no-report \
    ${LIMIT_ARG} || echo "  !! ${CONFIG} had failures; continuing" >&2

  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  trap - EXIT INT TERM
  echo "  done ${CONFIG}" >&2
done

echo "=== building report ===" >&2
cmb-indic report --run-dir "runs/${RUN_ID}"
echo "Report written for run_id=${RUN_ID}" >&2
