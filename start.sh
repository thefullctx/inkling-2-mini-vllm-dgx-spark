#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

_VARS=(MODEL_ID HOST PORT API_KEY HF_CACHE_DIR HF_TOKEN VLLM_IMAGE VLLM_MAX_MODEL_LEN VLLM_GPU_MEM_UTIL VLLM_NUM_SPEC_TOKENS)
declare -A _SAVED=()
for v in "${_VARS[@]}"; do
  [ -n "${!v+x}" ] && _SAVED[$v]="${!v}"
done

[ -f .env ] || cp .env.example .env
set -a && source .env && set +a

for v in "${!_SAVED[@]}"; do
  export "$v=${_SAVED[$v]}"
done

export HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE_DIR"

if [ "${HOST:-0.0.0.0}" = "0.0.0.0" ] && [ -z "${API_KEY:-}" ]; then
  echo "[WARN] Binding 0.0.0.0:${PORT:-8000} with no API_KEY set. Set API_KEY or HOST=127.0.0.1 in .env to restrict access."
fi

echo "Starting Inkling-2-Mini (vLLM + DSpark) on ${HOST:-0.0.0.0}:${PORT:-8000}..."
docker compose up -d
echo "Started. Check status with ./status.sh or logs with ./logs.sh"
