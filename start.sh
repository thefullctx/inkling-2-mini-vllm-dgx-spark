#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[ -f .env ] || cp .env.example .env
set -a && source .env && set +a
export HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE_DIR"

echo "Starting Inkling-2-Mini (vLLM + DSpark) on port ${PORT:-8000}..."
docker compose up -d
echo "Started. Logs: docker logs -f inkling2-mini-vllm"
