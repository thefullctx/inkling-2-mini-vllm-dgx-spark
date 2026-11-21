#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[ -f .env ] && { set -a; source .env; set +a; }
PORT="${PORT:-8000}"

docker ps --filter "name=^/inkling2-mini-vllm$" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "[OK] API ready at http://127.0.0.1:${PORT}/v1"
  curl -fsS "http://127.0.0.1:${PORT}/v1/models"
  echo ""
else
  echo "[STARTING/DOWN] http://127.0.0.1:${PORT}/v1/models not ready yet."
fi
