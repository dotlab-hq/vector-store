#!/usr/bin/env bash
set -euo pipefail

case "${1:-api}" in
  api)
    exec python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
    ;;
  worker)
    exec python -m apps.worker
    ;;
  smoke)
    exec python -m src.main
    ;;
  *)
    echo "Unknown role: $1" >&2
    echo "Usage: $0 {api|worker|smoke}" >&2
    exit 1
    ;;
esac
