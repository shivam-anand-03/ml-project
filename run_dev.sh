#!/usr/bin/env bash
# One-command local development runner for AI TriboPredict (no Docker needed).
#   ./run_dev.sh            -> backend on http://localhost:8000 (SQLite) + frontend on http://localhost:3000
#   ./run_dev.sh backend    -> backend only
#   ./run_dev.sh frontend   -> frontend only
# Stop with Ctrl-C (both processes are terminated).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
MODE="${1:-all}"
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null 2>&1 || PY=python3

start_backend() {
  cd "$BACKEND"
  if [ ! -x .venv/bin/python ]; then
    echo "[backend] creating virtual environment with $PY ..."
    if command -v uv >/dev/null 2>&1; then uv venv --python "$PY" .venv; else "$PY" -m venv .venv; fi
  fi
  if ! .venv/bin/python -c "import fastapi, sklearn, xgboost, shap, reportlab" >/dev/null 2>&1; then
    echo "[backend] installing dependencies ..."
    if command -v uv >/dev/null 2>&1; then uv pip install --python .venv/bin/python -r requirements.txt; else .venv/bin/pip install -r requirements.txt; fi
  fi
  [ -f .env ] || cp .env.example .env
  echo "[backend] starting FastAPI on http://localhost:8000 (docs: http://localhost:8000/docs)"
  .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
  BACK_PID=$!
}

start_frontend() {
  cd "$FRONTEND"
  if [ ! -d node_modules ]; then
    echo "[frontend] installing dependencies ..."
    npm install
  fi
  [ -f .env.local ] || echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
  echo "[frontend] starting Next.js on http://localhost:3000"
  npm run dev &
  FRONT_PID=$!
}

cleanup() {
  echo; echo "Stopping ..."
  [ -n "${BACK_PID:-}" ] && kill "$BACK_PID" 2>/dev/null || true
  [ -n "${FRONT_PID:-}" ] && kill "$FRONT_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

case "$MODE" in
  backend) start_backend ;;
  frontend) start_frontend ;;
  all) start_backend; start_frontend ;;
  *) echo "usage: $0 [all|backend|frontend]"; exit 1 ;;
esac

echo
echo "AI TriboPredict is starting. Open http://localhost:3000 — go to Datasets and click 'Import bundled Kolev 2023 dataset' to load the real data."
wait
