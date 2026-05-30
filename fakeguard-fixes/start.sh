#!/usr/bin/env bash
# FakeGuard — start both services simultaneously (Linux / macOS)
# Usage: bash start.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo ""
echo "  ============================================================"
echo "    FakeGuard  --  Fake News Detection System"
echo "  ============================================================"
echo ""

# ── Guard: must be run from project root ─────────────────────────
if [[ ! -f "src/app/main.py" ]]; then
    echo "  [ERROR] Run this script from the project root."
    exit 1
fi

export PYTHONPATH="$ROOT"

# ── Start FastAPI in background ───────────────────────────────────
echo "  ▶  Starting FastAPI backend on port 8000 ..."
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "  ⏳  Waiting 4 s for API to initialise ..."
sleep 4

# ── Start Streamlit in background ────────────────────────────────
echo "  ▶  Starting Streamlit dashboard on port 8501 ..."
streamlit run src/app/streamlit_app.py \
    --server.port 8501 \
    --server.headless false &
UI_PID=$!

echo ""
echo "  ============================================================"
echo "    API       :  http://localhost:8000"
echo "    API docs  :  http://localhost:8000/docs"
echo "    Dashboard :  http://localhost:8501"
echo ""
echo "    Press Ctrl+C to stop both services."
echo "  ============================================================"
echo ""

# ── Trap Ctrl+C and kill both processes ──────────────────────────
_shutdown() {
    echo ""
    echo "  Shutting down..."
    kill "$API_PID" 2>/dev/null || true
    kill "$UI_PID"  2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
    wait "$UI_PID"  2>/dev/null || true
    echo "  Done."
    exit 0
}
trap _shutdown SIGINT SIGTERM

# Keep script alive until either process exits unexpectedly
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$UI_PID" 2>/dev/null; do
    sleep 2
done

echo "  [WARN] A service exited unexpectedly. Shutting down the other."
_shutdown
