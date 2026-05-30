#!/usr/bin/env python3
"""
FakeGuard — cross-platform launcher.

Starts the FastAPI backend and the Streamlit dashboard simultaneously
in two child processes and keeps them alive until you press Ctrl+C.

Usage (run from the project root directory):
    python start_all.py

No extra dependencies — only the Python standard library is used.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.resolve()
API_PORT = 8000
UI_PORT  = 8501
STARTUP_WAIT = 4          # seconds to let the API bind before launching Streamlit

# Inherit the caller's env and ensure the project root is on PYTHONPATH
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _banner():
    print()
    print("  " + "=" * 58)
    print("    FakeGuard  --  Fake News Detection System")
    print("  " + "=" * 58)
    print()


def _check_root():
    if not (ROOT / "src" / "app" / "main.py").exists():
        sys.exit(
            "[ERROR] start_all.py must be run from the project root.\n"
            "        e.g.  cd C:\\Users\\A\\Desktop\\fake-news-detector\n"
            "              python start_all.py"
        )


def _popen(label: str, cmd: list[str]) -> subprocess.Popen:
    """Launch a subprocess, streaming its output to this terminal."""
    print(f"  ▶  {label}")
    return subprocess.Popen(
        cmd,
        env=ENV,
        cwd=ROOT,
        # On Windows, prevent child windows from capturing the console
        # On Unix this has no effect
        creationflags=0,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    _check_root()
    _banner()

    # ── 1. Start FastAPI ───────────────────────────────────────────────────────
    api_cmd = [
        sys.executable, "-m", "uvicorn",
        "src.app.main:app",
        "--host", "0.0.0.0",
        "--port", str(API_PORT),
    ]
    api_proc = _popen(
        f"FastAPI backend   →  http://localhost:{API_PORT}  "
        f"(docs: http://localhost:{API_PORT}/docs)",
        api_cmd,
    )

    # ── 2. Brief pause so the API can bind the port ────────────────────────────
    print(f"\n  ⏳  Waiting {STARTUP_WAIT} s for API to initialise ...\n")
    time.sleep(STARTUP_WAIT)

    # ── 3. Start Streamlit ─────────────────────────────────────────────────────
    st_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/app/streamlit_app.py",
        "--server.port", str(UI_PORT),
        "--server.headless", "false",
    ]
    st_proc = _popen(
        f"Streamlit dashboard →  http://localhost:{UI_PORT}",
        st_cmd,
    )

    print()
    print("  " + "=" * 58)
    print(f"    API       :  http://localhost:{API_PORT}")
    print(f"    API docs  :  http://localhost:{API_PORT}/docs")
    print(f"    Dashboard :  http://localhost:{UI_PORT}")
    print()
    print("    Press Ctrl+C once to stop both services.")
    print("  " + "=" * 58)
    print()

    # ── 4. Graceful shutdown on Ctrl+C / SIGTERM ──────────────────────────────
    def _shutdown(sig=None, frame=None):
        print("\n  Shutting down …")
        for proc in (api_proc, st_proc):
            if proc.poll() is None:          # still running
                proc.terminate()
        for proc in (api_proc, st_proc):
            try:
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("  All services stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # On Windows signal.SIGBREAK is available; map it to the same handler
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _shutdown)   # type: ignore[attr-defined]

    # ── 5. Watch-loop: restart if a process crashes ────────────────────────────
    while True:
        api_alive = api_proc.poll() is None
        st_alive  = st_proc.poll()  is None

        if not api_alive:
            print(
                f"\n  [WARN] API process exited with code {api_proc.returncode}.\n"
                "         Shutting down Streamlit too."
            )
            _shutdown()

        if not st_alive:
            print(
                f"\n  [WARN] Streamlit process exited with code {st_proc.returncode}.\n"
                "         Shutting down API too."
            )
            _shutdown()

        time.sleep(2)


if __name__ == "__main__":
    main()
