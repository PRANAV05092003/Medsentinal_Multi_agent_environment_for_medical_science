#!/usr/bin/env python3
"""
MedSentinel — Start Everything
================================

Starts both the Python backend API and the React UI together.

Usage:
  python start.py                    # Start both backend + frontend
  python start.py --backend-only     # Only the Python API (port 8000)
  python start.py --frontend-only    # Only the React UI (port 8080)

Requirements:
  Backend: pip install -r requirements.txt
  Frontend: cd ui && npm install (or bun install)

Then open: http://localhost:8080
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import webbrowser

REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
UI_DIR      = os.path.join(REPO_ROOT, "ui")
BACKEND_CMD = [sys.executable, "-m", "uvicorn", "api_server:app",
               "--host", "0.0.0.0", "--port", "8000", "--reload"]
def _frontend_dev_cmd():
    """Resolve bun/npm to full paths so subprocess works on Windows (CreateProcess)."""
    bun = shutil.which("bun")
    if bun:
        try:
            subprocess.run([bun, "--version"], capture_output=True, check=True)
            return [bun, "run", "dev"]
        except Exception:
            pass
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm:
        try:
            subprocess.run([npm, "--version"], capture_output=True, check=True)
            return [npm, "run", "dev"]
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Start MedSentinel")
    parser.add_argument("--backend-only",  action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    args = parser.parse_args()

    procs = []

    print("=" * 55)
    print("  MedSentinel -- Starting Up")
    print("=" * 55)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    print(
        "  Anthropic API key: "
        + ("set (real agents)" if api_key else "not set (rule-based fallback)")
    )
    print()

    if not args.frontend_only:
        print("  Starting Python backend on http://localhost:8000 ...")
        backend = subprocess.Popen(
            BACKEND_CMD,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
        )
        procs.append(backend)
        time.sleep(2)

    if not args.backend_only:
        frontend_cmd = _frontend_dev_cmd()
        if not frontend_cmd:
            print("  ERROR: Node.js / bun not found. Run: cd ui && npm install first")
        else:
            # Check if node_modules exists
            if not os.path.exists(os.path.join(UI_DIR, "node_modules")):
                print("  Installing UI dependencies...")
                subprocess.run(
                    [frontend_cmd[0], "install"],
                    cwd=UI_DIR, check=True
                )
            print("  Starting React UI on http://localhost:8080 ...")
            frontend = subprocess.Popen(
                frontend_cmd,
                cwd=UI_DIR,
                env=os.environ.copy(),
            )
            procs.append(frontend)
            time.sleep(3)
            webbrowser.open("http://localhost:8080")

    if not procs:
        print("Nothing started.")
        return

    print()
    print("=" * 55)
    print("  MedSentinel is running!")
    print()
    if not args.backend_only:
        print("  UI:      http://localhost:8080")
    if not args.frontend_only:
        print("  API:     http://localhost:8000")
        print("  Docs:    http://localhost:8000/docs")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 55)

    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
