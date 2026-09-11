#!/usr/bin/env python3
"""
run_dashboard.py — Launch the GTCRN Speech Enhancement Interactive Model Dashboard.

Usage:
    python scripts/run_dashboard.py [--port 8080] [--host 0.0.0.0]
"""
import argparse
import os
import sys
import webbrowser
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Auto-reexec with virtual environment if needed
if sys.prefix == sys.base_prefix:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = repo_root.parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)


def main():
    parser = argparse.ArgumentParser(description="Launch GTCRN Speech Enhancement Studio")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--host", "-H", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--open", action="store_true", help="Automatically open in web browser")
    args = parser.parse_args()

    # Verify model presence
    print("=" * 65)
    print("  🚀 SIH-26052: GTCRN Speech Enhancement Model Studio")
    print("=" * 65)

    ckpt_path = repo_root / "models" / "checkpoints" / "checkpoint_best.pth"
    onnx_path = repo_root / "models" / "gtcrn_finetuned_stream_int8.onnx"

    print(f" PyTorch Best Checkpoint : {'✓ Available' if ckpt_path.exists() else '✗ Missing'}")
    print(f" ONNX INT8 Quantized     : {'✓ Available' if onnx_path.exists() else '✗ Missing'}")
    print(f" Local Web Dashboard     : http://localhost:{args.port}")
    print("=" * 65)

    if args.open:
        try:
            webbrowser.open(f"http://localhost:{args.port}")
        except Exception:
            pass

    import uvicorn
    from sih26052.dashboard.server import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
