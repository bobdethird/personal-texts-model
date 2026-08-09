"""Ensure imessage_mlx resolves from src/ even when the venv's editable
.pth file is skipped (macOS/iCloud intermittently flags it hidden, and
Python silently ignores hidden .pth files)."""

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
