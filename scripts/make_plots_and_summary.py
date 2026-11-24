#!/usr/bin/env python3
"""Command-line entry point for :mod:`mqmas.reporting`."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from mqmas.reporting import main


if __name__ == "__main__":
    raise SystemExit(main())
