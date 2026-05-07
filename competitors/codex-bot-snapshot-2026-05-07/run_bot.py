#!/usr/bin/env python3
"""Convenience runner for the local package without installation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from morning_quant_bot.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

