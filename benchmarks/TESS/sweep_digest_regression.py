#!/usr/bin/env python3
"""TESS regression sweep digest CLI."""

from __future__ import annotations

import sys
from pathlib import Path

_B = Path(__file__).resolve().parent
if str(_B) not in sys.path:
    sys.path.insert(0, str(_B))

from sweep_digest.digest import main_reg

if __name__ == "__main__":
    main_reg()
