"""Ensure PythonClient root is on sys.path for EmbodiedMAS imports."""
from __future__ import annotations

import sys
from pathlib import Path

_PYTHONCLIENT_ROOT = Path(__file__).resolve().parent.parent


def ensure_pythonclient_on_path() -> Path:
    s = str(_PYTHONCLIENT_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)
    return _PYTHONCLIENT_ROOT
