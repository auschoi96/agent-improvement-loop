"""Pristine L1 check for the throwaway example fixture.

Run with ``cwd`` = the arm's workspace (set by the harness). Exits ``0`` iff the
agent's edited ``solution.py`` implements ``add`` as ``return a + b``. Reads the
file cwd-relative so it reflects whatever that arm actually produced.
"""

import sys
from pathlib import Path

solution = Path("solution.py")
text = solution.read_text(encoding="utf-8") if solution.exists() else ""
sys.exit(0 if "return a + b" in text else 1)
