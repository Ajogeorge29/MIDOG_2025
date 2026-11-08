"""Repository paths with environment-variable overrides."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("MIDOG_DATA_ROOT", REPO_ROOT / "data" / "raw")).resolve()
REPORT_ROOT = Path(
    os.getenv("MIDOG_REPORT_ROOT", REPO_ROOT / "reports" / "generated")
).resolve()
