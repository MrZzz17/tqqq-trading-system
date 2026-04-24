"""
B.5 #1 & #5: fail CI if known-stale user-facing numbers or internal layout debug tags reappear.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ROOT / "views" / "tqqq_dashboard.py"
APP = ROOT / "app.py"

# Stale marketing numbers from the QA report (must not be hardcoded again)
BANNED_SUBSTRINGS = [
    "$8.6M",
    "8.6M",
    "Our system lost only -18%",
    "72 trades over 16 years",
    "V3-FULL-ROW-BALANCE",
]

# B.5 #5 — internal version tags like V3-FOO-BAR in user-facing code
_DEBUG_TAG = re.compile(r"\bV[0-9]+-[A-Z0-9_]+-[A-Z0-9_]+\b")


def test_no_banned_stale_prose_in_dashboard():
    text = VIEWS.read_text()
    for s in BANNED_SUBSTRINGS:
        assert s not in text, f"banned stale phrase found: {s!r}"


def test_no_debug_version_tags_in_py():
    for p in (ROOT / "views").rglob("*.py"):
        t = p.read_text()
        m = _DEBUG_TAG.search(t)
        assert m is None, f"possible debug tag in {p}: {m.group(0)}"


def test_loaded_time_uses_manual_12h():
    t = VIEWS.read_text()
    assert "_h12" in t and "_loaded_et" in t
