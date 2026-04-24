"""
B.5 #4 — visual regression (633px / 1280px) is optional: requires Playwright + Streamlit
test server in CI. Enable when a frozen Streamlit + browser pipeline is set up.
"""
import pytest

pytestmark = pytest.mark.skip(
    reason="Playwright screenshot diffs not yet wired; run manually before major UI releases."
)


def test_dashboard_snapshot_633px():
    assert False  # placeholder for future Playwright + st run


def test_dashboard_snapshot_1280px():
    assert False
