"""Unit tests for core/dashboard_metrics.py (B.5 #3)."""
import pandas as pd
import pytest

from core.dashboard_metrics import compute_equity_max_drawdown, trade_count_breakdown


def test_max_drawdown_matches_pandas_cummax():
    idx = pd.date_range("2011-01-01", periods=5, freq="B")
    eq = {
        idx[0]: 100_000.0,
        idx[1]: 200_000.0,
        idx[2]: 100_000.0,
        idx[3]: 50_000.0,
        idx[4]: 80_000.0,
    }
    mdd, peak, trough = compute_equity_max_drawdown(eq)
    assert mdd < 0
    assert peak is not None and trough is not None
    s = pd.Series(eq).sort_index()
    rp = s.cummax()
    dd = (s / rp - 1) * 100.0
    assert mdd == pytest.approx(float(dd.min()))
    assert trough == dd.idxmin()


def test_trade_count_closed_plus_open():
    class R:
        def __init__(self, n, open_leg=None):
            self.num_trades = n
            self.open_leg = open_leg

    closed, op = trade_count_breakdown([R(2), R(3)], None)
    assert closed == 5
    assert op in (0, 1)
