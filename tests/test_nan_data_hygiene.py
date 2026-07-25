"""
Regression tests for NaN price rows (Yahoo weekend placeholder bars).

A single NaN Close in the newest row used to render "$nan" quote tiles,
"+nan%" MA distances, and poison rolling SMA windows across the dashboard.
Every fetch path must drop those rows before indicators are computed.
"""
import numpy as np
import pandas as pd


def _frame_with_nan_tail(n=60):
    idx = pd.bdate_range("2026-01-02", periods=n)
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 110, n),
            "High": np.linspace(101, 111, n),
            "Low": np.linspace(99, 109, n),
            "Close": np.linspace(100, 110, n),
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )
    # Yahoo-style placeholder: newest session exists but has no prices yet
    df.iloc[-1, df.columns.get_loc("Close")] = np.nan
    return df


def test_yfinance_normalize_drops_nan_close_rows():
    from core.data import _yfinance_normalize

    df = _frame_with_nan_tail()
    out = _yfinance_normalize(df)
    assert not out.empty
    assert not out["Close"].isna().any()
    # The NaN placeholder row (last bdate) must be gone
    assert out.index[-1] == df.index[-2]


def test_backtest_fetch_drops_nan_close(monkeypatch):
    """core/backtest._fetch applies the same hygiene: dropna(subset=['Close'])."""
    import core.backtest as bt

    df = _frame_with_nan_tail()
    monkeypatch.setattr(bt.yf, "download", lambda *a, **k: df.copy())
    out = bt._fetch("QQQ", "2026-01-01", "2026-04-01")
    assert not out["Close"].isna().any()
    assert out.index[-1] == df.index[-2]
