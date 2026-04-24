"""B.5 #6: equity vs TQQQ model chart use the same date windowing for Period=All."""
import datetime as dt

import pandas as pd
import pytest

import views.tqqq_dashboard as v


def _make_df(start: str, n: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 1e6},
        index=idx,
    )


def test_all_period_same_first_last_dates():
    tqqq = _make_df("2011-01-03", 4000)
    eq = {d: 100_000.0 * (1 + 0.0001 * i) for i, d in enumerate(tqqq.index)}

    d_eq, v_eq = v._filter_equity_series(eq, "All")
    tq = v._filter_tqqq_by_period(tqqq, "All")

    assert d_eq[0] == tq.index[0]
    assert d_eq[-1] == tq.index[-1]
