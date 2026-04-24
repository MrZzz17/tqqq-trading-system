"""
Testable math for the Streamlit dashboard (no UI imports).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from core.backtest import LiveSnapshot, YearResult  # noqa: F401


def compute_equity_max_drawdown(
    bt_equity: Dict[Any, float],
) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Running peak-to-trough max drawdown on the equity time series.

    Returns:
        mdd_pct: most negative (v/peak-1)*100, e.g. -48.8
        peak_d: high-water mark date (last peak on or before the max-DD trough)
        trough_d: date of the max drawdown point
    """
    if not bt_equity:
        return 0.0, None, None
    s = (
        pd.Series(
            {pd.Timestamp(k): float(v) for k, v in bt_equity.items()}
        )
        .sort_index()
    )
    running_peak = s.cummax()
    dd_pct = (s / running_peak - 1) * 100.0
    mdd = float(dd_pct.min())
    trough_d = pd.Timestamp(dd_pct.idxmin())
    sub = s.loc[:trough_d]
    peak_d = sub.idxmax()

    return mdd, pd.Timestamp(peak_d), trough_d


def trade_count_breakdown(
    bt_results: List,
    live: Any,
) -> Tuple[int, int]:
    """
    (closed_trades, open_lots) where open_lots is 0 or 1 if still long
    and not already represented as a closed round-trip in the year rows.
    """
    closed = sum(int(r.num_trades) for r in bt_results) if bt_results else 0
    has_open_leg = bool(bt_results) and any(
        getattr(r, "open_leg", None) for r in bt_results
    )
    in_live = bool(
        live
        and getattr(live, "in_position", False)
        and (float(getattr(live, "shares", 0) or 0) > 1e-6)
    )
    open_ = 1 if (has_open_leg or in_live) else 0
    return closed, open_


def year_result_for_year(bt_results: List, y: int):
    for r in bt_results:
        if r.year == y:
            return r
    return None
