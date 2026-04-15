#!/usr/bin/env python3
"""Rebuild data/backtest_cache.json from the V6 engine (finalize_open_position=False).

Run after logic changes:  python scripts/regenerate_backtest_cache.py
"""
import datetime as dt
import json
import os
import sys

import pandas as pd

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.backtest import _build_year_result, _run_continuous  # noqa: E402


def _trade_to_dict(t) -> dict:
    return {
        "entry_date": t.entry_date,
        "exit_date": t.exit_date,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "return_pct": t.return_pct,
        "signal_type": t.signal_type,
        "duration_days": t.duration_days,
        "outcome": t.outcome,
        "shares": t.shares,
        "cash_deployed": t.cash_deployed,
        "portfolio_before": t.portfolio_before,
        "portfolio_after": t.portfolio_after,
        "cash_after": t.cash_after,
    }


def main() -> None:
    current_year = dt.date.today().year
    equity, trades_by_year, tqqq, qqq, snap = _run_continuous(
        2011, current_year, finalize_open_position=False
    )
    if not equity:
        print("Engine returned no equity — aborting.", file=sys.stderr)
        sys.exit(1)

    out = []
    for year in range(2011, current_year + 1):
        r = _build_year_result(year, equity, trades_by_year.get(year, []), tqqq, qqq)
        if not r:
            continue
        ys, ye = f"{year}-01-01", f"{year + 1}-01-01"
        ts0, ts1 = pd.Timestamp(ys), pd.Timestamp(ye)
        yeq = {}
        for k, v in equity.items():
            if ts0 <= k < ts1:
                yeq[pd.Timestamp(k).isoformat()] = v

        out.append({
            "year": r.year,
            "total_return_pct": r.total_return_pct,
            "num_trades": r.num_trades,
            "win_rate_pct": r.win_rate_pct,
            "avg_win_pct": r.avg_win_pct,
            "avg_loss_pct": r.avg_loss_pct,
            "max_win_pct": r.max_win_pct,
            "max_loss_pct": r.max_loss_pct,
            "best_trade": r.best_trade,
            "worst_trade": r.worst_trade,
            "tqqq_buy_hold_pct": r.tqqq_buy_hold_pct,
            "qqq_buy_hold_pct": r.qqq_buy_hold_pct,
            "starting_value": r.starting_value,
            "ending_value": r.ending_value,
            "max_drawdown_pct": r.max_drawdown_pct,
            "trades": [_trade_to_dict(t) for t in r.trades],
            "equity": yeq,
        })

    path = os.path.join(ROOT, "data", "backtest_cache.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {path} ({len(out)} years). Live snapshot as_of={snap.as_of_date if snap else None}")


if __name__ == "__main__":
    main()
