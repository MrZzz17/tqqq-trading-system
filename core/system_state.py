"""
Single source of truth for everything the dashboard displays.

All fields are derived from the SAME data pipeline and rule functions the V6
trading engine uses (core/backtest.py): _fetch'd Yahoo bars, _indicators,
_make_weekly MACD, _find_ftd_signal, and the LiveSnapshot from the cached
engine run. No parallel signal implementations, no alternative regime
definitions. Context metrics that do NOT drive trading (distribution days)
are carried in explicitly-labeled context fields.
"""

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

import config
from core.backtest import (
    _fetch, _indicators, _make_weekly, _find_ftd_signal,
    get_dashboard_state, LiveSnapshot,
)
from core.indicators import count_distribution_days


@dataclass
class SystemState:
    """Engine-derived market + position state as of the last daily bar."""
    as_of_date: str
    tqqq_close: float
    qqq_close: float

    # ── Regime (exact engine entry rules: QQQ vs 200-day + weekly MACD sign) ──
    above_200: bool
    pct_above_200: float          # QQQ distance from its 200-day SMA, %
    sma200: float
    macd_pos: bool                # weekly MACD > 0 (engine's _make_weekly)
    macd_value: Optional[float]
    macd_rising: Optional[bool]   # MACD above its 9-week signal line
    regime_label: str             # "Strong Bull" | "Cautious Bull" | "Bear"
    regime_color: str
    target_alloc_pct: int         # 100 / 50 / 0 — engine target when entering

    # ── Entry signals (engine implementations) ──
    ftd_active: bool              # engine _find_ftd_signal on the last bar

    # ── Exit / safety rule states (engine rules, engine data) ──
    exit_200_active: bool         # QQQ closed below 200-day (engine exits on FIRST close)
    trail_armed: bool             # trailing stop armed (QQQ >3% above 200-day)
    trail_dd_pct: Optional[float] # portfolio drawdown from engine peak while long
    crash_detected: bool          # TQQQ -30% in 10 sessions
    tqqq_drop_10d: float

    # ── Engine cooldown state (from LiveSnapshot) ──
    cooldown_until: Optional[str]
    ftd_cooldown_until: Optional[str]

    # ── Position (engine LiveSnapshot passthrough) ──
    live: Optional[LiveSnapshot]

    # ── CONTEXT ONLY — never drives trades ──
    nasdaq_dist_days: int         # IBD-style distribution-day count on ^IXIC
    selling_pressure: str         # "Low" | "Elevated" | "High" (from dist days)


def _selling_pressure_label(count: int) -> str:
    if count >= config.DISTRIBUTION_DAY_CRITICAL:
        return "High"
    if count >= config.DISTRIBUTION_DAY_WARN:
        return "Elevated"
    return "Low"


@st.cache_data(ttl=config.STRATEGY_ENGINE_CACHE_SECONDS, show_spinner=False)
def _engine_frames(_cache_bust: int = 1):
    """Same tickers, same fetch window, same indicator warmup as _run_continuous."""
    start_year = 2011
    fetch_start = f"{start_year - 2}-01-01"
    fetch_end = (dt.date.today() + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    tqqq = _fetch("TQQQ", fetch_start, fetch_end)
    qqq = _fetch("QQQ", fetch_start, fetch_end)
    nasdaq = _fetch("^IXIC", fetch_start, fetch_end)
    if tqqq.empty or qqq.empty or nasdaq.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    tqqq = _indicators(tqqq)
    qqq = _indicators(qqq)
    nasdaq = _indicators(nasdaq)
    qqq_w = _make_weekly(qqq)
    return tqqq, qqq, nasdaq, qqq_w


def get_system_state() -> Optional[SystemState]:
    tqqq, qqq, nasdaq, qqq_w = _engine_frames()
    if tqqq.empty or qqq.empty or nasdaq.empty:
        return None

    live, _, _ = get_dashboard_state()

    last_d = tqqq.index[-1]
    price = float(tqqq.iloc[-1]["Close"])

    qq_idx = qqq.index.get_indexer([last_d], method="nearest")[0]
    qq_row = qqq.iloc[qq_idx]
    qq_close = float(qq_row["Close"])
    sma200_raw = qq_row.get("SMA_200")
    if sma200_raw is None or pd.isna(sma200_raw):
        return None
    sma200 = float(sma200_raw)
    above_200 = qq_close > sma200
    pct_above = (qq_close - sma200) / sma200 * 100

    # Weekly MACD exactly as the engine evaluates it (last completed weekly bar <= date)
    w_dates = qqq_w.index[qqq_w.index <= last_d]
    macd_value = float(qqq_w.loc[w_dates[-1], "MACD"]) if len(w_dates) else None
    macd_sig = float(qqq_w.loc[w_dates[-1], "MACD_signal"]) if len(w_dates) else None
    macd_pos = macd_value is not None and macd_value > 0
    macd_rising = (macd_value > macd_sig) if (macd_value is not None and macd_sig is not None) else None

    # Engine regime tiers (backtest.py: 1.0 if macd_pos and above_200 else 0.5; 0 below 200-day)
    if above_200 and macd_pos:
        regime_label, regime_color, target_alloc = "Strong Bull", "#17BF63", 100
    elif above_200:
        regime_label, regime_color, target_alloc = "Cautious Bull", "#FFAD1F", 50
    else:
        regime_label, regime_color, target_alloc = "Bear", "#E0245E", 0

    # Entry signal — the ENGINE's FTD detector, not the display-only variant
    nq_idx = nasdaq.index.get_indexer([last_d], method="nearest")[0]
    ftd_active = bool(_find_ftd_signal(nasdaq, nq_idx))

    # Exit / safety states, engine definitions
    exit_200_active = not above_200
    trail_armed = pct_above >= 3.0
    trail_dd = None
    if live and live.in_position and live.peak_portfolio > 0:
        trail_dd = (live.portfolio_value - live.peak_portfolio) / live.peak_portfolio * 100

    ti = len(tqqq) - 1
    drop_10d = 0.0
    if ti >= 10:
        rh = float(tqqq.iloc[ti - 10: ti + 1]["High"].max())
        drop_10d = (price - rh) / rh * 100
    crash_detected = drop_10d <= -30

    dist_count = len(count_distribution_days(nasdaq))

    return SystemState(
        as_of_date=last_d.strftime("%Y-%m-%d"),
        tqqq_close=round(price, 2),
        qqq_close=round(qq_close, 2),
        above_200=above_200,
        pct_above_200=round(pct_above, 2),
        sma200=round(sma200, 2),
        macd_pos=macd_pos,
        macd_value=round(macd_value, 2) if macd_value is not None else None,
        macd_rising=macd_rising,
        regime_label=regime_label,
        regime_color=regime_color,
        target_alloc_pct=target_alloc,
        ftd_active=ftd_active,
        exit_200_active=exit_200_active,
        trail_armed=trail_armed,
        trail_dd_pct=round(trail_dd, 2) if trail_dd is not None else None,
        crash_detected=crash_detected,
        tqqq_drop_10d=round(drop_10d, 2),
        cooldown_until=live.cooldown_until if live else None,
        ftd_cooldown_until=live.ftd_cooldown_until if live else None,
        live=live,
        nasdaq_dist_days=dist_count,
        selling_pressure=_selling_pressure_label(dist_count),
    )
