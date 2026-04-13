"""
Historical backtest of the TQQQ swing trading system.

Runs a CONTINUOUS multi-year simulation (no year-boundary resets) and slices
the equity curve into annual returns.  This correctly models positions that
span year-ends, which is critical for capturing long uptrend holds.

Key behavioral rules from Vibha Jha interviews / IBD Live:
- Enter on FTD (all-in), 3WK (half position), or pullback-to-MA (half)
- FTD counts TRADING DAYS from trough, not net-up days (IBD standard)
- Hold for weeks/months; 21-EMA is the primary trailing stop
- Sell signals are flags -- only nuclear (2 closes < 21-EMA) triggers full exit
- In bear markets (TQQQ below 200-day), only enter on FTD -- no dip-buying
- She maintains exposure almost continuously in bull markets
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    signal_type: str
    duration_days: int
    outcome: str


@dataclass
class YearResult:
    year: int
    total_return_pct: float
    num_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    max_win_pct: float
    max_loss_pct: float
    best_trade: str
    worst_trade: str
    tqqq_buy_hold_pct: float
    qqq_buy_hold_pct: float
    trades: List[Trade] = field(default_factory=list)


# ── Data ──────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Vol_SMA_50"] = df["Volume"].rolling(50).mean()
    return df


# ── Constants ─────────────────────────────────────────────────────

MIN_CORRECTION_PCT = 7.0
FTD_RALLY_DAY_MIN = 4
FTD_GAIN_MIN = 1.25
MIN_HOLD_DAYS = 15
PULLBACK_ENTRY_PCT = 8.0
COOLDOWN_BULL = 3
COOLDOWN_BEAR = 7


# ── Buy Signal Detection ─────────────────────────────────────────

def _is_bull_regime(tqqq: pd.DataFrame, idx: int) -> bool:
    """Bull = TQQQ above its 200-day SMA and the SMA is rising."""
    sma200 = tqqq.iloc[idx].get("SMA_200")
    if sma200 is None or pd.isna(sma200):
        return True
    close = float(tqqq.iloc[idx]["Close"])
    if close < float(sma200):
        return False
    if idx >= 20:
        sma200_prev = tqqq.iloc[idx - 20].get("SMA_200")
        if sma200_prev and not pd.isna(sma200_prev):
            return float(sma200) >= float(sma200_prev)
    return True


def _find_ftd_signal(nasdaq: pd.DataFrame, idx: int) -> bool:
    """
    Follow-Through Day: After a >= 7% Nasdaq decline, on trading day 4+
    from the trough (counting ALL days, not net-up days), the Nasdaq gains
    >= 1.25% on higher volume than the prior session.
    """
    if idx < 50:
        return False

    window = nasdaq.iloc[max(0, idx - 60): idx + 1]
    closes = window["Close"].values
    volumes = window["Volume"].values

    peak_idx = np.argmax(closes)
    peak_val = closes[peak_idx]

    post_peak = closes[peak_idx:]
    if len(post_peak) < 5:
        return False

    trough_offset = np.argmin(post_peak)
    trough_val = post_peak[trough_offset]
    trough_abs_idx = peak_idx + trough_offset

    decline_pct = ((trough_val - peak_val) / peak_val) * 100
    if decline_pct > -MIN_CORRECTION_PCT:
        return False

    days_from_trough = len(window) - 1 - trough_abs_idx
    if days_from_trough < FTD_RALLY_DAY_MIN:
        return False

    last_i = len(window) - 1

    for i in range(trough_abs_idx + 1, last_i + 1):
        if closes[i] < trough_val:
            return False

    daily_gain = (closes[last_i] / closes[last_i - 1] - 1) * 100
    vol_higher = volumes[last_i] > volumes[last_i - 1]

    return daily_gain >= FTD_GAIN_MIN and vol_higher


def _find_3wk_signal(qqq: pd.DataFrame, tqqq: pd.DataFrame, idx: int) -> bool:
    """
    3 White Knights: 3 consecutive days of higher highs AND higher lows on QQQ,
    but only after TQQQ has pulled back >= 8% from its recent high.

    Additional filter: either TQQQ is above its 200-day MA, or the 50-day SMA
    must be rising (trending up). This prevents 3WK entries in deep bear markets
    where bounces routinely fail.
    """
    if idx < 50:
        return False

    qqq_slice = qqq.iloc[idx - 2: idx + 1]
    if len(qqq_slice) < 3:
        return False

    higher_highs = all(
        float(qqq_slice.iloc[j]["High"]) > float(qqq_slice.iloc[j - 1]["High"])
        for j in range(1, 3)
    )
    higher_lows = all(
        float(qqq_slice.iloc[j]["Low"]) > float(qqq_slice.iloc[j - 1]["Low"])
        for j in range(1, 3)
    )
    if not (higher_highs and higher_lows):
        return False

    lookback = min(40, idx)
    recent_high = float(tqqq.iloc[idx - lookback: idx + 1]["High"].max())
    current_close = float(tqqq.iloc[idx]["Close"])
    pullback_pct = ((current_close - recent_high) / recent_high) * 100

    if pullback_pct > -PULLBACK_ENTRY_PCT:
        return False

    sma200 = tqqq.iloc[idx].get("SMA_200")
    above_200 = (sma200 is None or pd.isna(sma200) or current_close >= float(sma200))

    if above_200:
        return True

    sma50 = tqqq.iloc[idx].get("SMA_50")
    if sma50 and not pd.isna(sma50) and idx >= 10:
        sma50_prev = tqqq.iloc[idx - 10].get("SMA_50")
        if sma50_prev and not pd.isna(sma50_prev):
            if float(sma50) > float(sma50_prev):
                return True

    return False


def _find_pullback_entry(tqqq: pd.DataFrame, idx: int) -> bool:
    """
    Pullback entry (bull markets only): TQQQ has pulled back >= 8% from
    its recent high and is now turning up near a key moving average.

    Conditions:
    1. TQQQ pulled back >= 8% from recent 30-day high
    2. Current close > prior close (turning up)
    3. Higher low vs 2 days ago
    4. Price near or below the 50-day SMA or 21-EMA (not extended)
    """
    if idx < 50:
        return False

    lookback = min(30, idx)
    recent_high = float(tqqq.iloc[idx - lookback: idx + 1]["High"].max())
    close = float(tqqq.iloc[idx]["Close"])
    prev_close = float(tqqq.iloc[idx - 1]["Close"])
    pullback_pct = ((close - recent_high) / recent_high) * 100

    if pullback_pct > -PULLBACK_ENTRY_PCT:
        return False

    if close <= prev_close:
        return False

    if idx >= 3:
        if float(tqqq.iloc[idx]["Low"]) <= float(tqqq.iloc[idx - 2]["Low"]):
            return False

    ema21 = tqqq.iloc[idx].get("EMA_21")
    sma50 = tqqq.iloc[idx].get("SMA_50")
    near_ma = False
    if sma50 and not pd.isna(sma50):
        if close <= float(sma50) * 1.05:
            near_ma = True
    if ema21 and not pd.isna(ema21):
        if close <= float(ema21) * 1.03:
            near_ma = True

    return near_ma


def _find_ma_retake_entry(tqqq: pd.DataFrame, idx: int) -> bool:
    """
    MA retake entry (bull markets only): TQQQ was below the 21-EMA
    yesterday and closes above it today, signaling the pullback is over.

    From her interview: "I try to get in after it has retaken the 10-week
    or 50-day and the 50-day is starting to trend up."
    """
    if idx < 50:
        return False

    ema21 = tqqq.iloc[idx].get("EMA_21")
    prev_ema21 = tqqq.iloc[idx - 1].get("EMA_21")

    if ema21 is None or pd.isna(ema21) or prev_ema21 is None or pd.isna(prev_ema21):
        return False

    close = float(tqqq.iloc[idx]["Close"])
    prev_close = float(tqqq.iloc[idx - 1]["Close"])

    crossed_above = prev_close < float(prev_ema21) and close > float(ema21)
    if not crossed_above:
        return False

    sma50 = tqqq.iloc[idx].get("SMA_50")
    if sma50 and not pd.isna(sma50):
        if idx >= 10:
            sma50_prev = tqqq.iloc[idx - 10].get("SMA_50")
            if sma50_prev and not pd.isna(sma50_prev):
                if float(sma50) < float(sma50_prev):
                    return False

    lookback = min(20, idx)
    recent_high = float(tqqq.iloc[idx - lookback: idx + 1]["High"].max())
    dip_pct = ((close - recent_high) / recent_high) * 100
    return dip_pct <= -5.0


# ── Sell Signal Detection ─────────────────────────────────────────

def _check_nuclear_exit(tqqq: pd.DataFrame, idx: int) -> bool:
    """2 consecutive closes below the 21-day EMA."""
    if idx < 2:
        return False
    row = tqqq.iloc[idx]
    prev = tqqq.iloc[idx - 1]
    ema21 = row.get("EMA_21")
    prev_ema21 = prev.get("EMA_21")
    if ema21 and not pd.isna(ema21) and prev_ema21 and not pd.isna(prev_ema21):
        return (float(row["Close"]) < float(ema21) and
                float(prev["Close"]) < float(prev_ema21))
    return False


def _count_distribution_days(nasdaq: pd.DataFrame, idx: int) -> int:
    """Count distribution days in the last 25 sessions."""
    if idx < 25:
        return 0
    window = nasdaq.iloc[idx - 24: idx + 1]
    current_close = float(nasdaq.iloc[idx]["Close"])
    count = 0
    for i in range(1, len(window)):
        pct = (float(window.iloc[i]["Close"]) / float(window.iloc[i - 1]["Close"]) - 1) * 100
        vol_up = float(window.iloc[i]["Volume"]) > float(window.iloc[i - 1]["Volume"])
        if pct <= -0.2 and vol_up:
            close_on_dd = float(window.iloc[i]["Close"])
            rally_since = ((current_close - close_on_dd) / close_on_dd) * 100
            if rally_since < 5.0:
                count += 1
    return count


def _check_severe_weakness(tqqq: pd.DataFrame, idx: int) -> bool:
    """3 consecutive down days + rising volume + lower highs & lows."""
    if idx < 3:
        return False
    return (
        all(float(tqqq.iloc[idx - 2 + j]["Close"]) < float(tqqq.iloc[idx - 3 + j]["Close"]) for j in range(3))
        and all(float(tqqq.iloc[idx - 2 + j]["Volume"]) > float(tqqq.iloc[idx - 3 + j]["Volume"]) for j in range(3))
        and all(
            float(tqqq.iloc[idx - 2 + j]["High"]) < float(tqqq.iloc[idx - 3 + j]["High"])
            and float(tqqq.iloc[idx - 2 + j]["Low"]) < float(tqqq.iloc[idx - 3 + j]["Low"])
            for j in range(3)
        )
    )


def _evaluate_exit(tqqq: pd.DataFrame, nasdaq: pd.DataFrame, idx: int,
                   entry_price: float, entry_idx: int, rally_low: float) -> str:
    """
    Returns "full_exit", "trim_heavy", "trim_light", or "hold".
    """
    held_days = idx - entry_idx
    close = float(tqqq.iloc[idx]["Close"])

    if held_days <= 5:
        if ((close - entry_price) / entry_price) * 100 < -20.0:
            return "full_exit"
        return "hold"

    if close < rally_low:
        return "full_exit"

    if held_days < MIN_HOLD_DAYS:
        if ((close - entry_price) / entry_price) * 100 < -18.0:
            return "full_exit"
        return "hold"

    if _check_nuclear_exit(tqqq, idx):
        return "full_exit"

    nq_idx_arr = nasdaq.index.get_indexer([tqqq.index[idx]], method="nearest")
    nq_idx = nq_idx_arr[0] if len(nq_idx_arr) > 0 else idx
    dist_days = _count_distribution_days(nasdaq, nq_idx)
    severe = _check_severe_weakness(tqqq, idx)

    if dist_days >= 5 and severe:
        return "full_exit"

    if dist_days >= 5 or severe:
        return "trim_heavy"

    if dist_days >= 4:
        sma10 = tqqq.iloc[idx].get("SMA_10")
        vol_avg = tqqq.iloc[idx].get("Vol_SMA_50")
        if (sma10 and not pd.isna(sma10) and vol_avg and not pd.isna(vol_avg)
                and close < float(sma10)
                and float(tqqq.iloc[idx]["Volume"]) > float(vol_avg) * 1.1):
            return "trim_heavy"
        return "trim_light"

    return "hold"


# ── Continuous Multi-Year Backtest ────────────────────────────────

def _run_continuous(start_year: int, end_year: int):
    """
    Run a single continuous simulation from start_year to end_year.
    Returns (equity_by_date, trades_by_year, tqqq_df, qqq_df).
    """
    fetch_start = f"{start_year - 1}-01-01"
    end_dt = min(dt.date(end_year, 12, 31), dt.date.today())
    fetch_end = end_dt.strftime("%Y-%m-%d")

    tqqq_df = _fetch("TQQQ", fetch_start, fetch_end)
    qqq_df = _fetch("QQQ", fetch_start, fetch_end)
    nasdaq_df = _fetch("^IXIC", fetch_start, fetch_end)

    if tqqq_df.empty or qqq_df.empty or nasdaq_df.empty:
        return {}, {}, pd.DataFrame(), pd.DataFrame()

    tqqq_df = _indicators(tqqq_df)
    qqq_df = _indicators(qqq_df)
    nasdaq_df = _indicators(nasdaq_df)

    sim_start = f"{start_year}-01-01"
    all_idx = list(tqqq_df.index)
    sim_indices = [i for i, d in enumerate(all_idx) if d >= pd.Timestamp(sim_start)]

    if not sim_indices:
        return {}, {}, tqqq_df, qqq_df

    in_position = False
    position_size = 0.0
    entry_price = 0.0
    entry_date = ""
    entry_idx = 0
    signal_type = ""
    cooldown_until = 0
    rally_low = 0.0

    portfolio = 100.0
    equity_by_date: Dict[pd.Timestamp, float] = {}
    trades_by_year: Dict[int, List[Trade]] = {}

    for year in range(start_year, end_year + 1):
        trades_by_year[year] = []

    def _record_trade(exit_date_ts, exit_price, ret, sig, held, year_key):
        t = Trade(
            entry_date=entry_date,
            exit_date=exit_date_ts.strftime("%Y-%m-%d"),
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            return_pct=round(ret, 2),
            signal_type=sig,
            duration_days=held,
            outcome="Win" if ret > 0 else "Loss",
        )
        trades_by_year.setdefault(year_key, []).append(t)

    for idx in sim_indices:
        date = all_idx[idx]
        current_year = date.year

        nq_idx = nasdaq_df.index.get_indexer([date], method="nearest")[0]
        qq_idx = qqq_df.index.get_indexer([date], method="nearest")[0]

        if not in_position:
            if idx < cooldown_until:
                equity_by_date[date] = portfolio
                continue

            bull = _is_bull_regime(tqqq_df, idx)
            is_ftd = _find_ftd_signal(nasdaq_df, nq_idx)
            is_3wk = _find_3wk_signal(qqq_df, tqqq_df, idx)
            is_pullback = bull and _find_pullback_entry(tqqq_df, idx)
            is_ma_retake = bull and _find_ma_retake_entry(tqqq_df, idx)

            entered = False
            if is_ftd:
                position_size = 1.0
                signal_type = "FTD"
                entered = True
            elif is_3wk:
                position_size = 0.5
                signal_type = "3WK"
                entered = True
            elif is_pullback:
                position_size = 0.5
                signal_type = "Pullback"
                entered = True
            elif is_ma_retake:
                position_size = 0.5
                signal_type = "MA Retake"
                entered = True

            if entered:
                in_position = True
                entry_price = float(tqqq_df.iloc[idx]["Close"])
                entry_date = date.strftime("%Y-%m-%d")
                entry_idx = idx
                lookback = min(20, idx)
                rally_low = float(tqqq_df.iloc[idx - lookback: idx + 1]["Low"].min())

            equity_by_date[date] = portfolio

        else:
            action = _evaluate_exit(tqqq_df, nasdaq_df, idx,
                                    entry_price, entry_idx, rally_low)
            current_price = float(tqqq_df.iloc[idx]["Close"])
            ret_pct = ((current_price - entry_price) / entry_price) * 100
            held_days = idx - entry_idx

            if action == "full_exit":
                actual_ret = ret_pct * position_size
                portfolio *= (1 + actual_ret / 100)
                _record_trade(date, current_price, ret_pct, signal_type, held_days, current_year)
                in_position = False
                bull = _is_bull_regime(tqqq_df, idx)
                cooldown_until = idx + (COOLDOWN_BULL if bull else COOLDOWN_BEAR)
                equity_by_date[date] = portfolio

            elif action == "trim_heavy":
                trim = 0.4
                trimmed_ret = ret_pct * trim * position_size
                portfolio *= (1 + trimmed_ret / 100)
                position_size *= (1 - trim)
                if position_size < 0.15:
                    remainder_ret = ret_pct * position_size
                    portfolio *= (1 + remainder_ret / 100)
                    _record_trade(date, current_price, ret_pct, signal_type, held_days, current_year)
                    in_position = False
                    cooldown_until = idx + COOLDOWN_BULL
                    equity_by_date[date] = portfolio
                else:
                    unrealized = ((current_price - entry_price) / entry_price) * position_size * 100
                    equity_by_date[date] = portfolio * (1 + unrealized / 100)

            elif action == "trim_light":
                trim = 0.2
                trimmed_ret = ret_pct * trim * position_size
                portfolio *= (1 + trimmed_ret / 100)
                position_size *= (1 - trim)
                if position_size < 0.15:
                    remainder_ret = ret_pct * position_size
                    portfolio *= (1 + remainder_ret / 100)
                    _record_trade(date, current_price, ret_pct, signal_type, held_days, current_year)
                    in_position = False
                    cooldown_until = idx + COOLDOWN_BULL
                    equity_by_date[date] = portfolio
                else:
                    unrealized = ((current_price - entry_price) / entry_price) * position_size * 100
                    equity_by_date[date] = portfolio * (1 + unrealized / 100)

            else:
                unrealized = ((current_price - entry_price) / entry_price) * position_size * 100
                equity_by_date[date] = portfolio * (1 + unrealized / 100)

    if in_position:
        last_date = all_idx[sim_indices[-1]]
        current_price = float(tqqq_df.iloc[sim_indices[-1]]["Close"])
        ret_pct = ((current_price - entry_price) / entry_price) * 100
        actual_ret = ret_pct * position_size
        portfolio *= (1 + actual_ret / 100)
        held_days = sim_indices[-1] - entry_idx
        _record_trade(last_date, current_price, ret_pct, signal_type,
                      max(held_days, 1), last_date.year)
        equity_by_date[last_date] = portfolio

    return equity_by_date, trades_by_year, tqqq_df, qqq_df


def run_backtest_year(year: int) -> Optional[YearResult]:
    """Run a single-year backtest (standalone, no carryover)."""
    result = _run_continuous(year, year)
    equity, trades_by_year, tqqq_df, qqq_df = result
    if not equity:
        return None
    return _build_year_result(year, equity, trades_by_year.get(year, []), tqqq_df, qqq_df)


def _build_year_result(year: int, equity: Dict, trades: List[Trade],
                       tqqq_df: pd.DataFrame, qqq_df: pd.DataFrame) -> Optional[YearResult]:
    year_start = f"{year}-01-01"
    tqqq_year = tqqq_df[tqqq_df.index >= year_start]
    qqq_year = qqq_df[qqq_df.index >= year_start]

    if tqqq_year.empty or qqq_year.empty:
        return None

    year_end = f"{year + 1}-01-01"
    tqqq_year = tqqq_year[tqqq_year.index < year_end]
    qqq_year = qqq_year[qqq_year.index < year_end]

    if len(tqqq_year) < 5:
        return None

    tqqq_bh = (float(tqqq_year["Close"].iloc[-1]) / float(tqqq_year["Close"].iloc[0]) - 1) * 100
    qqq_bh = (float(qqq_year["Close"].iloc[-1]) / float(qqq_year["Close"].iloc[0]) - 1) * 100

    year_equity = {d: v for d, v in equity.items()
                   if d >= pd.Timestamp(year_start) and d < pd.Timestamp(year_end)}

    if not year_equity:
        return None

    sorted_dates = sorted(year_equity.keys())
    start_val = year_equity[sorted_dates[0]]
    end_val = year_equity[sorted_dates[-1]]

    if start_val == 0:
        return None
    total_return = ((end_val / start_val) - 1) * 100

    wins = [t for t in trades if t.return_pct > 0]
    losses = [t for t in trades if t.return_pct <= 0]
    best = max(trades, key=lambda t: t.return_pct) if trades else None
    worst = min(trades, key=lambda t: t.return_pct) if trades else None

    return YearResult(
        year=year,
        total_return_pct=round(total_return, 2),
        num_trades=len(trades),
        win_rate_pct=round(len(wins) / len(trades) * 100, 1) if trades else 0,
        avg_win_pct=round(np.mean([t.return_pct for t in wins]), 2) if wins else 0,
        avg_loss_pct=round(np.mean([t.return_pct for t in losses]), 2) if losses else 0,
        max_win_pct=round(max(t.return_pct for t in wins), 2) if wins else 0,
        max_loss_pct=round(min(t.return_pct for t in losses), 2) if losses else 0,
        best_trade=f"{best.entry_date} to {best.exit_date} ({best.return_pct:+.1f}%)" if best else "N/A",
        worst_trade=f"{worst.entry_date} to {worst.exit_date} ({worst.return_pct:+.1f}%)" if worst else "N/A",
        tqqq_buy_hold_pct=round(tqqq_bh, 2),
        qqq_buy_hold_pct=round(qqq_bh, 2),
        trades=trades,
    )


def run_all_backtests() -> List[YearResult]:
    """Run a continuous simulation across all years and return per-year results."""
    current_year = dt.date.today().year
    start_year = 2022
    end_year = current_year

    equity, trades_by_year, tqqq_df, qqq_df = _run_continuous(start_year, end_year)

    if not equity:
        return []

    results = []
    for year in range(start_year, end_year + 1):
        r = _build_year_result(year, equity, trades_by_year.get(year, []), tqqq_df, qqq_df)
        if r:
            results.append(r)

    return results
