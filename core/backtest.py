"""
Historical backtest of the TQQQ swing trading system.
Simulates the buy/sell rules on historical data to produce yearly returns.

Key behavioral rules derived from actual trading interviews:
- Enter on FTD (all-in) or 3WK after a pullback (partial position)
- Hold for weeks/months -- don't exit on a single sell signal
- Sell incrementally when 2-3 sell signals cluster together
- Full exit only on the nuclear signal: 2 closes below 21-EMA
- Minimum hold period of 10 trading days before sell rules activate
- Cooldown of 10 days after exit before looking for new entries
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional

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


# ── Buy Signal Detection ─────────────────────────────────────────

MIN_CORRECTION_PCT = 7.0
FTD_RALLY_DAY_MIN = 4
FTD_GAIN_MIN = 1.25
MIN_HOLD_DAYS = 10
POST_EXIT_COOLDOWN = 10


def _find_ftd_signal(nasdaq: pd.DataFrame, idx: int) -> bool:
    """
    Detect a Follow-Through Day at position idx.

    A valid FTD requires:
    1. The Nasdaq must have declined >= MIN_CORRECTION_PCT from a recent high
    2. A rally attempt begins (index starts making higher closes from the low)
    3. On day 4+ of the rally attempt, the index gains >= 1.25% on higher volume

    The FTD signal stays valid until the index undercuts the rally low.
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

    rally_day = 0
    rally_low = trough_val

    for i in range(trough_abs_idx + 1, len(window)):
        if closes[i] > closes[i - 1]:
            rally_day += 1
        else:
            if closes[i] < rally_low:
                rally_day = 0
                rally_low = closes[i]
                continue
            rally_day = max(rally_day - 1, 0)

        if rally_day >= FTD_RALLY_DAY_MIN and i == len(window) - 1:
            daily_gain = (closes[i] / closes[i - 1] - 1) * 100
            vol_higher = volumes[i] > volumes[i - 1]
            if daily_gain >= FTD_GAIN_MIN and vol_higher:
                return True

    return False


def _find_3wk_signal(qqq: pd.DataFrame, tqqq: pd.DataFrame, idx: int) -> bool:
    """
    Detect a 3 White Knights pattern, but ONLY after a meaningful pullback.

    Requirements:
    1. 3 consecutive days of higher highs AND higher lows on QQQ
    2. TQQQ must be within 5% above or below its 50-day or 21-day EMA
       (i.e., near a key moving average after a pullback, not extended)
    """
    if idx < 50:
        return False

    qqq_slice = qqq.iloc[idx - 2: idx + 1]
    if len(qqq_slice) < 3:
        return False

    higher_highs = all(
        qqq_slice.iloc[j]["High"] > qqq_slice.iloc[j - 1]["High"]
        for j in range(1, 3)
    )
    higher_lows = all(
        qqq_slice.iloc[j]["Low"] > qqq_slice.iloc[j - 1]["Low"]
        for j in range(1, 3)
    )
    if not (higher_highs and higher_lows):
        return False

    tqqq_row = tqqq.iloc[idx]
    close = float(tqqq_row["Close"])
    sma50 = tqqq_row.get("SMA_50")
    ema21 = tqqq_row.get("EMA_21")

    near_ma = False
    if sma50 and not pd.isna(sma50):
        dist = abs(close - float(sma50)) / float(sma50)
        if dist < 0.05 or close < float(sma50):
            near_ma = True
    if ema21 and not pd.isna(ema21):
        dist = abs(close - float(ema21)) / float(ema21)
        if dist < 0.03 or close < float(ema21):
            near_ma = True

    return near_ma


# ── Sell Signal Detection ─────────────────────────────────────────

def _count_sell_signals(tqqq: pd.DataFrame, nasdaq: pd.DataFrame, idx: int) -> tuple[int, bool]:
    """
    Count how many of the 9 sell rules are currently active.
    Returns (count, is_nuclear) where is_nuclear means 2 closes below 21-EMA.
    """
    if idx < 3:
        return 0, False

    row = tqqq.iloc[idx]
    prev = tqqq.iloc[idx - 1]
    close = float(row["Close"])
    signals = 0
    nuclear = False

    # Rule 1: Near 52-week high
    lookback_252 = min(252, idx + 1)
    high_52w = tqqq["High"].iloc[idx - lookback_252 + 1: idx + 1].max()
    if float(row["High"]) >= high_52w * 0.995:
        signals += 1

    # Rule 2: New high on declining volume (5-day context)
    if idx >= 5:
        recent = tqqq.iloc[idx - 4: idx + 1]
        making_high = close > float(recent["Close"].iloc[:-1].max())
        vol_down = float(row["Volume"]) < float(recent["Volume"].iloc[:-1].mean())
        if making_high and vol_down:
            signals += 1

    # Rule 3: 4+ distribution days on Nasdaq
    if idx >= 25:
        dist_count = 0
        nq_window = nasdaq.iloc[idx - 24: idx + 1]
        for i in range(1, len(nq_window)):
            pct = (nq_window.iloc[i]["Close"] / nq_window.iloc[i - 1]["Close"] - 1) * 100
            vol_up = nq_window.iloc[i]["Volume"] > nq_window.iloc[i - 1]["Volume"]
            if pct <= -0.2 and vol_up:
                dist_count += 1
        if dist_count >= 4:
            signals += 1
        if dist_count >= 5:
            signals += 1

    # Rule 4: 3 consecutive down days
    if idx >= 3:
        three_down = all(
            tqqq.iloc[idx - 2 + j]["Close"] < tqqq.iloc[idx - 3 + j]["Close"]
            for j in range(3)
        )
        if three_down:
            signals += 1

    # Rule 5: Below 10-day MA on above-average volume
    sma10 = row.get("SMA_10")
    vol_avg = row.get("Vol_SMA_50")
    if sma10 and not pd.isna(sma10) and vol_avg and not pd.isna(vol_avg):
        if close < float(sma10) and float(row["Volume"]) > float(vol_avg) * 1.1:
            signals += 1

    # Rule 6: 3 down days + rising volume + lower highs & lows
    if idx >= 3:
        three_down = all(
            tqqq.iloc[idx - 2 + j]["Close"] < tqqq.iloc[idx - 3 + j]["Close"]
            for j in range(3)
        )
        vol_rising = all(
            tqqq.iloc[idx - 2 + j]["Volume"] > tqqq.iloc[idx - 3 + j]["Volume"]
            for j in range(3)
        )
        lower_hl = all(
            tqqq.iloc[idx - 2 + j]["High"] < tqqq.iloc[idx - 3 + j]["High"]
            and tqqq.iloc[idx - 2 + j]["Low"] < tqqq.iloc[idx - 3 + j]["Low"]
            for j in range(3)
        )
        if three_down and vol_rising and lower_hl:
            signals += 1

    # Rule 9 (nuclear): 2 consecutive closes below 21-EMA
    ema21 = row.get("EMA_21")
    prev_ema21 = prev.get("EMA_21")
    if ema21 and not pd.isna(ema21) and prev_ema21 and not pd.isna(prev_ema21):
        if close < float(ema21) and float(prev["Close"]) < float(prev_ema21):
            nuclear = True
            signals += 2

    return signals, nuclear


# ── Main Backtest Loop ────────────────────────────────────────────

def run_backtest_year(year: int) -> Optional[YearResult]:
    start = f"{year - 1}-01-01"
    end_dt = min(dt.date(year, 12, 31), dt.date.today())
    end = end_dt.strftime("%Y-%m-%d")

    tqqq_df = _fetch("TQQQ", start, end)
    qqq_df = _fetch("QQQ", start, end)
    nasdaq_df = _fetch("^IXIC", start, end)

    if tqqq_df.empty or qqq_df.empty or nasdaq_df.empty:
        return None

    tqqq_df = _indicators(tqqq_df)
    qqq_df = _indicators(qqq_df)
    nasdaq_df = _indicators(nasdaq_df)

    year_start = f"{year}-01-01"
    tqqq_year = tqqq_df[tqqq_df.index >= year_start]
    if len(tqqq_year) < 10:
        return None

    tqqq_bh = (float(tqqq_year["Close"].iloc[-1]) / float(tqqq_year["Close"].iloc[0]) - 1) * 100
    qqq_year = qqq_df[qqq_df.index >= year_start]
    qqq_bh = (float(qqq_year["Close"].iloc[-1]) / float(qqq_year["Close"].iloc[0]) - 1) * 100

    trades: List[Trade] = []
    in_position = False
    position_size = 0.0
    entry_price = 0.0
    entry_date = ""
    entry_idx = 0
    signal_type = ""
    cooldown_until = 0

    portfolio = 100.0

    all_idx = list(tqqq_df.index)
    year_indices = [i for i, d in enumerate(all_idx) if d >= pd.Timestamp(year_start)]

    for idx in year_indices:
        date = all_idx[idx]
        nq_idx = nasdaq_df.index.get_indexer([date], method="nearest")[0]
        qq_idx = qqq_df.index.get_indexer([date], method="nearest")[0]

        if not in_position:
            if idx < cooldown_until:
                continue

            is_ftd = _find_ftd_signal(nasdaq_df, nq_idx)
            is_3wk = _find_3wk_signal(qqq_df, tqqq_df, idx)

            if is_ftd:
                in_position = True
                position_size = 1.0
                entry_price = float(tqqq_df.iloc[idx]["Close"])
                entry_date = date.strftime("%Y-%m-%d")
                entry_idx = idx
                signal_type = "FTD"
            elif is_3wk:
                in_position = True
                position_size = 0.5
                entry_price = float(tqqq_df.iloc[idx]["Close"])
                entry_date = date.strftime("%Y-%m-%d")
                entry_idx = idx
                signal_type = "3WK"
        else:
            held_days = idx - entry_idx
            if held_days < MIN_HOLD_DAYS:
                current_price = float(tqqq_df.iloc[idx]["Close"])
                loss_from_entry = ((current_price - entry_price) / entry_price) * 100
                if loss_from_entry < -15.0:
                    ret = loss_from_entry * position_size
                    portfolio *= (1 + ret / 100)
                    trades.append(Trade(
                        entry_date=entry_date,
                        exit_date=date.strftime("%Y-%m-%d"),
                        entry_price=round(entry_price, 2),
                        exit_price=round(current_price, 2),
                        return_pct=round(loss_from_entry, 2),
                        signal_type=signal_type,
                        duration_days=held_days,
                        outcome="Loss",
                    ))
                    in_position = False
                    cooldown_until = idx + POST_EXIT_COOLDOWN
                continue

            sig_count, is_nuclear = _count_sell_signals(tqqq_df, nasdaq_df, idx)

            if is_nuclear:
                current_price = float(tqqq_df.iloc[idx]["Close"])
                ret_pct = ((current_price - entry_price) / entry_price) * 100
                actual_ret = ret_pct * position_size
                portfolio *= (1 + actual_ret / 100)
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date.strftime("%Y-%m-%d"),
                    entry_price=round(entry_price, 2),
                    exit_price=round(current_price, 2),
                    return_pct=round(ret_pct, 2),
                    signal_type=signal_type,
                    duration_days=held_days,
                    outcome="Win" if ret_pct > 0 else "Loss",
                ))
                in_position = False
                cooldown_until = idx + POST_EXIT_COOLDOWN

            elif sig_count >= 3:
                trim = 0.3
                current_price = float(tqqq_df.iloc[idx]["Close"])
                ret_pct = ((current_price - entry_price) / entry_price) * 100
                trimmed_ret = ret_pct * trim * position_size
                portfolio *= (1 + trimmed_ret / 100)
                position_size *= (1 - trim)
                if position_size < 0.15:
                    trades.append(Trade(
                        entry_date=entry_date,
                        exit_date=date.strftime("%Y-%m-%d"),
                        entry_price=round(entry_price, 2),
                        exit_price=round(current_price, 2),
                        return_pct=round(ret_pct, 2),
                        signal_type=signal_type,
                        duration_days=held_days,
                        outcome="Win" if ret_pct > 0 else "Loss",
                    ))
                    in_position = False
                    cooldown_until = idx + POST_EXIT_COOLDOWN

            elif sig_count >= 2:
                trim = 0.1
                current_price = float(tqqq_df.iloc[idx]["Close"])
                ret_pct = ((current_price - entry_price) / entry_price) * 100
                trimmed_ret = ret_pct * trim * position_size
                portfolio *= (1 + trimmed_ret / 100)
                position_size *= (1 - trim)

    if in_position:
        current_price = float(tqqq_year["Close"].iloc[-1])
        ret_pct = ((current_price - entry_price) / entry_price) * 100
        actual_ret = ret_pct * position_size
        portfolio *= (1 + actual_ret / 100)
        held_days = len(tqqq_year) - max(0, entry_idx - year_indices[0])
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=tqqq_year.index[-1].strftime("%Y-%m-%d"),
            entry_price=round(entry_price, 2),
            exit_price=round(current_price, 2),
            return_pct=round(ret_pct, 2),
            signal_type=signal_type,
            duration_days=max(held_days, 1),
            outcome="Win" if ret_pct > 0 else "Loss",
        ))

    wins = [t for t in trades if t.return_pct > 0]
    losses = [t for t in trades if t.return_pct <= 0]
    total_return = portfolio - 100.0

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
    current_year = dt.date.today().year
    years = [2022, 2023, 2024, 2025]
    if current_year > 2025:
        years.append(current_year)
    results = []
    for y in years:
        r = run_backtest_year(y)
        if r:
            results.append(r)
    return results
