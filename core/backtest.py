"""
Historical backtest of the TQQQ swing trading system.
Simulates the buy/sell rules on historical data to produce yearly returns.
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import config


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    signal_type: str
    duration_days: int
    outcome: str  # "Win" or "Loss"


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


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_backtest_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Vol_SMA_50"] = df["Volume"].rolling(50).mean()
    return df


def _detect_ftd(nasdaq_df: pd.DataFrame, idx: int) -> bool:
    """Check if day at idx is a Follow-Through Day."""
    if idx < 10:
        return False
    lookback = nasdaq_df.iloc[max(0, idx - 30):idx + 1]
    if len(lookback) < 5:
        return False

    trough_pos = lookback["Close"].values.argmin()
    rally_day = 0
    for i in range(trough_pos + 1, len(lookback)):
        if lookback.iloc[i]["Close"] > lookback.iloc[i - 1]["Close"]:
            rally_day += 1
        else:
            rally_day = 0
        if i == len(lookback) - 1 and rally_day >= 4:
            pct = (lookback.iloc[i]["Close"] / lookback.iloc[i - 1]["Close"] - 1) * 100
            vol_higher = lookback.iloc[i]["Volume"] > lookback.iloc[i - 1]["Volume"]
            if pct >= 1.25 and vol_higher:
                return True
    return False


def _detect_3wk(qqq_df: pd.DataFrame, idx: int) -> bool:
    """Check if day at idx completes a 3 White Knights pattern."""
    if idx < 3:
        return False
    days = qqq_df.iloc[idx - 2:idx + 1]
    if len(days) < 3:
        return False
    higher_highs = all(days.iloc[j]["High"] > days.iloc[j - 1]["High"] for j in range(1, 3))
    higher_lows = all(days.iloc[j]["Low"] > days.iloc[j - 1]["Low"] for j in range(1, 3))
    return higher_highs and higher_lows


def _check_sell(tqqq_df: pd.DataFrame, nasdaq_df: pd.DataFrame, idx: int) -> bool:
    """Simplified sell check: 2 closes below 21-EMA or 10-day MA violation on volume."""
    if idx < 2:
        return False
    row = tqqq_df.iloc[idx]
    prev = tqqq_df.iloc[idx - 1]

    ema21 = row.get("EMA_21")
    if ema21 and not pd.isna(ema21):
        if row["Close"] < ema21 and prev["Close"] < prev.get("EMA_21", ema21):
            return True

    sma10 = row.get("SMA_10")
    vol_avg = row.get("Vol_SMA_50")
    if sma10 and not pd.isna(sma10) and vol_avg and not pd.isna(vol_avg):
        if row["Close"] < sma10 and row["Volume"] > vol_avg:
            return True

    return False


def run_backtest_year(year: int) -> Optional[YearResult]:
    """Run the trading system backtest for a single year."""
    start = f"{year - 1}-06-01"
    end_dt = min(dt.date(year, 12, 31), dt.date.today())
    end = end_dt.strftime("%Y-%m-%d")

    tqqq_df = _fetch_backtest_data("TQQQ", start, end)
    qqq_df = _fetch_backtest_data("QQQ", start, end)
    nasdaq_df = _fetch_backtest_data("^IXIC", start, end)

    if tqqq_df.empty or qqq_df.empty or nasdaq_df.empty:
        return None

    tqqq_df = _add_indicators(tqqq_df)
    qqq_df = _add_indicators(qqq_df)
    nasdaq_df = _add_indicators(nasdaq_df)

    year_start = f"{year}-01-01"
    tqqq_year = tqqq_df[tqqq_df.index >= year_start]
    qqq_year = qqq_df[qqq_df.index >= year_start]
    nasdaq_year = nasdaq_df[nasdaq_df.index >= year_start]

    if len(tqqq_year) < 10:
        return None

    tqqq_bh = (float(tqqq_year["Close"].iloc[-1]) / float(tqqq_year["Close"].iloc[0]) - 1) * 100
    qqq_bh = (float(qqq_year["Close"].iloc[-1]) / float(qqq_year["Close"].iloc[0]) - 1) * 100

    trades: List[Trade] = []
    in_position = False
    entry_price = 0.0
    entry_date = ""
    entry_idx = 0
    signal_type = ""

    portfolio = 100.0

    tqqq_full_idx = list(tqqq_df.index)
    year_indices = [i for i, d in enumerate(tqqq_full_idx) if d >= pd.Timestamp(year_start)]

    for idx in year_indices:
        date = tqqq_full_idx[idx]
        nasdaq_idx = nasdaq_df.index.get_indexer([date], method="nearest")[0]
        qqq_idx = qqq_df.index.get_indexer([date], method="nearest")[0]

        if not in_position:
            is_ftd = _detect_ftd(nasdaq_df, nasdaq_idx)
            is_3wk = _detect_3wk(qqq_df, qqq_idx)

            if is_ftd or is_3wk:
                in_position = True
                entry_price = float(tqqq_df.iloc[idx]["Close"])
                entry_date = date.strftime("%Y-%m-%d")
                entry_idx = idx
                signal_type = "FTD" if is_ftd else "3WK"
        else:
            should_sell = _check_sell(tqqq_df, nasdaq_df, idx)

            min_hold = 3
            held_days = idx - entry_idx
            if should_sell and held_days >= min_hold:
                exit_price = float(tqqq_df.iloc[idx]["Close"])
                ret = ((exit_price - entry_price) / entry_price) * 100
                portfolio *= (1 + ret / 100)
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date.strftime("%Y-%m-%d"),
                    entry_price=round(entry_price, 2),
                    exit_price=round(exit_price, 2),
                    return_pct=round(ret, 2),
                    signal_type=signal_type,
                    duration_days=held_days,
                    outcome="Win" if ret > 0 else "Loss",
                ))
                in_position = False

    if in_position:
        exit_price = float(tqqq_year["Close"].iloc[-1])
        ret = ((exit_price - entry_price) / entry_price) * 100
        portfolio *= (1 + ret / 100)
        held_days = len(tqqq_year) - (entry_idx - year_indices[0])
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=tqqq_year.index[-1].strftime("%Y-%m-%d"),
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            return_pct=round(ret, 2),
            signal_type=signal_type,
            duration_days=max(held_days, 1),
            outcome="Win" if ret > 0 else "Loss",
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
        best_trade=f"{best.entry_date} -> {best.exit_date} ({best.return_pct:+.1f}%)" if best else "N/A",
        worst_trade=f"{worst.entry_date} -> {worst.exit_date} ({worst.return_pct:+.1f}%)" if worst else "N/A",
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
