"""
TQQQ Swing Trading System — V3 Backtest Engine

Key changes from V2:
- Regime-aware exits: wide (50-day) in golden cross, tight (21-EMA) otherwise
- QQQ-only for all signals — no SPY gating, no TQQQ regime checks
- 75% allocation in confirmed bull (golden cross), 50% otherwise
- No adaptive scaling (it created hidden P&L distortions)
- QQQ below 200-day = immediate exit and no entries
- Start from 2011 for full TQQQ history
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


STARTING_CAPITAL = 100_000.0


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
    shares: float
    cash_deployed: float
    portfolio_before: float
    portfolio_after: float
    cash_after: float


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
    starting_value: float = 0.0
    ending_value: float = 0.0


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
MIN_HOLD_DAYS = 10
PULLBACK_ENTRY_PCT = 8.0
PULLBACK_BULL_QQQ_PCT = 4.0
PULLBACK_BEAR_QQQ_PCT = 5.0
COOLDOWN_WIN = 3
COOLDOWN_LOSS = 10
MAX_CONSECUTIVE_LOSSES = 2
ALLOC_BULL = 0.50
ALLOC_CAUTIOUS = 0.25


# ── QQQ Market Regime ─────────────────────────────────────────────

def _qqq_regime(qqq: pd.DataFrame, idx: int) -> str:
    """
    Returns the current QQQ market regime:
    - 'strong_bull': golden cross + above 200d + above 50d
    - 'bull': above 200d (no golden cross yet, or below 50d)
    - 'bear': below 200d
    """
    sma200 = qqq.iloc[idx].get("SMA_200")
    sma50 = qqq.iloc[idx].get("SMA_50")
    if sma200 is None or pd.isna(sma200):
        return "bull"
    close = float(qqq.iloc[idx]["Close"])
    if close < float(sma200):
        return "bear"
    if sma50 is not None and not pd.isna(sma50) and float(sma50) > float(sma200):
        return "strong_bull"
    return "bull"


def _qqq_death_cross(qqq: pd.DataFrame, idx: int) -> bool:
    sma50 = qqq.iloc[idx].get("SMA_50")
    sma200 = qqq.iloc[idx].get("SMA_200")
    if sma50 is None or pd.isna(sma50) or sma200 is None or pd.isna(sma200):
        return False
    return float(sma50) < float(sma200)


# ── Buy Signal Detection ─────────────────────────────────────────

def _find_ftd_signal(nasdaq: pd.DataFrame, idx: int) -> bool:
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


def _find_pullback_entry(qqq: pd.DataFrame, qq_idx: int, regime: str) -> bool:
    if qq_idx < 50:
        return False
    close = float(qqq.iloc[qq_idx]["Close"])
    prev_close = float(qqq.iloc[qq_idx - 1]["Close"])
    if close <= prev_close:
        return False
    if qq_idx >= 3:
        if float(qqq.iloc[qq_idx]["Low"]) <= float(qqq.iloc[qq_idx - 2]["Low"]):
            return False

    if regime == "strong_bull":
        lookback = min(20, qq_idx)
        recent_high = float(qqq.iloc[qq_idx - lookback: qq_idx + 1]["High"].max())
        recent_low = float(qqq.iloc[max(0, qq_idx - 3): qq_idx + 1]["Low"].min())
        dip_from_high = ((recent_low - recent_high) / recent_high) * 100
        return dip_from_high <= -PULLBACK_BULL_QQQ_PCT
    else:
        lookback = min(30, qq_idx)
        recent_high = float(qqq.iloc[qq_idx - lookback: qq_idx + 1]["High"].max())
        pullback_pct = ((close - recent_high) / recent_high) * 100
        if pullback_pct > -PULLBACK_BEAR_QQQ_PCT:
            return False
        ema21 = qqq.iloc[qq_idx].get("EMA_21")
        sma50 = qqq.iloc[qq_idx].get("SMA_50")
        near_ma = False
        if sma50 and not pd.isna(sma50):
            if close <= float(sma50) * 1.02:
                near_ma = True
        if ema21 and not pd.isna(ema21):
            if close <= float(ema21) * 1.01:
                near_ma = True
        return near_ma


def _find_ma_retake_entry(qqq: pd.DataFrame, qq_idx: int) -> bool:
    if qq_idx < 50:
        return False
    ema21 = qqq.iloc[qq_idx].get("EMA_21")
    prev_ema21 = qqq.iloc[qq_idx - 1].get("EMA_21")
    if ema21 is None or pd.isna(ema21) or prev_ema21 is None or pd.isna(prev_ema21):
        return False
    close = float(qqq.iloc[qq_idx]["Close"])
    prev_close = float(qqq.iloc[qq_idx - 1]["Close"])
    crossed_above = prev_close < float(prev_ema21) and close > float(ema21)
    if not crossed_above:
        return False
    sma50 = qqq.iloc[qq_idx].get("SMA_50")
    if sma50 and not pd.isna(sma50) and qq_idx >= 10:
        sma50_prev = qqq.iloc[qq_idx - 10].get("SMA_50")
        if sma50_prev and not pd.isna(sma50_prev):
            if float(sma50) < float(sma50_prev):
                return False
    lookback = min(20, qq_idx)
    recent_high = float(qqq.iloc[qq_idx - lookback: qq_idx + 1]["High"].max())
    dip_pct = ((close - recent_high) / recent_high) * 100
    return dip_pct <= -2.0


# ── Regime-Aware Exit Logic ───────────────────────────────────────

def _check_exit(qqq: pd.DataFrame, qq_idx: int, regime: str,
                entry_price: float, entry_idx: int, current_idx: int) -> str:
    """
    Regime-aware exit:
    - strong_bull: use WIDE exit (2 closes below 50-day SMA)
    - bull: use TIGHT exit (2 closes below 21-EMA)
    - bear: immediate exit
    """
    if qq_idx < 2:
        return "hold"

    close = float(qqq.iloc[qq_idx]["Close"])
    prev_close = float(qqq.iloc[qq_idx - 1]["Close"])
    held_days = current_idx - entry_idx

    # Hard stop: catastrophic loss from entry
    if held_days <= 5:
        tqqq_return = ((float(qqq.iloc[qq_idx]["Close"]) - entry_price) / entry_price) * 100
        if tqqq_return < -8.0:
            return "full_exit"
        return "hold"

    # Bear regime: immediate exit
    if regime == "bear":
        return "full_exit"

    # Strong bull: wide exit (2 closes below 50-day SMA)
    if regime == "strong_bull":
        sma50 = qqq.iloc[qq_idx].get("SMA_50")
        prev_sma50 = qqq.iloc[qq_idx - 1].get("SMA_50")
        if (sma50 and not pd.isna(sma50) and prev_sma50 and not pd.isna(prev_sma50)):
            if close < float(sma50) and prev_close < float(prev_sma50):
                return "full_exit"
        return "hold"

    # Bull (not golden cross): tight exit (2 closes below 21-EMA)
    ema21 = qqq.iloc[qq_idx].get("EMA_21")
    prev_ema21 = qqq.iloc[qq_idx - 1].get("EMA_21")
    if (ema21 and not pd.isna(ema21) and prev_ema21 and not pd.isna(prev_ema21)):
        if close < float(ema21) and prev_close < float(prev_ema21):
            return "full_exit"

    return "hold"


# ── Dollar-Based Portfolio ────────────────────────────────────────

class Portfolio:
    def __init__(self, starting_cash: float = STARTING_CAPITAL):
        self.cash = starting_cash
        self.shares = 0.0
        self.entry_price = 0.0
        self.signal_type = ""
        self.entry_date = ""
        self.entry_idx = 0
        self.rally_low = 0.0
        self.entry_shares = 0.0
        self.entry_cash_deployed = 0.0
        self.entry_portfolio_value = 0.0

    @property
    def in_position(self) -> bool:
        return self.shares > 0

    def total_value(self, current_price: float) -> float:
        return self.cash + self.shares * current_price

    def buy(self, price: float, allocation: float, date: str,
            signal: str, idx: int, rally_low: float):
        total = self.total_value(price)
        cash_to_deploy = total * allocation
        cash_to_deploy = min(cash_to_deploy, self.cash)
        if cash_to_deploy <= 0:
            return
        self.entry_portfolio_value = total
        self.shares = cash_to_deploy / price
        self.cash -= cash_to_deploy
        self.entry_price = price
        self.entry_date = date
        self.entry_idx = idx
        self.signal_type = signal
        self.rally_low = rally_low
        self.entry_shares = self.shares
        self.entry_cash_deployed = cash_to_deploy

    def sell_all(self, price: float) -> float:
        proceeds = self.shares * price
        self.cash += proceeds
        self.shares = 0.0
        return proceeds


# ── Continuous Multi-Year Backtest ────────────────────────────────

def _run_continuous(start_year: int, end_year: int):
    fetch_start = f"{start_year - 1}-06-01"
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

    pf = Portfolio(STARTING_CAPITAL)
    cooldown_until = 0
    consecutive_losses = 0
    last_trade_idx = 0

    equity_by_date: Dict[pd.Timestamp, float] = {}
    trades_by_year: Dict[int, List[Trade]] = {}
    for year in range(start_year, end_year + 1):
        trades_by_year[year] = []

    def _make_trade(exit_date_ts, exit_price):
        ret_pct = ((exit_price - pf.entry_price) / pf.entry_price) * 100
        return Trade(
            entry_date=pf.entry_date,
            exit_date=exit_date_ts.strftime("%Y-%m-%d"),
            entry_price=round(pf.entry_price, 2),
            exit_price=round(exit_price, 2),
            return_pct=round(ret_pct, 2),
            signal_type=pf.signal_type,
            duration_days=0,
            outcome="Win" if ret_pct > 0 else "Loss",
            shares=round(pf.entry_shares, 2),
            cash_deployed=round(pf.entry_cash_deployed, 2),
            portfolio_before=round(pf.entry_portfolio_value, 2),
            portfolio_after=0.0,
            cash_after=0.0,
        )

    def _close_position(close_idx, date, price, current_year):
        nonlocal consecutive_losses, cooldown_until, last_trade_idx
        t = _make_trade(date, price)
        t.duration_days = close_idx - pf.entry_idx
        pf.sell_all(price)
        t.portfolio_after = round(pf.total_value(price), 2)
        t.cash_after = round(pf.cash, 2)
        actual_pnl = t.portfolio_after - t.portfolio_before
        t.return_pct = round((actual_pnl / t.portfolio_before) * 100, 2) if t.portfolio_before > 0 else 0.0
        t.outcome = "Win" if actual_pnl > 0 else "Loss"
        is_loss = actual_pnl <= 0
        trades_by_year.setdefault(current_year, []).append(t)
        last_trade_idx = close_idx

        if is_loss:
            consecutive_losses += 1
            cooldown_until = close_idx + COOLDOWN_LOSS
        else:
            consecutive_losses = 0
            cooldown_until = close_idx + COOLDOWN_WIN

    for idx in sim_indices:
        date = all_idx[idx]
        current_year = date.year
        price = float(tqqq_df.iloc[idx]["Close"])

        nq_idx = nasdaq_df.index.get_indexer([date], method="nearest")[0]
        qq_idx = qqq_df.index.get_indexer([date], method="nearest")[0]

        regime = _qqq_regime(qqq_df, qq_idx)

        if not pf.in_position:
            if idx < cooldown_until:
                equity_by_date[date] = pf.total_value(price)
                continue

            if consecutive_losses > 0 and (idx - last_trade_idx) > 30:
                consecutive_losses = 0

            # No entries in bear market
            if regime == "bear":
                equity_by_date[date] = pf.total_value(price)
                continue

            death_cross = _qqq_death_cross(qqq_df, qq_idx)
            is_ftd = _find_ftd_signal(nasdaq_df, nq_idx)
            is_3wk = _find_3wk_signal(qqq_df, tqqq_df, idx)
            is_pullback = (regime != "bear") and _find_pullback_entry(qqq_df, qq_idx, regime)
            is_ma_retake = (regime != "bear") and _find_ma_retake_entry(qqq_df, qq_idx)

            alloc = 0.0
            signal = ""

            if is_ftd:
                alloc, signal = ALLOC_BULL, "FTD"
                consecutive_losses = 0
            elif is_3wk:
                alloc = ALLOC_CAUTIOUS if death_cross else ALLOC_BULL
                signal = "3WK"
            elif is_pullback:
                alloc, signal = ALLOC_BULL, "Pullback"
            elif is_ma_retake:
                alloc, signal = ALLOC_BULL, "MA Retake"

            if alloc > 0 and signal != "FTD" and consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                alloc = min(alloc, ALLOC_CAUTIOUS)

            if alloc > 0:
                lookback = min(20, idx)
                r_low = float(tqqq_df.iloc[idx - lookback: idx + 1]["Low"].min())
                pf.buy(price, alloc, date.strftime("%Y-%m-%d"),
                       signal, idx, r_low)

        else:
            entry_qq_idx = qqq_df.index.get_indexer(
                [tqqq_df.index[pf.entry_idx]], method="nearest")[0]
            entry_qq_price = float(qqq_df.iloc[entry_qq_idx]["Close"])
            exit_action = _check_exit(qqq_df, qq_idx, regime,
                                      entry_qq_price, entry_qq_idx, qq_idx)

            if exit_action == "full_exit":
                _close_position(idx, date, price, current_year)

        equity_by_date[date] = pf.total_value(price)

    if pf.in_position:
        last_date = all_idx[sim_indices[-1]]
        price = float(tqqq_df.iloc[sim_indices[-1]]["Close"])
        last_idx = sim_indices[-1]
        _close_position(last_idx, last_date, price, last_date.year)
        equity_by_date[last_date] = pf.total_value(price)

    return equity_by_date, trades_by_year, tqqq_df, qqq_df


def _build_year_result(year: int, equity: Dict, trades: List[Trade],
                       tqqq_df: pd.DataFrame, qqq_df: pd.DataFrame) -> Optional[YearResult]:
    year_start = f"{year}-01-01"
    year_end = f"{year + 1}-01-01"
    tqqq_year = tqqq_df[(tqqq_df.index >= year_start) & (tqqq_df.index < year_end)]
    qqq_year = qqq_df[(qqq_df.index >= year_start) & (qqq_df.index < year_end)]

    if tqqq_year.empty or qqq_year.empty or len(tqqq_year) < 5:
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
        starting_value=round(start_val, 2),
        ending_value=round(end_val, 2),
    )


def run_backtest_year(year: int) -> Optional[YearResult]:
    result = _run_continuous(year, year)
    equity, trades_by_year, tqqq_df, qqq_df = result
    if not equity:
        return None
    return _build_year_result(year, equity, trades_by_year.get(year, []), tqqq_df, qqq_df)


def run_all_backtests() -> List[YearResult]:
    current_year = dt.date.today().year
    start_year = 2021
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
