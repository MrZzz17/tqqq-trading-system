"""
TQQQ Trading System — V6 Engine

100% invested by default. Exit on breakdowns. FTD catches bottoms.
Crash detector prevents re-entry into freefall.

$100K starting capital, full TQQQ history from 2011.
"""

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


STARTING_CAPITAL = 100_000.0
SGOV_DAILY_YIELD = 0.045 / 252


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
    max_drawdown_pct: float = 0.0


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


def _indicators(df):
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Vol_SMA_50"] = df["Volume"].rolling(50).mean()
    return df


def _make_weekly(df):
    w = df.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna()
    w["EMA_12"] = w["Close"].ewm(span=12, adjust=False).mean()
    w["EMA_26"] = w["Close"].ewm(span=26, adjust=False).mean()
    w["MACD"] = w["EMA_12"] - w["EMA_26"]
    w["MACD_signal"] = w["MACD"].ewm(span=9, adjust=False).mean()
    return w


# ── FTD ───────────────────────────────────────────────────────────

def _find_ftd_signal(nasdaq, idx):
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
    if ((trough_val - peak_val) / peak_val) * 100 > -7.0:
        return False
    if len(window) - 1 - trough_abs_idx < 4:
        return False
    last_i = len(window) - 1
    for i in range(trough_abs_idx + 1, last_i + 1):
        if closes[i] < trough_val:
            return False
    daily_gain = (closes[last_i] / closes[last_i - 1] - 1) * 100
    return daily_gain >= 1.25 and volumes[last_i] > volumes[last_i - 1]


# ── Main Engine ───────────────────────────────────────────────────

def _run_continuous(start_year: int, end_year: int):
    fetch_start = f"{start_year - 2}-01-01"
    end_dt = min(dt.date(end_year, 12, 31), dt.date.today())

    tqqq = _fetch("TQQQ", fetch_start, end_dt.strftime("%Y-%m-%d"))
    qqq = _fetch("QQQ", fetch_start, end_dt.strftime("%Y-%m-%d"))
    nasdaq = _fetch("^IXIC", fetch_start, end_dt.strftime("%Y-%m-%d"))

    if tqqq.empty or qqq.empty or nasdaq.empty:
        return {}, {}, pd.DataFrame(), pd.DataFrame()

    tqqq = _indicators(tqqq)
    qqq = _indicators(qqq)
    nasdaq = _indicators(nasdaq)
    qqq_w = _make_weekly(qqq)

    sim_start = pd.Timestamp(f"{start_year}-01-01")
    dates = tqqq.index[tqqq.index >= sim_start]

    cash = STARTING_CAPITAL
    shares = 0.0
    peak_portfolio = STARTING_CAPITAL
    equity: Dict[pd.Timestamp, float] = {}
    trades_list: List[dict] = []
    entry_date = None
    entry_price = 0.0
    entry_value = 0.0
    entry_shares = 0.0
    entry_cash = 0.0
    entry_signal = ""

    exited = False
    cooldown_until = None
    ftd_cooldown_until = None

    for i, date in enumerate(dates):
        price = float(tqqq.loc[date, "Close"])
        total = cash + shares * price

        # Idle yield
        if cash > 0:
            cash *= (1 + SGOV_DAILY_YIELD)
            total = cash + shares * price

        qq_idx = qqq.index.get_indexer([date], method="nearest")[0]
        nq_idx = nasdaq.index.get_indexer([date], method="nearest")[0]

        qq_close = float(qqq.iloc[qq_idx]["Close"])
        sma200_raw = qqq.iloc[qq_idx].get("SMA_200")
        if sma200_raw is None or pd.isna(sma200_raw):
            equity[date] = total
            continue
        sma200 = float(sma200_raw)
        above_200 = qq_close > sma200
        pct_above = ((qq_close - sma200) / sma200) * 100

        w_dates = qqq_w.index[qqq_w.index <= date]
        macd_pos = len(w_dates) >= 1 and float(qqq_w.loc[w_dates[-1], "MACD"]) > 0

        ti = tqqq.index.get_indexer([date], method="nearest")[0]
        crash = False
        if ti >= 10:
            rh = float(tqqq.iloc[ti - 10: ti + 1]["High"].max())
            crash = ((price - rh) / rh * 100) <= -30

        is_ftd = _find_ftd_signal(nasdaq, nq_idx)
        ftd_blocked = ftd_cooldown_until is not None and date < ftd_cooldown_until

        if crash and shares == 0:
            cooldown_until = date + pd.Timedelta(days=40)
            ftd_cooldown_until = date + pd.Timedelta(days=40)

        # Determine target allocation based on today's close
        target = 0.0

        if cooldown_until and date < cooldown_until and shares == 0:
            if is_ftd and not ftd_blocked and not crash:
                cooldown_until = None
                exited = False
                target = 0.5
            else:
                target = 0.0
        elif shares == 0:
            if not above_200:
                if is_ftd and not ftd_blocked and not crash:
                    exited = False
                    target = 0.5
                else:
                    target = 0.0
            elif exited:
                if is_ftd:
                    exited = False
                    target = 0.5 if not macd_pos else 1.0
                elif macd_pos:
                    exited = False
                    target = 1.0
                else:
                    target = 0.0
            else:
                target = 1.0 if macd_pos else 0.5
        else:
            if total > peak_portfolio:
                peak_portfolio = total

            if pct_above >= 3.0:
                pdd = ((total - peak_portfolio) / peak_portfolio) * 100
                if pdd <= -12.0:
                    target = 0.0
                    exited = True
                    cooldown_until = date + pd.Timedelta(days=10)
                    ftd_cooldown_until = date + pd.Timedelta(days=15)

            if not above_200:
                target = 0.0
                exited = True
                cooldown_until = date + pd.Timedelta(days=10)
                ftd_cooldown_until = date + pd.Timedelta(days=15)

            if target != 0.0 or (target == 0.0 and not exited):
                target = 1.0 if (macd_pos and above_200) else 0.5

        # Execute at close (after-hours trading on Robinhood/Webull)
        current_alloc = (shares * price) / total if total > 0 else 0

        if target >= 0.1 and current_alloc < 0.1:
            deploy = min(total * target, cash)
            if deploy > 0:
                entry_shares = deploy / price
                shares += entry_shares
                cash -= deploy
                entry_date = date
                entry_price = price
                entry_value = total
                entry_cash = deploy
                entry_signal = "FTD" if is_ftd else ("MACD" if macd_pos else "Entry")
                peak_portfolio = max(peak_portfolio, total)

        elif target < 0.1 and current_alloc > 0.1:
            proceeds = shares * price
            cash += proceeds
            exit_total = cash
            if entry_date is not None:
                pnl = exit_total - entry_value
                trades_list.append({
                    "entry": entry_date.strftime("%Y-%m-%d"),
                    "exit": date.strftime("%Y-%m-%d"),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(price, 2),
                    "days": (date - entry_date).days,
                    "ret": round((pnl / entry_value) * 100, 2),
                    "pnl": round(pnl, 0),
                    "win": pnl > 0,
                    "signal": entry_signal,
                    "shares": round(entry_shares, 2),
                    "deployed": round(entry_cash, 0),
                    "before": round(entry_value, 0),
                    "after": round(exit_total, 0),
                    "cash_after": round(cash, 0),
                    "year": date.year,
                })
            shares = 0.0
            entry_date = None
            peak_portfolio = cash

        equity[date] = cash + shares * price

    # Close open position
    if shares > 0:
        last = dates[-1]
        price = float(tqqq.loc[last, "Close"])
        cash += shares * price
        if entry_date:
            pnl = cash - entry_value
            trades_list.append({
                "entry": entry_date.strftime("%Y-%m-%d"),
                "exit": last.strftime("%Y-%m-%d"),
                "entry_price": round(entry_price, 2),
                "exit_price": round(price, 2),
                "days": (last - entry_date).days,
                "ret": round((pnl / entry_value) * 100, 2),
                "pnl": round(pnl, 0),
                "win": pnl > 0,
                "signal": entry_signal,
                "shares": round(entry_shares, 2),
                "deployed": round(entry_cash, 0),
                "before": round(entry_value, 0),
                "after": round(cash, 0),
                "cash_after": round(cash, 0),
                "year": last.year,
            })
        shares = 0.0
        equity[last] = cash

    # Convert trades to Trade objects grouped by year
    trades_by_year: Dict[int, List[Trade]] = {}
    for year in range(start_year, end_year + 1):
        trades_by_year[year] = []
    for t in trades_list:
        trade = Trade(
            entry_date=t["entry"], exit_date=t["exit"],
            entry_price=t["entry_price"], exit_price=t["exit_price"],
            return_pct=t["ret"], signal_type=t["signal"],
            duration_days=t["days"], outcome="Win" if t["win"] else "Loss",
            shares=t["shares"], cash_deployed=t["deployed"],
            portfolio_before=t["before"], portfolio_after=t["after"],
            cash_after=t["cash_after"],
        )
        trades_by_year.setdefault(t["year"], []).append(trade)

    return equity, trades_by_year, tqqq, qqq


def _build_year_result(year, equity, trades, tqqq_df, qqq_df):
    ys, ye = f"{year}-01-01", f"{year + 1}-01-01"
    ty = tqqq_df[(tqqq_df.index >= ys) & (tqqq_df.index < ye)]
    qy = qqq_df[(qqq_df.index >= ys) & (qqq_df.index < ye)]
    if ty.empty or qy.empty or len(ty) < 5:
        return None
    tbh = (float(ty["Close"].iloc[-1]) / float(ty["Close"].iloc[0]) - 1) * 100
    qbh = (float(qy["Close"].iloc[-1]) / float(qy["Close"].iloc[0]) - 1) * 100
    yeq = {d: v for d, v in equity.items() if d >= pd.Timestamp(ys) and d < pd.Timestamp(ye)}
    if not yeq:
        return None
    sd = sorted(yeq.keys())
    sv, ev = yeq[sd[0]], yeq[sd[-1]]
    if sv == 0:
        return None
    ret = ((ev / sv) - 1) * 100
    wins = [t for t in trades if t.return_pct > 0]
    losses = [t for t in trades if t.return_pct <= 0]
    best = max(trades, key=lambda t: t.return_pct) if trades else None
    worst = min(trades, key=lambda t: t.return_pct) if trades else None
    # Max drawdown for this year
    peak_y = sv
    max_dd_y = 0.0
    for d in sd:
        v = yeq[d]
        if v > peak_y:
            peak_y = v
        dd = ((v - peak_y) / peak_y) * 100
        if dd < max_dd_y:
            max_dd_y = dd

    return YearResult(
        year=year, total_return_pct=round(ret, 2), num_trades=len(trades),
        win_rate_pct=round(len(wins) / len(trades) * 100, 1) if trades else 0,
        avg_win_pct=round(np.mean([t.return_pct for t in wins]), 2) if wins else 0,
        avg_loss_pct=round(np.mean([t.return_pct for t in losses]), 2) if losses else 0,
        max_win_pct=round(max(t.return_pct for t in wins), 2) if wins else 0,
        max_loss_pct=round(min(t.return_pct for t in losses), 2) if losses else 0,
        max_drawdown_pct=round(max_dd_y, 2),
        best_trade=f"{best.entry_date} to {best.exit_date} ({best.return_pct:+.1f}%)" if best else "N/A",
        worst_trade=f"{worst.entry_date} to {worst.exit_date} ({worst.return_pct:+.1f}%)" if worst else "N/A",
        tqqq_buy_hold_pct=round(tbh, 2), qqq_buy_hold_pct=round(qbh, 2),
        trades=trades, starting_value=round(sv, 2), ending_value=round(ev, 2),
    )


def run_all_backtests():
    """Returns (results, equity_curve_dict)."""
    current_year = dt.date.today().year
    equity, trades_by_year, tqqq, qqq = _run_continuous(2011, current_year)
    if not equity:
        return [], {}
    results = []
    for year in range(2011, current_year + 1):
        r = _build_year_result(year, equity, trades_by_year.get(year, []), tqqq, qqq)
        if r:
            results.append(r)
    return results, equity
