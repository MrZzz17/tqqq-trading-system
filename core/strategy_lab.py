"""
Strategy Lab: Test multiple TQQQ timing strategies head-to-head.
Does NOT modify existing backtest code. Self-contained.
"""

import datetime as dt
import os
import sys

os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st


def mock_cache_data(**kwargs):
    def decorator(func):
        return func
    return decorator


st.cache_data = mock_cache_data

import yfinance as yf

STARTING_CAPITAL = 100_000.0
SGOV_ANNUAL_YIELD = 0.045  # ~4.5% annual yield on idle cash


# ── Data ──────────────────────────────────────────────────────────

def fetch(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def add_indicators(df):
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["RSI_14"] = _compute_rsi(df["Close"], 14)
    return df


def _compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def make_weekly(df):
    weekly = df.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna()
    weekly["EMA_12"] = weekly["Close"].ewm(span=12, adjust=False).mean()
    weekly["EMA_26"] = weekly["Close"].ewm(span=26, adjust=False).mean()
    weekly["MACD"] = weekly["EMA_12"] - weekly["EMA_26"]
    weekly["MACD_signal"] = weekly["MACD"].ewm(span=9, adjust=False).mean()
    weekly["SMA_40"] = weekly["Close"].rolling(40).mean()  # ~200 day
    return weekly


# ── Strategy Framework ────────────────────────────────────────────

def run_strategy(name, tqqq, qqq, signal_func, start_year=2021):
    """Generic strategy runner. signal_func(qqq, tqqq, date) -> allocation (0-1)."""
    sim_start = pd.Timestamp(f"{start_year}-01-01")
    dates = tqqq.index[tqqq.index >= sim_start]

    cash = STARTING_CAPITAL
    shares = 0.0
    equity_curve = {}
    trades = []
    entry_date = None
    entry_price = 0.0
    entry_portfolio = 0.0

    prev_alloc = 0.0

    for date in dates:
        price = float(tqqq.loc[date, "Close"])
        target_alloc = signal_func(qqq, tqqq, date)

        current_value = cash + shares * price
        current_alloc = (shares * price) / current_value if current_value > 0 else 0

        # Idle yield on cash
        if cash > 0:
            daily_yield = SGOV_ANNUAL_YIELD / 252
            cash *= (1 + daily_yield)

        if target_alloc > 0 and current_alloc < 0.1:
            # Enter position
            deploy = current_value * target_alloc
            deploy = min(deploy, cash)
            if deploy > 0:
                new_shares = deploy / price
                shares += new_shares
                cash -= deploy
                entry_date = date
                entry_price = price
                entry_portfolio = current_value

        elif target_alloc == 0 and current_alloc > 0.1:
            # Exit position
            proceeds = shares * price
            cash += proceeds
            exit_value = cash
            if entry_date is not None:
                pnl = exit_value - entry_portfolio
                ret_pct = (pnl / entry_portfolio) * 100
                trades.append({
                    "entry": entry_date.strftime("%Y-%m-%d"),
                    "exit": date.strftime("%Y-%m-%d"),
                    "days": (date - entry_date).days,
                    "return_pct": round(ret_pct, 1),
                    "pnl": round(pnl, 0),
                    "outcome": "Win" if pnl > 0 else "Loss",
                })
            shares = 0.0
            entry_date = None

        elif target_alloc > current_alloc + 0.15:
            # Scale up
            additional = current_value * target_alloc - shares * price
            additional = min(additional, cash)
            if additional > 0:
                shares += additional / price
                cash -= additional

        elif target_alloc < current_alloc - 0.15 and target_alloc > 0:
            # Scale down
            target_val = current_value * target_alloc
            excess = shares * price - target_val
            if excess > 0:
                sell_shares = excess / price
                cash += sell_shares * price
                shares -= sell_shares

        equity_curve[date] = cash + shares * price
        prev_alloc = target_alloc

    # Close any open position at end
    if shares > 0:
        last_date = dates[-1]
        price = float(tqqq.loc[last_date, "Close"])
        proceeds = shares * price
        cash += proceeds
        if entry_date is not None:
            pnl = cash - entry_portfolio
            ret_pct = (pnl / entry_portfolio) * 100
            trades.append({
                "entry": entry_date.strftime("%Y-%m-%d"),
                "exit": last_date.strftime("%Y-%m-%d"),
                "days": (last_date - entry_date).days,
                "return_pct": round(ret_pct, 1),
                "pnl": round(pnl, 0),
                "outcome": "Win" if pnl > 0 else "Loss",
            })
        shares = 0.0

    return name, equity_curve, trades


def yearly_results(name, equity_curve, trades, tqqq):
    years = sorted(set(d.year for d in equity_curve.keys()))
    rows = []
    for year in years:
        year_eq = {d: v for d, v in equity_curve.items() if d.year == year}
        if not year_eq:
            continue
        sorted_dates = sorted(year_eq.keys())
        start_val = year_eq[sorted_dates[0]]
        end_val = year_eq[sorted_dates[-1]]
        ret = ((end_val / start_val) - 1) * 100 if start_val > 0 else 0

        tqqq_year = tqqq[(tqqq.index >= f"{year}-01-01") & (tqqq.index < f"{year+1}-01-01")]
        if len(tqqq_year) > 1:
            bh = (float(tqqq_year["Close"].iloc[-1]) / float(tqqq_year["Close"].iloc[0]) - 1) * 100
        else:
            bh = 0

        year_trades = [t for t in trades
                       if t["exit"][:4] == str(year) or t["entry"][:4] == str(year)]

        rows.append({
            "year": year,
            "return": round(ret, 1),
            "start": round(start_val, 0),
            "end": round(end_val, 0),
            "trades": len(year_trades),
            "wr": round(sum(1 for t in year_trades if t["outcome"] == "Win") /
                        max(len(year_trades), 1) * 100, 0),
            "tqqq_bh": round(bh, 1),
        })

    return rows


# ── Strategy Implementations ──────────────────────────────────────

def strategy_weekly_macd(qqq, tqqq, date):
    """Strategy A: Weekly MACD zero-line on QQQ."""
    qqq_weekly = make_weekly(qqq)
    # Find the most recent completed week
    week_dates = qqq_weekly.index[qqq_weekly.index <= date]
    if len(week_dates) < 2:
        return 0.0
    latest = qqq_weekly.loc[week_dates[-1]]
    macd = latest["MACD"]
    if pd.isna(macd):
        return 0.0
    return 1.0 if macd > 0 else 0.0


def strategy_macd_with_200sma_buffer(qqq, tqqq, date):
    """Strategy B: Weekly MACD + 200-day SMA asymmetric buffer."""
    qqq_weekly = make_weekly(qqq)
    week_dates = qqq_weekly.index[qqq_weekly.index <= date]
    if len(week_dates) < 2:
        return 0.0

    macd = qqq_weekly.loc[week_dates[-1], "MACD"]
    if pd.isna(macd):
        return 0.0

    # Daily 200-day SMA check with asymmetric buffer
    qqq_daily = qqq.loc[:date]
    if len(qqq_daily) < 200:
        return 0.0

    close = float(qqq_daily.iloc[-1]["Close"])
    sma200 = float(qqq_daily.iloc[-1]["SMA_200"])
    if pd.isna(sma200):
        return 0.0

    pct_from_200 = ((close - sma200) / sma200) * 100

    if macd > 0 and pct_from_200 > -3:
        return 1.0
    elif macd <= 0 or pct_from_200 < -3:
        return 0.0
    return 0.0


def strategy_macd_scaled(qqq, tqqq, date):
    """Strategy C: Weekly MACD + scale based on trend strength."""
    qqq_weekly = make_weekly(qqq)
    week_dates = qqq_weekly.index[qqq_weekly.index <= date]
    if len(week_dates) < 2:
        return 0.0

    latest = qqq_weekly.loc[week_dates[-1]]
    macd = latest["MACD"]
    macd_signal = latest["MACD_signal"]
    if pd.isna(macd) or pd.isna(macd_signal):
        return 0.0

    # Daily checks
    qqq_daily = qqq.loc[:date]
    if len(qqq_daily) < 200:
        return 0.0
    close = float(qqq_daily.iloc[-1]["Close"])
    sma200 = float(qqq_daily.iloc[-1]["SMA_200"])
    sma50 = float(qqq_daily.iloc[-1]["SMA_50"])
    if pd.isna(sma200) or pd.isna(sma50):
        return 0.0

    above_200 = close > sma200
    golden_cross = sma50 > sma200

    if macd <= 0:
        return 0.0

    # MACD above zero — scale based on trend confirmation
    if above_200 and golden_cross and macd > macd_signal:
        return 1.0  # Full conviction
    elif above_200 and macd > macd_signal:
        return 0.75
    elif macd > 0:
        return 0.5  # MACD above zero but weak confirmation
    return 0.0


def strategy_200sma_simple(qqq, tqqq, date):
    """Strategy D: Simple 200-day SMA on QQQ. Hold when above, cash when below."""
    qqq_daily = qqq.loc[:date]
    if len(qqq_daily) < 200:
        return 0.0
    close = float(qqq_daily.iloc[-1]["Close"])
    sma200 = float(qqq_daily.iloc[-1]["SMA_200"])
    if pd.isna(sma200):
        return 0.0
    return 1.0 if close > sma200 else 0.0


def strategy_200sma_asymmetric(qqq, tqqq, date):
    """Strategy E: 200-day SMA with asymmetric buffer (+5% buy / -3% sell)."""
    qqq_daily = qqq.loc[:date]
    if len(qqq_daily) < 200:
        return 0.0
    close = float(qqq_daily.iloc[-1]["Close"])
    sma200 = float(qqq_daily.iloc[-1]["SMA_200"])
    if pd.isna(sma200):
        return 0.0
    pct = ((close - sma200) / sma200) * 100

    # Need state: use a simple approach - buy above +5%, sell below -3%
    # For stateless: if above +2% hold/buy, if below -3% sell
    if pct > 2:
        return 1.0
    elif pct < -3:
        return 0.0
    return -1.0  # Hold current position (special: -1 = don't change)


def strategy_macd_rsi_combo(qqq, tqqq, date):
    """Strategy F: Weekly MACD + daily RSI oversold entries."""
    qqq_weekly = make_weekly(qqq)
    week_dates = qqq_weekly.index[qqq_weekly.index <= date]
    if len(week_dates) < 2:
        return 0.0

    macd = qqq_weekly.loc[week_dates[-1], "MACD"]
    if pd.isna(macd):
        return 0.0

    qqq_daily = qqq.loc[:date]
    if len(qqq_daily) < 200:
        return 0.0

    close = float(qqq_daily.iloc[-1]["Close"])
    sma200 = float(qqq_daily.iloc[-1]["SMA_200"])
    rsi = float(qqq_daily.iloc[-1]["RSI_14"])
    if pd.isna(sma200) or pd.isna(rsi):
        return 0.0

    above_200 = close > sma200

    if macd > 0:
        if above_200 and rsi > 30:
            return 1.0
        elif rsi < 30:
            return 0.5  # Oversold but MACD bullish — cautious
        return 0.75
    else:
        if rsi < 25 and above_200:
            return 0.25  # Deeply oversold mean-reversion
        return 0.0


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("Fetching data...")
    tqqq = fetch("TQQQ", "2019-06-01", dt.date.today().strftime("%Y-%m-%d"))
    qqq = fetch("QQQ", "2019-06-01", dt.date.today().strftime("%Y-%m-%d"))

    if tqqq.empty or qqq.empty:
        print("Failed to fetch data")
        return

    tqqq = add_indicators(tqqq)
    qqq = add_indicators(qqq)

    # Pre-compute weekly data once
    qqq_weekly_cache = make_weekly(qqq)

    # Optimized signal functions that use pre-computed weekly data
    def _weekly_macd_signal(qqq_df, tqqq_df, date):
        weeks = qqq_weekly_cache.index[qqq_weekly_cache.index <= date]
        if len(weeks) < 2:
            return 0.0
        macd = qqq_weekly_cache.loc[weeks[-1], "MACD"]
        return 1.0 if (not pd.isna(macd) and macd > 0) else 0.0

    def _macd_200_signal(qqq_df, tqqq_df, date):
        weeks = qqq_weekly_cache.index[qqq_weekly_cache.index <= date]
        if len(weeks) < 2:
            return 0.0
        macd = qqq_weekly_cache.loc[weeks[-1], "MACD"]
        if pd.isna(macd):
            return 0.0
        daily = qqq_df.loc[:date]
        if len(daily) < 200:
            return 0.0
        close = float(daily.iloc[-1]["Close"])
        sma200 = daily.iloc[-1].get("SMA_200")
        if sma200 is None or pd.isna(sma200):
            return 0.0
        pct = ((close - float(sma200)) / float(sma200)) * 100
        return 1.0 if (macd > 0 and pct > -3) else 0.0

    def _macd_scaled_signal(qqq_df, tqqq_df, date):
        weeks = qqq_weekly_cache.index[qqq_weekly_cache.index <= date]
        if len(weeks) < 2:
            return 0.0
        w = qqq_weekly_cache.loc[weeks[-1]]
        macd, macd_sig = w["MACD"], w["MACD_signal"]
        if pd.isna(macd) or pd.isna(macd_sig):
            return 0.0
        if macd <= 0:
            return 0.0
        daily = qqq_df.loc[:date]
        if len(daily) < 200:
            return 0.5
        close = float(daily.iloc[-1]["Close"])
        sma200 = daily.iloc[-1].get("SMA_200")
        sma50 = daily.iloc[-1].get("SMA_50")
        if sma200 is None or pd.isna(sma200) or sma50 is None or pd.isna(sma50):
            return 0.5
        above_200 = close > float(sma200)
        golden = float(sma50) > float(sma200)
        if above_200 and golden and macd > macd_sig:
            return 1.0
        elif above_200 and macd > macd_sig:
            return 0.75
        return 0.5

    def _200sma_signal(qqq_df, tqqq_df, date):
        daily = qqq_df.loc[:date]
        if len(daily) < 200:
            return 0.0
        close = float(daily.iloc[-1]["Close"])
        sma200 = daily.iloc[-1].get("SMA_200")
        if sma200 is None or pd.isna(sma200):
            return 0.0
        return 1.0 if close > float(sma200) else 0.0

    def _macd_rsi_signal(qqq_df, tqqq_df, date):
        weeks = qqq_weekly_cache.index[qqq_weekly_cache.index <= date]
        if len(weeks) < 2:
            return 0.0
        macd = qqq_weekly_cache.loc[weeks[-1], "MACD"]
        if pd.isna(macd):
            return 0.0
        daily = qqq_df.loc[:date]
        if len(daily) < 200:
            return 0.0
        close = float(daily.iloc[-1]["Close"])
        sma200 = daily.iloc[-1].get("SMA_200")
        rsi = daily.iloc[-1].get("RSI_14")
        if sma200 is None or pd.isna(sma200) or rsi is None or pd.isna(rsi):
            return 0.0
        rsi = float(rsi)
        above_200 = close > float(sma200)
        if macd > 0:
            if above_200 and rsi > 30:
                return 1.0
            elif rsi < 30:
                return 0.5
            return 0.75
        else:
            if rsi < 25 and above_200:
                return 0.25
            return 0.0

    strategies = [
        ("A: Weekly MACD", _weekly_macd_signal),
        ("B: MACD + 200SMA Buffer", _macd_200_signal),
        ("C: MACD Scaled", _macd_scaled_signal),
        ("D: 200SMA Simple", _200sma_signal),
        ("E: MACD + RSI Combo", _macd_rsi_signal),
    ]

    all_results = []

    for name, sig_func in strategies:
        print(f"Running {name}...")
        n, eq, trades = run_strategy(name, tqqq, qqq, sig_func, start_year=2021)
        rows = yearly_results(name, eq, trades, tqqq)
        all_results.append((name, rows, trades, eq))

    # Also run the current system for comparison
    print("Running current system...")
    from core.backtest import run_all_backtests
    current = run_all_backtests()

    # Print results
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON: $100K starting capital, 2021-2026")
    print("=" * 100)

    # Header
    print(f"\n{'Year':<8}", end="")
    for name, _, _, _ in all_results:
        print(f" {name:>22}", end="")
    print(f" {'Current System':>22} {'TQQQ B&H':>12}")
    print("-" * (8 + 23 * (len(all_results) + 1) + 13))

    # Year by year
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    for year in years:
        yr_label = f"{year} YTD" if year == 2026 else str(year)
        print(f"{yr_label:<8}", end="")
        for name, rows, _, _ in all_results:
            row = next((r for r in rows if r["year"] == year), None)
            if row:
                print(f" {row['return']:>+7.1f}% ({row['trades']}t)", end="")
            else:
                print(f" {'N/A':>22}", end="")

        cr = next((r for r in current if r.year == year), None)
        if cr:
            print(f" {cr.total_return_pct:>+7.1f}% ({cr.num_trades}t)", end="")
            print(f" {cr.tqqq_buy_hold_pct:>+10.1f}%", end="")
        print()

    # Overall
    print("-" * (8 + 23 * (len(all_results) + 1) + 13))
    print(f"{'$100K→':<8}", end="")
    for name, rows, _, eq in all_results:
        if eq:
            final = list(eq.values())[-1]
            print(f" ${final:>18,.0f}", end="")
        else:
            print(f" {'N/A':>22}", end="")

    cr_start = current[0].starting_value if current else 100000
    cr_end = current[-1].ending_value if current else 100000
    print(f" ${cr_end:>18,.0f}", end="")
    print()

    # Trade details for top strategies
    print("\n" + "=" * 100)
    print("TRADE DETAILS FOR TOP STRATEGIES")
    print("=" * 100)

    for name, rows, trades, eq in all_results:
        if not trades:
            continue
        wins = sum(1 for t in trades if t["outcome"] == "Win")
        total = len(trades)
        print(f"\n--- {name}: {total} trades, {wins}/{total} wins ({wins/max(total,1)*100:.0f}% WR) ---")
        for t in trades:
            print(f"  {t['entry']} -> {t['exit']} ({t['days']:3d}d) {t['return_pct']:+6.1f}% ${t['pnl']:+,.0f} {t['outcome']}")


if __name__ == "__main__":
    main()
