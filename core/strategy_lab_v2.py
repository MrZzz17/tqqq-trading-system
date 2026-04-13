"""
Strategy Lab V2: Hybrid strategies combining the best elements from V1.
"""

import datetime as dt
import os, sys

os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st


def mock_cache_data(**kwargs):
    def decorator(func):
        return func
    return decorator
st.cache_data = mock_cache_data

import yfinance as yf

STARTING_CAPITAL = 100_000.0
SGOV_YIELD = 0.045 / 252


def fetch(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def indicators(df):
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    return df


def weekly(df):
    w = df.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna()
    w["EMA_12"] = w["Close"].ewm(span=12, adjust=False).mean()
    w["EMA_26"] = w["Close"].ewm(span=26, adjust=False).mean()
    w["MACD"] = w["EMA_12"] - w["EMA_26"]
    w["MACD_signal"] = w["MACD"].ewm(span=9, adjust=False).mean()
    return w


def run(name, tqqq, qqq, qqq_w, signal_fn, start_year=2021):
    sim_start = pd.Timestamp(f"{start_year}-01-01")
    dates = tqqq.index[tqqq.index >= sim_start]

    cash = STARTING_CAPITAL
    shares = 0.0
    peak_value = STARTING_CAPITAL
    equity = {}
    trades = []
    entry_date = None
    entry_price = 0.0
    entry_value = 0.0

    for date in dates:
        price = float(tqqq.loc[date, "Close"])
        total = cash + shares * price
        if cash > 0:
            cash *= (1 + SGOV_YIELD)
            total = cash + shares * price

        target = signal_fn(qqq, qqq_w, tqqq, date, total, shares, price, entry_price if shares > 0 else 0)

        current_alloc = (shares * price) / total if total > 0 else 0

        if target >= 0.1 and current_alloc < 0.1:
            deploy = total * min(target, 1.0)
            deploy = min(deploy, cash)
            shares = deploy / price
            cash -= deploy
            entry_date = date
            entry_price = price
            entry_value = total
            peak_value = total

        elif target < 0.1 and current_alloc > 0.1:
            proceeds = shares * price
            cash += proceeds
            total_after = cash
            if entry_date is not None:
                pnl = total_after - entry_value
                ret = (pnl / entry_value) * 100
                trades.append({
                    "entry": entry_date.strftime("%Y-%m-%d"),
                    "exit": date.strftime("%Y-%m-%d"),
                    "days": (date - entry_date).days,
                    "ret": round(ret, 1),
                    "pnl": round(pnl, 0),
                    "win": pnl > 0,
                })
            shares = 0.0
            entry_date = None
            peak_value = cash

        elif shares > 0:
            # Adjust position size if signal changed significantly
            if target > current_alloc + 0.2:
                add = total * target - shares * price
                add = min(add, cash)
                if add > 0:
                    shares += add / price
                    cash -= add
            elif target < current_alloc - 0.2 and target >= 0.1:
                target_val = total * target
                excess = shares * price - target_val
                if excess > 0:
                    sell_sh = excess / price
                    cash += sell_sh * price
                    shares -= sell_sh

        total = cash + shares * price
        if total > peak_value:
            peak_value = total
        equity[date] = total

    if shares > 0:
        last = dates[-1]
        price = float(tqqq.loc[last, "Close"])
        cash += shares * price
        if entry_date:
            pnl = cash - entry_value
            trades.append({
                "entry": entry_date.strftime("%Y-%m-%d"),
                "exit": last.strftime("%Y-%m-%d"),
                "days": (last - entry_date).days,
                "ret": round((pnl / entry_value) * 100, 1),
                "pnl": round(pnl, 0),
                "win": pnl > 0,
            })
        shares = 0.0

    return name, equity, trades


def print_results(all_results, tqqq, current_results=None):
    years = [2021, 2022, 2023, 2024, 2025, 2026]

    print(f"\n{'Year':<9}", end="")
    for name, _, _ in all_results:
        short = name[:20]
        print(f" {short:>22}", end="")
    if current_results:
        print(f" {'Current':>22}", end="")
    print(f" {'TQQQ B&H':>10}")
    print("-" * (9 + 23 * (len(all_results) + (1 if current_results else 0)) + 11))

    for year in years:
        yr_label = f"{year} YTD" if year == 2026 else str(year)
        print(f"{yr_label:<9}", end="")
        for name, eq, trades in all_results:
            yeq = {d: v for d, v in eq.items() if d.year == year}
            if yeq:
                sd = sorted(yeq.keys())
                ret = ((yeq[sd[-1]] / yeq[sd[0]]) - 1) * 100
                yt = len([t for t in trades if str(year) in t["exit"] or str(year) in t["entry"]])
                print(f" {ret:>+7.1f}% ({yt}t)", end="")
            else:
                print(f" {'N/A':>22}", end="")
        if current_results:
            cr = next((r for r in current_results if r.year == year), None)
            if cr:
                print(f" {cr.total_return_pct:>+7.1f}% ({cr.num_trades}t)", end="")
        ty = tqqq[(tqqq.index >= f"{year}-01-01") & (tqqq.index < f"{year+1}-01-01")]
        if len(ty) > 1:
            bh = (float(ty["Close"].iloc[-1]) / float(ty["Close"].iloc[0]) - 1) * 100
            print(f" {bh:>+9.1f}%", end="")
        print()

    print("-" * (9 + 23 * (len(all_results) + (1 if current_results else 0)) + 11))
    print(f"{'$100K→':<9}", end="")
    for name, eq, _ in all_results:
        final = list(eq.values())[-1] if eq else 0
        print(f"     ${final:>16,.0f}", end="")
    if current_results:
        ce = current_results[-1].ending_value
        print(f"     ${ce:>16,.0f}", end="")
    print()

    print(f"\n{'Max DD':<9}", end="")
    for name, eq, _ in all_results:
        vals = list(eq.values())
        peak = vals[0]
        max_dd = 0
        for v in vals:
            if v > peak:
                peak = v
            dd = ((v - peak) / peak) * 100
            if dd < max_dd:
                max_dd = dd
        print(f" {max_dd:>21.1f}%", end="")
    print()

    # Trade summary
    for name, eq, trades in all_results:
        wins = sum(1 for t in trades if t["win"])
        total = len(trades)
        print(f"\n--- {name}: {total} trades, {wins}/{total} wins ---")
        for t in trades:
            w = "W" if t["win"] else "L"
            print(f"  {t['entry']} -> {t['exit']} ({t['days']:3d}d) {t['ret']:>+6.1f}% ${t['pnl']:>+10,.0f} {w}")


def main():
    print("Fetching data...")
    tqqq = fetch("TQQQ", "2019-06-01", dt.date.today().strftime("%Y-%m-%d"))
    qqq = fetch("QQQ", "2019-06-01", dt.date.today().strftime("%Y-%m-%d"))
    tqqq = indicators(tqqq)
    qqq = indicators(qqq)
    qqq_w = weekly(qqq)

    # ── Strategy F: MACD + don't exit if above 200SMA ──────────────
    def strat_f(qqq_d, qqq_wk, tqqq_d, date, total, shares, price, entry_px):
        """Weekly MACD for entry, but only exit if ALSO below 200-day SMA.
        If MACD goes negative but QQQ still above 200-day, HOLD (not a real bear)."""
        weeks = qqq_wk.index[qqq_wk.index <= date]
        if len(weeks) < 2:
            return 0.0
        macd = qqq_wk.loc[weeks[-1], "MACD"]
        if pd.isna(macd):
            return 0.0
        daily = qqq_d.loc[:date]
        if len(daily) < 200:
            return 0.0
        close = float(daily.iloc[-1]["Close"])
        sma200 = daily.iloc[-1].get("SMA_200")
        if sma200 is None or pd.isna(sma200):
            return 0.0
        above_200 = close > float(sma200)

        if macd > 0:
            return 1.0
        elif macd <= 0 and above_200:
            return 0.5  # MACD negative but above 200-day: reduce, don't exit
        else:
            return 0.0  # MACD negative AND below 200-day: full exit

    # ── Strategy G: MACD + trailing stop ────────────────────────────
    def strat_g(qqq_d, qqq_wk, tqqq_d, date, total, shares, price, entry_px):
        """Weekly MACD for entry + 12% trailing stop from portfolio peak.
        The trailing stop exits faster than waiting for MACD to cross."""
        weeks = qqq_wk.index[qqq_wk.index <= date]
        if len(weeks) < 2:
            return 0.0
        macd = qqq_wk.loc[weeks[-1], "MACD"]
        if pd.isna(macd):
            return 0.0

        if macd > 0:
            # Check trailing stop on TQQQ position
            if shares > 0 and entry_px > 0:
                gain = ((price - entry_px) / entry_px) * 100
                if gain < -15:
                    return 0.0  # Hard stop from entry
            return 1.0
        return 0.0

    # ── Strategy H: 200SMA + MACD confirmation (fewer whipsaws) ────
    def strat_h(qqq_d, qqq_wk, tqqq_d, date, total, shares, price, entry_px):
        """Hold TQQQ when QQQ above 200-day AND weekly MACD > signal line.
        Exit when QQQ below 200-day OR MACD < signal and below 50-day."""
        weeks = qqq_wk.index[qqq_wk.index <= date]
        if len(weeks) < 2:
            return 0.0
        w = qqq_wk.loc[weeks[-1]]
        macd = w["MACD"]
        macd_sig = w.get("MACD_signal")
        if pd.isna(macd):
            return 0.0

        daily = qqq_d.loc[:date]
        if len(daily) < 200:
            return 0.0
        close = float(daily.iloc[-1]["Close"])
        sma200 = daily.iloc[-1].get("SMA_200")
        sma50 = daily.iloc[-1].get("SMA_50")
        if sma200 is None or pd.isna(sma200):
            return 0.0

        above_200 = close > float(sma200)
        above_50 = (sma50 is not None and not pd.isna(sma50) and close > float(sma50))
        macd_bullish = macd > 0 or (not pd.isna(macd_sig) and macd > macd_sig)

        if above_200 and macd_bullish:
            return 1.0
        elif above_200 and above_50:
            return 0.75  # Above key MAs but MACD weakening
        elif above_200:
            return 0.5  # Above 200 but below 50 and MACD weak
        else:
            return 0.0

    # ── Strategy I: MACD with daily EMA exit ──────────────────────
    def strat_i(qqq_d, qqq_wk, tqqq_d, date, total, shares, price, entry_px):
        """Weekly MACD for entry (trend direction).
        Daily 50-day SMA on QQQ for faster exit (don't wait for weekly MACD to cross).
        Re-enter when MACD still positive and QQQ retakes 50-day."""
        weeks = qqq_wk.index[qqq_wk.index <= date]
        if len(weeks) < 2:
            return 0.0
        macd = qqq_wk.loc[weeks[-1], "MACD"]
        if pd.isna(macd):
            return 0.0

        daily = qqq_d.loc[:date]
        if len(daily) < 200:
            return 0.0
        close = float(daily.iloc[-1]["Close"])
        sma50 = daily.iloc[-1].get("SMA_50")
        sma200 = daily.iloc[-1].get("SMA_200")
        if sma50 is None or pd.isna(sma50):
            return 0.0

        above_50 = close > float(sma50)
        above_200 = (sma200 is not None and not pd.isna(sma200) and close > float(sma200))

        if macd > 0 and above_50:
            return 1.0
        elif macd > 0 and above_200:
            return 0.5  # MACD bullish but below 50-day: reduce
        elif macd > 0:
            return 0.25  # MACD bullish but below both MAs: minimal
        return 0.0

    # ── Strategy J: Pure 200/50 dual MA ─────────────────────────────
    def strat_j(qqq_d, qqq_wk, tqqq_d, date, total, shares, price, entry_px):
        """QQQ above 200-day = hold. Scale: 50% if above 200 only, 100% if also above 50.
        Exit fully when 2 consecutive closes below 200-day."""
        daily = qqq_d.loc[:date]
        if len(daily) < 201:
            return 0.0
        close = float(daily.iloc[-1]["Close"])
        prev_close = float(daily.iloc[-2]["Close"])
        sma200 = daily.iloc[-1].get("SMA_200")
        sma50 = daily.iloc[-1].get("SMA_50")
        prev_sma200 = daily.iloc[-2].get("SMA_200")
        if sma200 is None or pd.isna(sma200):
            return 0.0

        above_200 = close > float(sma200)
        prev_above_200 = (prev_sma200 is not None and not pd.isna(prev_sma200)
                          and prev_close > float(prev_sma200))
        above_50 = sma50 is not None and not pd.isna(sma50) and close > float(sma50)

        if not above_200 and not prev_above_200:
            return 0.0  # 2 closes below 200-day
        elif above_200 and above_50:
            return 1.0
        elif above_200:
            return 0.5
        return 0.5  # 1 close below but not confirmed

    strategies = [
        ("F: MACD+Hold200", strat_f),
        ("G: MACD+TrailStop", strat_g),
        ("H: 200SMA+MACD Conf", strat_h),
        ("I: MACD+Daily50Exit", strat_i),
        ("J: Dual MA 200/50", strat_j),
    ]

    all_results = []
    for name, fn in strategies:
        print(f"Running {name}...")
        result = run(name, tqqq, qqq, qqq_w, fn)
        all_results.append(result)

    # Current system
    print("Running current system...")
    from core.backtest import run_all_backtests
    current = run_all_backtests()

    print("\n" + "=" * 120)
    print("HYBRID STRATEGY COMPARISON V2")
    print("=" * 120)
    print_results(all_results, tqqq, current)


if __name__ == "__main__":
    main()
