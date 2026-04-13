"""
Compare 3 staged-entry approaches for the TQQQ swing trading system.

Approach A: Quick Scale-In (3-day confirmation)
Approach B: Confirmed Bull = 100% immediately
Approach C: Adaptive Allocation (trailing scale-up/down)
"""

import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
import streamlit as st
def mock_cache_data(**kwargs):
    def decorator(func):
        return func
    return decorator
st.cache_data = mock_cache_data

import datetime as dt
from typing import Dict, List, Optional
from copy import deepcopy

import numpy as np
import pandas as pd

from backtest import (
    STARTING_CAPITAL, COOLDOWN_BULL, COOLDOWN_BEAR, MAX_CONSECUTIVE_LOSSES,
    MIN_HOLD_DAYS, Trade, YearResult, Portfolio,
    _fetch, _indicators,
    _find_ftd_signal, _find_3wk_signal, _find_pullback_entry,
    _find_ma_retake_entry, _find_bull_reentry,
    _check_nuclear_exit, _count_distribution_days, _check_severe_weakness,
    _evaluate_exit,
    _is_bull_regime, _qqq_below_200, _qqq_death_cross, _is_confirmed_bull,
)


# ── Shared data fetch (run once) ─────────────────────────────────

def fetch_all_data(start_year=2021, end_year=2026):
    fetch_start = f"{start_year - 1}-01-01"
    end_dt = min(dt.date(end_year, 12, 31), dt.date.today())
    fetch_end = end_dt.strftime("%Y-%m-%d")

    tqqq = _indicators(_fetch("TQQQ", fetch_start, fetch_end))
    qqq = _indicators(_fetch("QQQ", fetch_start, fetch_end))
    nasdaq = _indicators(_fetch("^IXIC", fetch_start, fetch_end))

    return tqqq, qqq, nasdaq, start_year, end_year


# ── Helper: build year summaries ─────────────────────────────────

def summarize(equity, trades_by_year, tqqq_df, qqq_df, start_year, end_year):
    results = []
    for year in range(start_year, end_year + 1):
        year_start = f"{year}-01-01"
        year_end = f"{year + 1}-01-01"
        tqqq_year = tqqq_df[(tqqq_df.index >= year_start) & (tqqq_df.index < year_end)]
        qqq_year = qqq_df[(qqq_df.index >= year_start) & (qqq_df.index < year_end)]
        if tqqq_year.empty or qqq_year.empty or len(tqqq_year) < 5:
            continue

        tqqq_bh = (float(tqqq_year["Close"].iloc[-1]) / float(tqqq_year["Close"].iloc[0]) - 1) * 100
        year_eq = {d: v for d, v in equity.items()
                   if pd.Timestamp(year_start) <= d < pd.Timestamp(year_end)}
        if not year_eq:
            continue
        sorted_d = sorted(year_eq.keys())
        sv = year_eq[sorted_d[0]]
        ev = year_eq[sorted_d[-1]]
        if sv == 0:
            continue
        ret = ((ev / sv) - 1) * 100
        trades = trades_by_year.get(year, [])
        wins = [t for t in trades if t.return_pct > 0]
        wr = round(len(wins) / len(trades) * 100, 1) if trades else 0.0
        results.append({
            "year": year, "ret": round(ret, 1), "sv": round(sv, 0),
            "ev": round(ev, 0), "trades": len(trades), "wr": wr,
            "tqqq_bh": round(tqqq_bh, 1),
        })
    return results


# ── Shared close-position helper ─────────────────────────────────

def close_position(pf, idx, date, price, current_year, trades_by_year,
                   tqqq_df, state):
    held_days = idx - pf.entry_idx
    ret_pct = ((price - pf.entry_price) / pf.entry_price) * 100
    t = Trade(
        entry_date=pf.entry_date,
        exit_date=date.strftime("%Y-%m-%d"),
        entry_price=round(pf.entry_price, 2),
        exit_price=round(price, 2),
        return_pct=round(ret_pct, 2),
        signal_type=pf.signal_type,
        duration_days=held_days,
        outcome="Win" if ret_pct > 0 else "Loss",
        shares=round(pf.entry_shares, 2),
        cash_deployed=round(pf.entry_cash_deployed, 2),
        portfolio_before=round(pf.entry_portfolio_value, 2),
        portfolio_after=0.0,
        cash_after=0.0,
    )
    pf.sell_all(price)
    t.portfolio_after = round(pf.total_value(price), 2)
    t.cash_after = round(pf.cash, 2)
    trades_by_year.setdefault(current_year, []).append(t)
    state["last_trade_idx"] = idx

    if ret_pct <= 0:
        state["consecutive_losses"] += 1
    else:
        state["consecutive_losses"] = 0

    bull = _is_bull_regime(tqqq_df, idx)
    state["cooldown_until"] = idx + (COOLDOWN_BULL if bull else COOLDOWN_BEAR)


# ── Approach A: Quick Scale-In ───────────────────────────────────

def run_approach_a(tqqq_df, qqq_df, nasdaq_df, start_year, end_year):
    sim_start = f"{start_year}-01-01"
    all_idx = list(tqqq_df.index)
    sim_indices = [i for i, d in enumerate(all_idx) if d >= pd.Timestamp(sim_start)]
    if not sim_indices:
        return {}, {}

    pf = Portfolio(STARTING_CAPITAL)
    state = {"cooldown_until": 0, "consecutive_losses": 0, "last_trade_idx": 0}
    equity = {}
    trades_by_year = {}
    scale_pending = False
    scale_check_idx = 0
    initial_entry_price = 0.0

    for idx in sim_indices:
        date = all_idx[idx]
        yr = date.year
        price = float(tqqq_df.iloc[idx]["Close"])
        nq_idx = nasdaq_df.index.get_indexer([date], method="nearest")[0]
        qq_idx = qqq_df.index.get_indexer([date], method="nearest")[0]

        if not pf.in_position:
            scale_pending = False
            if idx < state["cooldown_until"]:
                equity[date] = pf.total_value(price)
                continue

            if state["consecutive_losses"] > 0 and (idx - state["last_trade_idx"]) > 30:
                state["consecutive_losses"] = 0

            death_cross = _qqq_death_cross(qqq_df, qq_idx)
            below_200 = _qqq_below_200(qqq_df, qq_idx)
            confirmed_bull = _is_confirmed_bull(qqq_df, qq_idx)
            bull = _is_bull_regime(tqqq_df, idx)

            is_ftd = _find_ftd_signal(nasdaq_df, nq_idx)
            is_3wk = _find_3wk_signal(qqq_df, tqqq_df, idx)
            is_pullback = bull and _find_pullback_entry(tqqq_df, idx)
            is_ma_retake = bull and _find_ma_retake_entry(tqqq_df, idx)
            is_bull_reentry = confirmed_bull and _find_bull_reentry(tqqq_df, idx)

            alloc = 0.0
            signal = ""

            if is_ftd:
                alloc = 0.5
                signal = "FTD"
                state["consecutive_losses"] = 0
            elif is_3wk:
                alloc = 0.5
                signal = "3WK"
            elif is_pullback:
                alloc = 0.5
                signal = "Pullback"
            elif is_ma_retake:
                alloc = 0.5
                signal = "MA Retake"
            elif is_bull_reentry:
                alloc = 0.5
                signal = "Bull Re-entry"

            if alloc > 0 and signal != "FTD" and state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                alloc = min(alloc, 0.25)

            if alloc > 0:
                lookback = min(20, idx)
                r_low = float(tqqq_df.iloc[idx - lookback: idx + 1]["Low"].min())
                pf.buy(price, alloc, date.strftime("%Y-%m-%d"), signal, idx, r_low)
                scale_pending = True
                scale_check_idx = idx + 3
                initial_entry_price = price

        else:
            # Scale-in check: after 3 days, if profitable → buy remaining cash
            if scale_pending and idx >= scale_check_idx:
                if price > initial_entry_price and pf.cash > 0:
                    additional_shares = pf.cash / price
                    pf.shares += additional_shares
                    pf.entry_cash_deployed += pf.cash
                    pf.cash = 0.0
                scale_pending = False

            action = _evaluate_exit(tqqq_df, nasdaq_df, idx,
                                    pf.entry_price, pf.entry_idx, pf.rally_low)
            if action == "full_exit":
                close_position(pf, idx, date, price, yr, trades_by_year,
                               tqqq_df, state)
            elif action in ("trim_heavy", "trim_light"):
                trim_frac = 0.4 if action == "trim_heavy" else 0.2
                pf.trim(price, trim_frac)
                if pf.shares * price < pf.total_value(price) * 0.10:
                    close_position(pf, idx, date, price, yr, trades_by_year,
                                   tqqq_df, state)

        equity[date] = pf.total_value(price)

    if pf.in_position:
        last_date = all_idx[sim_indices[-1]]
        price = float(tqqq_df.iloc[sim_indices[-1]]["Close"])
        close_position(pf, sim_indices[-1], last_date, price,
                       last_date.year, trades_by_year, tqqq_df, state)
        equity[last_date] = pf.total_value(price)

    return equity, trades_by_year


# ── Approach B: Confirmed Bull = 100% ────────────────────────────

def run_approach_b(tqqq_df, qqq_df, nasdaq_df, start_year, end_year):
    sim_start = f"{start_year}-01-01"
    all_idx = list(tqqq_df.index)
    sim_indices = [i for i, d in enumerate(all_idx) if d >= pd.Timestamp(sim_start)]
    if not sim_indices:
        return {}, {}

    pf = Portfolio(STARTING_CAPITAL)
    state = {"cooldown_until": 0, "consecutive_losses": 0, "last_trade_idx": 0}
    equity = {}
    trades_by_year = {}

    for idx in sim_indices:
        date = all_idx[idx]
        yr = date.year
        price = float(tqqq_df.iloc[idx]["Close"])
        nq_idx = nasdaq_df.index.get_indexer([date], method="nearest")[0]
        qq_idx = qqq_df.index.get_indexer([date], method="nearest")[0]

        if not pf.in_position:
            if idx < state["cooldown_until"]:
                equity[date] = pf.total_value(price)
                continue

            if state["consecutive_losses"] > 0 and (idx - state["last_trade_idx"]) > 30:
                state["consecutive_losses"] = 0

            death_cross = _qqq_death_cross(qqq_df, qq_idx)
            below_200 = _qqq_below_200(qqq_df, qq_idx)
            confirmed_bull = _is_confirmed_bull(qqq_df, qq_idx)
            bull = _is_bull_regime(tqqq_df, idx)

            is_ftd = _find_ftd_signal(nasdaq_df, nq_idx)
            is_3wk = _find_3wk_signal(qqq_df, tqqq_df, idx)
            is_pullback = bull and _find_pullback_entry(tqqq_df, idx)
            is_ma_retake = bull and _find_ma_retake_entry(tqqq_df, idx)
            is_bull_reentry = confirmed_bull and _find_bull_reentry(tqqq_df, idx)

            alloc = 0.0
            signal = ""

            if is_ftd:
                if confirmed_bull:
                    alloc = 1.0
                elif below_200:
                    alloc = 0.5
                else:
                    alloc = 1.0
                signal = "FTD"
                state["consecutive_losses"] = 0
            elif is_3wk:
                alloc = 1.0 if confirmed_bull else (0.25 if below_200 else 0.5)
                signal = "3WK"
            elif is_pullback:
                alloc = 1.0 if confirmed_bull else (0.25 if below_200 else 0.5)
                signal = "Pullback"
            elif is_ma_retake:
                alloc = 1.0 if confirmed_bull else (0.25 if below_200 else 0.5)
                signal = "MA Retake"
            elif is_bull_reentry:
                alloc = 1.0
                signal = "Bull Re-entry"

            if alloc > 0 and signal != "FTD" and state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                alloc = min(alloc, 0.25)

            if alloc > 0:
                lookback = min(20, idx)
                r_low = float(tqqq_df.iloc[idx - lookback: idx + 1]["Low"].min())
                pf.buy(price, alloc, date.strftime("%Y-%m-%d"), signal, idx, r_low)

        else:
            action = _evaluate_exit(tqqq_df, nasdaq_df, idx,
                                    pf.entry_price, pf.entry_idx, pf.rally_low)
            if action == "full_exit":
                close_position(pf, idx, date, price, yr, trades_by_year,
                               tqqq_df, state)
            elif action in ("trim_heavy", "trim_light"):
                trim_frac = 0.4 if action == "trim_heavy" else 0.2
                pf.trim(price, trim_frac)
                if pf.shares * price < pf.total_value(price) * 0.10:
                    close_position(pf, idx, date, price, yr, trades_by_year,
                                   tqqq_df, state)

        equity[date] = pf.total_value(price)

    if pf.in_position:
        last_date = all_idx[sim_indices[-1]]
        price = float(tqqq_df.iloc[sim_indices[-1]]["Close"])
        close_position(pf, sim_indices[-1], last_date, price,
                       last_date.year, trades_by_year, tqqq_df, state)
        equity[last_date] = pf.total_value(price)

    return equity, trades_by_year


# ── Approach C: Adaptive Allocation ──────────────────────────────

def run_approach_c(tqqq_df, qqq_df, nasdaq_df, start_year, end_year):
    sim_start = f"{start_year}-01-01"
    all_idx = list(tqqq_df.index)
    sim_indices = [i for i, d in enumerate(all_idx) if d >= pd.Timestamp(sim_start)]
    if not sim_indices:
        return {}, {}

    pf = Portfolio(STARTING_CAPITAL)
    state = {"cooldown_until": 0, "consecutive_losses": 0, "last_trade_idx": 0}
    equity = {}
    trades_by_year = {}

    profitable_days = 0
    peak_position_value = 0.0
    current_tier = 0  # 0=no position, 1=50%, 2=75%, 3=100%
    was_trimmed_back = False

    for idx in sim_indices:
        date = all_idx[idx]
        yr = date.year
        price = float(tqqq_df.iloc[idx]["Close"])
        nq_idx = nasdaq_df.index.get_indexer([date], method="nearest")[0]
        qq_idx = qqq_df.index.get_indexer([date], method="nearest")[0]

        if not pf.in_position:
            profitable_days = 0
            peak_position_value = 0.0
            current_tier = 0
            was_trimmed_back = False

            if idx < state["cooldown_until"]:
                equity[date] = pf.total_value(price)
                continue

            if state["consecutive_losses"] > 0 and (idx - state["last_trade_idx"]) > 30:
                state["consecutive_losses"] = 0

            death_cross = _qqq_death_cross(qqq_df, qq_idx)
            below_200 = _qqq_below_200(qqq_df, qq_idx)
            confirmed_bull = _is_confirmed_bull(qqq_df, qq_idx)
            bull = _is_bull_regime(tqqq_df, idx)

            is_ftd = _find_ftd_signal(nasdaq_df, nq_idx)
            is_3wk = _find_3wk_signal(qqq_df, tqqq_df, idx)
            is_pullback = bull and _find_pullback_entry(tqqq_df, idx)
            is_ma_retake = bull and _find_ma_retake_entry(tqqq_df, idx)
            is_bull_reentry = confirmed_bull and _find_bull_reentry(tqqq_df, idx)

            alloc = 0.0
            signal = ""

            if is_ftd:
                alloc = 0.5
                signal = "FTD"
                state["consecutive_losses"] = 0
            elif is_3wk:
                alloc = 0.5
                signal = "3WK"
            elif is_pullback:
                alloc = 0.5
                signal = "Pullback"
            elif is_ma_retake:
                alloc = 0.5
                signal = "MA Retake"
            elif is_bull_reentry:
                alloc = 0.5
                signal = "Bull Re-entry"

            if alloc > 0 and signal != "FTD" and state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                alloc = min(alloc, 0.25)

            if alloc > 0:
                lookback = min(20, idx)
                r_low = float(tqqq_df.iloc[idx - lookback: idx + 1]["Low"].min())
                pf.buy(price, alloc, date.strftime("%Y-%m-%d"), signal, idx, r_low)
                current_tier = 1
                peak_position_value = pf.shares * price

        else:
            position_value = pf.shares * price
            peak_position_value = max(peak_position_value, position_value)

            # Track profitable days since entry
            if price > pf.entry_price:
                profitable_days += 1
            else:
                profitable_days = max(0, profitable_days - 1)

            # Trailing drawdown check: if position drops 5% from peak, trim to 50%
            if peak_position_value > 0:
                drawdown_from_peak = (position_value - peak_position_value) / peak_position_value
                if drawdown_from_peak < -0.05 and current_tier > 1 and not was_trimmed_back:
                    total_val = pf.total_value(price)
                    target_position = total_val * 0.5
                    current_position = pf.shares * price
                    if current_position > target_position:
                        excess_shares = (current_position - target_position) / price
                        pf.cash += excess_shares * price
                        pf.shares -= excess_shares
                    current_tier = 1
                    was_trimmed_back = True
                    peak_position_value = pf.shares * price

            # Scale UP: 5 profitable days → 75%, 10 profitable days → 100%
            if not was_trimmed_back:
                if current_tier == 1 and profitable_days >= 5 and pf.cash > 0:
                    total_val = pf.total_value(price)
                    target = total_val * 0.75
                    current_pos = pf.shares * price
                    if current_pos < target:
                        add_cash = min(target - current_pos, pf.cash)
                        add_shares = add_cash / price
                        pf.shares += add_shares
                        pf.entry_cash_deployed += add_cash
                        pf.cash -= add_cash
                    current_tier = 2
                    peak_position_value = pf.shares * price

                elif current_tier == 2 and profitable_days >= 10 and pf.cash > 0:
                    add_shares = pf.cash / price
                    pf.shares += add_shares
                    pf.entry_cash_deployed += pf.cash
                    pf.cash = 0.0
                    current_tier = 3
                    peak_position_value = pf.shares * price

            action = _evaluate_exit(tqqq_df, nasdaq_df, idx,
                                    pf.entry_price, pf.entry_idx, pf.rally_low)
            if action == "full_exit":
                close_position(pf, idx, date, price, yr, trades_by_year,
                               tqqq_df, state)
            elif action in ("trim_heavy", "trim_light"):
                trim_frac = 0.4 if action == "trim_heavy" else 0.2
                pf.trim(price, trim_frac)
                if pf.shares * price < pf.total_value(price) * 0.10:
                    close_position(pf, idx, date, price, yr, trades_by_year,
                                   tqqq_df, state)

        equity[date] = pf.total_value(price)

    if pf.in_position:
        last_date = all_idx[sim_indices[-1]]
        price = float(tqqq_df.iloc[sim_indices[-1]]["Close"])
        close_position(pf, sim_indices[-1], last_date, price,
                       last_date.year, trades_by_year, tqqq_df, state)
        equity[last_date] = pf.total_value(price)

    return equity, trades_by_year


# ── Main ─────────────────────────────────────────────────────────

def print_results(label, rows):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  {'Year':<6} {'Return':>8} {'Start':>12} {'End':>12} {'Trades':>7} {'WR':>6} {'TQQQ B&H':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*7} {'-'*6} {'-'*10}")
    for r in rows:
        print(f"  {r['year']:<6} {r['ret']:>+7.1f}% {r['sv']:>11,.0f} {r['ev']:>11,.0f} {r['trades']:>7} {r['wr']:>5.1f}% {r['tqqq_bh']:>+9.1f}%")

    if rows:
        first_sv = rows[0]["sv"]
        last_ev = rows[-1]["ev"]
        total_ret = ((last_ev / first_sv) - 1) * 100
        total_trades = sum(r["trades"] for r in rows)
        print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*7} {'-'*6} {'-'*10}")
        print(f"  {'TOTAL':<6} {total_ret:>+7.1f}% {first_sv:>11,.0f} {last_ev:>11,.0f} {total_trades:>7}")
        print(f"\n  $100K → ${last_ev:,.0f}  ({total_trades} trades)")


if __name__ == "__main__":
    print("Fetching data...")
    tqqq_df, qqq_df, nasdaq_df, sy, ey = fetch_all_data(2021, 2026)
    print(f"Data loaded: TQQQ {len(tqqq_df)} rows, QQQ {len(qqq_df)} rows, NASDAQ {len(nasdaq_df)} rows")
    print(f"Date range: {tqqq_df.index[0].date()} to {tqqq_df.index[-1].date()}")

    print("\n>>> Running Approach A: Quick Scale-In (50% → 100% after 3 profitable days)...")
    eq_a, tr_a = run_approach_a(tqqq_df, qqq_df, nasdaq_df, sy, ey)
    rows_a = summarize(eq_a, tr_a, tqqq_df, qqq_df, sy, ey)

    print(">>> Running Approach B: Confirmed Bull = 100% immediately...")
    eq_b, tr_b = run_approach_b(tqqq_df, qqq_df, nasdaq_df, sy, ey)
    rows_b = summarize(eq_b, tr_b, tqqq_df, qqq_df, sy, ey)

    print(">>> Running Approach C: Adaptive Allocation (50 → 75 → 100, trailing trim)...")
    eq_c, tr_c = run_approach_c(tqqq_df, qqq_df, nasdaq_df, sy, ey)
    rows_c = summarize(eq_c, tr_c, tqqq_df, qqq_df, sy, ey)

    print_results("APPROACH A: Quick Scale-In (50% → 100% after 3 profitable days)", rows_a)
    print_results("APPROACH B: Confirmed Bull = 100% Immediately", rows_b)
    print_results("APPROACH C: Adaptive Allocation (50 → 75 → 100 + trailing trim)", rows_c)

    print(f"\n{'='*80}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Year':<6} {'A':>10} {'B':>10} {'C':>10} {'TQQQ B&H':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i in range(max(len(rows_a), len(rows_b), len(rows_c))):
        yr = rows_a[i]["year"] if i < len(rows_a) else (rows_b[i]["year"] if i < len(rows_b) else rows_c[i]["year"])
        a_ret = f"{rows_a[i]['ret']:+.1f}%" if i < len(rows_a) else "N/A"
        b_ret = f"{rows_b[i]['ret']:+.1f}%" if i < len(rows_b) else "N/A"
        c_ret = f"{rows_c[i]['ret']:+.1f}%" if i < len(rows_c) else "N/A"
        bh = f"{rows_a[i]['tqqq_bh']:+.1f}%" if i < len(rows_a) else "N/A"
        print(f"  {yr:<6} {a_ret:>10} {b_ret:>10} {c_ret:>10} {bh:>10}")

    if rows_a and rows_b and rows_c:
        a_final = rows_a[-1]["ev"]
        b_final = rows_b[-1]["ev"]
        c_final = rows_c[-1]["ev"]
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'$100K→':<6} {'$'+f'{a_final:,.0f}':>10} {'$'+f'{b_final:,.0f}':>10} {'$'+f'{c_final:,.0f}':>10}")
        best = max([("A", a_final), ("B", b_final), ("C", c_final)], key=lambda x: x[1])
        print(f"\n  >>> WINNER: Approach {best[0]} with ${best[1]:,.0f}")
