"""
Read-only measurement harness.

Faithfully reproduces core.backtest._run_continuous and adds toggles so we can
measure the CAGR vs max-drawdown trade-off of anti-whipsaw re-entry rules.

Nothing here modifies the engine or any state. Safe to delete.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as dt
import numpy as np
import pandas as pd

from core.backtest import (
    _fetch, _indicators, _make_weekly, _find_ftd_signal,
    STARTING_CAPITAL, SGOV_DAILY_YIELD,
)


def simulate(
    start_year=2011, end_year=2026,
    reentry_confirm=False,      # after a stop, MACD re-entry requires QQQ > SMA_50
    trail_cooldown=10,          # days blocked after 12% trail exit
    macd_cooldown_after_exit=0, # extra days MACD re-entry is blocked after ANY exit
    trail_gate=3.0,             # arm the trailing stop only when pct_above >= this
    trail_pct=12.0,             # trailing-stop drawdown threshold (%)
    trim_gate=None,             # if set, trim to trim_to when pct_above >= this
    trim_to=0.5,
    adaptive_cd=False,          # penalty box: cooldown grows with recent stop count
    cd_base=10, cd_extra=20, cd_window=90,
    breakout_reentry=0,         # after a stop, MACD re-entry needs an N-day closing high
):
    fetch_start = f"{start_year - 2}-01-01"
    end_dt = min(dt.date(end_year, 12, 31), dt.date.today())
    fetch_end = (end_dt + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    tqqq = _indicators(_fetch("TQQQ", fetch_start, fetch_end))
    qqq = _indicators(_fetch("QQQ", fetch_start, fetch_end))
    nasdaq = _indicators(_fetch("^IXIC", fetch_start, fetch_end))
    qqq_w = _make_weekly(qqq)
    qqq_hiN = qqq["Close"].rolling(breakout_reentry).max().shift(1) if breakout_reentry else None

    sim_start = pd.Timestamp(f"{start_year}-01-01")
    dates = tqqq.index[tqqq.index >= sim_start]

    cash = STARTING_CAPITAL
    shares = 0.0
    peak_portfolio = STARTING_CAPITAL
    equity = {}
    expo = {}  # actual TQQQ exposure held into the NEXT day, per date
    trades = []
    entry_date = entry_price = entry_value = entry_shares = entry_cash = None
    entry_signal = ""
    exited = False
    cooldown_until = None
    ftd_cooldown_until = None
    macd_block_until = None  # anti-whipsaw: MACD re-entry blocked until this date
    stop_dates = []          # dates of trail/200 stops, for adaptive penalty box

    def stop_cooldown(date):
        if not adaptive_cd:
            return cd_base
        recent = sum(1 for d in stop_dates if d != date and (date - d).days <= cd_window)
        return cd_base + cd_extra * recent

    for i, date in enumerate(dates):
        price = float(tqqq.loc[date, "Close"])
        total = cash + shares * price
        if cash > 0 and i > 0:
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
        sma50_raw = qqq.iloc[qq_idx].get("SMA_50")
        sma50 = float(sma50_raw) if sma50_raw is not None and not pd.isna(sma50_raw) else sma200
        above_200 = qq_close > sma200
        pct_above = ((qq_close - sma200) / sma200) * 100
        above_50 = qq_close > sma50

        w_dates = qqq_w.index[qqq_w.index <= date]
        macd_pos = len(w_dates) >= 1 and float(qqq_w.loc[w_dates[-1], "MACD"]) > 0

        ti = tqqq.index.get_indexer([date], method="nearest")[0]
        crash = False
        if ti >= 10:
            rh = float(tqqq.iloc[ti - 10: ti + 1]["High"].max())
            crash = ((price - rh) / rh * 100) <= -30

        is_ftd = _find_ftd_signal(nasdaq, nq_idx)
        ftd_blocked = ftd_cooldown_until is not None and date < ftd_cooldown_until
        macd_blocked = macd_block_until is not None and date < macd_block_until

        if crash and shares == 0:
            cooldown_until = date + pd.Timedelta(days=40)
            ftd_cooldown_until = date + pd.Timedelta(days=40)

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
                if breakout_reentry:
                    hv = qqq_hiN.iloc[qq_idx]
                    breakout_ok = (hv is not None) and (not pd.isna(hv)) and (qq_close >= float(hv))
                else:
                    breakout_ok = True
                if is_ftd:
                    exited = False
                    target = 0.5 if not macd_pos else 1.0
                elif macd_pos and not macd_blocked and (not reentry_confirm or above_50) and breakout_ok:
                    exited = False
                    target = 1.0
                else:
                    target = 0.0
            else:
                target = 1.0 if macd_pos else 0.5
        else:
            if total > peak_portfolio:
                peak_portfolio = total
            if pct_above >= trail_gate:
                pdd = ((total - peak_portfolio) / peak_portfolio) * 100
                if pdd <= -trail_pct:
                    target = 0.0
                    exited = True
                    cd = stop_cooldown(date) if adaptive_cd else trail_cooldown
                    if not stop_dates or stop_dates[-1] != date:
                        stop_dates.append(date)
                    cooldown_until = date + pd.Timedelta(days=cd)
                    ftd_cooldown_until = date + pd.Timedelta(days=15)
                    macd_block_until = date + pd.Timedelta(days=macd_cooldown_after_exit)
            if not above_200:
                target = 0.0
                exited = True
                cd = stop_cooldown(date) if adaptive_cd else 10
                if not stop_dates or stop_dates[-1] != date:
                    stop_dates.append(date)
                cooldown_until = date + pd.Timedelta(days=cd)
                ftd_cooldown_until = date + pd.Timedelta(days=15)
                macd_block_until = date + pd.Timedelta(days=macd_cooldown_after_exit)
            if target != 0.0 or (target == 0.0 and not exited):
                target = 1.0 if (macd_pos and above_200) else 0.5

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
                trades.append({
                    "entry": entry_date.strftime("%Y-%m-%d"),
                    "exit": date.strftime("%Y-%m-%d"),
                    "ret": round((pnl / entry_value) * 100, 2),
                    "before": round(entry_value, 0),
                    "after": round(exit_total, 0),
                    "signal": entry_signal,
                    "year": date.year,
                })
            shares = 0.0
            entry_date = None
            peak_portfolio = cash

        eq_now = cash + shares * price
        equity[date] = eq_now
        expo[date] = (shares * price) / eq_now if eq_now > 0 else 0.0

    # finalize open position for fair final-value comparison
    if shares > 0:
        last = dates[-1]
        equity[last] = cash + shares * float(tqqq.loc[last, "Close"])

    s = pd.Series({pd.Timestamp(k): v for k, v in equity.items()}).sort_index()
    final = float(s.iloc[-1])
    dd = (s / s.cummax() - 1) * 100
    mdd = float(dd.min())
    n_years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (final / STARTING_CAPITAL) ** (1 / n_years) - 1
    s26 = s["2026-01-01":]
    dd26 = float((s26 / s26.cummax() - 1).min() * 100) if len(s26) else 0.0
    return {
        "final": final, "mult": final / STARTING_CAPITAL, "cagr": cagr * 100,
        "mdd": mdd, "dd_2026": dd26, "trades": trades,
        "n_trades": len(trades),
        "wins": sum(1 for t in trades if t["ret"] > 0),
        "expo": expo,
        "tqqq_close": {d: float(tqqq.loc[d, "Close"]) for d in dates},
    }


def overlay(base_run, mode="cap", k=0.6, vol_target=0.55, vol_win=20):
    """First-order sizing overlay on the strategy's own daily exposure series.

    Rebuilds equity where each day's TQQQ exposure e_t is transformed:
      - cap : e' = e_t * k              (proportional de-lever)
      - vol : e' = e_t * min(1, vt/rv)  (volatility targeting on TQQQ realized vol)
    Cash earns SGOV yield. Ignores the trail/peak feedback loop (clearly first-order).
    """
    dates = sorted(base_run["expo"].keys())
    px = base_run["tqqq_close"]
    rets = pd.Series({d: px[d] for d in dates}).pct_change().fillna(0.0)
    # 20d annualized realized vol of TQQQ
    rv = rets.rolling(vol_win).std() * np.sqrt(252)
    val = STARTING_CAPITAL
    eqs = {}
    prev = None
    for d in dates:
        if prev is not None:
            e = base_run["expo"][prev]
            if mode == "cap":
                e2 = e * k
            else:
                rvd = rv.get(prev)
                scale = 1.0 if (rvd is None or pd.isna(rvd) or rvd == 0) else min(1.0, vol_target / rvd)
                e2 = e * scale
            r = rets[d]
            val = val * (e2 * (1 + r) + (1 - e2) * (1 + SGOV_DAILY_YIELD))
        eqs[d] = val
        prev = d
    s = pd.Series(eqs).sort_index()
    final = float(s.iloc[-1])
    mdd = float((s / s.cummax() - 1).min() * 100)
    n_years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = ((final / STARTING_CAPITAL) ** (1 / n_years) - 1) * 100
    s26 = s["2026-01-01":]
    dd26 = float((s26 / s26.cummax() - 1).min() * 100) if len(s26) else 0.0
    return {"final": final, "mult": final / STARTING_CAPITAL, "cagr": cagr,
            "mdd": mdd, "dd_2026": dd26, "n_trades": 0, "wins": 0, "trades": []}


def show(name, r):
    print(f"\n=== {name} ===")
    print(f"  final ${r['final']:,.0f}  ({r['mult']:.0f}x)   CAGR {r['cagr']:.1f}%")
    print(f"  max DD {r['mdd']:.1f}%   2026 DD {r['dd_2026']:.1f}%   trades {r['n_trades']} ({r['wins']}W)")
    t26 = [t for t in r["trades"] if t["year"] == 2026]
    for t in t26:
        print(f"    {t['entry']} -> {t['exit']}  {t['signal']:5s} {t['ret']:+6.1f}%  {t['before']:,.0f} -> {t['after']:,.0f}")


if __name__ == "__main__":
    base = simulate()
    show("BASELINE (current engine)", base)

    confirm = simulate(reentry_confirm=True)
    show("A1: MACD re-entry requires QQQ > 50DMA", confirm)

    cd25 = simulate(macd_cooldown_after_exit=25)
    show("A2: MACD re-entry blocked 25d after any exit", cd25)

    combo = simulate(reentry_confirm=True, macd_cooldown_after_exit=15)
    show("A3: confirm 50DMA + 15d MACD cooldown", combo)

    show("B1: trail always armed (gate 0), 12%", simulate(trail_gate=0.0, trail_pct=12.0))
    show("B2: trail always armed (gate 0), 10%", simulate(trail_gate=0.0, trail_pct=10.0))
    show("B3: trail always armed (gate 0), 8%",  simulate(trail_gate=0.0, trail_pct=8.0))
    show("B4: keep gate 3%, tighten to 10%",     simulate(trail_gate=3.0, trail_pct=10.0))
    show("B5: keep gate 3%, tighten to 8%",      simulate(trail_gate=3.0, trail_pct=8.0))

    # First-order sizing overlays on the SAME signals as baseline
    show("C0: overlay sanity (k=1.0, should ~= baseline)", overlay(base, mode="cap", k=1.0))
    show("C1: proportional de-lever to 60% (~1.8x eff)", overlay(base, mode="cap", k=0.60))
    show("C2: proportional de-lever to 40% (~1.2x eff)", overlay(base, mode="cap", k=0.40))
    show("C3: volatility targeting (target 55% ann vol)", overlay(base, mode="vol", vol_target=0.55))
    show("C4: volatility targeting (target 45% ann vol)", overlay(base, mode="vol", vol_target=0.45))

    # D: regime-aware "don't re-enter as fast in chop" (FTD always still allowed)
    show("D1: adaptive penalty box +20d/stop (90d window)",
         simulate(adaptive_cd=True, cd_base=10, cd_extra=20, cd_window=90))
    show("D2: adaptive penalty box +30d/stop (120d window)",
         simulate(adaptive_cd=True, cd_base=10, cd_extra=30, cd_window=120))
    show("D3: breakout re-entry (20-day QQQ closing high)",
         simulate(breakout_reentry=20))
    show("D4: breakout re-entry (10-day QQQ closing high)",
         simulate(breakout_reentry=10))
    show("D5: adaptive +20d + 20d breakout (combo)",
         simulate(adaptive_cd=True, cd_base=10, cd_extra=20, cd_window=90, breakout_reentry=20))
