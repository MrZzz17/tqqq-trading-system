"""
Different trading PRINCIPLES vs the trend-following V6 family:
  M1  mean reversion standalone (RSI2 dips in bull regime)
  M2  ensemble: honest V6 + vol gate, with MR sleeve deployed only when V6 is flat
  V1  VIX term-structure regime (contango/backwardation)
  R1  dual momentum rotation TQQQ / TMF / cash

All under honest assumptions (real cash yield, 5 bps per side).
Read-only research harness. Safe to delete.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime as dt
import numpy as np
import pandas as pd

from core.backtest import _fetch, _indicators, STARTING_CAPITAL
from experiments.exp_v7_research import load_data, metrics, print_table, sim, START_YEAR, END_YEAR

TC = 0.0005


def fetch_extra():
    fetch_start = f"{START_YEAR - 2}-01-01"
    end_dt = min(dt.date(END_YEAR, 12, 31), dt.date.today())
    fetch_end = (end_dt + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    vix = _fetch("^VIX", fetch_start, fetch_end)
    vix3m = _fetch("^VIX3M", fetch_start, fetch_end)
    tmf = _fetch("TMF", fetch_start, fetch_end)
    return vix, vix3m, tmf


def rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def cash_rate(D, date):
    return float(D["cash_yield"].loc[date])


# ── M1: standalone mean reversion ────────────────────────────────

def sim_mean_reversion(D, size=1.0, rsi_buy=10, rsi_sell=65, max_hold=5,
                       regime="above200"):
    tqqq, qqq = D["tqqq"], D["qqq"]
    dates = tqqq.index[tqqq.index >= pd.Timestamp(f"{START_YEAR}-01-01")]
    r2 = rsi(qqq["Close"], 2)
    cash, shares = STARTING_CAPITAL, 0.0
    hold_days = 0
    equity, trades = {}, []
    entry_val = 0.0
    entry_d = None
    for i, date in enumerate(dates):
        price = float(tqqq.loc[date, "Close"])
        if cash > 0 and i > 0:
            cash *= (1 + cash_rate(D, date))
        total = cash + shares * price
        qrow = qqq.loc[date] if date in qqq.index else None
        if qrow is None or pd.isna(qrow.get("SMA_200")):
            equity[date] = total
            continue
        bull = float(qrow["Close"]) > float(qrow["SMA_200"])
        if regime == "always":
            bull = True
        rv = r2.loc[date]

        if shares == 0:
            if bull and rv <= rsi_buy:
                deploy = total * size
                shares = deploy * (1 - TC) / price
                cash -= deploy
                hold_days = 0
                entry_val = total
                entry_d = date
        else:
            hold_days += 1
            exit_now = (rv >= rsi_sell) or (hold_days >= max_hold) or (not bull)
            if exit_now:
                cash += shares * price * (1 - TC)
                shares = 0.0
                trades.append({"entry": entry_d.strftime("%Y-%m-%d"),
                               "exit": date.strftime("%Y-%m-%d"),
                               "ret": round((cash + 0 - entry_val) / entry_val * 100, 2),
                               "year": date.year})
        equity[date] = cash + shares * price
    if shares > 0:
        equity[dates[-1]] = cash + shares * float(tqqq.loc[dates[-1], "Close"])
    m = metrics(equity)
    m["n_trades"] = len(trades)
    wins = sum(1 for t in trades if t["ret"] > 0)
    m["wr"] = wins / len(trades) * 100 if trades else 0
    m["trades"] = trades
    return m


# ── M2: ensemble — V6 trend core + MR sleeve when flat ───────────

def sim_ensemble(D, vol_gate=0.28, mr_size=0.5, rsi_buy=10, rsi_sell=65, max_hold=5):
    """Recompute honest V6 (+vol gate) exposure path, then run an MR sleeve on the
    cash whenever the trend core is 100% flat. MR only trades when core expo == 0,
    so it cannot interfere with the core's trailing stop."""
    core = sim(D, indexer="ffill", real_cash_yield=True, tc=TC, vol_gate=vol_gate,
               return_expo=True)
    expo = core["expo"]
    tqqq, qqq = D["tqqq"], D["qqq"]
    dates = sorted(expo.keys())
    px = pd.Series({d: float(tqqq.loc[d, "Close"]) for d in dates})
    rets = px.pct_change().fillna(0.0)
    r2 = rsi(qqq["Close"], 2)
    bull = (qqq["Close"] > qqq["SMA_200"]).reindex(px.index).fillna(False)

    val = STARTING_CAPITAL
    equity = {}
    mr_in = False
    hold_days = 0
    n_mr = 0
    mr_wins = 0
    mr_entry_val = None
    prev = None
    for d in dates:
        if prev is not None:
            e_core = expo[prev]
            e_mr = mr_size if (mr_in and e_core == 0.0) else 0.0
            e = e_core + e_mr
            r = rets[d]
            val = val * (e * (1 + r) + (1 - e) * (1 + cash_rate(D, d)))
        equity[d] = val
        # MR state machine (signal at close of d, applies to next day)
        e_core_now = expo[d]
        if e_core_now == 0.0 and bool(bull.loc[d]):
            rv = r2.loc[d] if d in r2.index else 50.0
            if not mr_in and rv <= rsi_buy:
                mr_in = True
                hold_days = 0
                n_mr += 1
                mr_entry_val = val
                val -= val * mr_size * TC  # entry cost
            elif mr_in:
                hold_days += 1
                if rv >= rsi_sell or hold_days >= max_hold:
                    val -= val * mr_size * TC  # exit cost
                    if mr_entry_val is not None and val > mr_entry_val:
                        mr_wins += 1
                    mr_in = False
        else:
            if mr_in:
                val -= val * mr_size * TC
                mr_in = False
        prev = d
    m = metrics(equity)
    m["n_trades"] = core["n_trades"]
    m["n_mr"] = n_mr
    m["mr_wins"] = mr_wins
    return m


# ── V1: VIX term structure ───────────────────────────────────────

def sim_vix_structure(D, vix, vix3m, use_200=True, enter_ratio=1.0, exit_ratio=0.97):
    tqqq, qqq = D["tqqq"], D["qqq"]
    dates = tqqq.index[tqqq.index >= pd.Timestamp(f"{START_YEAR}-01-01")]
    ratio = (vix3m["Close"] / vix["Close"]).reindex(tqqq.index, method="ffill")
    above = (qqq["Close"] > qqq["SMA_200"]).reindex(tqqq.index, method="ffill").fillna(False)
    cash, shares = STARTING_CAPITAL, 0.0
    equity = {}
    invested = False
    n_switch = 0
    for i, date in enumerate(dates):
        price = float(tqqq.loc[date, "Close"])
        if cash > 0 and i > 0:
            cash *= (1 + cash_rate(D, date))
        rt = ratio.loc[date]
        if pd.isna(rt):
            equity[date] = cash + shares * price
            continue
        contango_on = float(rt) >= enter_ratio
        contango_off = float(rt) <= exit_ratio
        bull_ok = bool(above.loc[date]) or not use_200
        if not invested and contango_on and bull_ok:
            total = cash + shares * price
            shares = total * (1 - TC) / price
            cash = 0.0
            invested = True
            n_switch += 1
        elif invested and (contango_off or not bull_ok):
            cash = shares * price * (1 - TC)
            shares = 0.0
            invested = False
            n_switch += 1
        equity[date] = cash + shares * price
    m = metrics(equity)
    m["n_trades"] = n_switch // 2
    return m


# ── R1: dual momentum TQQQ / TMF / cash ──────────────────────────

def sim_dual_momentum(D, tmf, lookback=63, rebal=21):
    tqqq = D["tqqq"]
    dates = tqqq.index[tqqq.index >= pd.Timestamp(f"{START_YEAR}-01-01")]
    tq_px = tqqq["Close"]
    tm_px = tmf["Close"].reindex(tqqq.index, method="ffill")
    cash, units, asset = STARTING_CAPITAL, 0.0, None  # asset in {None,'TQQQ','TMF'}
    equity = {}
    n_switch = 0
    for i, date in enumerate(dates):
        p_tq = float(tq_px.loc[date])
        p_tm = float(tm_px.loc[date]) if not pd.isna(tm_px.loc[date]) else None
        if asset is None and cash > 0 and i > 0:
            cash *= (1 + cash_rate(D, date))
        if i % rebal == 0 and i >= lookback:
            mom_tq = p_tq / float(tq_px.iloc[max(0, tq_px.index.get_loc(date) - lookback)]) - 1
            mom_tm = (p_tm / float(tm_px.iloc[max(0, tm_px.index.get_indexer([date])[0] - lookback)]) - 1) if p_tm else -1
            cash_mom = float(D["cash_yield"].loc[date]) * lookback
            best = max([("TQQQ", mom_tq), ("TMF", mom_tm), (None, cash_mom)], key=lambda x: x[1])[0]
            if best != asset:
                # liquidate
                if asset == "TQQQ":
                    cash = units * p_tq * (1 - TC)
                elif asset == "TMF" and p_tm:
                    cash = units * p_tm * (1 - TC)
                units = 0.0
                # buy
                if best == "TQQQ":
                    units = cash * (1 - TC) / p_tq
                    cash = 0.0
                elif best == "TMF" and p_tm:
                    units = cash * (1 - TC) / p_tm
                    cash = 0.0
                asset = best
                n_switch += 1
        v = cash
        if asset == "TQQQ":
            v = units * p_tq
        elif asset == "TMF" and p_tm:
            v = units * p_tm
        equity[date] = v
    m = metrics(equity)
    m["n_trades"] = n_switch
    return m


if __name__ == "__main__":
    D = load_data()
    vix, vix3m, tmf = fetch_extra()
    print(f"data: VIX rows={len(vix)}, VIX3M rows={len(vix3m)}, TMF rows={len(tmf)}")

    rows = []
    r = sim(D, indexer="ffill", real_cash_yield=True, tc=TC)
    r["name"] = "REF honest V6 baseline"; rows.append(r)
    r = sim(D, indexer="ffill", real_cash_yield=True, tc=TC, vol_gate=0.28)
    r["name"] = "REF V6 + vol gate 28%"; rows.append(r)

    m = sim_mean_reversion(D, size=1.0)
    m["name"] = f"M1 mean reversion RSI2 (100%, {m['n_trades']}tr {m['wr']:.0f}%W)"; rows.append(m)
    m = sim_mean_reversion(D, size=1.0, regime="always")
    m["name"] = "M1b MR no regime filter (danger check)"; rows.append(m)

    m = sim_ensemble(D, vol_gate=0.28, mr_size=0.5)
    m["name"] = f"M2 ensemble: gate28 core + MR50% when flat ({m['n_mr']}mr)"; rows.append(m)
    m = sim_ensemble(D, vol_gate=0.28, mr_size=1.0)
    m["name"] = "M2b ensemble with MR at 100% size"; rows.append(m)

    if not vix.empty and not vix3m.empty:
        m = sim_vix_structure(D, vix, vix3m, use_200=True)
        m["name"] = f"V1 VIX3M/VIX contango + 200SMA ({m['n_trades']}sw)"; rows.append(m)
        m = sim_vix_structure(D, vix, vix3m, use_200=False)
        m["name"] = "V1b contango only (no 200SMA)"; rows.append(m)

    if not tmf.empty:
        m = sim_dual_momentum(D, tmf)
        m["name"] = f"R1 dual momentum TQQQ/TMF/cash ({m['n_trades']}sw)"; rows.append(m)

    print()
    print_table(rows)
