"""
Phase 1: empirical QA of core/backtest.py assumptions (lookahead, cash yield,
transaction costs, dead allocation tiers).
Phase 2: new algorithm constructions — vol-regime gates, composite-score tiers
with partial rebalancing, wired-in 21-EMA exit, and simple benchmarks.

Read-only research harness. Safe to delete.
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

START_YEAR, END_YEAR = 2011, 2026


# ── data ──────────────────────────────────────────────────────────

def load_data():
    fetch_start = f"{START_YEAR - 2}-01-01"
    end_dt = min(dt.date(END_YEAR, 12, 31), dt.date.today())
    fetch_end = (end_dt + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    tqqq = _indicators(_fetch("TQQQ", fetch_start, fetch_end))
    qqq = _indicators(_fetch("QQQ", fetch_start, fetch_end))
    nasdaq = _indicators(_fetch("^IXIC", fetch_start, fetch_end))
    irx = _fetch("^IRX", fetch_start, fetch_end)  # 13-week T-bill discount rate, %
    qqq_w = _make_weekly(qqq)

    # realistic daily cash yield series aligned to tqqq dates
    if not irx.empty:
        y = (irx["Close"] / 100.0 / 252.0).reindex(tqqq.index, method="ffill").fillna(0.0)
    else:
        y = pd.Series(SGOV_DAILY_YIELD, index=tqqq.index)

    # QQQ 20d realized vol (annualized), and 200SMA slope over 20d
    rv20 = qqq["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    sma200_slope = qqq["SMA_200"].pct_change(20)

    return dict(tqqq=tqqq, qqq=qqq, nasdaq=nasdaq, qqq_w=qqq_w,
                cash_yield=y, rv20=rv20, sma200_slope=sma200_slope)


# ── metrics ───────────────────────────────────────────────────────

def metrics(equity: dict, name=""):
    s = pd.Series({pd.Timestamp(k): v for k, v in equity.items()}).sort_index()
    final = float(s.iloc[-1])
    dd = (s / s.cummax() - 1) * 100
    mdd = float(dd.min())
    n_years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = ((final / STARTING_CAPITAL) ** (1 / n_years) - 1) * 100
    dr = s.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    s26 = s["2026-01-01":]
    dd26 = float((s26 / s26.cummax() - 1).min() * 100) if len(s26) else 0.0

    def sub(a, b):
        ss = s[a:b]
        if len(ss) < 50:
            return 0.0, 0.0
        yrs = (ss.index[-1] - ss.index[0]).days / 365.25
        c = ((float(ss.iloc[-1]) / float(ss.iloc[0])) ** (1 / yrs) - 1) * 100
        m = float((ss / ss.cummax() - 1).min() * 100)
        return c, m

    c1, m1 = sub("2011-01-01", "2018-12-31")
    c2, m2 = sub("2019-01-01", "2026-12-31")
    return dict(name=name, final=final, mult=final / STARTING_CAPITAL, cagr=cagr,
                mdd=mdd, dd26=dd26, calmar=cagr / abs(mdd) if mdd else 0.0,
                sharpe=sharpe, cagr_a=c1, mdd_a=m1, cagr_b=c2, mdd_b=m2)


def print_table(rows):
    hdr = f"{'variant':44s} {'mult':>6s} {'CAGR':>6s} {'maxDD':>7s} {'26DD':>7s} {'Calmar':>6s} {'Sharpe':>6s} | {'11-18':>12s} {'19-26':>12s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:44s} {r['mult']:5.0f}x {r['cagr']:5.1f}% {r['mdd']:6.1f}% {r['dd26']:6.1f}% "
              f"{r['calmar']:6.2f} {r['sharpe']:6.2f} | {r['cagr_a']:4.0f}%/{r['mdd_a']:4.0f}% {r['cagr_b']:4.0f}%/{r['mdd_b']:4.0f}%")


# ── unified simulator ─────────────────────────────────────────────

def sim(D,
        indexer="nearest",        # "nearest" (engine) or "ffill" (no lookahead)
        real_cash_yield=False,    # ^IRX series instead of flat 4.5%
        tc=0.0,                   # one-way transaction cost, fraction of notional
        ema21_exit=False,         # wire in advertised rule 9: 2 TQQQ closes < 21-EMA
        vol_gate=None,            # cap target at 0.5 when QQQ rv20 > this (e.g. 0.28)
        adaptive_cd=False, cd_base=10, cd_extra=20, cd_window=90,
        tier_mode=False,          # composite-score tiers + partial rebalancing
        rebalance_band=0.15,
        return_expo=False,        # include daily exposure series in the result
        ):
    tqqq, qqq, nasdaq, qqq_w = D["tqqq"], D["qqq"], D["nasdaq"], D["qqq_w"]
    dates = tqqq.index[tqqq.index >= pd.Timestamp(f"{START_YEAR}-01-01")]

    cash = STARTING_CAPITAL
    shares = 0.0
    peak_portfolio = STARTING_CAPITAL
    equity, trades = {}, []
    expo = {}
    entry_date = None
    entry_value = 0.0
    entry_signal = ""
    exited = False
    cooldown_until = None
    ftd_cooldown_until = None
    stop_dates = []
    n_rebals = 0

    def stop_cd(date):
        if not adaptive_cd:
            return cd_base
        recent = sum(1 for d in stop_dates if d != date and (date - d).days <= cd_window)
        return cd_base + cd_extra * recent

    def idx_for(df, date):
        if indexer == "ffill":
            j = df.index.get_indexer([date], method="ffill")[0]
            return j if j >= 0 else 0
        return df.index.get_indexer([date], method="nearest")[0]

    for i, date in enumerate(dates):
        price = float(tqqq.loc[date, "Close"])
        if cash > 0 and i > 0:
            cy = float(D["cash_yield"].loc[date]) if real_cash_yield else SGOV_DAILY_YIELD
            cash *= (1 + cy)
        total = cash + shares * price

        qq_idx = idx_for(qqq, date)
        nq_idx = idx_for(nasdaq, date)
        qq_row = qqq.iloc[qq_idx]
        qq_close = float(qq_row["Close"])
        sma200_raw = qq_row.get("SMA_200")
        if sma200_raw is None or pd.isna(sma200_raw):
            equity[date] = total
            continue
        sma200 = float(sma200_raw)
        above_200 = qq_close > sma200
        pct_above = ((qq_close - sma200) / sma200) * 100
        sma50_raw = qq_row.get("SMA_50")
        sma50_above = (not pd.isna(sma50_raw)) and float(sma50_raw) > sma200

        w_dates = qqq_w.index[qqq_w.index <= date]
        macd_pos = len(w_dates) >= 1 and float(qqq_w.loc[w_dates[-1], "MACD"]) > 0

        ti = idx_for(tqqq, date)
        crash = False
        if ti >= 10:
            rh = float(tqqq.iloc[ti - 10: ti + 1]["High"].max())
            crash = ((price - rh) / rh * 100) <= -30

        is_ftd = _find_ftd_signal(nasdaq, nq_idx)
        ftd_blocked = ftd_cooldown_until is not None and date < ftd_cooldown_until

        rv = D["rv20"].iloc[qq_idx]
        rv = float(rv) if not pd.isna(rv) else 0.0
        vol_calm = (vol_gate is None) or (rv <= vol_gate)

        if crash and shares == 0:
            cooldown_until = date + pd.Timedelta(days=40)
            ftd_cooldown_until = date + pd.Timedelta(days=40)

        # ── target allocation ──
        if tier_mode:
            # Composite score: each condition adds conviction; stops/crash still rule.
            score = (int(above_200) + int(sma50_above) + int(macd_pos)
                     + int(rv <= (vol_gate or 0.30)))
            tier_target = {4: 1.0, 3: 0.75, 2: 0.5}.get(score, 0.0)
            if not above_200 and not (is_ftd and not ftd_blocked and not crash):
                tier_target = 0.0
            if cooldown_until and date < cooldown_until:
                if is_ftd and not ftd_blocked and not crash:
                    cooldown_until = None
                    tier_target = max(tier_target, 0.5)
                else:
                    tier_target = 0.0
            target = tier_target
            # trailing stop on portfolio, same as engine
            if shares > 0:
                if total > peak_portfolio:
                    peak_portfolio = total
                if pct_above >= 3.0:
                    pdd = ((total - peak_portfolio) / peak_portfolio) * 100
                    if pdd <= -12.0:
                        target = 0.0
                        if not stop_dates or stop_dates[-1] != date:
                            stop_dates.append(date)
                        cooldown_until = date + pd.Timedelta(days=stop_cd(date))
                        ftd_cooldown_until = date + pd.Timedelta(days=15)
        else:
            # V6 logic (faithful), with optional QA/feature toggles
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
                        if not stop_dates or stop_dates[-1] != date:
                            stop_dates.append(date)
                        cooldown_until = date + pd.Timedelta(days=stop_cd(date))
                        ftd_cooldown_until = date + pd.Timedelta(days=15)
                if not above_200:
                    target = 0.0
                    exited = True
                    if not stop_dates or stop_dates[-1] != date:
                        stop_dates.append(date)
                    cooldown_until = date + pd.Timedelta(days=stop_cd(date))
                    ftd_cooldown_until = date + pd.Timedelta(days=15)
                # advertised rule 9, optionally wired in
                if ema21_exit and target != 0.0 or (ema21_exit and not exited and target == 0.0):
                    ema21 = tqqq.iloc[ti].get("EMA_21")
                    ema21_prev = tqqq.iloc[ti - 1].get("EMA_21") if ti >= 1 else None
                    c_prev = float(tqqq.iloc[ti - 1]["Close"]) if ti >= 1 else price
                    if (ema21 is not None and not pd.isna(ema21) and ema21_prev is not None
                            and not pd.isna(ema21_prev)
                            and price < float(ema21) and c_prev < float(ema21_prev)):
                        target = 0.0
                        exited = True
                        if not stop_dates or stop_dates[-1] != date:
                            stop_dates.append(date)
                        cooldown_until = date + pd.Timedelta(days=stop_cd(date))
                        ftd_cooldown_until = date + pd.Timedelta(days=15)
                if target != 0.0 or (target == 0.0 and not exited):
                    target = 1.0 if (macd_pos and above_200) else 0.5

            if not vol_calm and target > 0.5:
                target = 0.5

        # ── execution ──
        current_alloc = (shares * price) / total if total > 0 else 0.0
        do_trade = False
        if tier_mode:
            do_trade = abs(target - current_alloc) > rebalance_band or (target == 0.0 and current_alloc > 0.01)
        else:
            do_trade = (target >= 0.1 and current_alloc < 0.1) or (target < 0.1 and current_alloc > 0.1)

        if do_trade:
            desired_val = total * target
            delta = desired_val - shares * price
            if delta > 0:
                deploy = min(delta, cash)
                cost = deploy * tc
                bought = (deploy - cost) / price
                shares += bought
                cash -= deploy
                n_rebals += 1
                if current_alloc < 0.1:
                    entry_date = date
                    entry_value = total
                    entry_signal = "FTD" if is_ftd else ("MACD" if macd_pos else "Entry")
                    peak_portfolio = max(peak_portfolio, total)
            elif delta < 0:
                sell_val = min(-delta, shares * price)
                sold_shares = sell_val / price
                shares -= sold_shares
                cash += sell_val * (1 - tc)
                n_rebals += 1
                if target < 0.1:
                    exit_total = cash + shares * price
                    if entry_date is not None:
                        pnl = exit_total - entry_value
                        trades.append({"entry": entry_date.strftime("%Y-%m-%d"),
                                       "exit": date.strftime("%Y-%m-%d"),
                                       "ret": round(pnl / entry_value * 100, 2),
                                       "signal": entry_signal, "year": date.year})
                    shares = 0.0
                    entry_date = None
                    peak_portfolio = cash

        eq_now = cash + shares * price
        equity[date] = eq_now
        expo[date] = (shares * price) / eq_now if eq_now > 0 else 0.0

    if shares > 0:
        last = dates[-1]
        equity[last] = cash + shares * float(tqqq.loc[last, "Close"])

    m = metrics(equity)
    m["n_trades"] = len(trades)
    m["n_rebals"] = n_rebals
    m["trades"] = trades
    if return_expo:
        m["expo"] = expo
    return m


# ── benchmarks ────────────────────────────────────────────────────

def bench_sma200(D, confirm=2, tc=0.0005, real_cash_yield=True):
    """Classic: hold TQQQ while QQQ > 200SMA (N-day confirm both ways), else cash."""
    tqqq, qqq = D["tqqq"], D["qqq"]
    dates = tqqq.index[tqqq.index >= pd.Timestamp(f"{START_YEAR}-01-01")]
    above = (qqq["Close"] > qqq["SMA_200"]).reindex(tqqq.index, method="ffill").fillna(False)
    cash, shares = STARTING_CAPITAL, 0.0
    equity = {}
    streak_above = streak_below = 0
    for i, date in enumerate(dates):
        price = float(tqqq.loc[date, "Close"])
        if cash > 0 and i > 0:
            cy = float(D["cash_yield"].loc[date]) if real_cash_yield else SGOV_DAILY_YIELD
            cash *= (1 + cy)
        if bool(above.loc[date]):
            streak_above += 1; streak_below = 0
        else:
            streak_below += 1; streak_above = 0
        if streak_above >= confirm and shares == 0 and cash > 0:
            shares = cash * (1 - tc) / price
            cash = 0.0
        elif streak_below >= confirm and shares > 0:
            cash = shares * price * (1 - tc)
            shares = 0.0
        equity[date] = cash + shares * price
    return metrics(equity)


def bench_buyhold(D, ticker="tqqq"):
    df = D[ticker]
    dates = df.index[df.index >= pd.Timestamp(f"{START_YEAR}-01-01")]
    p0 = float(df.loc[dates[0], "Close"])
    equity = {d: STARTING_CAPITAL * float(df.loc[d, "Close"]) / p0 for d in dates}
    return metrics(equity)


# ── QA checks ─────────────────────────────────────────────────────

def qa_checks(D):
    print("── QA: calendar alignment (lookahead check) ──")
    tq, qq, nd = D["tqqq"].index, D["qqq"].index, D["nasdaq"].index
    miss_q = tq.difference(qq)
    miss_n = tq.difference(nd)
    print(f"  TQQQ dates missing from QQQ:    {len(miss_q)}")
    print(f"  TQQQ dates missing from ^IXIC:  {len(miss_n)}")
    fwd = 0
    for d in list(miss_q) + list(miss_n):
        src = qq if d in miss_q else nd
        j = src.get_indexer([d], method="nearest")[0]
        if src[j] > d:
            fwd += 1
            print(f"    FUTURE-BAR RESOLUTION: {d.date()} -> {src[j].date()}")
    if fwd == 0:
        print("  method='nearest' never resolved to a future bar on this data. Latent risk only.")

    print("\n── QA: cash yield assumption ──")
    y = D["cash_yield"]
    print(f"  flat engine assumption: 4.50% annualized, every year 2011-2026")
    for yr in (2012, 2015, 2018, 2021, 2024, 2026):
        yy = y[f"{yr}-01-01":f"{yr}-12-31"]
        if len(yy):
            print(f"  actual ^IRX {yr}: {float(yy.mean()) * 252 * 100:5.2f}%")


if __name__ == "__main__":
    D = load_data()
    qa_checks(D)

    rows = []
    base = sim(D)
    base["name"] = "Q0 baseline replication (engine as-is)"
    rows.append(base)

    r = sim(D, indexer="ffill"); r["name"] = "Q1 ffill indexer (no lookahead risk)"; rows.append(r)
    r = sim(D, real_cash_yield=True); r["name"] = "Q2 realistic cash yield (^IRX)"; rows.append(r)
    r = sim(D, tc=0.0005); r["name"] = "Q3 + 5bps/side transaction cost"; rows.append(r)
    r = sim(D, indexer="ffill", real_cash_yield=True, tc=0.0005)
    r["name"] = "Q4 honest baseline (all three)"; rows.append(r)

    print("\n── Phase 1: QA impact on reported performance ──")
    print_table(rows)

    # Phase 2 — all variants on the HONEST base (ffill, real yield, 5bps)
    kw = dict(indexer="ffill", real_cash_yield=True, tc=0.0005)
    rows2 = [dict(rows[-1], name="honest V6 baseline")]

    r = sim(D, ema21_exit=True, **kw); r["name"] = "E1 wire in 21-EMA x2 exit (advertised)"; rows2.append(r)
    r = sim(D, vol_gate=0.28, **kw); r["name"] = "E2 vol gate: cap 50% when QQQ rv20>28%"; rows2.append(r)
    r = sim(D, vol_gate=0.35, **kw); r["name"] = "E3 vol gate at 35%"; rows2.append(r)
    r = sim(D, adaptive_cd=True, **kw); r["name"] = "E4 D1 penalty box (honest)"; rows2.append(r)
    r = sim(D, adaptive_cd=True, vol_gate=0.28, **kw); r["name"] = "E5 penalty box + vol gate 28%"; rows2.append(r)

    r = sim(D, tier_mode=True, vol_gate=0.28, **kw); r["name"] = "T1 composite tiers (0/50/75/100) band 15%"; rows2.append(r)
    r = sim(D, tier_mode=True, vol_gate=0.28, adaptive_cd=True, **kw)
    r["name"] = "T2 tiers + penalty box"; rows2.append(r)
    r = sim(D, tier_mode=True, vol_gate=0.35, adaptive_cd=True, **kw)
    r["name"] = "T3 tiers (vol 35%) + penalty box"; rows2.append(r)

    b = bench_sma200(D, confirm=2); b["name"] = "S1 simple: TQQQ if QQQ>200SMA (2d confirm)"; rows2.append(b)
    b = bench_buyhold(D, "tqqq"); b["name"] = "S2 TQQQ buy & hold"; rows2.append(b)
    b = bench_buyhold(D, "qqq"); b["name"] = "S3 QQQ buy & hold"; rows2.append(b)

    print("\n── Phase 2: new constructions (honest assumptions) ──")
    print_table(rows2)

    # detail: 2026 trades for the most interesting variants
    for nm in ("E5 penalty box + vol gate 28%", "T2 tiers + penalty box"):
        rr = [x for x in rows2 if x["name"] == nm]
        if rr and rr[0].get("trades"):
            print(f"\n  2026 trades — {nm}")
            for t in rr[0]["trades"]:
                if t["year"] == 2026:
                    print(f"    {t['entry']} -> {t['exit']}  {t['signal']:5s} {t['ret']:+6.1f}%")
