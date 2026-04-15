"""
TQQQ Swing Trading Dashboard -- Phase 1
Rules-based TQQQ buy/sell signal system.
"""

import datetime as dt
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

# Yahoo Finance–style equity chart ranges (daily backtest series)
EQUITY_PERIOD_OPTIONS = ["1D", "5D", "1M", "6M", "YTD", "1Y", "3Y", "5Y", "All"]


def _equity_cutoff_date(period: str, last_d: dt.date, first_d: dt.date) -> Optional[dt.date]:
    """Return first calendar date to show for this period, or None for full history."""
    if period == "All":
        return None
    today = dt.date.today()
    cy = today.year
    if period == "1D":
        return last_d - dt.timedelta(days=14)
    if period == "5D":
        return last_d - dt.timedelta(days=21)
    if period == "1M":
        return last_d - dt.timedelta(days=35)
    if period == "6M":
        return last_d - dt.timedelta(days=190)
    if period == "YTD":
        return max(dt.date(cy, 1, 1), first_d)
    if period == "1Y":
        return last_d - dt.timedelta(days=372)
    if period == "3Y":
        return last_d - dt.timedelta(days=3 * 372)
    if period == "5Y":
        return last_d - dt.timedelta(days=5 * 372)
    return first_d


def _filter_equity_series(bt_equity: dict, period: str) -> Tuple[List, List[float]]:
    """Sorted dates and values for the selected period (falls back if slice is empty)."""
    eq_dates = sorted(bt_equity.keys())
    eq_vals = [float(bt_equity[d]) for d in eq_dates]
    if not eq_dates:
        return [], []
    last_d = pd.Timestamp(eq_dates[-1]).date()
    first_d = pd.Timestamp(eq_dates[0]).date()
    cutoff = _equity_cutoff_date(period, last_d, first_d)
    if cutoff is None:
        return eq_dates, eq_vals
    eq_dates_f = [d for d in eq_dates if pd.Timestamp(d).date() >= cutoff]
    eq_vals_f = [float(bt_equity[d]) for d in eq_dates_f]
    if not eq_dates_f:
        n = min(60, len(eq_dates))
        eq_dates_f = eq_dates[-n:]
        eq_vals_f = [float(bt_equity[d]) for d in eq_dates_f]
    return eq_dates_f, eq_vals_f

from core.data import (
    get_tqqq_data, get_qqq_data, get_nasdaq_data, get_sp500_data,
    get_52_week_high, get_current_price, get_latest_date,
)
from core.indicators import detect_market_regime
from core.signals import (
    detect_follow_through_day, detect_three_white_knights,
    check_all_sell_signals, compute_alert_level,
)
from core.swing_tracker import detect_swings
from core.charts import build_tqqq_chart
from core.backtest import run_all_backtests, STARTING_CAPITAL, Trade, YearResult
import config
import json
import os


def _load_backtest_cached():
    """Load backtest from pre-committed JSON cache (instant, no recompute).
    Returns (results, equity_dict)."""
    cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "backtest_cache.json")

    try:
        with open(cache_path) as f:
            data = json.load(f)
        results = []
        equity_data = {}
        for r in data:
            trades = [Trade(**t) for t in r["trades"]]
            results.append(YearResult(
                year=r["year"], total_return_pct=r["total_return_pct"],
                num_trades=r["num_trades"], win_rate_pct=r["win_rate_pct"],
                avg_win_pct=r["avg_win_pct"], avg_loss_pct=r["avg_loss_pct"],
                max_win_pct=r["max_win_pct"], max_loss_pct=r["max_loss_pct"],
                best_trade=r["best_trade"], worst_trade=r["worst_trade"],
                tqqq_buy_hold_pct=r["tqqq_buy_hold_pct"],
                qqq_buy_hold_pct=r["qqq_buy_hold_pct"],
                trades=trades, starting_value=r["starting_value"],
                ending_value=r["ending_value"],
                max_drawdown_pct=r.get("max_drawdown_pct", 0.0),
            ))
            for d, v in r.get("equity", {}).items():
                equity_data[pd.Timestamp(d)] = v
        return results, equity_data
    except Exception:
        return [], {}


SEVERITY_ICONS = {"watch": "👀", "warning": "⚠️", "sell": "🔴"}
REGIME_ICONS = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


LOGO_URL = "https://pbs.twimg.com/profile_images/1959893019509071872/Xa6rYCoN_400x400.jpg"
TWITTER_URL = "https://x.com/__MrZzz__"


def _styled_card(title: str, content: str, border_color: str = "#38444D") -> str:
    return f"""<div style="border: 1px solid {border_color}; border-radius: 12px;
        padding: 20px; margin-bottom: 12px; background: rgba(29,161,242,0.03);">
        <div style="font-weight: 700; font-size: 1.05em; margin-bottom: 8px; color: #E7E9EA;">{title}</div>
        <div style="font-size: 0.9em; color: #8899A6; line-height: 1.7;">{content}</div>
    </div>"""


def render():
    st.markdown(f"""
        <div style="text-align: center; padding: 32px 0 20px 0; position: relative;">
            <div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%);
                width: 400px; height: 200px; background: radial-gradient(ellipse,
                rgba(99,102,241,0.12), transparent 70%); pointer-events: none;"></div>
            <a href="{TWITTER_URL}" target="_blank" style="text-decoration: none; position: relative;">
                <img src="{LOGO_URL}" alt="MrZzz"
                     style="width: 52px; height: 52px; border-radius: 50%;
                            border: 2px solid rgba(99,102,241,0.6); margin-bottom: 14px;
                            box-shadow: 0 0 40px rgba(99,102,241,0.2), 0 0 80px rgba(99,102,241,0.05);">
            </a>
            <h1 style="margin-bottom: 0; font-size: 1.9em; color: #f5f5f5;
                        letter-spacing: -0.03em; font-weight: 800; position: relative;">
                TQQQ Trading System
            </h1>
            <p style="color: #6b7280; font-size: 0.88em; margin-top: 8px; font-weight: 400;
                position: relative;">
                Rules-based swing trading &nbsp;&#183;&nbsp; 3x leveraged Nasdaq 100
                &nbsp;&#183;&nbsp;
                <a href="{TWITTER_URL}" target="_blank"
                   style="color: #a5b4fc; text-decoration: none; font-weight: 600;">
                    by @MrZzz</a>
            </p>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading market data..."):
        tqqq = get_tqqq_data()
        qqq = get_qqq_data()
        nasdaq = get_nasdaq_data()
        sp500 = get_sp500_data()

    if tqqq.empty or nasdaq.empty:
        st.error("Unable to fetch market data. Check your internet connection.")
        return

    data_date = get_latest_date(tqqq)
    tqqq_price = get_current_price(tqqq)
    qqq_price = get_current_price(qqq)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(f"""
            <div style="text-align: center; padding: 16px 0 20px 0;">
                <img src="{LOGO_URL}" style="width: 44px; height: 44px; border-radius: 50%;
                     border: 2px solid rgba(99,102,241,0.4);
                     box-shadow: 0 0 20px rgba(99,102,241,0.1);">
                <p style="color: #f0f0f0; font-weight: 700; margin: 10px 0 2px 0; font-size: 0.9em;
                    letter-spacing: -0.01em;">TQQQ System</p>
                <p style="color: #6b7280; font-size: 0.72em; margin: 0;">by @MrZzz</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("##### Settings")
        chart_lookback = st.slider("Chart lookback (days)", 30, 365, 120)
        bulls_pct = st.number_input(
            "Bulls % (AAII sentiment)",
            min_value=0.0, max_value=100.0, value=0.0, step=1.0,
            help="Enter the latest AAII bullish sentiment %. Leave 0 if unknown.",
        )
        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.caption("Data: Yahoo Finance (delayed)")
        st.caption("Not financial advice.")
    bulls_input = bulls_pct if bulls_pct > 0 else None

    # ── Load backtest from cache (instant) or recompute if stale ──
    bt_results, bt_equity = _load_backtest_cached()

    # ══════════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ══════════════════════════════════════════════════════════════
    tab_dash, tab_how, tab_guide, tab_perf = st.tabs([
        "📊 Dashboard",
        "📖 How the System Works",
        "🧭 How to Use This Site",
        "📈 Historical Performance",
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1: DASHBOARD
    # ══════════════════════════════════════════════════════════════
    with tab_dash:
        # ── Market Status + Action (VERY TOP) ──
        nasdaq_regime = detect_market_regime(nasdaq)
        sp_regime = detect_market_regime(sp500)
        tqqq_delta = (tqqq.iloc[-1]['Close'] - tqqq.iloc[-2]['Close']) / tqqq.iloc[-2]['Close'] * 100
        qqq_delta = (qqq.iloc[-1]['Close'] - qqq.iloc[-2]['Close']) / qqq.iloc[-2]['Close'] * 100
        REGIME_SHORT = {"Confirmed Uptrend": "Uptrend", "Uptrend Under Pressure": "Caution",
                        "Market in Correction": "Correction"}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TQQQ", f"${tqqq_price:.2f}", delta=f"{tqqq_delta:+.2f}%")
        c2.metric("QQQ", f"${qqq_price:.2f}", delta=f"{qqq_delta:+.2f}%")
        nq_short = REGIME_SHORT.get(nasdaq_regime.status, nasdaq_regime.status)
        sp_short = REGIME_SHORT.get(sp_regime.status, sp_regime.status)
        nq_icon = REGIME_ICONS.get(nasdaq_regime.color, '')
        sp_icon = REGIME_ICONS.get(sp_regime.color, '')
        c3.metric("Nasdaq", f"{nq_icon} {nq_short}")
        c4.metric("SPY", f"{sp_icon} {sp_short}")
        st.caption(f"Data as of {data_date.strftime('%b %d, %Y')}")

        # Live action status
        qqq_close_val = float(qqq.iloc[-1]["Close"])
        qqq_sma200_val = qqq.iloc[-1].get("SMA_200")
        qqq_above_200_now = (qqq_sma200_val is not None and not pd.isna(qqq_sma200_val)
                             and qqq_close_val > float(qqq_sma200_val))
        w_macd_val = qqq.iloc[-1].get("Weekly_MACD") if "Weekly_MACD" in qqq.columns else None
        macd_pos_now = w_macd_val is not None and not pd.isna(w_macd_val) and float(w_macd_val) > 0

        # Regime computation (needed for position sizing bar under BUY/SELL card)
        qqq_sma50_val = qqq.iloc[-1].get("SMA_50")
        above_200 = (qqq_sma200_val is not None and not pd.isna(qqq_sma200_val)
                     and qqq_close_val > float(qqq_sma200_val))
        golden = (qqq_sma50_val is not None and not pd.isna(qqq_sma50_val)
                  and qqq_sma200_val is not None and not pd.isna(qqq_sma200_val)
                  and float(qqq_sma50_val) > float(qqq_sma200_val))
        if golden and above_200:
            regime_str = "Strong Bull"
            regime_color = "#17BF63"
            exit_desc = "Holding through normal pullbacks. Exit on 2 closes below QQQ 50-day."
        elif above_200:
            regime_str = "Bull"
            regime_color = "#FFAD1F"
            exit_desc = "Cautious hold. Exit on 2 closes below QQQ 21-day EMA."
        else:
            regime_str = "Bear"
            regime_color = "#E0245E"
            exit_desc = "QQQ below 200-day. Stay in cash."
        alloc_label = "100%" if (golden and above_200) else ("50%" if above_200 else "0% (cash)")

        all_trades_flat = [t for r in bt_results for t in r.trades]
        lt = all_trades_flat[-1] if all_trades_flat else None
        lt_color = "#34d399" if (lt and lt.return_pct > 0) else "#f87171"
        today_str = dt.date.today().strftime("%B %d, %Y")

        # Determine if we're currently in a position or flat
        if lt:
            last_data_date = data_date.strftime("%Y-%m-%d")
            trade_is_open = lt.exit_date >= last_data_date
            pct_deployed = lt.cash_deployed / lt.portfolio_before * 100 if lt.portfolio_before > 0 else 0
            unrealized = ((tqqq_price - lt.entry_price) / lt.entry_price * 100) if lt.entry_price > 0 else 0
            unr_color = "#34d399" if unrealized >= 0 else "#f87171"

            if trade_is_open:
                days_in = (dt.date.today() - dt.datetime.strptime(lt.entry_date, "%Y-%m-%d").date()).days
                why_text = ('Weekly MACD crossed above zero — bullish trend confirmed.'
                            if lt.signal_type == 'MACD'
                            else ('Follow-Through Day — Nasdaq gained 1.25%+ on day 4+ of rally.'
                                  if lt.signal_type == 'FTD'
                                  else 'System defaults to invested in uptrend.'))
                st.markdown(f"""<div style="border: 2px solid #34d39944; border-radius: 16px;
                    padding: 20px 24px; background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(129,140,248,0.04));
                    margin: 8px 0 16px 0;">
                    <div style="display: grid; grid-template-columns: auto 1fr 1fr 1fr 1fr 1fr; gap: 10px; align-items: center;">
                        <div style="text-align: center; padding-right: 10px;">
                            <div style="font-size: 3em; font-weight: 900; color: #34d399;
                                letter-spacing: -0.02em; line-height: 1;">BUY</div>
                            <div style="font-size: 1.1em; font-weight: 700; color: #f0f0f0;
                                font-family: 'JetBrains Mono', monospace;">{lt.entry_date}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;">Position</div>
                            <div style="font-size: 1.8em; font-weight: 900; color: #818cf8;
                                font-family: 'JetBrains Mono', monospace;">{pct_deployed:.0f}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;">Entry</div>
                            <div style="font-size: 1.4em; font-weight: 700; color: #f0f0f0;
                                font-family: 'JetBrains Mono', monospace;">${lt.entry_price:.2f}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;">Now</div>
                            <div style="font-size: 1.4em; font-weight: 700; color: #f0f0f0;
                                font-family: 'JetBrains Mono', monospace;">${tqqq_price:.2f}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;">P&L</div>
                            <div style="font-size: 1.8em; font-weight: 900; color: {unr_color};
                                font-family: 'JetBrains Mono', monospace;">{unrealized:+.1f}%</div>
                        </div>
                        <div style="border-left: 2px solid rgba(52,211,153,0.3);
                            padding-left: 16px;">
                            <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;
                                letter-spacing: 0.08em;">Why</div>
                            <div style="font-size: 1.0em; color: #f0f0f0; margin-top: 4px; line-height: 1.6; font-weight: 600;">
                                {why_text} QQQ above 200-day SMA.</div>
                            <div style="font-size: 0.88em; color: #d1d5db; margin-top: 8px; line-height: 1.5;">
                                <b style="color: {regime_color};">{regime_str}:</b> {exit_desc} Allocation: <b>{alloc_label}</b></div>
                            <div style="font-size: 0.78em; color: #6b7280; margin-top: 6px;">
                                {days_in} days · {lt.shares:,.0f} shares</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                act_color = "#f87171" if not qqq_above_200_now else "#fbbf24"
                trade_pnl = lt.portfolio_after - lt.portfolio_before
                sell_why = ('QQQ closed below 200-day SMA for 2 consecutive days — bear market confirmed.'
                            if not qqq_above_200_now
                            else ('12% trailing stop triggered — portfolio dropped from peak.'
                                  if lt.return_pct < -5
                                  else 'QQQ broke below 200-day SMA — exited to protect capital.'))
                sell_next = ('Watching for re-entry signal' if qqq_above_200_now
                             else 'Staying in cash until QQQ reclaims 200-day')
                st.markdown(f"""<div style="border: 2px solid {act_color}44; border-radius: 16px;
                    padding: 20px 24px; background: linear-gradient(135deg, {act_color}08, rgba(255,255,255,0.02));
                    margin: 8px 0 16px 0;">
                    <div style="display: grid; grid-template-columns: auto 1fr 1fr 1fr 1fr 1fr; gap: 10px; align-items: center;">
                        <div style="text-align: center; padding-right: 10px; min-width: 120px;">
                            <div style="font-size: 2.8em; font-weight: 900; color: {act_color};
                                letter-spacing: -0.02em; line-height: 1;">SELL</div>
                            <div style="font-size: 1.1em; font-weight: 700; color: #f0f0f0;
                                font-family: 'JetBrains Mono', monospace;">{lt.exit_date}</div>
                            <div style="font-size: 0.72em; color: #6b7280; text-transform: uppercase; margin-top: 10px;">Sell price</div>
                            <div style="font-size: 1.25em; font-weight: 700; color: #f0f0f0;
                                font-family: 'JetBrains Mono', monospace;">${lt.exit_price:.2f}</div>
                            <div style="font-size: 0.72em; color: #6b7280; text-transform: uppercase; margin-top: 8px;">Sell time</div>
                            <div style="font-size: 0.82em; color: #9ca3af; line-height: 1.35;">4:00 PM ET<br><span style="color:#6b7280;font-size:0.9em;">Regular close (modeled)</span></div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;">Position</div>
                            <div style="font-size: 1.5em; font-weight: 900; color: #9ca3af;
                                font-family: 'JetBrains Mono', monospace;">0%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;">Result</div>
                            <div style="font-size: 1.4em; font-weight: 900; color: {lt_color};
                                font-family: 'JetBrains Mono', monospace;">{lt.return_pct:+.1f}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;">P&L</div>
                            <div style="font-size: 1.4em; font-weight: 700; color: {lt_color};
                                font-family: 'JetBrains Mono', monospace;">${trade_pnl:+,.0f}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;">Held</div>
                            <div style="font-size: 1.4em; font-weight: 700; color: #f0f0f0;
                                font-family: 'JetBrains Mono', monospace;">{lt.duration_days}d</div>
                        </div>
                        <div style="border-left: 2px solid {act_color}44;
                            padding-left: 16px;">
                            <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;
                                letter-spacing: 0.08em;">Why</div>
                            <div style="font-size: 1.0em; color: #f0f0f0; margin-top: 4px; line-height: 1.6; font-weight: 600;">
                                {sell_why}</div>
                            <div style="font-size: 0.88em; color: #d1d5db; margin-top: 8px; line-height: 1.5;">
                                <b style="color: {regime_color};">{regime_str}:</b> {exit_desc} Allocation: <b>{alloc_label}</b></div>
                            <div style="font-size: 0.78em; color: #6b7280; margin-top: 6px;">
                                {sell_next}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.caption(
                    "Execution model: entries and exits use the **official daily closing price** on that date "
                    "(after-hours fills), not the next session’s open. The dashboard shows SELL when the last trade "
                    "in the backtest is **closed** — often right after the data updates for a new day, not because "
                    "we sold at the open."
                )

        # ── Hero: Lifetime Performance ──
        current_year = dt.date.today().year
        if bt_results:
            lifetime_start = bt_results[0].starting_value
            lifetime_end = bt_results[-1].ending_value
            lifetime_mult = lifetime_end / lifetime_start
            start_year_bt = bt_results[0].year
            ytd_result = next((r for r in bt_results if r.year == current_year), None)
            prior_year_result = next((r for r in bt_results if r.year == current_year - 1), None)

            ytd_pct = ytd_result.total_return_pct if ytd_result else 0
            ytd_color = "#17BF63" if ytd_pct >= 0 else "#E0245E"
            py_pct = prior_year_result.total_return_pct if prior_year_result else 0
            py_color = "#17BF63" if py_pct >= 0 else "#E0245E"

            py_color = "#34d399" if py_pct >= 0 else "#f87171"
            ytd_color = "#34d399" if ytd_pct >= 0 else "#f87171"

            st.markdown(f"""<div style="border: 1px solid rgba(255,255,255,0.06); border-radius: 20px;
                padding: 28px 24px; background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(139,92,246,0.03));
                margin-bottom: 20px; position: relative; overflow: hidden;">
                <div style="position: absolute; top: -40px; right: -40px; width: 200px; height: 200px;
                    background: radial-gradient(circle, rgba(99,102,241,0.08), transparent 70%);
                    border-radius: 50%;"></div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; align-items: center;
                    position: relative;">
                    <div style="text-align: center;">
                        <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;
                            letter-spacing: 0.12em; font-weight: 500;">Return {start_year_bt} – YTD {current_year}</div>
                        <div style="font-size: 2.8em; font-weight: 900; color: #34d399;
                            letter-spacing: -0.04em; line-height: 1.1;
                            font-family: 'JetBrains Mono', monospace;">${lifetime_end:,.0f}</div>
                        <div style="font-size: 0.82em; color: #9ca3af; margin-top: 6px;">
                            <span style="color: #818cf8; font-weight: 700;">{lifetime_mult:,.0f}x</span> from $100K</div>
                    </div>
                    <div style="text-align: center; border-left: 1px solid rgba(255,255,255,0.06);
                        border-right: 1px solid rgba(255,255,255,0.06); padding: 0 16px;">
                        <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;
                            letter-spacing: 0.12em; font-weight: 500;">{current_year - 1} Return</div>
                        <div style="font-size: 2.2em; font-weight: 800; color: {py_color};
                            line-height: 1.2; font-family: 'JetBrains Mono', monospace;">{py_pct:+.1f}%</div>
                        <div style="font-size: 0.75em; color: #6b7280; margin-top: 6px;">
                            vs B&H {prior_year_result.tqqq_buy_hold_pct:+.1f}%</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 0.85em; color: #6b7280; text-transform: uppercase;
                            letter-spacing: 0.12em; font-weight: 500;">{current_year} YTD</div>
                        <div style="font-size: 2.2em; font-weight: 800; color: {ytd_color};
                            line-height: 1.2; font-family: 'JetBrains Mono', monospace;">{ytd_pct:+.1f}%</div>
                        <div style="font-size: 0.75em; color: #6b7280; margin-top: 6px;">
                            vs B&H {ytd_result.tqqq_buy_hold_pct:+.1f}%</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Equity Chart + Key Stats on main page ──
        if bt_equity and bt_results:
            import plotly.graph_objects as go

            total_trades = sum(r.num_trades for r in bt_results)
            overall_wr = sum(r.win_rate_pct * r.num_trades for r in bt_results if r.num_trades > 0) / max(total_trades, 1)
            total_max_dd = 0.0
            max_dd_date = None
            pk = list(bt_equity.values())[0]
            for d_eq, v in sorted(bt_equity.items()):
                if v > pk: pk = v
                dd = ((v - pk) / pk) * 100
                if dd < total_max_dd:
                    total_max_dd = dd
                    max_dd_date = d_eq
            max_dd_label = max_dd_date.strftime("%b'%y") if max_dd_date else ""

            # Stats row ABOVE the chart
            st.markdown(f"""<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
                margin-bottom: 12px;">
                <div style="text-align: center; padding: 12px; border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 12px; background: rgba(255,255,255,0.02);">
                    <div style="font-size: 0.68em; color: #6b7280; text-transform: uppercase;
                        letter-spacing: 0.08em;">Starting Capital</div>
                    <div style="font-size: 1.4em; font-weight: 800; color: #f0f0f0;
                        font-family: 'JetBrains Mono', monospace;">$100,000</div>
                </div>
                <div style="text-align: center; padding: 12px; border: 1px solid rgba(52,211,153,0.15);
                    border-radius: 12px; background: rgba(52,211,153,0.04);">
                    <div style="font-size: 0.68em; color: #6b7280; text-transform: uppercase;
                        letter-spacing: 0.08em;">Current Value</div>
                    <div style="font-size: 1.4em; font-weight: 800; color: #34d399;
                        font-family: 'JetBrains Mono', monospace;">${bt_results[-1].ending_value:,.0f}</div>
                </div>
                <div style="text-align: center; padding: 12px; border: 1px solid rgba(248,113,113,0.15);
                    border-radius: 12px; background: rgba(248,113,113,0.04);">
                    <div style="font-size: 0.68em; color: #6b7280; text-transform: uppercase;
                        letter-spacing: 0.08em;">Max Drawdown</div>
                    <div style="font-size: 1.4em; font-weight: 800; color: #f87171;
                        font-family: 'JetBrains Mono', monospace;">{total_max_dd:.1f}%</div>
                    <div style="font-size: 0.72em; color: #6b7280; margin-top: 2px;">{max_dd_label}</div>
                </div>
                <div style="text-align: center; padding: 12px; border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 12px; background: rgba(255,255,255,0.02);">
                    <div style="font-size: 0.68em; color: #6b7280; text-transform: uppercase;
                        letter-spacing: 0.08em;">{total_trades} Trades</div>
                    <div style="font-size: 1.4em; font-weight: 800; color: #f0f0f0;
                        font-family: 'JetBrains Mono', monospace;">{overall_wr:.0f}% WR</div>
                </div>
            </div>""", unsafe_allow_html=True)

            eq_period = st.segmented_control(
                "Period",
                options=EQUITY_PERIOD_OPTIONS,
                default="All",
                key="eq_range_main",
            ) or "All"
            eq_dates_f, eq_vals_f = _filter_equity_series(bt_equity, eq_period)

            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                x=eq_dates_f, y=eq_vals_f,
                mode="lines",
                line=dict(color="#818cf8", width=2.5),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>$%{y:,.0f}<extra></extra>",
            ))
            y_min = min(eq_vals_f) * 0.95 if eq_vals_f else 0
            y_max = max(eq_vals_f) * 1.05 if eq_vals_f else 100000
            if eq_vals_f and (y_max - y_min) < max(abs(y_max) * 1e-9, 1.0):
                pad = max(abs(y_max) * 0.02, 1000.0)
                y_min, y_max = y_min - pad, y_max + pad
            eq_fig.update_layout(
                template="plotly_dark",
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,15,26,1)",
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                           tickprefix="$", tickformat=",",
                           showgrid=True, zeroline=False,
                           range=[y_min, y_max], fixedrange=True),
                xaxis=dict(gridcolor="rgba(255,255,255,0.03)",
                           showgrid=False, fixedrange=True),
                showlegend=False,
            )
            st.caption("Use Period above to change the time window — chart Y-axis matches that window.")
            st.plotly_chart(
                eq_fig,
                use_container_width=True,
                config={
                    "scrollZoom": False,
                    "displayModeBar": False,
                },
            )

        # (Last Trade + Allocation moved to top)

        # ── Market Health Panel ──
        st.markdown("### Market Health")

        def _ma_status(df, col, label):
            val = df.iloc[-1].get(col)
            if val is None or pd.isna(val):
                return label, "N/A", "N/A", "#657786"
            v = float(val)
            price = float(df.iloc[-1]["Close"])
            dist = ((price - v) / v) * 100
            above = price > v
            return label, f"${v:.2f}", f"{dist:+.1f}%", "#17BF63" if above else "#E0245E"

        def _cross_status(df):
            sma50 = df.iloc[-1].get("SMA_50")
            sma200 = df.iloc[-1].get("SMA_200")
            if sma50 is None or pd.isna(sma50) or sma200 is None or pd.isna(sma200):
                return "Unknown", "#657786"
            if float(sma50) > float(sma200):
                return "Golden Cross", "#17BF63"
            return "Death Cross", "#E0245E"

        def _drawdown_from_peak(df, lookback=252):
            lb = min(lookback, len(df))
            peak = float(df["High"].iloc[-lb:].max())
            current = float(df["Close"].iloc[-1])
            dd = ((current - peak) / peak) * 100
            return dd, peak

        def _health_card(ticker, df):
            price = float(df.iloc[-1]["Close"])
            cross_label, cross_color = _cross_status(df)
            dd_pct, peak = _drawdown_from_peak(df)

            _, sma50_val, sma50_dist, sma50_color = _ma_status(df, "SMA_50", "50d")
            _, sma200_val, sma200_dist, sma200_color = _ma_status(df, "SMA_200", "200d")
            _, ema21_val, ema21_dist, ema21_color = _ma_status(df, "EMA_21", "21d")

            above_200 = float(df.iloc[-1].get("SMA_200", 0) or 0)
            regime_color = "#17BF63" if price > above_200 and above_200 > 0 else "#E0245E"
            regime_label = "BULL" if price > above_200 and above_200 > 0 else "BEAR"

            dd_color = "#34d399" if dd_pct >= 0 else "#f87171"

            st.markdown(f"""<div style="border: 1px solid rgba(29,161,242,0.15); border-radius: 12px;
                padding: 18px; background: rgba(29,161,242,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <div>
                        <span style="font-size: 1.2em; font-weight: 800; color: #E7E9EA;">{ticker}</span>
                        <span style="font-size: 1.2em; font-weight: 700; color: #E7E9EA; margin-left: 10px;">${price:.2f}</span>
                    </div>
                    <div>
                        <span style="background: {regime_color}; color: white; padding: 3px 12px;
                            border-radius: 20px; font-size: 0.78em; font-weight: 700;">{regime_label}</span>
                        <span style="background: {cross_color}22; color: {cross_color}; padding: 3px 12px;
                            border-radius: 20px; font-size: 0.78em; font-weight: 600; margin-left: 6px;
                            border: 1px solid {cross_color}44;">{cross_label}</span>
                    </div>
                </div>
                <div style="display: flex; justify-content: center; gap: 20px; font-size: 1.0em; margin-top: 8px;">
                    <div style="text-align: center;">
                        <div style="color: #6b7280; font-size: 0.72em; margin-bottom: 2px;">21-EMA</div>
                        <div style="color: {ema21_color}; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{ema21_dist}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #6b7280; font-size: 0.72em; margin-bottom: 2px;">50-day</div>
                        <div style="color: {sma50_color}; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{sma50_dist}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #6b7280; font-size: 0.72em; margin-bottom: 2px;">200-day</div>
                        <div style="color: {sma200_color}; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{sma200_dist}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #6b7280; font-size: 0.72em; margin-bottom: 2px;">From Peak</div>
                        <div style="color: {dd_color}; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{dd_pct:+.1f}%</div>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 0.78em; color: #8899A6;">
                    52w high: ${peak:.2f}
                </div>
            </div>""", unsafe_allow_html=True)

        # Weekly MACD detail for health grid
        w_macd = qqq.iloc[-1].get("Weekly_MACD")
        w_macd_sig = qqq.iloc[-1].get("Weekly_MACD_Signal")
        macd_val = float(w_macd) if w_macd is not None and not pd.isna(w_macd) else None
        macd_sig_val = float(w_macd_sig) if w_macd_sig is not None and not pd.isna(w_macd_sig) else None

        exit_mode = ("Wide exit (QQQ 50-day SMA)" if (golden and above_200)
                     else ("Tight exit (QQQ 21-EMA)" if above_200 else "No positions"))

        macd_color = "#17BF63" if (macd_val and macd_val > 0) else "#E0245E"
        macd_label = "Bullish" if (macd_val and macd_val > 0) else "Bearish"
        macd_trend = "Rising" if (macd_val and macd_sig_val and macd_val > macd_sig_val) else "Falling"

        st.markdown(f"""<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 16px;">
            <div style="border: 1px solid {regime_color}33; border-radius: 12px; padding: 14px;
                background: {regime_color}08; text-align: center;">
                <div style="font-size: 0.75em; color: #8899A6; text-transform: uppercase;">Market Regime</div>
                <div style="font-size: 1.4em; font-weight: 800; color: {regime_color};">{regime_str}</div>
                <div style="font-size: 0.75em; color: #8899A6; margin-top: 4px;">{exit_mode}</div>
            </div>
            <div style="border: 1px solid {macd_color}33; border-radius: 12px; padding: 14px;
                background: {macd_color}08; text-align: center;">
                <div style="font-size: 0.75em; color: #8899A6; text-transform: uppercase;">Weekly MACD</div>
                <div style="font-size: 1.4em; font-weight: 800; color: {macd_color};">{macd_label}</div>
                <div style="font-size: 0.75em; color: #8899A6; margin-top: 4px;">{macd_trend} · {macd_val:+.1f}</div>
            </div>
            <div style="border: 1px solid rgba(29,161,242,0.2); border-radius: 12px; padding: 14px;
                background: rgba(29,161,242,0.03); text-align: center;">
                <div style="font-size: 0.82em; color: #8899A6; text-transform: uppercase;">Exit Strategy</div>
                <div style="font-size: 1.1em; font-weight: 600; color: #E7E9EA; margin-top: 6px;">{exit_desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        hc1, hc2 = st.columns(2)
        with hc1:
            _health_card("QQQ", qqq)
        with hc2:
            _health_card("SPY", sp500)

        # ── System Signals (what actually drives decisions) ──
        st.markdown("### System Signals")

        ftd = detect_follow_through_day(nasdaq)

        # Entry signals
        s1, s2, s3 = st.columns(3)
        with s1:
            ftd_active = ftd is not None
            ftd_bg = "#34d39922" if ftd_active else "rgba(255,255,255,0.02)"
            ftd_border = "#34d39944" if ftd_active else "rgba(255,255,255,0.06)"
            ftd_badge = "DETECTED" if ftd_active else "CLEAR"
            ftd_badge_color = "#34d399" if ftd_active else "#6b7280"
            st.markdown(f"""<div style="border: 1px solid {ftd_border}; background: {ftd_bg};
                border-radius: 14px; padding: 16px; min-height: 120px;">
                <div style="font-weight: 700; font-size: 0.88em; color: #f0f0f0;">
                    Follow-Through Day</div>
                <div style="font-size: 0.75em; color: #9ca3af; margin: 6px 0;">
                    {'Nasdaq gained ' + f'{ftd.details}' if ftd_active else 'No FTD in last 30 sessions. Requires 7%+ Nasdaq correction + 1.25% rally on day 4+.'}</div>
                <span style="background: {ftd_badge_color}22; color: {ftd_badge_color}; padding: 3px 12px;
                    border-radius: 20px; font-size: 0.72em; font-weight: 600;
                    border: 1px solid {ftd_badge_color}44;">{ftd_badge}</span>
                <span style="color: #6b7280; font-size: 0.68em; margin-left: 6px;">ENTRY SIGNAL</span>
            </div>""", unsafe_allow_html=True)

        with s2:
            macd_bg = "#34d39922" if macd_pos_now else "#f8717122"
            macd_border = "#34d39944" if macd_pos_now else "#f8717144"
            macd_badge = "POSITIVE" if macd_pos_now else "NEGATIVE"
            macd_badge_c = "#34d399" if macd_pos_now else "#f87171"
            macd_desc = "Weekly MACD above zero — bullish trend confirmed. System at 100%." if macd_pos_now else "Weekly MACD below zero — trend weakening. System at 50% or cash."
            st.markdown(f"""<div style="border: 1px solid {macd_border}; background: {macd_bg};
                border-radius: 14px; padding: 16px; min-height: 120px;">
                <div style="font-weight: 700; font-size: 0.88em; color: #f0f0f0;">
                    Weekly MACD</div>
                <div style="font-size: 0.75em; color: #9ca3af; margin: 6px 0;">{macd_desc}</div>
                <span style="background: {macd_badge_c}22; color: {macd_badge_c}; padding: 3px 12px;
                    border-radius: 20px; font-size: 0.72em; font-weight: 600;
                    border: 1px solid {macd_badge_c}44;">{macd_badge}</span>
                <span style="color: #6b7280; font-size: 0.68em; margin-left: 6px;">TREND</span>
            </div>""", unsafe_allow_html=True)

        with s3:
            above_c = "#34d399" if above_200 else "#f87171"
            above_bg = "#34d39922" if above_200 else "#f8717122"
            above_border = "#34d39944" if above_200 else "#f8717144"
            above_badge = "ABOVE" if above_200 else "BELOW"
            sma200_val = float(qqq_sma200_val) if qqq_sma200_val is not None and not pd.isna(qqq_sma200_val) else 0
            pct_from_200 = ((qqq_close_val - sma200_val) / sma200_val * 100) if sma200_val > 0 else 0
            above_desc = f"QQQ at ${qqq_close_val:.2f}, {pct_from_200:+.1f}% from 200-day (${sma200_val:.2f}). {'Bull market — entries allowed.' if above_200 else 'Bear market — cash only (except FTD).'}"
            st.markdown(f"""<div style="border: 1px solid {above_border}; background: {above_bg};
                border-radius: 14px; padding: 16px; min-height: 120px;">
                <div style="font-weight: 700; font-size: 0.88em; color: #f0f0f0;">
                    QQQ vs 200-Day SMA</div>
                <div style="font-size: 0.75em; color: #9ca3af; margin: 6px 0;">{above_desc}</div>
                <span style="background: {above_c}22; color: {above_c}; padding: 3px 12px;
                    border-radius: 20px; font-size: 0.72em; font-weight: 600;
                    border: 1px solid {above_c}44;">{above_badge}</span>
                <span style="color: #6b7280; font-size: 0.68em; margin-left: 6px;">REGIME</span>
            </div>""", unsafe_allow_html=True)

        # Exit signals
        st.markdown("")
        e1, e2, e3 = st.columns(3)

        with e1:
            # Check consecutive closes below 200-day
            below_count = 0
            if len(qqq) >= 2:
                for j in range(1, min(3, len(qqq))):
                    row = qqq.iloc[-j]
                    s = row.get("SMA_200")
                    if s is not None and not pd.isna(s) and float(row["Close"]) < float(s):
                        below_count += 1
                    else:
                        break
            exit_200_triggered = below_count >= 2
            exit_200_c = "#f87171" if exit_200_triggered else ("#fbbf24" if below_count == 1 else "#34d399")
            exit_200_badge = "EXIT" if exit_200_triggered else (f"{below_count}/2 CLOSES" if below_count > 0 else "CLEAR")
            st.markdown(f"""<div style="border: 1px solid {exit_200_c}44; background: {exit_200_c}11;
                border-radius: 14px; padding: 16px; min-height: 120px;">
                <div style="font-weight: 700; font-size: 0.88em; color: #f0f0f0;">
                    200-Day SMA Exit</div>
                <div style="font-size: 0.75em; color: #9ca3af; margin: 6px 0;">
                    {'EXIT TRIGGERED — QQQ closed below 200-day for 2 consecutive days.' if exit_200_triggered else
                     (f'Warning: {below_count} close below 200-day. One more triggers exit.' if below_count == 1 else
                      'QQQ holding above 200-day SMA. No exit signal.')}</div>
                <span style="background: {exit_200_c}22; color: {exit_200_c}; padding: 3px 12px;
                    border-radius: 20px; font-size: 0.72em; font-weight: 600;
                    border: 1px solid {exit_200_c}44;">{exit_200_badge}</span>
                <span style="color: #6b7280; font-size: 0.68em; margin-left: 6px;">EXIT SIGNAL</span>
            </div>""", unsafe_allow_html=True)

        with e2:
            # Trailing stop status
            stop_active = pct_from_200 >= 3.0 if above_200 else False
            stop_c = "#34d399" if not stop_active else "#818cf8"
            st.markdown(f"""<div style="border: 1px solid {stop_c}44; background: {stop_c}11;
                border-radius: 14px; padding: 16px; min-height: 120px;">
                <div style="font-weight: 700; font-size: 0.88em; color: #f0f0f0;">
                    12% Trailing Stop</div>
                <div style="font-size: 0.75em; color: #9ca3af; margin: 6px 0;">
                    {'ACTIVE — QQQ is ' + f'{pct_from_200:.1f}% above 200-day (>3% threshold). Exit fires if portfolio drops 12% from peak.' if stop_active else
                     'INACTIVE — QQQ is within 3% of 200-day. Trailing stop disabled to avoid chop.'}</div>
                <span style="background: {stop_c}22; color: {stop_c}; padding: 3px 12px;
                    border-radius: 20px; font-size: 0.72em; font-weight: 600;
                    border: 1px solid {stop_c}44;">{'ACTIVE' if stop_active else 'INACTIVE'}</span>
                <span style="color: #6b7280; font-size: 0.68em; margin-left: 6px;">TRAILING STOP</span>
            </div>""", unsafe_allow_html=True)

        with e3:
            # Crash detector
            tqqq_10d_high = float(tqqq.iloc[-10:]["High"].max()) if len(tqqq) >= 10 else tqqq_price
            tqqq_drop_10d = ((tqqq_price - tqqq_10d_high) / tqqq_10d_high) * 100
            crash_detected = tqqq_drop_10d <= -30
            crash_c = "#f87171" if crash_detected else "#34d399"
            st.markdown(f"""<div style="border: 1px solid {crash_c}44; background: {crash_c}11;
                border-radius: 14px; padding: 16px; min-height: 120px;">
                <div style="font-weight: 700; font-size: 0.88em; color: #f0f0f0;">
                    Crash Detector</div>
                <div style="font-size: 0.75em; color: #9ca3af; margin: 6px 0;">
                    {'CRASH DETECTED — TQQQ dropped ' + f'{tqqq_drop_10d:.0f}% in 10 days. All entries blocked for 40 days.' if crash_detected else
                     f'TQQQ 10-day drawdown: {tqqq_drop_10d:+.1f}% (threshold: -30%). No crash detected.'}</div>
                <span style="background: {crash_c}22; color: {crash_c}; padding: 3px 12px;
                    border-radius: 20px; font-size: 0.72em; font-weight: 600;
                    border: 1px solid {crash_c}44;">{'CRASH' if crash_detected else 'CLEAR'}</span>
                <span style="color: #6b7280; font-size: 0.68em; margin-left: 6px;">SAFETY</span>
            </div>""", unsafe_allow_html=True)

        # Chart
        st.markdown("### TQQQ Price Chart")
        year_now = dt.datetime.now().year
        swings = detect_swings(tqqq, min_move_pct=5.0, year_filter=year_now - 1)
        fig = build_tqqq_chart(tqqq, swings=swings, lookback_days=chart_lookback)
        st.plotly_chart(fig, key="main_chart", use_container_width=True,
                        config={"scrollZoom": True})

        # MA Table
        latest = tqqq.iloc[-1]
        ma_rows = []
        for col, label in [("SMA_10", "10-day SMA"), ("EMA_21", "21-day EMA"), ("SMA_50", "50-day SMA"), ("SMA_200", "200-day SMA")]:
            if col in tqqq.columns and not pd.isna(latest[col]):
                v = float(latest[col])
                d = ((tqqq_price - v) / v) * 100
                pos = "✅ Above" if tqqq_price > v else "❌ Below"
                ma_rows.append({"MA": label, "Value": f"${v:.2f}", "Position": pos, "Distance": f"{d:+.1f}%"})
        if ma_rows:
            st.markdown("##### Moving Average Positioning")
            st.dataframe(pd.DataFrame(ma_rows), use_container_width=True, hide_index=True)



    # ══════════════════════════════════════════════════════════════
    # TAB 2: HOW THE SYSTEM WORKS
    # ══════════════════════════════════════════════════════════════
    with tab_how:
        st.markdown("## How the Trading System Works")
        st.markdown("""
This system is **100% invested in TQQQ by default** during bull markets and exits
to cash when conditions deteriorate. The philosophy: **ride all the ups, dodge the
crashes.** Inspired by Vibha Jha's TQQQ swing trading approach, enhanced with
quantitative signals.
""")

        st.markdown("### The Core Idea")
        h1, h2 = st.columns(2)
        with h1:
            st.markdown(_styled_card(
                "TQQQ = 3x Daily Nasdaq 100",
                """TQQQ delivers <b>3x the daily return</b> of QQQ.
                A 10% QQQ rally = ~30% TQQQ gain. But a 10% QQQ drop = ~30% TQQQ loss.
                The system aims to be fully invested during rallies and in cash during drops.
                Backtested: <b>$100K → $8.6M</b> over 16 years (2011-2026)."""
            ), unsafe_allow_html=True)
        with h2:
            st.markdown(_styled_card(
                "Why Not Just Buy & Hold?",
                """TQQQ buy-and-hold lost <b>-79.7% in 2022</b>. Our system lost only -18%.
                In a 3x leveraged ETF, avoiding the big crashes matters more than catching
                every rally. Best executed in <b>IRA/Roth accounts</b> for tax-free compounding.
                Idle cash earns ~4.5% in SGOV while waiting."""
            ), unsafe_allow_html=True)

        st.markdown("### Entry Rules")
        st.markdown("""
The system uses **three entry signals**, all evaluated on QQQ/Nasdaq (not TQQQ):

| # | Signal | When It Fires | Allocation |
|---|--------|--------------|------------|
| 1 | **Default Entry** | QQQ is above 200-day SMA + Weekly MACD positive | **100%** |
| 2 | **Follow-Through Day (FTD)** | After a 7%+ Nasdaq correction, the Nasdaq gains 1.25%+ on higher volume on day 4+ of a rally attempt | **50-100%** (50% if below 200-day) |
| 3 | **MACD Crossover** | Weekly MACD crosses above zero on QQQ after being negative | **100%** |

**Key rules:**
- FTD can enter **even below the 200-day SMA** — it's the one signal strong enough to catch market bottoms early
- When QQQ is above 200-day but MACD is negative, the system holds at **50%** (cautious)
- After an exit, re-entry requires either an FTD or MACD turning positive (no blind re-entry)
""")

        st.markdown("### Exit Rules")
        st.markdown("""
The system has **three exit triggers**, designed to catch crashes early:

| # | Exit Signal | Condition | Speed |
|---|------------|-----------|-------|
| 1 | **200-day SMA Break** | QQQ closes below its 200-day SMA for **2 consecutive days** | Primary exit — catches bear markets |
| 2 | **12% Trailing Stop** | Portfolio drops 12% from its peak (only active when QQQ is >3% above 200-day) | Catches slow rollovers from bull market highs |
| 3 | **Crash Detector** | TQQQ drops 30%+ in 10 trading days → blocks ALL entries for 40 days | Prevents re-entering during freefall (COVID, flash crashes) |

**How exits work in practice:**
- Evaluate signals at **market close** (4:00 PM ET)
- Execute the trade in **after-hours** (4:00-8:00 PM ET) or at **next day's open**
- After any exit, the system enters a **10-day cooldown** before considering re-entry
- FTD can override the cooldown (it's a strong enough signal to re-enter early)
""")

        st.markdown("### Position Sizing")
        st.markdown("""
| Market Condition | QQQ Status | Weekly MACD | Allocation |
|-----------------|------------|------------|------------|
| **Strong Bull** | Above 200-day SMA | Positive | **100%** |
| **Cautious Bull** | Above 200-day SMA | Negative | **50%** |
| **FTD Below 200-day** | Below 200-day SMA | Any | **50%** (probe entry) |
| **Bear Market** | Below 200-day SMA | No FTD | **0% (cash)** |

Cash earns ~4.5% annualized in SGOV (short-term Treasuries) while waiting.
""")

        st.markdown("### Key Concepts")

        with st.expander("What is a Follow-Through Day (FTD)?"):
            st.markdown("""The primary signal that a new market uptrend is beginning after a correction.

**Requirements:**
1. The Nasdaq must have declined **7%+ from a recent high** (establishes a correction)
2. A **rally attempt** begins (the index starts making higher closes from the low)
3. On **day 4 or later** of the rally, the Nasdaq gains **at least 1.25%**
4. Volume must be **higher than the previous session** (institutional buying)

Not all FTDs succeed, but almost every major market rally has started with one.
The system enters TQQQ on an FTD — even below the 200-day SMA — at 50% allocation
as a probe. If the rally proves real, MACD will go positive and the system scales to 100%.""")

        with st.expander("What is the Weekly MACD?"):
            st.markdown("""The **Moving Average Convergence Divergence (MACD)** on QQQ's weekly chart
is the system's primary trend indicator.

- **MACD = 12-week EMA minus 26-week EMA** of QQQ's closing price
- When MACD is **above zero**: the intermediate-term trend is bullish → 100% invested
- When MACD is **below zero**: the trend is bearish → reduce to 50% or exit

The weekly timeframe filters out daily noise. The MACD only crosses zero
**2-4 times per year**, capturing major trend changes while ignoring minor dips.
This is why the system holds through normal 5-8% pullbacks without exiting.""")

        with st.expander("What is the 200-day SMA?"):
            st.markdown("""The **200-day Simple Moving Average** of QQQ is the dividing line between
bull and bear markets.

- **QQQ above 200-day**: Bull market — the system is invested
- **QQQ below 200-day**: Bear market — the system is in cash

Two consecutive closes below the 200-day confirms the transition. This is
the system's **hard exit** — it doesn't wait for MACD or other signals.
In 2022, this got the system out in January, avoiding the -79.7% TQQQ crash.""")

        with st.expander("What is the 12% Trailing Stop?"):
            st.markdown("""The trailing stop protects against **slow rollovers from bull market highs**
— like the Nov 2021 → Jan 2022 decline where QQQ stayed above its 200-day
while slowly grinding down.

**How it works:**
- Tracks the **portfolio's peak value** while in a position
- If the portfolio drops **12% from that peak**, the system exits
- Only active when QQQ is **>3% above its 200-day** (prevents firing in choppy markets near the 200-day)

At 100% TQQQ allocation, a 12% portfolio drop ≈ 4% QQQ drop — which catches
the start of corrections before they become crashes.""")

        with st.expander("What is the Crash Detector?"):
            st.markdown("""The crash detector prevents the system from re-entering during a **freefall**
like COVID (March 2020) where TQQQ dropped 70%+ in weeks.

**How it works:**
- If TQQQ drops **30%+ in 10 trading days**, a crash is detected
- ALL entries are **blocked for 40 days** — even FTD signals
- This prevents the system from buying a bear market rally that gets crushed

In COVID, this blocked the March 5 FTD that would have re-entered right before
the final crash leg. The system waited until April, catching the real bottom.""")

        with st.expander("Why trade TQQQ in an IRA?"):
            st.markdown("""TQQQ swing trades generate frequent short-term capital gains. In a
taxable account, these can be taxed at **35-50%+** (federal + state).

- **Traditional IRA:** Gains are tax-deferred until withdrawal
- **Roth IRA:** Gains are **tax-free** forever

This makes IRAs the ideal vehicle — 100% of gains compound without tax drag.
The system's 72 trades over 16 years would create significant tax liability
in a taxable account.""")

    # ══════════════════════════════════════════════════════════════
    # TAB 3: HOW TO USE THIS SITE
    # ══════════════════════════════════════════════════════════════
    with tab_guide:
        st.markdown("## How to Use This Dashboard")

        st.markdown("### Daily Routine (30 seconds)")
        st.markdown("""
1. **Check the alert bar** at the top — it tells you the overall risk level
   (CLEAR → WATCH → ELEVATED → HIGH → CRITICAL → FULL EXIT)
2. **Glance at the sell signal scoreboard** — how many of the 9 rules are triggered?
3. **Check the market regime** — Is the Nasdaq in a "Confirmed Uptrend" (green light to buy)
   or "Market in Correction" (stay in cash)?
4. If all clear, **no action needed**. If signals are firing, consider trimming per the rules.
""")

        st.markdown("### Weekly Routine (15 minutes)")
        st.markdown("""
1. **Review the swing tracker** — Where are we in the current swing? How far from the last
   trough or peak?
2. **Check distribution days** — Are they clustering? Is the count approaching 4-5?
3. **Review the chart** — Is TQQQ above or below key moving averages?
4. **Update the Bulls % in the sidebar** — Check [AAII Sentiment Survey](https://www.aaii.com/sentimentsurvey)
   for the latest reading and enter it in the sidebar.
""")

        st.markdown("### Reading the Dashboard")
        st.markdown("""
| Section | What It Tells You | Action |
|---------|-------------------|--------|
| **Alert Bar** | Overall risk level based on all 9 sell rules | Green = hold. Yellow = watch. Red = sell. |
| **Buy Signals** | Whether FTD or 3WK patterns have appeared | If active, consider entering TQQQ |
| **Sell Scoreboard** | Which specific sell rules are triggered | Trim 10% per triggered signal |
| **Price Chart** | TQQQ with moving averages and swing points | Visual confirmation of signals |
| **MA Table** | Distance from each key moving average | Below 21-EMA is danger zone |
""")

        st.markdown("### Sidebar Controls")
        st.markdown("""
- **Chart lookback** — How many days of price history to display (30-365)
- **Bulls %** — Manually enter the latest AAII bullish sentiment percentage. When bulls
  exceed 60%, it triggers sell rule #8 as a secondary caution signal.
- **Refresh Data** — Force a fresh fetch from Yahoo Finance (data caches for 4 hours)
""")

    # ══════════════════════════════════════════════════════════════
    # TAB 4: HISTORICAL PERFORMANCE
    # ══════════════════════════════════════════════════════════════
    with tab_perf:
        st.markdown("## Historical Performance")
        st.caption("Backtested using the system's buy/sell rules on historical TQQQ data. Past performance does not guarantee future results.")

        results = bt_results

        if not results:
            st.warning("Unable to run backtests — data unavailable.")
        else:
            current_year = dt.date.today().year

            # ── Equity Curve Chart ──
            if bt_equity:
                import plotly.graph_objects as go

                st.markdown("#### Equity Curve — Cumulative Portfolio Value")
                eq_period2 = st.segmented_control(
                    "Period",
                    options=EQUITY_PERIOD_OPTIONS,
                    default="All",
                    key="eq_range_hist",
                ) or "All"
                eq_dates2_f, eq_vals2_f = _filter_equity_series(bt_equity, eq_period2)

                eq_fig2 = go.Figure()
                eq_fig2.add_trace(go.Scatter(
                    x=eq_dates2_f, y=eq_vals2_f,
                    mode="lines", name="Strategy",
                    line=dict(color="#818cf8", width=2.5),
                ))
                y_min2 = min(eq_vals2_f) * 0.95 if eq_vals2_f else 0
                y_max2 = max(eq_vals2_f) * 1.05 if eq_vals2_f else 100000
                if eq_vals2_f and (y_max2 - y_min2) < max(abs(y_max2) * 1e-9, 1.0):
                    pad2 = max(abs(y_max2) * 0.02, 1000.0)
                    y_min2, y_max2 = y_min2 - pad2, y_max2 + pad2
                eq_fig2.update_layout(
                    template="plotly_dark",
                    height=450,
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10,15,26,1)",
                    yaxis=dict(
                        gridcolor="rgba(255,255,255,0.04)",
                        tickprefix="$", tickformat=",",
                        range=[y_min2, y_max2], fixedrange=True,
                    ),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", fixedrange=True),
                )
                st.caption("Use Period above to change the time window — chart Y-axis matches that window.")
                st.plotly_chart(
                    eq_fig2,
                    use_container_width=True,
                    config={
                        "scrollZoom": False,
                        "displayModeBar": False,
                    },
                )

            # ── Summary table with max drawdown ──
            summary_rows = []
            for r in results:
                label = f"{r.year} YTD" if r.year == current_year else str(r.year)
                summary_rows.append({
                    "Year": label,
                    "System Return": f"{r.total_return_pct:+.1f}%",
                    "Max Drawdown": f"{r.max_drawdown_pct:.1f}%",
                    "TQQQ B&H": f"{r.tqqq_buy_hold_pct:+.1f}%",
                    "Trades": r.num_trades,
                    "Win Rate": f"{r.win_rate_pct:.0f}%",
                    "Avg Win": f"{r.avg_win_pct:+.1f}%",
                    "Avg Loss": f"{r.avg_loss_pct:+.1f}%" if r.avg_loss_pct != 0 else "N/A",
                })

            st.markdown("### Year-by-Year Summary")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            # Total max drawdown
            if bt_equity:
                all_vals = sorted(bt_equity.items())
                peak_v = all_vals[0][1]
                total_max_dd = 0.0
                for d, v in all_vals:
                    if v > peak_v:
                        peak_v = v
                    dd = ((v - peak_v) / peak_v) * 100
                    if dd < total_max_dd:
                        total_max_dd = dd
                st.markdown(f"**Total Max Drawdown (all-time): {total_max_dd:.1f}%**")

            # Key metrics
            total_trades = sum(r.num_trades for r in results)
            overall_wr = sum(r.win_rate_pct * r.num_trades for r in results if r.num_trades > 0) / max(total_trades, 1)
            start_val = results[0].starting_value if results else STARTING_CAPITAL
            end_val = results[-1].ending_value if results else STARTING_CAPITAL
            cumulative_pct = ((end_val / start_val) - 1) * 100 if start_val > 0 else 0

            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Starting Capital", f"${start_val:,.0f}")
            ic2.metric("Current Value", f"${end_val:,.0f}")
            ic3.metric("Total Return", f"{cumulative_pct:+.1f}%")
            ic4.metric(f"Trades ({total_trades})", f"{overall_wr:.0f}% Win Rate")

            # Per-year portfolio trace
            st.markdown("### Portfolio Trace by Year")
            for r in results:
                label = f"{r.year} YTD" if r.year == current_year else str(r.year)
                color = "#17BF63" if r.total_return_pct > 0 else "#E0245E"
                with st.expander(
                    f"{label} — {r.total_return_pct:+.1f}% · "
                    f"${r.starting_value:,.0f} → ${r.ending_value:,.0f} · "
                    f"{r.num_trades} trades · {r.win_rate_pct:.0f}% WR"
                ):
                    if r.trades:
                        ym1, ym2, ym3 = st.columns(3)
                        ym1.metric("Start", f"${r.starting_value:,.0f}")
                        ym2.metric("End", f"${r.ending_value:,.0f}")
                        pnl_val = r.ending_value - r.starting_value
                        ym3.metric("P&L", f"${pnl_val:+,.0f}", delta=f"{r.total_return_pct:+.1f}%")

                        # Build event list: each BUY and SELL is a separate event
                        events = []
                        for ti, t in enumerate(r.trades):
                            pct_deployed = t.cash_deployed / t.portfolio_before * 100 if t.portfolio_before > 0 else 0
                            trade_pnl = t.portfolio_after - t.portfolio_before

                            # ── BUY event ──
                            if t.signal_type == "FTD":
                                buy_trigger = "Follow-Through Day (FTD) detected on the Nasdaq"
                                buy_conditions = [
                                    "Nasdaq corrected 7%+ from recent high",
                                    "Rally attempt reached day 4+",
                                    "Nasdaq gained 1.25%+ on the day",
                                    "Volume was higher than prior session",
                                ]
                            elif t.signal_type == "MACD":
                                buy_trigger = "Weekly MACD crossed above zero on QQQ"
                                buy_conditions = [
                                    "QQQ 12-week EMA crossed above 26-week EMA",
                                    "QQQ was above its 200-day SMA",
                                    "Intermediate-term trend confirmed bullish",
                                ]
                            elif t.signal_type == "Entry":
                                buy_trigger = "Standard entry — QQQ in confirmed uptrend"
                                buy_conditions = [
                                    "QQQ was above its 200-day SMA",
                                    "System defaults to invested in bull markets",
                                ]
                            else:
                                buy_trigger = f"{t.signal_type} signal"
                                buy_conditions = ["Signal conditions met"]

                            if pct_deployed > 80:
                                alloc_reason = "100% — Full conviction: MACD positive + QQQ above 200-day"
                            elif pct_deployed > 40:
                                alloc_reason = "50% — Half position: either MACD negative or FTD below 200-day (probe)"
                            else:
                                alloc_reason = f"{pct_deployed:.0f}% — Cautious: limited conviction"

                            events.append({
                                "type": "BUY",
                                "date": t.entry_date,
                                "icon": "🟢",
                                "title": f"BUY {t.entry_date} — {t.signal_type}",
                                "trigger": buy_trigger,
                                "conditions": buy_conditions,
                                "details": [
                                    f"**Price:** ${t.entry_price:.2f} (TQQQ)",
                                    f"**Shares:** {t.shares:,.0f}",
                                    f"**Deployed:** ${t.cash_deployed:,.0f}",
                                    f"**Allocation:** {alloc_reason}",
                                    f"**Portfolio:** ${t.portfolio_before:,.0f}",
                                ],
                            })

                            # ── SELL event ──
                            if t.duration_days <= 3 and t.return_pct <= 0:
                                sell_trigger = "Quick exit — entry signal failed"
                                sell_conditions = [
                                    "QQQ broke below 200-day SMA within days of entry",
                                    "2 consecutive closes below 200-day confirmed",
                                    "Position closed to preserve capital",
                                ]
                            elif t.duration_days <= 20 and t.return_pct < -8:
                                sell_trigger = "12% trailing stop fired"
                                sell_conditions = [
                                    "Portfolio dropped 12% from its recent peak",
                                    "QQQ was >3% above 200-day (trailing stop was active)",
                                    "Stop protects against slow rollovers from bull highs",
                                ]
                            elif t.return_pct <= 0:
                                sell_trigger = "200-day SMA breakdown"
                                sell_conditions = [
                                    "QQQ closed below its 200-day SMA",
                                    "Second consecutive close below confirmed the break",
                                    "Market transitioned from bull to bear regime",
                                ]
                            elif t.duration_days > 100:
                                sell_trigger = "200-day SMA exit after long hold"
                                sell_conditions = [
                                    f"Held for {t.duration_days} days through normal pullbacks",
                                    "QQQ eventually closed below 200-day SMA for 2 consecutive days",
                                    "The wide exit allowed riding the full uptrend",
                                ]
                            else:
                                sell_trigger = "Exit conditions met"
                                sell_conditions = [
                                    "Either QQQ broke below 200-day for 2 days",
                                    "Or 12% trailing stop fired from peak",
                                ]

                            t_color = "#17BF63" if t.return_pct > 0 else "#E0245E"
                            events.append({
                                "type": "SELL",
                                "date": t.exit_date,
                                "icon": "🔴" if t.return_pct <= 0 else "🟢",
                                "trigger": sell_trigger,
                                "conditions": sell_conditions,
                                "ret": t.return_pct,
                                "details": [
                                    f"**Closed:** {t.exit_date} in after-hours (4:00-8:00 PM ET)",
                                    f"**Sell price:** ${t.exit_price:.2f} (TQQQ)",
                                    f"**Entered:** {t.entry_date} at ${t.entry_price:.2f}",
                                    f"**Held:** {t.duration_days} days",
                                    f"**Return:** {t.return_pct:+.1f}%",
                                    f"**P&L:** ${trade_pnl:+,.0f}",
                                    f"**Portfolio after:** ${t.portfolio_after:,.0f}",
                                ],
                                "color": t_color,
                            })

                        # Render each event as its own expander
                        for ev in events:
                            badge = "BUY" if ev["type"] == "BUY" else "SELL"
                            ret_label = f" · **{ev['ret']:+.1f}%**" if "ret" in ev else ""
                            with st.expander(f"{ev['icon']} **{badge}** · {ev['date']} · {ev['trigger']}{ret_label}"):
                                st.markdown(f"**Trigger:** {ev['trigger']}")
                                st.markdown("**Conditions met:**")
                                for c in ev["conditions"]:
                                    st.markdown(f"- {c}")
                                st.markdown("---")
                                for d in ev["details"]:
                                    st.markdown(d)

                        st.markdown(
                            f"**Best:** {r.best_trade} · **Worst:** {r.worst_trade} · "
                            f"**TQQQ B&H:** {r.tqqq_buy_hold_pct:+.1f}% · "
                            f"**QQQ B&H:** {r.qqq_buy_hold_pct:+.1f}%"
                        )
                    else:
                        st.info("No trades generated for this period.")

            st.markdown("---")
            st.markdown("""
##### Important Disclaimers
- Backtest uses **simplified signal detection** — real-time signals may differ slightly due to
  look-ahead bias and implementation details
- **No slippage or commissions** are modeled — real returns would be slightly lower
- The system assumes **full position on each entry** — actual sizing varies by conviction
- **TQQQ suffers from volatility decay** in sideways/declining markets — buy-and-hold
  comparisons are included to show the value of active management
- Past performance does **not** guarantee future results
""")

    # Footer
    st.markdown(f"""
        <div style="border-top: 1px solid rgba(255,255,255,0.04); margin-top: 48px; padding: 28px 0;
                     text-align: center;">
            <a href="{TWITTER_URL}" target="_blank" style="text-decoration: none;">
                <img src="{LOGO_URL}" alt="MrZzz"
                     style="width: 28px; height: 28px; border-radius: 50%;
                            border: 1px solid rgba(255,255,255,0.1); vertical-align: middle;
                            opacity: 0.8;">
                <span style="color: #818cf8; font-weight: 600; margin-left: 8px;
                              vertical-align: middle; font-size: 0.85em;">
                    @MrZzz</span>
            </a>
            <p style="color: #4b5563; font-size: 0.72em; margin-top: 10px; letter-spacing: 0.02em;">
                Data: TradingView / Yahoo Finance &nbsp;&#183;&nbsp;
                Not financial advice &nbsp;&#183;&nbsp;
                For educational purposes only
            </p>
        </div>
    """, unsafe_allow_html=True)
