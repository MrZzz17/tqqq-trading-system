"""
TQQQ Swing Trading Dashboard -- Phase 1
Rules-based TQQQ buy/sell signal system.
"""

import datetime as dt

import pandas as pd
import streamlit as st

from core.data import (
    get_tqqq_data, get_qqq_data, get_nasdaq_data, get_sp500_data,
    get_52_week_high, get_current_price, get_latest_date,
)
from core.indicators import count_distribution_days, detect_market_regime
from core.signals import (
    detect_follow_through_day, detect_three_white_knights,
    check_all_sell_signals, compute_alert_level,
)
from core.swing_tracker import detect_swings, current_swing_stats, swing_summary_stats
from core.charts import build_tqqq_chart, build_distribution_chart
from core.backtest import run_all_backtests
import config


SEVERITY_ICONS = {"watch": "👀", "warning": "⚠️", "sell": "🔴"}
REGIME_ICONS = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


def _styled_card(title: str, content: str, border_color: str = "#333") -> str:
    return f"""<div style="border: 1px solid {border_color}; border-radius: 10px;
        padding: 20px; margin-bottom: 12px; background: rgba(255,255,255,0.02);">
        <div style="font-weight: 700; font-size: 1.05em; margin-bottom: 8px; color: #eee;">{title}</div>
        <div style="font-size: 0.9em; color: #bbb; line-height: 1.6;">{content}</div>
    </div>"""


def render():
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 5px 0;">
            <h1 style="margin-bottom: 0; font-size: 2.4em;">TQQQ Trading System</h1>
            <p style="color: #888; font-size: 1.05em; margin-top: 4px;">
                Rules-based swing trading &nbsp;|&nbsp; CAN SLIM + 3x leveraged ETF strategy
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
    st.sidebar.markdown("### Settings")
    chart_lookback = st.sidebar.slider("Chart lookback (days)", 30, 365, 120)
    swing_min_pct = st.sidebar.slider("Swing min % move", 3.0, 15.0, 5.0, 0.5)
    bulls_pct = st.sidebar.number_input(
        "Bulls % (AAII sentiment)",
        min_value=0.0, max_value=100.0, value=0.0, step=1.0,
        help="Enter the latest AAII bullish sentiment %. Leave 0 if unknown.",
    )
    bulls_input = bulls_pct if bulls_pct > 0 else None
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

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
        # Market Status
        nasdaq_regime = detect_market_regime(nasdaq)
        sp_regime = detect_market_regime(sp500)
        tqqq_delta = (tqqq.iloc[-1]['Close'] - tqqq.iloc[-2]['Close']) / tqqq.iloc[-2]['Close'] * 100
        qqq_delta = (qqq.iloc[-1]['Close'] - qqq.iloc[-2]['Close']) / qqq.iloc[-2]['Close'] * 100

        REGIME_SHORT = {
            "Confirmed Uptrend": "Uptrend",
            "Uptrend Under Pressure": "Under Pressure",
            "Market in Correction": "Correction",
        }
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TQQQ", f"${tqqq_price:.2f}", delta=f"{tqqq_delta:+.2f}%")
        c2.metric("QQQ", f"${qqq_price:.2f}", delta=f"{qqq_delta:+.2f}%")
        nq_short = REGIME_SHORT.get(nasdaq_regime.status, nasdaq_regime.status)
        sp_short = REGIME_SHORT.get(sp_regime.status, sp_regime.status)
        c3.metric("Nasdaq", f"{REGIME_ICONS.get(nasdaq_regime.color, '')} {nq_short}",
                   delta=f"{nasdaq_regime.dist_day_count} dist. days", delta_color="off")
        c4.metric("S&P 500", f"{REGIME_ICONS.get(sp_regime.color, '')} {sp_short}",
                   delta=f"{sp_regime.dist_day_count} dist. days", delta_color="off")
        st.caption(f"Data as of {data_date.strftime('%b %d, %Y')} · Yahoo Finance (delayed)")

        # Alert Bar
        sell_signals = check_all_sell_signals(tqqq, nasdaq, bulls_input)
        alert_level, alert_color, alert_action = compute_alert_level(sell_signals)
        triggered_count = sum(1 for s in sell_signals if s.triggered)
        color_hex = {"green": "#4CAF50", "yellow": "#FFC107", "orange": "#FF9800", "red": "#F44336"}.get(alert_color, "#FFC107")

        st.markdown(f"""<div style="
            background: linear-gradient(135deg, {color_hex}18, {color_hex}08);
            border: 1px solid {color_hex}44; border-left: 5px solid {color_hex};
            padding: 18px 24px; border-radius: 10px; margin: 12px 0 20px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 1.5em; font-weight: 800; color: {color_hex};">
                        {alert_level}</span>
                    <span style="font-size: 0.95em; color: #aaa; margin-left: 12px;">
                        {triggered_count} of 9 sell signals active</span>
                </div>
                <div style="font-size: 0.95em; color: #ddd; margin-top: 4px;">{alert_action}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Buy Signals
        st.markdown("### Buy Signals")
        bc1, bc2 = st.columns(2)
        ftd = detect_follow_through_day(nasdaq)
        with bc1:
            if ftd:
                st.success(f"**Follow-Through Day** detected {ftd.date.strftime('%b %d')} ({ftd.strength})")
                st.markdown(f"{ftd.details}  \nSuggested size: **{ftd.suggested_size:.0%}**")
            else:
                st.info("**Follow-Through Day** — Not detected in last 30 sessions")
        wk = detect_three_white_knights(qqq)
        with bc2:
            if wk:
                st.success(f"**3 White Knights** detected {wk.date.strftime('%b %d')} ({wk.strength})")
                st.markdown(f"{wk.details}  \nSuggested size: **{wk.suggested_size:.0%}**")
            else:
                st.info("**3 White Knights** — Not detected in last 10 sessions")

        # Sell Signal Scoreboard
        st.markdown("### Sell Signal Scoreboard")
        cols = st.columns(3)
        for i, sig in enumerate(sell_signals):
            with cols[i % 3]:
                icon = SEVERITY_ICONS.get(sig.severity, "")
                badge_bg = "#F44336" if sig.triggered else "#4CAF50"
                card_border = "#F4434488" if sig.triggered else "#33333388"
                card_bg = "rgba(244,67,54,0.06)" if sig.triggered else "transparent"
                st.markdown(f"""<div style="border: 1px solid {card_border}; background: {card_bg};
                    border-radius: 10px; padding: 14px; margin-bottom: 10px; min-height: 110px;">
                    <div style="font-weight: 700; font-size: 0.88em; color: #ddd;">
                        {icon} #{sig.rule_number} {sig.name}</div>
                    <div style="font-size: 0.78em; color: #999; margin: 6px 0;">{sig.details}</div>
                    <span style="background: {badge_bg}; color: white; padding: 2px 10px;
                        border-radius: 4px; font-size: 0.72em; font-weight: 600;">
                        {'TRIGGERED' if sig.triggered else 'CLEAR'}</span>
                    <span style="color: #666; font-size: 0.7em; margin-left: 6px;">
                        {sig.severity.upper()}</span>
                </div>""", unsafe_allow_html=True)

        # Chart
        st.markdown("### TQQQ Price Chart")
        year_now = dt.datetime.now().year
        swings = detect_swings(tqqq, min_move_pct=swing_min_pct, year_filter=year_now - 1)
        fig = build_tqqq_chart(tqqq, swings=swings, lookback_days=chart_lookback)
        st.plotly_chart(fig, key="main_chart", use_container_width=True)

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

        # Swing Tracker
        st.markdown("### Swing Tracker")
        all_swings = detect_swings(tqqq, min_move_pct=swing_min_pct)
        ytd_swings = detect_swings(tqqq, min_move_pct=swing_min_pct, year_filter=year_now)

        current = current_swing_stats(tqqq, all_swings)
        if "label" in current:
            icon = "📈" if current.get("direction") == "up" else "📉"
            st.info(f"{icon} {current['label']}")

        summary = swing_summary_stats(all_swings)
        if summary:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Swings (2yr)", summary["total_swings"])
            m2.metric("Avg Rally", f"{summary['avg_up_move']:+.1f}%")
            m3.metric("Avg Pullback", f"{summary['avg_down_move']:.1f}%")
            m4.metric("Best Rally", f"{summary['max_up_move']:+.1f}%")
            m5.metric("Avg Duration", f"{summary['avg_duration_days']:.0f}d")

        stab1, stab2 = st.tabs([f"{year_now} YTD", "All Swings (2yr)"])
        for tab, swing_list in [(stab1, ytd_swings), (stab2, all_swings)]:
            with tab:
                if swing_list:
                    rows = [{"Date": s.date.strftime("%Y-%m-%d"), "Type": s.point_type.title(),
                             "Price": f"${s.price:.2f}", "Move": f"{s.pct_move:+.1f}%",
                             "Days": s.trading_days, "vs 50d": s.vs_sma_50.title(),
                             "vs 21d": s.vs_ema_21.title()} for s in swing_list]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No swings detected with current sensitivity.")

        # Distribution Days
        st.markdown("### Distribution Days")
        dc1, dc2 = st.columns(2)
        for col_widget, title, df_data in [(dc1, "Nasdaq Composite", nasdaq), (dc2, "S&P 500", sp500)]:
            with col_widget:
                st.markdown(f"##### {title}")
                dist = count_distribution_days(df_data)
                fig_d = build_distribution_chart(df_data, dist)
                st.plotly_chart(fig_d, use_container_width=True)
                if dist:
                    rows = [{"Date": d.date.strftime("%Y-%m-%d"), "Decline": f"{d.pct_change:.2f}%",
                             "Vol Ratio": f"{d.volume_vs_prior:.2f}x"} for d in dist]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.success("No distribution days in last 25 sessions.")

    # ══════════════════════════════════════════════════════════════
    # TAB 2: HOW THE SYSTEM WORKS
    # ══════════════════════════════════════════════════════════════
    with tab_how:
        st.markdown("## How the Trading System Works")
        st.markdown("""
This system is a **rules-based, hybrid swing trading strategy** that combines two
complementary approaches to generate returns in different market environments.
""")

        st.markdown("### The Core Concept")
        h1, h2 = st.columns(2)
        with h1:
            st.markdown(_styled_card(
                "Strategy A: TQQQ Swing Trading",
                """Trade TQQQ (3x leveraged Nasdaq 100 ETF) to capture 20-30% swings
                that occur multiple times per year. Best executed in <b>tax-advantaged accounts</b>
                (IRA/Roth) to avoid short-term capital gains taxes. This is the primary
                return driver when individual stock setups are scarce."""
            ), unsafe_allow_html=True)
        with h2:
            st.markdown(_styled_card(
                "Strategy B: CAN SLIM Stock Picking",
                """Hold 6-8 individual growth stocks with the potential to double or
                triple over 12-18 months. Screened using CAN SLIM fundamentals
                (25%+ earnings/sales growth, RS 95+) with technical entry timing.
                TQQQ is sold down to fund these positions as setups emerge."""
            ), unsafe_allow_html=True)

        st.markdown("### Buy Rules (Only 2)")
        st.markdown("""
| # | Signal | Description | Position Size |
|---|--------|-------------|---------------|
| 1 | **Follow-Through Day (FTD)** | After a market correction, on day 4+ of a rally attempt, the Nasdaq gains >1.25% on volume higher than the prior day. This is the highest-conviction entry. | Up to 100% in IRA |
| 2 | **3 White Knights** | 3 consecutive days of higher highs AND higher lows on QQQ (not TQQQ). Used when an FTD hasn't occurred but the market appears to be turning. | 25-75% based on conviction |
""")

        st.markdown("### Sell Rules (9 Signals)")
        st.markdown("""
Positions are trimmed in **10% chunks** as signals accumulate. More signals = more aggressive selling.

| # | Signal | Severity | What It Means |
|---|--------|----------|---------------|
| 1 | New 52-week high | Watch | Potential resistance — stay alert |
| 2 | New high on declining volume | Warning | Institutions aren't participating in the move up |
| 3 | 4-5 distribution days | Warning/Sell | Heavy institutional selling in the broader market |
| 4 | 3 consecutive down days | Warning | Short-term momentum shifting |
| 5 | 10-day MA violated on rising volume | Sell | Short-term trend broken with conviction |
| 6 | 3 down days + rising volume + lower H/L | Sell | Severe weakness pattern |
| 7 | Triple rejection at resistance | Warning | Price can't break through — exhaustion |
| 8 | Bulls vs Bears >60% | Watch | Sentiment too bullish — contrarian caution |
| 9 | **2 closes below 21-day EMA** | **Full Exit** | **Nuclear sell signal — exit immediately** |
""")

        st.markdown("### Position Sizing & Risk Management")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(_styled_card(
                "TQQQ Sizing",
                """<b>FTD entry:</b> Up to 100% (IRA accounts)<br>
                <b>3WK entry:</b> 25% initial, scale to 50-75%<br>
                <b>Exit:</b> Sell in 10% increments as sell signals fire<br>
                <b>Hard stop:</b> If price undercuts the rally day low, exit immediately<br>
                <b>No adding:</b> Enter at the turn, don't add to positions"""
            ), unsafe_allow_html=True)
        with r2:
            st.markdown(_styled_card(
                "Individual Stock Sizing",
                """<b>Initial position:</b> 10% of portfolio<br>
                <b>Max position:</b> 12.5-15% after adding on pullbacks<br>
                <b>Stop-loss:</b> Technical stop at 50-day/10-week MA<br>
                <b>100% rule:</b> When a stock doubles, sell half<br>
                <b>Max holdings:</b> 6-8 core positions"""
            ), unsafe_allow_html=True)

        st.markdown("### Key Concepts")

        with st.expander("What is a Distribution Day?"):
            st.markdown("""A distribution day occurs when a major index (Nasdaq or S&P 500) declines
more than 0.2% on volume higher than the prior session. This indicates
**institutional selling** — the big money is unloading shares. When 4-5
distribution days cluster within a 25-session window, the market is under
significant selling pressure and corrections often follow.

Distribution days expire after 25 trading sessions or when the index rallies
5% above the distribution day's close.""")

        with st.expander("What is a Follow-Through Day?"):
            st.markdown("""A Follow-Through Day (FTD) is the primary signal that a new market
uptrend is beginning after a correction. The criteria:

1. The market must first establish a **rally attempt** — the index makes
   a low and begins moving higher
2. On **day 4 or later** of the rally attempt, the Nasdaq must gain
   **at least 1.25%** on volume **higher than the previous session**

Not all FTDs succeed, but almost every major market rally has started with
one. The system buys TQQQ aggressively on an FTD, with a stop-loss below
the rally attempt's low.""")

        with st.expander("What is the 3 White Knights pattern?"):
            st.markdown("""The 3 White Knights pattern is a secondary buy signal consisting of
**3 consecutive trading days** where QQQ makes **both a higher high AND
a higher low** compared to the previous day. This pattern suggests the
market is establishing a floor and beginning to trend upward.

It's used as an earlier, lower-conviction entry before an official FTD
is confirmed. Position sizes are smaller (25-50%) compared to FTD entries.""")

        with st.expander("Why TQQQ in an IRA?"):
            st.markdown("""TQQQ swing trades generate frequent short-term capital gains. In a
taxable account, these gains can be taxed at 35-50%+ (federal + state),
significantly eroding returns. In an IRA or Roth IRA:

- **Traditional IRA:** Gains are tax-deferred until withdrawal
- **Roth IRA:** Gains are **tax-free** forever

This makes IRAs the ideal vehicle for frequent TQQQ swing trading, as
100% of the gains compound without tax drag. Individual stocks with
12-18 month holds can go in taxable accounts where they qualify for
lower long-term capital gains rates.""")

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
| **Swing Tracker** | Historical peak/trough data with % moves | Context for current position |
| **Distribution Days** | Institutional selling pressure | 4+ days = elevated caution |
""")

        st.markdown("### Sidebar Controls")
        st.markdown("""
- **Chart lookback** — How many days of price history to display (30-365)
- **Swing min % move** — Sensitivity for peak/trough detection. Lower = more swings detected.
  Default 5% works well for TQQQ.
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

        with st.spinner("Running backtests on 2022-2026 data..."):
            results = run_all_backtests()

        if not results:
            st.warning("Unable to run backtests — data unavailable.")
        else:
            current_year = dt.date.today().year

            # Summary table
            summary_rows = []
            for r in results:
                label = f"{r.year} YTD" if r.year == current_year else str(r.year)
                summary_rows.append({
                    "Year": label,
                    "System Return": f"{r.total_return_pct:+.1f}%",
                    "TQQQ Buy & Hold": f"{r.tqqq_buy_hold_pct:+.1f}%",
                    "QQQ Buy & Hold": f"{r.qqq_buy_hold_pct:+.1f}%",
                    "Trades": r.num_trades,
                    "Win Rate": f"{r.win_rate_pct:.0f}%",
                    "Avg Win": f"{r.avg_win_pct:+.1f}%",
                    "Avg Loss": f"{r.avg_loss_pct:+.1f}%" if r.avg_loss_pct != 0 else "N/A",
                })

            st.markdown("### Year-by-Year Summary")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            # Key insight
            total_system = 100
            total_bh = 100
            for r in results:
                total_system *= (1 + r.total_return_pct / 100)
                total_bh *= (1 + r.tqqq_buy_hold_pct / 100)
            cumulative_system = total_system - 100
            cumulative_bh = total_bh - 100
            total_trades = sum(r.num_trades for r in results)
            overall_wr = sum(r.win_rate_pct * r.num_trades for r in results if r.num_trades > 0) / max(total_trades, 1)

            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Cumulative Return", f"{cumulative_system:+.1f}%")
            ic2.metric("vs TQQQ B&H", f"{cumulative_bh:+.1f}%")
            ic3.metric("Total Trades", total_trades)
            ic4.metric("Avg Win Rate", f"{overall_wr:.0f}%")

            # Per-year trade details
            st.markdown("### Trade Details by Year")
            for r in results:
                label = f"{r.year} YTD" if r.year == current_year else str(r.year)
                with st.expander(f"{label} — {r.total_return_pct:+.1f}% ({r.num_trades} trades, {r.win_rate_pct:.0f}% win rate)"):
                    if r.trades:
                        trade_rows = [{
                            "Entry": t.entry_date,
                            "Exit": t.exit_date,
                            "Signal": t.signal_type,
                            "Entry $": f"${t.entry_price:.2f}",
                            "Exit $": f"${t.exit_price:.2f}",
                            "Return": f"{t.return_pct:+.1f}%",
                            "Days": t.duration_days,
                            "Result": t.outcome,
                        } for t in r.trades]
                        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

                        mc1, mc2 = st.columns(2)
                        mc1.markdown(f"**Best trade:** {r.best_trade}")
                        mc2.markdown(f"**Worst trade:** {r.worst_trade}")
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
    st.markdown("---")
    st.caption("Data: Yahoo Finance (delayed) • Not financial advice • For educational purposes only")
