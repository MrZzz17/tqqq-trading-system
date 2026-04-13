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
from core.backtest import run_all_backtests, STARTING_CAPITAL
import config


SEVERITY_ICONS = {"watch": "👀", "warning": "⚠️", "sell": "🔴"}
REGIME_ICONS = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


LOGO_URL = "https://pbs.twimg.com/profile_images/1959893019509071872/Xa6rYCoN_400x400.jpg"
TWITTER_URL = "https://x.com/MrZzz17"


def _styled_card(title: str, content: str, border_color: str = "#38444D") -> str:
    return f"""<div style="border: 1px solid {border_color}; border-radius: 12px;
        padding: 20px; margin-bottom: 12px; background: rgba(29,161,242,0.03);">
        <div style="font-weight: 700; font-size: 1.05em; margin-bottom: 8px; color: #E7E9EA;">{title}</div>
        <div style="font-size: 0.9em; color: #8899A6; line-height: 1.7;">{content}</div>
    </div>"""


def render():
    st.markdown(f"""
        <div style="text-align: center; padding: 20px 0 10px 0;">
            <a href="{TWITTER_URL}" target="_blank" style="text-decoration: none;">
                <img src="{LOGO_URL}" alt="MrZzz"
                     style="width: 64px; height: 64px; border-radius: 50%;
                            border: 2px solid #1DA1F2; margin-bottom: 10px;
                            box-shadow: 0 0 20px rgba(29,161,242,0.3);">
            </a>
            <h1 style="margin-bottom: 0; font-size: 2.2em; color: #E7E9EA;
                        letter-spacing: -0.02em;">
                TQQQ Trading System
            </h1>
            <p style="color: #8899A6; font-size: 1.0em; margin-top: 6px;">
                Rules-based swing trading &nbsp;&#183;&nbsp; 3x leveraged Nasdaq 100
                &nbsp;&#183;&nbsp;
                <a href="{TWITTER_URL}" target="_blank"
                   style="color: #1DA1F2; text-decoration: none; font-weight: 600;">
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
            <div style="text-align: center; padding: 10px 0 16px 0;">
                <img src="{LOGO_URL}" style="width: 48px; height: 48px; border-radius: 50%;
                     border: 2px solid #1DA1F2;">
                <p style="color: #E7E9EA; font-weight: 700; margin: 8px 0 2px 0; font-size: 0.95em;">
                    TQQQ System</p>
                <p style="color: #657786; font-size: 0.75em; margin: 0;">by @MrZzz</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("##### Settings")
        chart_lookback = st.slider("Chart lookback (days)", 30, 365, 120)
        swing_min_pct = st.slider("Swing min % move", 3.0, 15.0, 5.0, 0.5)
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
        color_hex = {"green": "#17BF63", "yellow": "#FFAD1F", "orange": "#FF6F00", "red": "#E0245E"}.get(alert_color, "#FFAD1F")

        st.markdown(f"""<div style="
            background: linear-gradient(135deg, {color_hex}15, {color_hex}08);
            border: 1px solid {color_hex}33; border-left: 5px solid {color_hex};
            padding: 18px 24px; border-radius: 12px; margin: 12px 0 20px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 1.5em; font-weight: 800; color: {color_hex};
                                 letter-spacing: -0.02em;">
                        {alert_level}</span>
                    <span style="font-size: 0.9em; color: #8899A6; margin-left: 12px;">
                        {triggered_count} of 9 sell signals active</span>
                </div>
                <div style="font-size: 0.95em; color: #E7E9EA; margin-top: 4px;">{alert_action}</div>
            </div>
        </div>""", unsafe_allow_html=True)

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

        def _health_card(ticker, df, dist_days_count):
            price = float(df.iloc[-1]["Close"])
            cross_label, cross_color = _cross_status(df)
            dd_pct, peak = _drawdown_from_peak(df)

            _, sma50_val, sma50_dist, sma50_color = _ma_status(df, "SMA_50", "50d")
            _, sma200_val, sma200_dist, sma200_color = _ma_status(df, "SMA_200", "200d")
            _, ema21_val, ema21_dist, ema21_color = _ma_status(df, "EMA_21", "21d")

            above_200 = float(df.iloc[-1].get("SMA_200", 0) or 0)
            regime_color = "#17BF63" if price > above_200 and above_200 > 0 else "#E0245E"
            regime_label = "BULL" if price > above_200 and above_200 > 0 else "BEAR"

            dd_color = "#17BF63" if dd_pct > -5 else ("#FFAD1F" if dd_pct > -10 else "#E0245E")

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
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 0.82em;">
                    <div style="text-align: center;">
                        <div style="color: #657786; margin-bottom: 2px;">21-EMA</div>
                        <div style="color: {ema21_color}; font-weight: 600;">{ema21_dist}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #657786; margin-bottom: 2px;">50-day</div>
                        <div style="color: {sma50_color}; font-weight: 600;">{sma50_dist}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #657786; margin-bottom: 2px;">200-day</div>
                        <div style="color: {sma200_color}; font-weight: 600;">{sma200_dist}</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #657786; margin-bottom: 2px;">From Peak</div>
                        <div style="color: {dd_color}; font-weight: 600;">{dd_pct:+.1f}%</div>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 0.78em; color: #8899A6;">
                    Dist. days: <b style="color: {'#E0245E' if dist_days_count >= 4 else '#FFAD1F' if dist_days_count >= 3 else '#17BF63'}">{dist_days_count}</b>/25 sessions
                    &nbsp;·&nbsp; 52w high: ${peak:.2f}
                </div>
            </div>""", unsafe_allow_html=True)

        hc1, hc2 = st.columns(2)
        qqq_dist = count_distribution_days(qqq)
        sp_dist = count_distribution_days(sp500)
        with hc1:
            _health_card("QQQ", qqq, len(qqq_dist))
        with hc2:
            _health_card("S&P 500", sp500, len(sp_dist))

        # Position sizing guidance based on market health
        qqq_sma200 = qqq.iloc[-1].get("SMA_200")
        qqq_below_200 = (qqq_sma200 is not None and not pd.isna(qqq_sma200)
                         and float(qqq.iloc[-1]["Close"]) < float(qqq_sma200))
        qqq_sma50 = qqq.iloc[-1].get("SMA_50")
        qqq_death_cross = (qqq_sma50 is not None and not pd.isna(qqq_sma50)
                           and qqq_sma200 is not None and not pd.isna(qqq_sma200)
                           and float(qqq_sma50) < float(qqq_sma200))

        if qqq_death_cross:
            sizing_color = "#E0245E"
            sizing_msg = ("**Death Cross active** — QQQ 50-day is below 200-day. "
                          "Only FTD entries recommended (at 50% allocation). "
                          "Non-FTD entries capped at 25%.")
        elif qqq_below_200:
            sizing_color = "#FF6F00"
            sizing_msg = ("**QQQ below 200-day** — Bear market conditions. "
                          "FTD entries capped at 50%. Use caution with all entries.")
        else:
            sizing_color = "#17BF63"
            sizing_msg = ("**QQQ above 200-day** — Bull market conditions. "
                          "Full allocation on FTD signals. Normal sizing for all entries.")

        st.markdown(f"""<div style="
            background: {sizing_color}08; border: 1px solid {sizing_color}22;
            border-left: 4px solid {sizing_color}; border-radius: 10px;
            padding: 14px 18px; margin: 12px 0 20px 0; font-size: 0.9em; color: #E7E9EA;">
            📐 <b>Position Sizing:</b> {sizing_msg}
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
                if sig.triggered:
                    badge_bg = "#E0245E"
                    card_border = "rgba(224,36,94,0.4)"
                    card_bg = "rgba(224,36,94,0.06)"
                else:
                    badge_bg = "#17BF63"
                    card_border = "rgba(56,68,77,0.6)"
                    card_bg = "rgba(29,161,242,0.02)"
                st.markdown(f"""<div style="border: 1px solid {card_border}; background: {card_bg};
                    border-radius: 12px; padding: 14px; margin-bottom: 10px; min-height: 110px;">
                    <div style="font-weight: 700; font-size: 0.88em; color: #E7E9EA;">
                        {icon} #{sig.rule_number} {sig.name}</div>
                    <div style="font-size: 0.78em; color: #8899A6; margin: 6px 0;">{sig.details}</div>
                    <span style="background: {badge_bg}; color: white; padding: 2px 10px;
                        border-radius: 20px; font-size: 0.72em; font-weight: 600;">
                        {'TRIGGERED' if sig.triggered else 'CLEAR'}</span>
                    <span style="color: #657786; font-size: 0.7em; margin-left: 6px;">
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
This is a **rules-based swing trading system** for TQQQ, the 3x leveraged
Nasdaq 100 ETF. It captures **20-30% swings** that occur multiple times per year
by timing entries at market turns and exiting incrementally as weakness appears.
""")

        st.markdown("### What is TQQQ?")
        h1, h2 = st.columns(2)
        with h1:
            st.markdown(_styled_card(
                "TQQQ = 3x Daily Nasdaq 100",
                """TQQQ delivers <b>3x the daily return</b> of the Nasdaq 100 (QQQ).
                If QQQ rises 1%, TQQQ rises ~3%. If QQQ falls 1%, TQQQ falls ~3%.
                This amplification creates large swings — typically <b>20-30%</b> moves
                multiple times per year — which is what this system trades."""
            ), unsafe_allow_html=True)
        with h2:
            st.markdown(_styled_card(
                "Why Swing Trade, Not Buy & Hold?",
                """TQQQ suffers from <b>volatility decay</b> in sideways or declining markets.
                A buy-and-hold approach can lose money even when QQQ is flat. Active swing
                trading avoids the decay by <b>exiting during pullbacks</b> and only holding
                during confirmed uptrends. Best executed in <b>IRA/Roth accounts</b> for
                tax-free compounding."""
            ), unsafe_allow_html=True)

        st.markdown("### Buy Rules (Only 2)")
        st.markdown("""
| # | Signal | Description | Position Size |
|---|--------|-------------|---------------|
| 1 | **Follow-Through Day (FTD)** | After a market correction, on day 4+ of a rally attempt, the Nasdaq gains >1.25% on volume higher than the prior day. Highest-conviction entry. | Up to 100% in IRA |
| 2 | **3 White Knights** | 3 consecutive days of higher highs AND higher lows on QQQ (not TQQQ). Earlier signal before an FTD is confirmed. | 25-75% based on conviction |
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
| 6 | 3 down days + rising vol + lower H/L | Sell | Severe weakness pattern |
| 7 | Triple rejection at resistance | Warning | Price can't break through — exhaustion |
| 8 | Bulls vs Bears >60% | Watch | Sentiment too bullish — contrarian caution |
| 9 | **2 closes below 21-day EMA** | **Full Exit** | **Nuclear sell signal — exit entire position** |
""")

        st.markdown("### Position Sizing & Risk Management")
        st.markdown(_styled_card(
            "How to Size TQQQ Positions",
            """<b>On a Follow-Through Day:</b> Up to 100% of TQQQ allocation (in IRA accounts)<br>
            <b>On 3 White Knights (no FTD yet):</b> Start at 25%, scale to 50-75% based on conviction<br>
            <b>Selling:</b> Trim in 10% increments as sell signals fire — don't dump all at once<br>
            <b>Hard stop:</b> If price undercuts the low of the rally attempt's first day, exit immediately<br>
            <b>No adding:</b> Enter at the market turn, do not add to the position after entry<br>
            <b>Target:</b> Capture 20%+ swings, then sell into strength as signals appear"""
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

        with st.expander("What is a Follow-Through Day (FTD)?"):
            st.markdown("""A Follow-Through Day is the primary signal that a new market uptrend
is beginning after a correction:

1. The market must first establish a **rally attempt** — the index makes
   a low and begins moving higher
2. On **day 4 or later** of the rally attempt, the Nasdaq must gain
   **at least 1.25%** on volume **higher than the previous session**

Not all FTDs succeed, but almost every major market rally has started with
one. The system buys TQQQ aggressively on an FTD, with a stop-loss below
the rally attempt's low.""")

        with st.expander("What is the 3 White Knights pattern?"):
            st.markdown("""3 consecutive trading days where QQQ makes **both a higher high AND
a higher low** compared to the previous day. This suggests the market is
establishing a floor and beginning to trend upward.

It's an earlier, lower-conviction entry used before an official FTD is
confirmed. Position sizes are smaller (25-50%) compared to FTD entries.""")

        with st.expander("Why trade TQQQ in an IRA?"):
            st.markdown("""TQQQ swing trades generate frequent short-term capital gains. In a
taxable account, these can be taxed at **35-50%+** (federal + state),
significantly eroding returns. In an IRA or Roth IRA:

- **Traditional IRA:** Gains are tax-deferred until withdrawal
- **Roth IRA:** Gains are **tax-free** forever

This makes IRAs the ideal vehicle for frequent TQQQ swing trading — 100%
of gains compound without tax drag.""")

        with st.expander("What is volatility decay?"):
            st.markdown("""Leveraged ETFs like TQQQ reset their leverage **daily**. In a volatile,
sideways market, this daily reset erodes value even if the underlying index
ends flat. Example:

- Day 1: QQQ drops 5% → TQQQ drops 15% (from $100 to $85)
- Day 2: QQQ rises 5.26% (back to even) → TQQQ rises 15.8% ($85 → $98.43)
- **QQQ is flat, but TQQQ lost 1.6%**

This is why buy-and-hold doesn't work with TQQQ in choppy markets. The
system avoids decay by **exiting during pullbacks** and only holding during
confirmed, trending uptrends.""")

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
                        pnl = r.ending_value - r.starting_value
                        ym3.metric("P&L", f"${pnl:+,.0f}", delta=f"{r.total_return_pct:+.1f}%")

                        trade_rows = []
                        for t in r.trades:
                            trade_rows.append({
                                "Entry": t.entry_date,
                                "Exit": t.exit_date,
                                "Signal": t.signal_type,
                                "Days": t.duration_days,
                                "Buy @": f"${t.entry_price:.2f}",
                                "Sell @": f"${t.exit_price:.2f}",
                                "Shares": f"{t.shares:,.0f}",
                                "Deployed": f"${t.cash_deployed:,.0f}",
                                "Return": f"{t.return_pct:+.1f}%",
                                "Portfolio After": f"${t.portfolio_after:,.0f}",
                                "Cash After": f"${t.cash_after:,.0f}",
                                "Result": t.outcome,
                            })
                        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

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
        <div style="border-top: 1px solid #38444D; margin-top: 40px; padding: 24px 0;
                     text-align: center;">
            <a href="{TWITTER_URL}" target="_blank" style="text-decoration: none;">
                <img src="{LOGO_URL}" alt="MrZzz"
                     style="width: 32px; height: 32px; border-radius: 50%;
                            border: 1px solid #38444D; vertical-align: middle;">
                <span style="color: #1DA1F2; font-weight: 600; margin-left: 8px;
                              vertical-align: middle; font-size: 0.9em;">
                    @MrZzz</span>
            </a>
            <p style="color: #657786; font-size: 0.78em; margin-top: 10px;">
                Data: Yahoo Finance (delayed) &nbsp;&#183;&nbsp;
                Not financial advice &nbsp;&#183;&nbsp;
                For educational purposes only
            </p>
        </div>
    """, unsafe_allow_html=True)
