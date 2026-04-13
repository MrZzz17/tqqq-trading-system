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
import config


SEVERITY_ICONS = {"watch": "👀", "warning": "⚠️", "sell": "🔴"}
REGIME_ICONS = {"green": "🟢", "yellow": "🟡", "red": "🔴"}


def render():
    st.title("TQQQ Swing Trading Dashboard")
    st.caption("Rules-based swing trading system | CAN SLIM + leveraged ETF strategy")

    with st.spinner("Fetching market data..."):
        tqqq = get_tqqq_data()
        qqq = get_qqq_data()
        nasdaq = get_nasdaq_data()
        sp500 = get_sp500_data()

    if tqqq.empty or nasdaq.empty:
        st.error("Unable to fetch market data. Please check your internet connection and try again.")
        return

    data_date = get_latest_date(tqqq)
    tqqq_price = get_current_price(tqqq)
    qqq_price = get_current_price(qqq)

    # ── Sidebar controls ──
    st.sidebar.markdown("### Settings")
    chart_lookback = st.sidebar.slider("Chart lookback (days)", 30, 365, 120)
    swing_min_pct = st.sidebar.slider("Swing min % move", 3.0, 15.0, 5.0, 0.5)
    bulls_pct = st.sidebar.number_input(
        "Bulls % (AAII/0BULL -- enter manually)",
        min_value=0.0, max_value=100.0, value=0.0, step=1.0,
        help="Enter the latest AAII bullish sentiment %. Leave 0 if unknown.",
    )
    bulls_input = bulls_pct if bulls_pct > 0 else None

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # ── Market Status Bar ──
    st.markdown("---")
    nasdaq_regime = detect_market_regime(nasdaq)
    sp_regime = detect_market_regime(sp500)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TQQQ", f"${tqqq_price:.2f}",
                   delta=f"{((tqqq.iloc[-1]['Close'] - tqqq.iloc[-2]['Close']) / tqqq.iloc[-2]['Close'] * 100):.2f}%")
    with col2:
        st.metric("QQQ", f"${qqq_price:.2f}",
                   delta=f"{((qqq.iloc[-1]['Close'] - qqq.iloc[-2]['Close']) / qqq.iloc[-2]['Close'] * 100):.2f}%")
    with col3:
        icon = REGIME_ICONS.get(nasdaq_regime.color, "")
        st.metric("Nasdaq Status", f"{icon} {nasdaq_regime.status}")
        st.caption(f"{nasdaq_regime.dist_day_count} distribution days")
    with col4:
        icon_sp = REGIME_ICONS.get(sp_regime.color, "")
        st.metric("S&P 500 Status", f"{icon_sp} {sp_regime.status}")
        st.caption(f"{sp_regime.dist_day_count} distribution days")

    st.caption(f"Data as of {data_date.strftime('%B %d, %Y')}")

    # ── Alert Level ──
    sell_signals = check_all_sell_signals(tqqq, nasdaq, bulls_input)
    alert_level, alert_color, alert_action = compute_alert_level(sell_signals)
    triggered_count = sum(1 for s in sell_signals if s.triggered)

    alert_color_map = {"green": "#4CAF50", "yellow": "#FFC107", "orange": "#FF9800", "red": "#F44336"}
    hex_color = alert_color_map.get(alert_color, "#FFC107")
    st.markdown(
        f"""<div style="
            background: linear-gradient(135deg, {hex_color}22, {hex_color}11);
            border-left: 4px solid {hex_color};
            padding: 16px 20px; border-radius: 8px; margin: 16px 0;">
            <span style="font-size: 1.4em; font-weight: 700; color: {hex_color};">
                ALERT: {alert_level}</span>
            <span style="font-size: 1.0em; color: #ccc; margin-left: 16px;">
                {triggered_count}/9 sell signals active</span>
            <br><span style="color: #eee;">{alert_action}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Buy Signals ──
    st.markdown("## Buy Signals")
    buy_col1, buy_col2 = st.columns(2)

    ftd = detect_follow_through_day(nasdaq)
    with buy_col1:
        st.markdown("### Follow-Through Day")
        if ftd:
            st.success(f"**{ftd.strength} FTD detected** on {ftd.date.strftime('%b %d, %Y')}")
            st.markdown(ftd.details)
            st.markdown(f"Suggested TQQQ position: **{ftd.suggested_size:.0%}**")
        else:
            st.info("No Follow-Through Day detected in the last 30 sessions.")

    wk = detect_three_white_knights(qqq)
    with buy_col2:
        st.markdown("### 3 White Knights")
        if wk:
            st.success(f"**{wk.strength} signal** on {wk.date.strftime('%b %d, %Y')}")
            st.markdown(wk.details)
            st.markdown(f"Suggested TQQQ position: **{wk.suggested_size:.0%}**")
        else:
            st.info("No 3 White Knights pattern detected in the last 10 sessions.")

    # ── Sell Signal Scoreboard ──
    st.markdown("## Sell Signal Scoreboard")
    st.caption("Sell TQQQ in 10% chunks as signals stack up. More signals = more aggressive trimming.")

    sell_cols = st.columns(3)
    for i, signal in enumerate(sell_signals):
        col = sell_cols[i % 3]
        with col:
            icon = SEVERITY_ICONS.get(signal.severity, "")
            status_color = "#F44336" if signal.triggered else "#4CAF5066"
            border = f"2px solid {status_color}"
            bg = f"{status_color}15" if signal.triggered else "transparent"

            st.markdown(
                f"""<div style="border: {border}; background: {bg};
                    border-radius: 8px; padding: 12px; margin-bottom: 8px; min-height: 120px;">
                    <div style="font-weight: 600; font-size: 0.9em;">
                        {icon} Rule {signal.rule_number}: {signal.name}
                    </div>
                    <div style="font-size: 0.8em; color: #aaa; margin-top: 4px;">
                        {signal.details}
                    </div>
                    <div style="margin-top: 8px;">
                        <span style="background: {'#F44336' if signal.triggered else '#4CAF50'};
                            color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75em;">
                            {'TRIGGERED' if signal.triggered else 'CLEAR'}
                        </span>
                        <span style="color: #888; font-size: 0.75em; margin-left: 8px;">
                            [{signal.severity.upper()}]
                        </span>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── TQQQ Chart ──
    st.markdown("## TQQQ Chart")

    year_now = dt.datetime.now().year
    swings = detect_swings(tqqq, min_move_pct=swing_min_pct, year_filter=year_now - 1)
    fig = build_tqqq_chart(tqqq, swings=swings, lookback_days=chart_lookback)
    st.plotly_chart(fig, width="stretch")

    # ── MA Position Summary ──
    latest = tqqq.iloc[-1]
    ma_data = []
    for ma_col, label in [("SMA_10", "10-day SMA"), ("EMA_21", "21-day EMA"), ("SMA_50", "50-day SMA"), ("SMA_200", "200-day SMA")]:
        if ma_col in tqqq.columns and not pd.isna(latest[ma_col]):
            ma_val = float(latest[ma_col])
            pct_from = ((tqqq_price - ma_val) / ma_val) * 100
            position = "Above" if tqqq_price > ma_val else "Below"
            ma_data.append({"Moving Average": label, "Value": f"${ma_val:.2f}", "TQQQ Position": position, "Distance": f"{pct_from:+.2f}%"})

    if ma_data:
        st.markdown("### Price vs Moving Averages")
        ma_df = pd.DataFrame(ma_data)
        st.dataframe(ma_df, width="stretch", hide_index=True)

    # ── Swing Tracker ──
    st.markdown("## Swing Tracker")
    st.caption("Automated peak/trough detection -- % moves, durations, and MA relationships.")

    current_year_swings = detect_swings(tqqq, min_move_pct=swing_min_pct, year_filter=year_now)
    all_swings = detect_swings(tqqq, min_move_pct=swing_min_pct)

    current = current_swing_stats(tqqq, all_swings)
    if "label" in current:
        direction_icon = "📈" if current.get("direction") == "up" else "📉"
        st.info(f"{direction_icon} {current['label']}")

    summary = swing_summary_stats(all_swings)
    if summary:
        scol1, scol2, scol3, scol4, scol5 = st.columns(5)
        scol1.metric("Total Swings", summary["total_swings"])
        scol2.metric("Avg Up Move", f"{summary['avg_up_move']:+.1f}%")
        scol3.metric("Avg Down Move", f"{summary['avg_down_move']:.1f}%")
        scol4.metric("Max Up Move", f"{summary['max_up_move']:+.1f}%")
        scol5.metric("Avg Duration", f"{summary['avg_duration_days']:.0f} days")

    tab_ytd, tab_all = st.tabs([f"{year_now} Swings", "All Swings (2yr)"])

    with tab_ytd:
        if current_year_swings:
            swing_rows = [{
                "Date": s.date.strftime("%Y-%m-%d"),
                "Type": s.point_type.title(),
                "Price": f"${s.price:.2f}",
                "% Move": f"{s.pct_move:+.1f}%",
                "Days": s.trading_days,
                "vs 50d MA": s.vs_sma_50.title(),
                "vs 21d EMA": s.vs_ema_21.title(),
            } for s in current_year_swings]
            st.dataframe(pd.DataFrame(swing_rows), width="stretch", hide_index=True)
        else:
            st.info(f"No swings detected in {year_now} with the current sensitivity setting.")

    with tab_all:
        if all_swings:
            swing_rows = [{
                "Date": s.date.strftime("%Y-%m-%d"),
                "Type": s.point_type.title(),
                "Price": f"${s.price:.2f}",
                "% Move": f"{s.pct_move:+.1f}%",
                "Days": s.trading_days,
                "vs 50d MA": s.vs_sma_50.title(),
                "vs 21d EMA": s.vs_ema_21.title(),
            } for s in all_swings]
            st.dataframe(pd.DataFrame(swing_rows), width="stretch", hide_index=True)
        else:
            st.info("No swings detected with the current sensitivity setting.")

    # ── Distribution Days Detail ──
    st.markdown("## Distribution Days Detail")

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.markdown("### Nasdaq Composite")
        nasdaq_dist = count_distribution_days(nasdaq)
        fig_nasdaq = build_distribution_chart(nasdaq, nasdaq_dist)
        st.plotly_chart(fig_nasdaq, width="stretch")
        if nasdaq_dist:
            dist_rows = [{
                "Date": d.date.strftime("%Y-%m-%d"),
                "Decline": f"{d.pct_change:.2f}%",
                "Vol vs Prior": f"{d.volume_vs_prior:.2f}x",
            } for d in nasdaq_dist]
            st.dataframe(pd.DataFrame(dist_rows), width="stretch", hide_index=True)
        else:
            st.success("No distribution days in the last 25 sessions.")

    with dist_col2:
        st.markdown("### S&P 500")
        sp_dist = count_distribution_days(sp500)
        fig_sp = build_distribution_chart(sp500, sp_dist)
        st.plotly_chart(fig_sp, width="stretch")
        if sp_dist:
            dist_rows = [{
                "Date": d.date.strftime("%Y-%m-%d"),
                "Decline": f"{d.pct_change:.2f}%",
                "Vol vs Prior": f"{d.volume_vs_prior:.2f}x",
            } for d in sp_dist]
            st.dataframe(pd.DataFrame(dist_rows), width="stretch", hide_index=True)
        else:
            st.success("No distribution days in the last 25 sessions.")

    # ── Quick Reference ──
    with st.expander("📋 TQQQ Rules -- Quick Reference"):
        st.markdown("""
**BUY RULES (only 2):**
1. **Follow-Through Day** -- All-in on TQQQ. Stop if price undercuts the rally day low.
2. **3 White Knights** -- 3 consecutive higher highs & higher lows on QQQ. Enter 25-50%.

**SELL RULES (sell in 10% chunks as signals stack):**
1. New 52-week high -- watch closely
2. New high on declining volume
3. 4-5 distribution days on Nasdaq
4. 3 consecutive down days
5. 10-day MA violated on rising volume
6. 3 down days + rising volume + lower highs/lows
7. Triple rejection at resistance
8. Bulls vs Bears >60% (secondary)
9. **2 closes below 21-day EMA -- FULL EXIT**

**POSITION SIZING:**
- FTD: Up to 100% in IRA
- 3WK without FTD: 25-50-75% based on conviction
- Sell down TQQQ to fund individual stock positions as setups emerge

**KEY INSIGHT:** TQQQ typically delivers 20-30% swings multiple times per year.
""")

    st.markdown("---")
    st.caption("Data sourced from Yahoo Finance. This tool is for educational purposes only and does not constitute financial advice.")
