"""
Plotly chart builders for the TQQQ dashboard.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

from core.swing_tracker import SwingPoint


def build_tqqq_chart(
    df: pd.DataFrame,
    swings: Optional[List[SwingPoint]] = None,
    lookback_days: int = 120,
) -> go.Figure:
    """
    Build the main TQQQ candlestick chart with MAs and volume.
    """
    plot_df = df.iloc[-lookback_days:] if len(df) > lookback_days else df

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"],
        high=plot_df["High"],
        low=plot_df["Low"],
        close=plot_df["Close"],
        name="TQQQ",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    ma_configs = [
        ("SMA_10", "10-day MA", "#FF9800", 1),
        ("EMA_21", "21-day EMA", "#2196F3", 1.5),
        ("SMA_50", "50-day MA", "#E040FB", 2),
        ("SMA_200", "200-day MA", "#F44336", 2),
    ]
    for col_name, label, color, width in ma_configs:
        if col_name in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df[col_name],
                name=label,
                line=dict(color=color, width=width),
                opacity=0.8,
            ), row=1, col=1)

    colors = [
        "#26a69a" if plot_df.iloc[i]["Close"] >= plot_df.iloc[i]["Open"] else "#ef5350"
        for i in range(len(plot_df))
    ]
    fig.add_trace(go.Bar(
        x=plot_df.index,
        y=plot_df["Volume"],
        name="Volume",
        marker_color=colors,
        opacity=0.5,
    ), row=2, col=1)

    if "Vol_SMA_50" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df["Vol_SMA_50"],
            name="50-day Avg Vol",
            line=dict(color="#FFC107", width=1, dash="dot"),
        ), row=2, col=1)

    if swings:
        for s in swings:
            if s.date in plot_df.index or (s.date >= plot_df.index[0] and s.date <= plot_df.index[-1]):
                color = "#FF5722" if s.point_type == "peak" else "#4CAF50"
                symbol = "triangle-down" if s.point_type == "peak" else "triangle-up"
                fig.add_trace(go.Scatter(
                    x=[s.date],
                    y=[s.price],
                    mode="markers",
                    marker=dict(size=12, color=color, symbol=symbol, line=dict(width=1, color="white")),
                    name=f"{'Peak' if s.point_type == 'peak' else 'Trough'} ${s.price:.2f}",
                    showlegend=False,
                    hovertext=f"{s.point_type.title()}: ${s.price:.2f}<br>{s.pct_move:+.1f}% | {s.trading_days}d",
                ), row=1, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,1)",
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")

    return fig


def build_distribution_chart(nasdaq_df: pd.DataFrame, dist_days: list, lookback: int = 60) -> go.Figure:
    """Build a mini chart showing distribution days on the Nasdaq."""
    plot_df = nasdaq_df.iloc[-lookback:]
    dist_dates = {d.date for d in dist_days}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["Close"],
        name="Nasdaq",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.1)",
    ))

    dist_in_range = [d for d in dist_days if d.date >= plot_df.index[0]]
    if dist_in_range:
        fig.add_trace(go.Scatter(
            x=[d.date for d in dist_in_range],
            y=[float(plot_df.loc[d.date, "Close"]) for d in dist_in_range if d.date in plot_df.index],
            mode="markers",
            marker=dict(size=10, color="#F44336", symbol="x", line=dict(width=2, color="#F44336")),
            name="Distribution Day",
        ))

    fig.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,1)",
    )
    return fig
