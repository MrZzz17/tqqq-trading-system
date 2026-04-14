"""
Plotly chart builders for the TQQQ dashboard.
Twitter-dark color scheme: #15202B background, #1DA1F2 accent.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

from core.swing_tracker import SwingPoint

BG_COLOR = "rgba(10,15,26,1)"
GRID_COLOR = "rgba(255,255,255,0.04)"
GREEN = "#34d399"
RED = "#f87171"
BLUE = "#818cf8"


def build_tqqq_chart(
    df: pd.DataFrame,
    swings: Optional[List[SwingPoint]] = None,
    lookback_days: int = 120,
) -> go.Figure:
    # Use ALL data so zooming out reveals more history
    plot_df = df

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
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        increasing_fillcolor=GREEN,
        decreasing_fillcolor=RED,
    ), row=1, col=1)

    ma_configs = [
        ("SMA_10", "10d MA", "#fbbf24", 1),
        ("EMA_21", "21d EMA", BLUE, 1.5),
        ("SMA_50", "50d MA", "#c084fc", 2),
        ("SMA_200", "200d MA", "#fb923c", 2),
    ]
    for col_name, label, color, width in ma_configs:
        if col_name in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df.index,
                y=plot_df[col_name],
                name=label,
                line=dict(color=color, width=width),
                opacity=0.85,
            ), row=1, col=1)

    colors = [
        GREEN if plot_df.iloc[i]["Close"] >= plot_df.iloc[i]["Open"] else RED
        for i in range(len(plot_df))
    ]
    fig.add_trace(go.Bar(
        x=plot_df.index,
        y=plot_df["Volume"],
        name="Volume",
        marker_color=colors,
        opacity=0.3,
    ), row=2, col=1)

    if "Vol_SMA_50" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df["Vol_SMA_50"],
            name="50d Avg Vol",
            line=dict(color="#fbbf24", width=1, dash="dot"),
        ), row=2, col=1)

    if swings:
        for s in swings:
            if s.date >= plot_df.index[0] and s.date <= plot_df.index[-1]:
                color = RED if s.point_type == "peak" else GREEN
                symbol = "triangle-down" if s.point_type == "peak" else "triangle-up"
                fig.add_trace(go.Scatter(
                    x=[s.date],
                    y=[s.price],
                    mode="markers",
                    marker=dict(size=12, color=color, symbol=symbol,
                                line=dict(width=1, color="#f0f0f0")),
                    name=f"{'Peak' if s.point_type == 'peak' else 'Trough'} ${s.price:.2f}",
                    showlegend=False,
                    hovertext=f"{s.point_type.title()}: ${s.price:.2f}<br>{s.pct_move:+.1f}% | {s.trading_days}d",
                ), row=1, col=1)

    # Set default visible range to lookback_days, but allow zooming to see all data
    if len(plot_df) > lookback_days:
        x_start = plot_df.index[-lookback_days]
        x_end = plot_df.index[-1]
    else:
        x_start = plot_df.index[0]
        x_end = plot_df.index[-1]

    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        xaxis=dict(range=[x_start, x_end]),
        yaxis=dict(autorange=True, fixedrange=False),
        yaxis2=dict(autorange=True, fixedrange=False),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#9ca3af", size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG_COLOR,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, autorange=True)

    return fig


def build_distribution_chart(nasdaq_df: pd.DataFrame, dist_days: list, lookback: int = 60) -> go.Figure:
    plot_df = nasdaq_df.iloc[-lookback:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["Close"],
        name="Nasdaq",
        line=dict(color=BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(29,161,242,0.08)",
    ))

    dist_in_range = [d for d in dist_days if d.date >= plot_df.index[0]]
    if dist_in_range:
        fig.add_trace(go.Scatter(
            x=[d.date for d in dist_in_range],
            y=[float(plot_df.loc[d.date, "Close"]) for d in dist_in_range if d.date in plot_df.index],
            mode="markers",
            marker=dict(size=10, color=RED, symbol="x", line=dict(width=2, color=RED)),
            name="Distribution Day",
        ))

    fig.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#9ca3af", size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=BG_COLOR,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig
