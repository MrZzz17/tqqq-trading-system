"""
Plotly chart builders for the TQQQ dashboard.
Twitter-dark color scheme: #15202B background, #1DA1F2 accent.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG_COLOR = "rgba(10,15,26,1)"
GRID_COLOR = "rgba(255,255,255,0.04)"
GREEN = "#34d399"
RED = "#f87171"
BLUE = "#818cf8"
QQQ_CANDLE_UP = "#38bdf8"
QQQ_CANDLE_DOWN = "#0ea5e9"


def _rangebreaks_kw():
    return dict(bounds=["sat", "mon"])


def _align_qqq_to_tqqq(tqqq_df: pd.DataFrame, qqq_df: pd.DataFrame) -> pd.DataFrame:
    """Reindex QQQ to TQQQ trading dates (ffill for rare misaligned holidays)."""
    if qqq_df is None or qqq_df.empty or tqqq_df is None or tqqq_df.empty:
        return pd.DataFrame()
    q = qqq_df.reindex(tqqq_df.index).ffill()
    return q


def build_qqq_tqqq_model_chart(
    tqqq_df: pd.DataFrame,
    qqq_df: Optional[pd.DataFrame] = None,
    trade_markers: Optional[List[Dict[str, Any]]] = None,
    label_mode: str = "price",
    tqqq_yaxis_log: bool = False,
) -> go.Figure:
    """
    QQQ (regime / trend context) + TQQQ (traded vehicle) + TQQQ volume.

    trade_markers: list of dicts with keys ts (Timestamp), price (float),
    kind ('entry'|'exit'), optional signal (str) — backtest / model fills on TQQQ.
    label_mode: "full" (text+signal), "price" (price only; signal on hover), "none" (markers only).
    tqqq_yaxis_log: if True, log scale on TQQQ price panel (useful for very long "All" windows).
    """
    plot_t = tqqq_df if tqqq_df is not None and not tqqq_df.empty else pd.DataFrame()
    if plot_t.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BG_COLOR,
            annotations=[dict(text="No data", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")],
        )
        return fig

    markers = trade_markers or []
    q_aligned = _align_qqq_to_tqqq(plot_t, qqq_df) if qqq_df is not None else pd.DataFrame()
    show_qqq = not q_aligned.empty and q_aligned["Close"].notna().any()

    if show_qqq:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.22, 0.58, 0.20],
        )
        row_qqq, row_tqqq, row_vol = 1, 2, 3
    else:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.78, 0.22],
        )
        row_qqq, row_tqqq, row_vol = None, 1, 2

    # ── QQQ row (context: same MA stack the model uses on QQQ) ──
    if show_qqq:
        fig.add_trace(
            go.Candlestick(
                x=q_aligned.index,
                open=q_aligned["Open"],
                high=q_aligned["High"],
                low=q_aligned["Low"],
                close=q_aligned["Close"],
                name="QQQ",
                increasing_line_color=QQQ_CANDLE_UP,
                decreasing_line_color=QQQ_CANDLE_DOWN,
                increasing_fillcolor=QQQ_CANDLE_UP,
                decreasing_fillcolor=QQQ_CANDLE_DOWN,
            ),
            row=row_qqq,
            col=1,
        )
        for col_name, label, color, width in [
            ("EMA_21", "QQQ 21d EMA", "#93c5fd", 1),
            ("SMA_50", "QQQ 50d MA", "#c084fc", 1.5),
            ("SMA_200", "QQQ 200d MA", "#fb923c", 1.5),
        ]:
            if col_name in q_aligned.columns and q_aligned[col_name].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=q_aligned.index,
                        y=q_aligned[col_name],
                        name=label,
                        line=dict(color=color, width=width),
                        opacity=0.9,
                    ),
                    row=row_qqq,
                    col=1,
                )

    # ── TQQQ row (execution + full MA ribbon) ──
    fig.add_trace(
        go.Candlestick(
            x=plot_t.index,
            open=plot_t["Open"],
            high=plot_t["High"],
            low=plot_t["Low"],
            close=plot_t["Close"],
            name="TQQQ",
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            increasing_fillcolor=GREEN,
            decreasing_fillcolor=RED,
        ),
        row=row_tqqq,
        col=1,
    )

    ma_configs = [
        ("SMA_10", "TQQQ 10d MA", "#fbbf24", 1),
        ("EMA_21", "TQQQ 21d EMA", BLUE, 1.5),
        ("SMA_50", "TQQQ 50d MA", "#c084fc", 2),
        ("SMA_200", "TQQQ 200d MA", "#fb923c", 2),
    ]
    for col_name, label, color, width in ma_configs:
        if col_name in plot_t.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_t.index,
                    y=plot_t[col_name],
                    name=label,
                    line=dict(color=color, width=width),
                    opacity=0.85,
                ),
                row=row_tqqq,
                col=1,
            )

    # Model entry / exit on TQQQ (backtest + open leg passed in from dashboard)
    ent_x, ent_y, ent_txt, ent_hint = [], [], [], []
    ex_x, ex_y, ex_txt, ex_hint = [], [], [], []
    for m in markers:
        ts = m.get("ts")
        if ts is None:
            continue
        ts = pd.Timestamp(ts)
        if ts < plot_t.index[0] or ts > plot_t.index[-1]:
            continue
        price = float(m.get("price") or 0)
        if price <= 0:
            continue
        sig = (m.get("signal") or "").strip()
        kind = (m.get("kind") or "").lower()
        short_buy = f"Buy ${price:.2f}"
        short_sell = f"Sell ${price:.2f}"
        if kind == "entry":
            ent_x.append(ts)
            ent_y.append(price)
            if label_mode == "none":
                ent_txt.append("")
            elif label_mode == "full":
                ent_txt.append(short_buy + (f"<br>({sig})" if sig else ""))
            else:  # price
                ent_txt.append(short_buy)
            ent_hint.append(sig or "—")
        elif kind == "exit":
            ex_x.append(ts)
            ex_y.append(price)
            if label_mode == "none":
                ex_txt.append("")
            elif label_mode == "full":
                ex_txt.append(short_sell + (f"<br>({sig})" if sig else ""))
            else:
                ex_txt.append(short_sell)
            ex_hint.append(sig or "—")

    _ent_mode = "markers+text" if label_mode != "none" else "markers"
    _ex_mode = "markers+text" if label_mode != "none" else "markers"
    _ht_ent = (
        "Entry $%{y:,.2f}<br>Signal: %{customdata}<extra></extra>" if label_mode == "none"
        else (
            "%{text}<extra></extra>" if label_mode == "full"
            else "%{text}<br>Signal: %{customdata}<extra></extra>"
        )
    )
    _ht_ex = (
        "Exit $%{y:,.2f}<br>Signal: %{customdata}<extra></extra>" if label_mode == "none"
        else (
            "%{text}<extra></extra>" if label_mode == "full"
            else "%{text}<br>Signal: %{customdata}<extra></extra>"
        )
    )
    if ent_x:
        fig.add_trace(
            go.Scatter(
                x=ent_x,
                y=ent_y,
                mode=_ent_mode,
                name="Model entry",
                legendgroup="marks",
                marker=dict(size=11, color=GREEN, symbol="triangle-up", line=dict(width=1, color="#ecfdf5")),
                text=ent_txt if label_mode != "none" else None,
                textposition="top center",
                textfont=dict(size=10, color="#a7f3d0"),
                customdata=ent_hint,
                cliponaxis=False,
                hovertemplate=_ht_ent,
            ),
            row=row_tqqq,
            col=1,
        )
    if ex_x:
        fig.add_trace(
            go.Scatter(
                x=ex_x,
                y=ex_y,
                mode=_ex_mode,
                name="Model exit",
                legendgroup="marks",
                marker=dict(size=11, color=RED, symbol="triangle-down", line=dict(width=1, color="#fee2e2")),
                text=ex_txt if label_mode != "none" else None,
                textposition="bottom center",
                textfont=dict(size=10, color="#fecaca"),
                customdata=ex_hint,
                cliponaxis=False,
                hovertemplate=_ht_ex,
            ),
            row=row_tqqq,
            col=1,
        )

    # ── Volume: TQQQ only, bar height = daily volume (no secondary fake “strip”) ──
    vol_colors = [
        GREEN if plot_t.iloc[i]["Close"] >= plot_t.iloc[i]["Open"] else RED for i in range(len(plot_t))
    ]
    fig.add_trace(
        go.Bar(
            x=plot_t.index,
            y=plot_t["Volume"],
            name="TQQQ volume",
            marker_color=vol_colors,
            marker_line_width=0,
            opacity=0.72,
            showlegend=True,
        ),
        row=row_vol,
        col=1,
    )
    if "Vol_SMA_50" in plot_t.columns and plot_t["Vol_SMA_50"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=plot_t.index,
                y=plot_t["Vol_SMA_50"],
                name="50d avg vol",
                line=dict(color="#fbbf24", width=1.2, dash="dot"),
            ),
            row=row_vol,
            col=1,
        )

    x_start = plot_t.index[0]
    x_end = plot_t.index[-1]

    fig.update_layout(
        template="plotly_dark",
        height=560 if show_qqq else 480,
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#9ca3af", size=10),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,26,1)",
    )
    # Candlestick rangeslider draws a misleading flat strip under the chart — disable on every pane
    fig.update_xaxes(
        rangeslider_visible=False,
        range=[x_start, x_end],
        showgrid=False,
        gridcolor="rgba(255,255,255,0.03)",
        fixedrange=True,
        zeroline=False,
        rangebreaks=[_rangebreaks_kw()],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.04)",
        zeroline=False,
        fixedrange=True,
        type="log" if tqqq_yaxis_log else "linear",
        title="TQQQ (log)" if tqqq_yaxis_log else None,
        row=row_tqqq,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.04)",
        zeroline=False,
        fixedrange=True,
        rangemode="tozero",
        row=row_vol,
        col=1,
    )
    if show_qqq:
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            fixedrange=True,
            row=row_qqq,
            col=1,
        )

    return fig
