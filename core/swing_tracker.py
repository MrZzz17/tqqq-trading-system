"""
TQQQ Swing Tracker -- replicates Vibha's spreadsheet.
Automatically detects peaks and troughs, calculates % moves and durations.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

import config


@dataclass
class SwingPoint:
    date: pd.Timestamp
    price: float
    point_type: str      # "peak" or "trough"
    pct_move: float      # % change from prior swing point
    trading_days: int    # days since prior swing point
    vs_sma_50: str       # "above" or "below"
    vs_ema_21: str       # "above" or "below"


def detect_swings(
    df: pd.DataFrame,
    min_move_pct: float = config.SWING_MIN_MOVE_PCT,
    year_filter: Optional[int] = None,
) -> List[SwingPoint]:
    """
    Detect peaks and troughs using a zig-zag algorithm with a minimum
    percentage threshold. Mirrors Vibha's manual tracking process.
    """
    if len(df) < 10:
        return []

    if year_filter:
        df = df[df.index.year >= year_filter]
        if len(df) < 5:
            return []

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    dates = df.index

    swings: List[SwingPoint] = []

    direction = 0  # 1 = looking for peak, -1 = looking for trough
    last_high_idx = 0
    last_low_idx = 0
    last_high = highs[0]
    last_low = lows[0]

    for i in range(1, len(df)):
        if highs[i] > last_high:
            last_high = highs[i]
            last_high_idx = i

        if lows[i] < last_low:
            last_low = lows[i]
            last_low_idx = i

        if direction <= 0:
            decline_from_high = ((lows[i] - last_high) / last_high) * 100
            if decline_from_high <= -min_move_pct and last_high_idx < i:
                sma50_val = df.iloc[last_high_idx].get("SMA_50")
                ema21_val = df.iloc[last_high_idx].get("EMA_21")
                peak = SwingPoint(
                    date=dates[last_high_idx],
                    price=round(float(last_high), 2),
                    point_type="peak",
                    pct_move=0.0,
                    trading_days=0,
                    vs_sma_50="above" if (sma50_val and not pd.isna(sma50_val) and last_high > sma50_val) else "below",
                    vs_ema_21="above" if (ema21_val and not pd.isna(ema21_val) and last_high > ema21_val) else "below",
                )
                if swings:
                    prev = swings[-1]
                    peak.pct_move = round(((peak.price - prev.price) / prev.price) * 100, 2)
                    trading_days = len(df.loc[prev.date:peak.date]) - 1
                    peak.trading_days = max(trading_days, 0)
                swings.append(peak)
                last_low = lows[i]
                last_low_idx = i
                direction = 1

        if direction >= 0:
            rally_from_low = ((highs[i] - last_low) / last_low) * 100
            if rally_from_low >= min_move_pct and last_low_idx < i:
                sma50_val = df.iloc[last_low_idx].get("SMA_50")
                ema21_val = df.iloc[last_low_idx].get("EMA_21")
                trough = SwingPoint(
                    date=dates[last_low_idx],
                    price=round(float(last_low), 2),
                    point_type="trough",
                    pct_move=0.0,
                    trading_days=0,
                    vs_sma_50="above" if (sma50_val and not pd.isna(sma50_val) and last_low > sma50_val) else "below",
                    vs_ema_21="above" if (ema21_val and not pd.isna(ema21_val) and last_low > ema21_val) else "below",
                )
                if swings:
                    prev = swings[-1]
                    trough.pct_move = round(((trough.price - prev.price) / prev.price) * 100, 2)
                    trading_days = len(df.loc[prev.date:trough.date]) - 1
                    trough.trading_days = max(trading_days, 0)
                swings.append(trough)
                last_high = highs[i]
                last_high_idx = i
                direction = -1

    return swings


def current_swing_stats(df: pd.DataFrame, swings: List[SwingPoint]) -> dict:
    """Compute stats about the current (in-progress) swing."""
    if not swings:
        return {"status": "Insufficient data"}

    last_swing = swings[-1]
    current_price = float(df["Close"].iloc[-1])
    current_date = df.index[-1]

    pct_from_last = round(((current_price - last_swing.price) / last_swing.price) * 100, 2)
    days_since = len(df.loc[last_swing.date:current_date]) - 1

    if last_swing.point_type == "trough":
        direction = "up"
        label = f"Currently rallying +{pct_from_last:.1f}% off the {last_swing.date.strftime('%b %d')} low (${last_swing.price:.2f}) over {days_since} trading days."
    else:
        direction = "down"
        label = f"Currently pulling back {pct_from_last:.1f}% from the {last_swing.date.strftime('%b %d')} high (${last_swing.price:.2f}) over {days_since} trading days."

    return {
        "direction": direction,
        "pct_from_last_swing": pct_from_last,
        "days_since_last_swing": days_since,
        "last_swing_type": last_swing.point_type,
        "last_swing_date": last_swing.date,
        "last_swing_price": last_swing.price,
        "current_price": current_price,
        "label": label,
    }


def swing_summary_stats(swings: List[SwingPoint]) -> dict:
    """Aggregate statistics across all detected swings."""
    if len(swings) < 2:
        return {}

    up_moves = [s.pct_move for s in swings if s.pct_move > 0]
    down_moves = [s.pct_move for s in swings if s.pct_move < 0]
    durations = [s.trading_days for s in swings if s.trading_days > 0]

    return {
        "total_swings": len(swings),
        "avg_up_move": round(np.mean(up_moves), 1) if up_moves else 0,
        "avg_down_move": round(np.mean(down_moves), 1) if down_moves else 0,
        "max_up_move": round(max(up_moves), 1) if up_moves else 0,
        "max_down_move": round(min(down_moves), 1) if down_moves else 0,
        "avg_duration_days": round(np.mean(durations), 0) if durations else 0,
    }
