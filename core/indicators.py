"""
Distribution day counter and market regime detection.
Implements IBD-style distribution day logic used in Vibha's system.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

import config


@dataclass
class DistributionDay:
    date: pd.Timestamp
    pct_change: float
    volume_vs_prior: float  # ratio: today's vol / yesterday's vol


def count_distribution_days(df: pd.DataFrame, window: int = config.DISTRIBUTION_DAY_WINDOW) -> List[DistributionDay]:
    """
    Count distribution days in the last `window` trading sessions.
    A distribution day = index declines >0.2% on volume higher than the prior session.
    Expired: distribution days older than 25 sessions or where the index has risen
    5% from the distribution day's close are removed (IBD convention).
    """
    if len(df) < window + 1:
        return []

    recent = df.iloc[-(window + 1):].copy()
    recent["Pct_Change"] = recent["Close"].pct_change() * 100
    current_close = float(df["Close"].iloc[-1])

    dist_days = []
    for i in range(1, len(recent)):
        row = recent.iloc[i]
        prev = recent.iloc[i - 1]
        pct = row["Pct_Change"]
        vol_ratio = row["Volume"] / prev["Volume"] if prev["Volume"] > 0 else 0

        if pct <= config.DISTRIBUTION_DAY_DECLINE and vol_ratio > 1.0:
            close_on_dist_day = row["Close"]
            rally_since = ((current_close - close_on_dist_day) / close_on_dist_day) * 100
            if rally_since < 5.0:
                dist_days.append(DistributionDay(
                    date=recent.index[i],
                    pct_change=round(pct, 2),
                    volume_vs_prior=round(vol_ratio, 2),
                ))

    return dist_days


@dataclass
class MarketRegime:
    status: str          # "Confirmed Uptrend", "Uptrend Under Pressure", "Market in Correction"
    color: str           # green, yellow, red
    dist_day_count: int
    description: str


def detect_market_regime(df: pd.DataFrame) -> MarketRegime:
    """
    Approximate IBD Market Pulse using distribution day count
    and price relative to key MAs.
    """
    dist_days = count_distribution_days(df)
    count = len(dist_days)
    current = float(df["Close"].iloc[-1])

    has_sma50 = "SMA_50" in df.columns and not pd.isna(df["SMA_50"].iloc[-1])
    above_50 = current > float(df["SMA_50"].iloc[-1]) if has_sma50 else True

    has_sma200 = "SMA_200" in df.columns and not pd.isna(df["SMA_200"].iloc[-1])
    above_200 = current > float(df["SMA_200"].iloc[-1]) if has_sma200 else True

    if count >= config.DISTRIBUTION_DAY_CRITICAL or not above_200:
        return MarketRegime(
            status="Market in Correction",
            color="red",
            dist_day_count=count,
            description=f"{count} distribution days. High institutional selling pressure.",
        )
    elif count >= config.DISTRIBUTION_DAY_WARN or not above_50:
        return MarketRegime(
            status="Uptrend Under Pressure",
            color="yellow",
            dist_day_count=count,
            description=f"{count} distribution days. Caution -- reduce exposure.",
        )
    else:
        return MarketRegime(
            status="Confirmed Uptrend",
            color="green",
            dist_day_count=count,
            description=f"{count} distribution days. Green light for new buys.",
        )
