"""
TQQQ buy and sell signal detection.
Implements the 2 buy rules and 9 sell rules checklist.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config
from core.indicators import count_distribution_days


# ---------------------------------------------------------------------------
# Buy signals
# ---------------------------------------------------------------------------

@dataclass
class BuySignal:
    signal_type: str   # "FTD" or "3_WHITE_KNIGHTS"
    date: pd.Timestamp
    strength: str      # "Strong", "Moderate", "Weak"
    details: str
    suggested_size: float  # fraction of portfolio


def detect_follow_through_day(nasdaq_df: pd.DataFrame, lookback: int = 30) -> Optional[BuySignal]:
    """
    Detect the most recent Follow-Through Day.
    FTD = After a market decline, on day 4+ of a rally attempt,
    the Nasdaq gains >1.25% on volume higher than the previous day.
    """
    if len(nasdaq_df) < lookback:
        return None

    recent = nasdaq_df.iloc[-lookback:].copy()
    recent["Pct_Change"] = recent["Close"].pct_change() * 100

    trough_idx = recent["Close"].idxmin()
    trough_pos = recent.index.get_loc(trough_idx)

    rally_day = 0
    for i in range(trough_pos + 1, len(recent)):
        row = recent.iloc[i]
        if row["Close"] > recent.iloc[i - 1]["Close"]:
            rally_day += 1
        else:
            rally_day = 0
            continue

        if rally_day < config.FTD_MIN_RALLY_DAY:
            continue

        pct = row["Pct_Change"]
        vol_higher = row["Volume"] > recent.iloc[i - 1]["Volume"]

        if pct >= config.FTD_MIN_GAIN_PCT and vol_higher:
            strength = "Strong" if pct >= 2.0 else "Moderate"
            return BuySignal(
                signal_type="FTD",
                date=recent.index[i],
                strength=strength,
                details=f"Day {rally_day} of rally. Nasdaq gained {pct:.1f}% on higher volume.",
                suggested_size=config.TQQQ_INITIAL_POSITION_FTD,
            )
    return None


def detect_three_white_knights(qqq_df: pd.DataFrame) -> Optional[BuySignal]:
    """
    Detect 3 consecutive days of higher highs AND higher lows on QQQ.
    Uses QQQ (not TQQQ) for this signal.
    """
    if len(qqq_df) < 5:
        return None

    recent = qqq_df.iloc[-10:]
    for end in range(len(recent) - 1, 2, -1):
        days = recent.iloc[end - 2: end + 1]
        higher_highs = all(
            days.iloc[j]["High"] > days.iloc[j - 1]["High"]
            for j in range(1, 3)
        )
        higher_lows = all(
            days.iloc[j]["Low"] > days.iloc[j - 1]["Low"]
            for j in range(1, 3)
        )
        if higher_highs and higher_lows:
            avg_vol = days["Volume"].mean()
            prior_avg_vol = recent.iloc[max(0, end - 5): end - 2]["Volume"].mean()
            vol_strong = avg_vol > prior_avg_vol if prior_avg_vol > 0 else False
            strength = "Strong" if vol_strong else "Moderate"
            return BuySignal(
                signal_type="3_WHITE_KNIGHTS",
                date=days.index[-1],
                strength=strength,
                details=f"3 consecutive higher highs & higher lows on QQQ. Volume {'above' if vol_strong else 'below'} average.",
                suggested_size=config.TQQQ_INITIAL_POSITION_3WK if strength == "Moderate" else 0.50,
            )
    return None


# ---------------------------------------------------------------------------
# Sell signals
# ---------------------------------------------------------------------------

@dataclass
class SellSignal:
    rule_number: int
    name: str
    triggered: bool
    severity: str       # "watch", "warning", "sell"
    details: str


def check_all_sell_signals(
    tqqq_df: pd.DataFrame,
    nasdaq_df: pd.DataFrame,
    bulls_pct: Optional[float] = None,
) -> List[SellSignal]:
    """Run all 9 TQQQ sell rules and return their status."""
    signals = []
    latest = tqqq_df.iloc[-1]
    prev = tqqq_df.iloc[-2] if len(tqqq_df) >= 2 else latest

    high_52w = tqqq_df["High"].iloc[-252:].max() if len(tqqq_df) >= 252 else tqqq_df["High"].max()

    # Rule 1: New 52-week high
    at_52w_high = float(latest["High"]) >= high_52w * 0.995
    signals.append(SellSignal(
        rule_number=1,
        name="New 52-week high",
        triggered=at_52w_high,
        severity="watch",
        details=f"TQQQ high: ${latest['High']:.2f} vs 52w high: ${high_52w:.2f}",
    ))

    # Rule 2: New high on declining volume
    recent_5 = tqqq_df.iloc[-5:]
    making_highs = float(latest["Close"]) > float(recent_5["Close"].iloc[:-1].max())
    vol_declining = float(latest["Volume"]) < float(recent_5["Volume"].iloc[:-1].mean())
    new_high_low_vol = making_highs and vol_declining
    signals.append(SellSignal(
        rule_number=2,
        name="New high on declining volume",
        triggered=new_high_low_vol,
        severity="warning",
        details=f"Price rising with {'declining' if vol_declining else 'rising'} volume vs 5-day avg.",
    ))

    # Rule 3: 4-5 distribution days
    dist_days = count_distribution_days(nasdaq_df)
    dist_count = len(dist_days)
    signals.append(SellSignal(
        rule_number=3,
        name="4-5 distribution days",
        triggered=dist_count >= config.DISTRIBUTION_DAY_WARN,
        severity="warning" if dist_count == 4 else ("sell" if dist_count >= 5 else "watch"),
        details=f"Nasdaq has {dist_count} distribution days in the last 25 sessions.",
    ))

    # Rule 4: 3 consecutive down days
    last_3 = tqqq_df.iloc[-3:]
    three_down = all(
        last_3.iloc[i]["Close"] < last_3.iloc[i - 1]["Close"]
        for i in range(1, 3)
    ) and last_3.iloc[0]["Close"] < tqqq_df.iloc[-4]["Close"]
    signals.append(SellSignal(
        rule_number=4,
        name="3 consecutive down days",
        triggered=three_down,
        severity="warning",
        details="Last 3 closes: " + ", ".join(f"${r['Close']:.2f}" for _, r in last_3.iterrows()),
    ))

    # Rule 5: 10-day MA violation on rising volume
    below_10ma = float(latest["Close"]) < float(latest["SMA_10"]) if not pd.isna(latest.get("SMA_10")) else False
    vol_rising = float(latest["Volume"]) > float(latest.get("Vol_SMA_50", 0))
    ten_ma_violation = below_10ma and vol_rising
    signals.append(SellSignal(
        rule_number=5,
        name="10-day MA violated on rising volume",
        triggered=ten_ma_violation,
        severity="sell",
        details=f"Close ${latest['Close']:.2f} vs 10-day MA ${latest.get('SMA_10', 0):.2f}. Volume {'above' if vol_rising else 'below'} 50-day avg.",
    ))

    # Rule 6: 3 down days + rising volume + lower highs/lows
    if len(tqqq_df) >= 4:
        last_3_rows = tqqq_df.iloc[-3:]
        prev_row = tqqq_df.iloc[-4]
        three_down_vol = (
            three_down
            and all(last_3_rows.iloc[i]["Volume"] > last_3_rows.iloc[i - 1]["Volume"] for i in range(1, 3))
            and all(last_3_rows.iloc[i]["High"] < last_3_rows.iloc[i - 1]["High"] for i in range(1, 3))
            and all(last_3_rows.iloc[i]["Low"] < last_3_rows.iloc[i - 1]["Low"] for i in range(1, 3))
        )
    else:
        three_down_vol = False
    signals.append(SellSignal(
        rule_number=6,
        name="3 down days + rising volume + lower H/L",
        triggered=three_down_vol,
        severity="sell",
        details="Severe weakness pattern: consecutive declines with increasing selling pressure.",
    ))

    # Rule 7: Triple rejection at resistance
    if len(tqqq_df) >= 20:
        recent_20 = tqqq_df.iloc[-20:]
        resistance = recent_20["High"].max()
        resistance_zone = resistance * 0.99  # within 1% of resistance
        touches = (recent_20["High"] >= resistance_zone).sum()
        rejections = touches >= 3 and float(latest["Close"]) < resistance_zone
    else:
        rejections = False
        resistance = 0
    signals.append(SellSignal(
        rule_number=7,
        name="Triple rejection at resistance",
        triggered=rejections,
        severity="warning",
        details=f"Price has tested resistance near ${resistance:.2f} multiple times and been rejected.",
    ))

    # Rule 8: Bulls vs Bears >60%
    bulls_frothy = bulls_pct is not None and bulls_pct > config.BULLS_BEARS_FROTHY
    signals.append(SellSignal(
        rule_number=8,
        name="Bulls vs Bears >60%",
        triggered=bulls_frothy,
        severity="watch",
        details=f"Bulls at {bulls_pct:.0f}%" if bulls_pct is not None else "No sentiment data -- enter manually.",
    ))

    # Rule 9: 2 closes below 21-day EMA (FULL EXIT)
    if len(tqqq_df) >= 3 and "EMA_21" in tqqq_df.columns:
        last_2 = tqqq_df.iloc[-2:]
        two_below_21ema = all(
            float(last_2.iloc[i]["Close"]) < float(last_2.iloc[i]["EMA_21"])
            for i in range(2)
        )
    else:
        two_below_21ema = False
    signals.append(SellSignal(
        rule_number=9,
        name="2 closes below 21-day EMA -- FULL EXIT",
        triggered=two_below_21ema,
        severity="sell",
        details=f"Close ${latest['Close']:.2f} vs 21-EMA ${latest.get('EMA_21', 0):.2f}. This is the nuclear exit signal.",
    ))

    return signals


def compute_alert_level(signals: List[SellSignal]) -> Tuple[str, str, str]:
    """
    Returns (level, color, action) based on triggered sell signals.
    """
    triggered = [s for s in signals if s.triggered]
    sell_count = sum(1 for s in triggered if s.severity == "sell")
    warn_count = sum(1 for s in triggered if s.severity == "warning")
    total = len(triggered)

    full_exit = any(s.rule_number == 9 and s.triggered for s in signals)
    if full_exit:
        return "FULL EXIT", "red", "Exit entire TQQQ position immediately."

    if sell_count >= 2 or total >= 5:
        return "CRITICAL", "red", "Sell aggressively -- multiple sell signals firing."
    elif sell_count >= 1 or total >= 3:
        return "HIGH", "orange", "Trim 20-30% of position. Watch closely."
    elif warn_count >= 2 or total >= 2:
        return "ELEVATED", "yellow", "Trim 10% and tighten stops."
    elif total >= 1:
        return "WATCH", "yellow", "Monitor closely -- early warning."
    else:
        return "CLEAR", "green", "Hold position. No sell signals active."
