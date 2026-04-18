"""
Configuration constants for the TQQQ Trading System.
Thresholds derived from CAN SLIM methodology and rules-based swing trading.
"""

# --- Tickers ---
TQQQ = "TQQQ"
QQQ = "QQQ"
NASDAQ_COMPOSITE = "^IXIC"
SP500 = "SPY"

# --- Moving Average Periods ---
MA_10 = 10
MA_21 = 21
MA_50 = 50
MA_200 = 200
MA_10_WEEK = 50  # 10 weeks * 5 trading days

# --- TQQQ Buy Signals ---
FTD_MIN_RALLY_DAY = 4          # FTD must be day 4+ of rally attempt
FTD_MIN_GAIN_PCT = 1.25        # Nasdaq must gain >1.25% on the day
WHITE_KNIGHTS_DAYS = 3         # 3 consecutive higher highs + higher lows

# --- TQQQ Sell Signal Thresholds ---
DISTRIBUTION_DAY_DECLINE = -0.2    # >0.2% decline on higher volume
DISTRIBUTION_DAY_WINDOW = 25       # Rolling 25-session window
DISTRIBUTION_DAY_WARN = 4          # 4+ distribution days = warning
DISTRIBUTION_DAY_CRITICAL = 5      # 5+ = aggressive sell
CONSECUTIVE_DOWN_DAYS = 3          # 3 consecutive down days
BULLS_BEARS_FROTHY = 60            # Bulls >60% = frothy

# --- TQQQ Position Sizing ---
TQQQ_INITIAL_POSITION_FTD = 1.0    # 100% on FTD (in IRA)
TQQQ_INITIAL_POSITION_3WK = 0.25   # 25% on 3 White Knights
TQQQ_SELL_CHUNK = 0.10             # Sell in 10% blocks

# --- Individual Stock Criteria ---
MIN_EPS_GROWTH = 25         # Quarterly EPS growth %
PREFERRED_EPS_GROWTH = 35
MIN_REVENUE_GROWTH = 25     # Quarterly revenue growth %
MIN_RS_RATING = 80          # Approximate RS rating (1-99)
PREFERRED_RS_RATING = 95
MIN_COMPOSITE = 95          # IBD Composite (manual input)
MAX_POSITIONS = 8           # 6-8 core holdings
INITIAL_POSITION_SIZE = 0.10  # 10% of portfolio
MAX_POSITION_SIZE = 0.15      # 12.5-15% after adding
STOP_LOSS_MAX_DISTANCE = 0.08  # Buy within 8% of technical stop
DOUBLE_SELL_HALF = 1.0         # Sell half at 100% gain
PULLBACK_TOLERANCE = 0.08      # Tolerate 5-8% pullbacks

# --- Swing Tracker ---
SWING_MIN_MOVE_PCT = 5.0       # Minimum % move to count as a swing
TQQQ_TYPICAL_SWING_PCT = 20.0  # Typical TQQQ swing = 20-30%

# --- Data ---
CACHE_EXPIRY_HOURS = 4
LOOKBACK_DAYS = 365
# Yahoo Finance daily bars for the V6 engine (must stay in sync with dashboard live state)
STRATEGY_ENGINE_CACHE_SECONDS = 300

# Shown on Market Health — bump when that panel changes (confirms deploy picked up UI)
DASHBOARD_MARKET_HEALTH_REV = "v2-st-columns"
