# TQQQ Trading System

A rules-based TQQQ swing trading dashboard combining CAN SLIM stock screening with leveraged ETF swing trading.

## Quick Start

```bash
cd tqqq-trading-system
pip install -r requirements.txt
streamlit run app.py
```

## Phase 1: TQQQ Dashboard (Current)

- **Market Status Panel** — Distribution day counter for Nasdaq and S&P 500 with regime detection
- **Buy Signal Detection** — Follow-Through Day and 3 White Knights automatic detection
- **Sell Signal Scoreboard** — All 9 sell rules with live triggered/clear status
- **Alert System** — Green/Yellow/Orange/Red alert level with suggested action
- **Swing Tracker** — Automated peak/trough detection with % moves and durations
- **Interactive Charts** — Candlestick chart with 10/21/50/200-day MAs and volume

## Phase 2: Stock Screener (Planned)

- Automated screening of US stocks with approximate RS ratings and earnings growth
- Manual IBD rating input for watchlist stocks
- Individual stock buy/sell signal dashboard
- Portfolio tracker with position sizing

## Data

All market data is fetched from Yahoo Finance via `yfinance`. Data is cached for 4 hours. Click the refresh button in the sidebar to force a fresh fetch.

## Disclaimer

This tool is for educational purposes only and does not constitute financial advice. Past performance does not guarantee future results.
