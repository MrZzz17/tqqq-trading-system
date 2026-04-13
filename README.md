# Vibha Jha Trading System

A trading dashboard implementing Vibha Jha's hybrid CAN SLIM + TQQQ swing trading system — the strategy that earned her consistent top finishes in the U.S. Investing Championship (100% in 2021, 70% in 2023, 78% in 2024).

## Quick Start

```bash
cd vibha-trading-system
pip install -r requirements.txt
streamlit run app.py
```

## Phase 1: TQQQ Dashboard (Current)

- **Market Status Panel** — Distribution day counter for Nasdaq and S&P 500 with regime detection (Confirmed Uptrend / Under Pressure / Correction)
- **Buy Signal Detection** — Follow-Through Day and 3 White Knights automatic detection
- **Sell Signal Scoreboard** — All 9 of Vibha's sell rules with live triggered/clear status
- **Alert System** — Green/Yellow/Orange/Red alert level with suggested action
- **Swing Tracker** — Automated peak/trough detection with % moves and durations (replaces Vibha's manual spreadsheet)
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
