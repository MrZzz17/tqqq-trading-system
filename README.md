# TQQQ Trading System

A rules-based TQQQ swing trading dashboard combining CAN SLIM stock screening with leveraged ETF swing trading.

## Quick Start

```bash
cd tqqq-trading-system
pip install -r requirements.txt
streamlit run app.py
```

## Phase 1: TQQQ Dashboard (Current)

- **Market pulse** — Nasdaq & SPY regime labels (distribution days + MA structure)
- **V6 live card** — Buy / Sell / Flat from the last daily close; open-position detail when long
- **Backtest hero & equity curve** — Cumulative performance with Period control (1D … All)
- **Market Health** — QQQ/SPY vs MAs; regime + weekly MACD explanation tiles
- **System Signals** — FTD, weekly MACD, QQQ vs 200-day (entries); 200-day exit, 12% trail, crash detector (risk)
- **QQQ and TQQQ model chart** — QQQ row (regime MAs), TQQQ row (candles, MAs, backtest entry/exit markers), TQQQ volume (same Period as equity chart)

## Phase 2: Stock Screener (Planned)

- Automated screening of US stocks with approximate RS ratings and earnings growth
- Manual IBD rating input for watchlist stocks
- Individual stock buy/sell signal dashboard
- Portfolio tracker with position sizing

## Data

The **V6 strategy engine** (alerts, equity curve, trade history) uses Yahoo Finance daily bars via `yfinance`, refreshed on a short interval (`STRATEGY_ENGINE_CACHE_SECONDS` in `config.py`, default 5 minutes). The **quote panel** uses `core/data.py` with a longer cache (default 4 hours) and may use TradingView when credentials are set — if so, top-of-page prices can differ slightly from the engine; trust the engine dates for signals. Use **Refresh Data** to force both paths to update immediately.

## Disclaimer

This tool is for educational purposes only and does not constitute financial advice. Past performance does not guarantee future results.
