"""
Data fetching and caching layer.
Tries TradingView (tvdatafeed) first for cleaner data, falls back to yfinance.

To use your paid TradingView account for premium data, set environment vars:
    export TV_USERNAME="your_tradingview_username"
    export TV_PASSWORD="your_tradingview_password"
Or add them to a .env file in the project root.
"""

import datetime as dt
import os
import logging
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# TradingView exchange mapping
_TV_EXCHANGES = {
    "TQQQ": "NASDAQ",
    "QQQ": "NASDAQ",
    "SPY": "AMEX",
}


def _try_tvdatafeed(ticker: str, n_bars: int = 500, interval: str = "daily") -> pd.DataFrame:
    """Try fetching from TradingView. Returns empty DataFrame on failure."""
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    try:
        from tvDatafeed import TvDatafeed, Interval

        tv_user = os.environ.get("TV_USERNAME")
        tv_pass = os.environ.get("TV_PASSWORD")

        if tv_user and tv_pass:
            tv = TvDatafeed(username=tv_user, password=tv_pass)
        else:
            tv = TvDatafeed()

        interval_map = {
            "daily": Interval.in_daily,
            "weekly": Interval.in_weekly,
        }
        tv_interval = interval_map.get(interval, Interval.in_daily)

        clean_ticker = ticker.replace("^", "")
        exchange = _TV_EXCHANGES.get(ticker, "NASDAQ")

        if ticker == "^IXIC":
            clean_ticker = "IXIC"
            exchange = "NASDAQ"
        elif ticker == "^GSPC":
            clean_ticker = "SPX"
            exchange = "SP"

        df = tv.get_hist(
            symbol=clean_ticker,
            exchange=exchange,
            interval=tv_interval,
            n_bars=n_bars,
        )

        if df is not None and not df.empty:
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in df.columns:
                    col_lower = col.lower()
                    if col_lower in df.columns:
                        df = df.rename(columns={col_lower: col})

            if "symbol" in df.columns:
                df = df.drop(columns=["symbol"])

            df.index = pd.to_datetime(df.index)
            df.index = df.index.normalize()
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        logger.debug(f"tvdatafeed failed for {ticker}: {e}")

    return pd.DataFrame()


def _fetch_yfinance(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


@st.cache_data(ttl=config.CACHE_EXPIRY_HOURS * 3600, show_spinner=False)
def fetch_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch daily OHLCV. Tries TradingView first, falls back to yfinance."""
    n_bars = {"1y": 252, "2y": 504, "5y": 1260, "max": 5000}.get(period, 504)
    df = _try_tvdatafeed(ticker, n_bars=n_bars, interval="daily")
    if not df.empty:
        return df

    return _fetch_yfinance(ticker, period=period, interval="1d")


@st.cache_data(ttl=config.CACHE_EXPIRY_HOURS * 3600, show_spinner=False)
def fetch_weekly(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch weekly OHLCV. Tries TradingView first, falls back to yfinance."""
    n_bars = {"1y": 52, "2y": 104, "5y": 260, "max": 1000}.get(period, 104)
    df = _try_tvdatafeed(ticker, n_bars=n_bars, interval="weekly")
    if not df.empty:
        return df

    return _fetch_yfinance(ticker, period=period, interval="1wk")


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(window=config.MA_10).mean()
    df["EMA_21"] = df["Close"].ewm(span=config.MA_21, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(window=config.MA_50).mean()
    df["SMA_200"] = df["Close"].rolling(window=config.MA_200).mean()
    df["Vol_SMA_50"] = df["Volume"].rolling(window=50).mean()
    return df


def add_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_SMA_50"]
    df["Vol_Change"] = df["Volume"].pct_change()
    return df


def add_weekly_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Add weekly MACD to a daily DataFrame by resampling."""
    df = df.copy()
    weekly = df["Close"].resample("W-FRI").last().dropna()
    w_ema12 = weekly.ewm(span=12, adjust=False).mean()
    w_ema26 = weekly.ewm(span=26, adjust=False).mean()
    w_macd = w_ema12 - w_ema26
    w_signal = w_macd.ewm(span=9, adjust=False).mean()
    df["Weekly_MACD"] = w_macd.reindex(df.index, method="ffill")
    df["Weekly_MACD_Signal"] = w_signal.reindex(df.index, method="ffill")
    return df


def get_tqqq_data() -> pd.DataFrame:
    df = fetch_daily(config.TQQQ, period="2y")
    if df.empty:
        return df
    df = add_moving_averages(df)
    df = add_volume_metrics(df)
    return df


def get_qqq_data() -> pd.DataFrame:
    df = fetch_daily(config.QQQ, period="2y")
    if df.empty:
        return df
    df = add_moving_averages(df)
    df = add_volume_metrics(df)
    df = add_weekly_macd(df)
    return df


def get_nasdaq_data() -> pd.DataFrame:
    df = fetch_daily(config.NASDAQ_COMPOSITE, period="2y")
    if df.empty:
        return df
    df = add_moving_averages(df)
    df = add_volume_metrics(df)
    return df


def get_sp500_data() -> pd.DataFrame:
    df = fetch_daily(config.SP500, period="2y")
    if df.empty:
        return df
    df = add_moving_averages(df)
    df = add_volume_metrics(df)
    return df


def get_52_week_high(df: pd.DataFrame) -> float:
    lookback = min(252, len(df))
    return df["High"].iloc[-lookback:].max()


def get_current_price(df: pd.DataFrame) -> float:
    return float(df["Close"].iloc[-1])


def get_latest_date(df: pd.DataFrame) -> dt.date:
    return df.index[-1].date()
