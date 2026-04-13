"""
Data fetching and caching layer.
Uses yfinance for price/volume data with in-memory + TTL caching.
"""

import datetime as dt
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import config


@st.cache_data(ttl=config.CACHE_EXPIRY_HOURS * 3600, show_spinner=False)
def fetch_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch daily OHLCV data. Returns DataFrame with DatetimeIndex."""
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


@st.cache_data(ttl=config.CACHE_EXPIRY_HOURS * 3600, show_spinner=False)
def fetch_weekly(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch weekly OHLCV data."""
    df = yf.download(ticker, period=period, interval="1wk", progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA and EMA columns used by the trading system."""
    df = df.copy()
    df["SMA_10"] = df["Close"].rolling(window=config.MA_10).mean()
    df["EMA_21"] = df["Close"].ewm(span=config.MA_21, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(window=config.MA_50).mean()
    df["SMA_200"] = df["Close"].rolling(window=config.MA_200).mean()
    df["Vol_SMA_50"] = df["Volume"].rolling(window=50).mean()
    return df


def add_volume_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-relative metrics."""
    df = df.copy()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_SMA_50"]
    df["Vol_Change"] = df["Volume"].pct_change()
    return df


def get_tqqq_data() -> pd.DataFrame:
    """Fetch TQQQ daily data with all indicators."""
    df = fetch_daily(config.TQQQ, period="2y")
    if df.empty:
        return df
    df = add_moving_averages(df)
    df = add_volume_metrics(df)
    return df


def get_qqq_data() -> pd.DataFrame:
    """Fetch QQQ daily data with indicators (used for signal detection on the unleveraged chart)."""
    df = fetch_daily(config.QQQ, period="2y")
    if df.empty:
        return df
    df = add_moving_averages(df)
    df = add_volume_metrics(df)
    return df


def get_nasdaq_data() -> pd.DataFrame:
    """Fetch Nasdaq Composite daily data (for distribution day counting)."""
    df = fetch_daily(config.NASDAQ_COMPOSITE, period="2y")
    if df.empty:
        return df
    df = add_moving_averages(df)
    df = add_volume_metrics(df)
    return df


def get_sp500_data() -> pd.DataFrame:
    """Fetch S&P 500 daily data (for distribution day counting)."""
    df = fetch_daily(config.SP500, period="2y")
    if df.empty:
        return df
    df = add_moving_averages(df)
    df = add_volume_metrics(df)
    return df


def get_52_week_high(df: pd.DataFrame) -> float:
    """Return the 52-week (252 trading day) high."""
    lookback = min(252, len(df))
    return df["High"].iloc[-lookback:].max()


def get_current_price(df: pd.DataFrame) -> float:
    return float(df["Close"].iloc[-1])


def get_latest_date(df: pd.DataFrame) -> dt.date:
    return df.index[-1].date()
