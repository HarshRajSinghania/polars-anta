"""polars-ta: Ultra-fast Technical Analysis & Factor Library powered by Polars."""

from __future__ import annotations

from . import namespace  # noqa: F401 - triggers Polars namespace registration
from .momentum import macd, rsi, stochastic, williams_r
from .trend import ema, hma, sma, wma
from .volatility import atr, bollinger_bands, keltner_channels, true_range
from .volume import adl, cmf, obv, vwap, vwma

__version__ = "0.1.0"

__all__ = [
    # Trend
    "sma",
    "ema",
    "wma",
    "hma",
    # Momentum
    "rsi",
    "macd",
    "stochastic",
    "williams_r",
    # Volatility
    "true_range",
    "atr",
    "bollinger_bands",
    "keltner_channels",
    # Volume
    "obv",
    "vwap",
    "vwma",
    "cmf",
    "adl",
]
