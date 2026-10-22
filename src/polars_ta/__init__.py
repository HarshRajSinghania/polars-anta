"""polars-ta: High-performance, zero-C technical analysis and factor library powered by Polars."""

from __future__ import annotations

from . import namespace  # noqa: F401 - triggers Polars namespace registration
from .momentum import (
    awesome_oscillator,
    cci,
    macd,
    roc,
    rsi,
    stoch_rsi,
    stochastic,
    tsi,
    williams_r,
)
from .trend import (
    adx,
    dema,
    ema,
    hma,
    sma,
    tema,
    vortex,
    wma,
    zlema,
)
from .volatility import (
    atr,
    bollinger_bands,
    donchian_channels,
    keltner_channels,
    stddev,
    true_range,
    ulcer_index,
)
from .volume import (
    adl,
    cmf,
    eom,
    mfi,
    obv,
    vwap,
    vwma,
)

__version__ = "0.1.0"

__all__ = [
    # Trend (9)
    "sma",
    "ema",
    "wma",
    "hma",
    "dema",
    "tema",
    "zlema",
    "vortex",
    "adx",
    # Momentum (9)
    "rsi",
    "macd",
    "stochastic",
    "williams_r",
    "cci",
    "roc",
    "stoch_rsi",
    "awesome_oscillator",
    "tsi",
    # Volatility (7)
    "true_range",
    "atr",
    "bollinger_bands",
    "keltner_channels",
    "donchian_channels",
    "stddev",
    "ulcer_index",
    # Volume (7)
    "obv",
    "vwap",
    "vwma",
    "cmf",
    "adl",
    "mfi",
    "eom",
]
