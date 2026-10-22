"""Volatility indicators implemented natively for Polars."""

from __future__ import annotations

import polars as pl

from .core import IntoExpr, parse_expr, wilder_smooth
from .trend import ema


def true_range(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
) -> pl.Expr:
    """True Range (TR).

    TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :return: Polars expression
    """
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    prev_close = c.shift(1)

    hl = hi - lo
    hc = (hi - prev_close).abs()
    lc = (lo - prev_close).abs()

    return pl.max_horizontal(hl, hc, lc)


def atr(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Average True Range (ATR) using Wilder's smoothing.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 14)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    tr = true_range(high, low, close)
    return wilder_smooth(tr, period)


def bollinger_bands(
    close: IntoExpr,
    period: int = 20,
    multiplier: float = 2.0,
    num_std: float | None = None,
) -> pl.Expr:
    """Bollinger Bands (BBANDS).

    Returns a struct with fields:
        - 'upper': middle + multiplier * std
        - 'middle': SMA of close
        - 'lower': middle - multiplier * std
        - 'bandwidth': (upper - lower) / middle
        - 'percent_b': (close - lower) / (upper - lower)

    :param close: price column or expression
    :param period: lookback period (default 20)
    :param multiplier: standard deviation multiplier (default 2.0)
    :param num_std: alias for multiplier
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    mult = float(num_std) if num_std is not None else float(multiplier)
    c = parse_expr(close)
    middle = c.rolling_mean(window_size=period)
    std = c.rolling_std(window_size=period)

    upper = middle + mult * std
    lower = middle - mult * std
    spread = upper - lower

    bandwidth = pl.when(middle > 0.0).then(spread / middle).otherwise(0.0)
    percent_b = pl.when(spread > 0.0).then((c - lower) / spread).otherwise(0.5)

    return pl.struct(
        [
            upper.alias("upper"),
            middle.alias("middle"),
            lower.alias("lower"),
            bandwidth.alias("bandwidth"),
            percent_b.alias("percent_b"),
        ]
    )


def keltner_channels(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
    ema_period: int | None = None,
) -> pl.Expr:
    """Keltner Channels (KC).

    Returns a struct with fields:
        - 'upper': EMA(close, period) + multiplier * ATR(atr_period)
        - 'middle': EMA(close, period)
        - 'lower': EMA(close, period) - multiplier * ATR(atr_period)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: EMA lookback period (default 20)
    :param atr_period: ATR lookback period (default 10)
    :param multiplier: ATR multiplier (default 2.0)
    :param ema_period: alias for period
    :return: Polars Struct expression
    """
    effective_period = ema_period if ema_period is not None else period
    if effective_period <= 0 or atr_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")

    c = parse_expr(close)
    middle = ema(c, period=effective_period)
    band_range = atr(high, low, close, period=atr_period) * float(multiplier)

    upper = middle + band_range
    lower = middle - band_range

    return pl.struct(
        [
            upper.alias("upper"),
            middle.alias("middle"),
            lower.alias("lower"),
        ]
    )
