"""Momentum indicators implemented natively for Polars."""

from __future__ import annotations

import polars as pl

from .core import IntoExpr, parse_expr, wilder_smooth


def rsi(close: IntoExpr, period: int = 14) -> pl.Expr:
    """Relative Strength Index (RSI) using Wilder's smoothing.

    :param close: price column name or expression
    :param period: lookback period (default 14)
    :return: Polars expression producing RSI values in [0, 100]
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    c = parse_expr(close)
    diff = c.diff(1)
    gain = pl.when(diff > 0).then(diff).otherwise(0.0)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0)

    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)

    rs = avg_gain / avg_loss
    return (
        pl.when(avg_loss == 0.0)
        .then(100.0)
        .when(avg_gain == 0.0)
        .then(0.0)
        .otherwise(100.0 - (100.0 / (1.0 + rs)))
    )


def macd(
    close: IntoExpr,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pl.Expr:
    """Moving Average Convergence Divergence (MACD).

    Returns a struct with fields:
        - 'macd': difference between fast and slow EMA
        - 'signal': EMA of the MACD line
        - 'histogram': MACD line minus signal line

    :param close: price column name or expression
    :param fast_period: fast EMA period (default 12)
    :param slow_period: slow EMA period (default 26)
    :param signal_period: signal line EMA period (default 9)
    :return: Polars Struct expression
    """
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    c = parse_expr(close)
    macd_line = c.ewm_mean(span=fast_period, adjust=False) - c.ewm_mean(
        span=slow_period, adjust=False
    )
    signal_line = macd_line.ewm_mean(span=signal_period, adjust=False)
    histogram = macd_line - signal_line

    return pl.struct(
        [
            macd_line.alias("macd"),
            signal_line.alias("signal"),
            histogram.alias("histogram"),
        ]
    )


def stochastic(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pl.Expr:
    """Stochastic Oscillator (%K, %D).

    Returns a struct with fields:
        - 'k': smoothed %K line
        - 'd': %D signal line (SMA of %K)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param k_period: lookback window (default 14)
    :param d_period: %D smoothing window (default 3)
    :param smooth_k: %K smoothing window (default 3)
    :return: Polars Struct expression
    """
    if k_period <= 0 or d_period <= 0 or smooth_k <= 0:
        raise ValueError("periods must be positive integers greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    lowest_low = lo.rolling_min(window_size=k_period)
    highest_high = hi.rolling_max(window_size=k_period)
    spread = highest_high - lowest_low

    fast_k = pl.when(spread > 0.0).then(100.0 * (c - lowest_low) / spread).otherwise(50.0)
    k_line = fast_k.rolling_mean(window_size=smooth_k)
    d_line = k_line.rolling_mean(window_size=d_period)

    return pl.struct(
        [
            k_line.alias("k"),
            d_line.alias("d"),
        ]
    )


def williams_r(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Williams %R momentum indicator.

    Values oscillate in [-100, 0].

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 14)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    highest_high = hi.rolling_max(window_size=period)
    lowest_low = lo.rolling_min(window_size=period)
    spread = highest_high - lowest_low

    return pl.when(spread > 0.0).then(-100.0 * (highest_high - c) / spread).otherwise(-50.0)
