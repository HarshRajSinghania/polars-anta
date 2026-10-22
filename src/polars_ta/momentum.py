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


def cci(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 20,
    constant: float = 0.015,
) -> pl.Expr:
    """Commodity Channel Index (CCI).

    Measures current price level relative to an average price level over a given period.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 20)
    :param constant: scaling factor (default 0.015)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)

    tp = (hi + lo + c) / 3.0
    sma_tp = tp.rolling_mean(window_size=period)
    mad = sum((tp.shift(i) - sma_tp).abs() for i in range(period)) / float(period)

    return pl.when(mad > 0.0).then((tp - sma_tp) / (constant * mad)).otherwise(0.0)


def roc(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Rate of Change (ROC).

    Calculates the percentage change between current price and price n periods ago.

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    c = parse_expr(close)
    prev_c = c.shift(period)
    return pl.when(prev_c != 0.0).then(100.0 * (c - prev_c) / prev_c).otherwise(0.0)


def stoch_rsi(
    close: IntoExpr,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> pl.Expr:
    """Stochastic RSI (StochRSI).

    Applies the Stochastic Oscillator formula to RSI values rather than raw price.

    Returns a struct with fields:
        - 'k': smoothed %K line in [0, 100]
        - 'd': %D signal line in [0, 100]

    :param close: price column or expression
    :param rsi_period: RSI lookback period (default 14)
    :param stoch_period: Stochastic lookback period (default 14)
    :param k_period: %K smoothing window (default 3)
    :param d_period: %D smoothing window (default 3)
    :return: Polars Struct expression
    """
    if rsi_period <= 0 or stoch_period <= 0 or k_period <= 0 or d_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")

    rsi_val = rsi(close, period=rsi_period)
    lowest_rsi = rsi_val.rolling_min(window_size=stoch_period)
    highest_rsi = rsi_val.rolling_max(window_size=stoch_period)
    spread = highest_rsi - lowest_rsi

    fast_k = pl.when(spread > 0.0).then(100.0 * (rsi_val - lowest_rsi) / spread).otherwise(50.0)
    k_line = fast_k.rolling_mean(window_size=k_period)
    d_line = k_line.rolling_mean(window_size=d_period)

    return pl.struct(
        [
            k_line.alias("k"),
            d_line.alias("d"),
        ]
    )


def awesome_oscillator(
    high: IntoExpr,
    low: IntoExpr,
    fast_period: int = 5,
    slow_period: int = 34,
) -> pl.Expr:
    """Awesome Oscillator (AO).

    Difference between a fast (default 5) and slow (default 34) SMA of the median price (High + Low) / 2.

    :param high: high price column or expression
    :param low: low price column or expression
    :param fast_period: fast SMA period (default 5)
    :param slow_period: slow SMA period (default 34)
    :return: Polars expression
    """
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    hi = parse_expr(high)
    lo = parse_expr(low)
    median_price = (hi + lo) / 2.0

    fast_sma = median_price.rolling_mean(window_size=fast_period)
    slow_sma = median_price.rolling_mean(window_size=slow_period)

    return fast_sma - slow_sma


def tsi(
    close: IntoExpr,
    long_period: int = 25,
    short_period: int = 13,
) -> pl.Expr:
    """True Strength Index (TSI).

    Momentum oscillator using double-smoothed exponential moving averages of price changes.
    Values oscillate between -100 and +100.

    :param close: price column or expression
    :param long_period: first smoothing span (default 25)
    :param short_period: second smoothing span (default 13)
    :return: Polars expression
    """
    if long_period <= 0 or short_period <= 0:
        raise ValueError("periods must be positive integers greater than zero")

    c = parse_expr(close)
    diff = c.diff(1)
    abs_diff = diff.abs()

    smooth_1 = diff.ewm_mean(span=long_period, adjust=False)
    smooth_2 = smooth_1.ewm_mean(span=short_period, adjust=False)

    abs_smooth_1 = abs_diff.ewm_mean(span=long_period, adjust=False)
    abs_smooth_2 = abs_smooth_1.ewm_mean(span=short_period, adjust=False)

    return pl.when(abs_smooth_2 > 0.0).then(100.0 * smooth_2 / abs_smooth_2).otherwise(0.0)
