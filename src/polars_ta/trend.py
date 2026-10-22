"""Trend indicators implemented natively for Polars."""

from __future__ import annotations

import math

import polars as pl

from .core import IntoExpr, parse_expr, true_range, wilder_smooth


def sma(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Simple Moving Average (SMA).

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    return parse_expr(close).rolling_mean(window_size=period)


def ema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Exponential Moving Average (EMA).

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    return parse_expr(close).ewm_mean(span=period, adjust=False)


def wma(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Weighted Moving Average (WMA).

    Weights linearly increase: [1, 2, ..., period].

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    if period == 1:
        return parse_expr(close)

    total_weight = (period * (period + 1)) // 2
    c = parse_expr(close)
    weighted_sum = sum(c.shift(period - weight) * weight for weight in range(1, period + 1))
    return weighted_sum / total_weight


def hma(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Hull Moving Average (HMA).

    Formula:
        HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(math.sqrt(period)))

    c = parse_expr(close)
    diff = 2.0 * wma(c, half_period) - wma(c, period)
    return wma(diff, sqrt_period)


def dema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Double Exponential Moving Average (DEMA).

    Formula:
        DEMA = 2 * EMA(close, n) - EMA(EMA(close, n), n)

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    e1 = ema(close, period=period)
    e2 = ema(e1, period=period)
    return 2.0 * e1 - e2


def tema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Triple Exponential Moving Average (TEMA).

    Formula:
        TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
        where EMA1 = EMA(close), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    e1 = ema(close, period=period)
    e2 = ema(e1, period=period)
    e3 = ema(e2, period=period)
    return 3.0 * e1 - 3.0 * e2 + e3


def zlema(close: IntoExpr, period: int = 10) -> pl.Expr:
    """Zero-Lag Exponential Moving Average (ZLEMA).

    Formula:
        lag = (period - 1) // 2
        ZLEMA = EMA(2 * close - close.shift(lag), period)

    :param close: price column name or expression
    :param period: lookback period (default 10)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    lag = (period - 1) // 2
    c = parse_expr(close)
    adjusted = 2.0 * c - c.shift(lag)
    return ema(adjusted, period=period)


def vortex(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Vortex Indicator (VI).

    Returns a struct with fields:
        - 'vi_plus': positive vortex line
        - 'vi_minus': negative vortex line

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback period (default 14)
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)
    tr = true_range(hi, lo, close)

    vm_plus = (hi - lo.shift(1)).abs()
    vm_minus = (lo - hi.shift(1)).abs()

    sum_tr = tr.rolling_sum(window_size=period)
    sum_vm_plus = vm_plus.rolling_sum(window_size=period)
    sum_vm_minus = vm_minus.rolling_sum(window_size=period)

    vi_plus = pl.when(sum_tr > 0.0).then(sum_vm_plus / sum_tr).otherwise(1.0)
    vi_minus = pl.when(sum_tr > 0.0).then(sum_vm_minus / sum_tr).otherwise(1.0)

    return pl.struct(
        [
            vi_plus.alias("vi_plus"),
            vi_minus.alias("vi_minus"),
        ]
    )


def adx(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Average Directional Index (ADX) with Directional Movement (+DI, -DI).

    Returns a struct with fields:
        - 'adx': Average Directional Index [0, 100]
        - 'plus_di': Plus Directional Indicator (+DI)
        - 'minus_di': Minus Directional Indicator (-DI)

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param period: lookback smoothing period (default 14)
    :return: Polars Struct expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)

    up_move = hi - hi.shift(1)
    down_move = lo.shift(1) - lo

    plus_dm = pl.when((up_move > down_move) & (up_move > 0.0)).then(up_move).otherwise(0.0)
    minus_dm = pl.when((down_move > up_move) & (down_move > 0.0)).then(down_move).otherwise(0.0)

    tr = true_range(hi, lo, close)
    atr_smooth = wilder_smooth(tr, period)
    smooth_plus_dm = wilder_smooth(plus_dm, period)
    smooth_minus_dm = wilder_smooth(minus_dm, period)

    plus_di = pl.when(atr_smooth > 0.0).then(100.0 * smooth_plus_dm / atr_smooth).otherwise(0.0)
    minus_di = pl.when(atr_smooth > 0.0).then(100.0 * smooth_minus_dm / atr_smooth).otherwise(0.0)

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()

    dx = pl.when(di_sum > 0.0).then(100.0 * di_diff / di_sum).otherwise(0.0)
    adx_line = wilder_smooth(dx, period)

    return pl.struct(
        [
            adx_line.alias("adx"),
            plus_di.alias("plus_di"),
            minus_di.alias("minus_di"),
        ]
    )
