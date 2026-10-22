"""Trend indicators implemented natively for Polars."""

from __future__ import annotations

import math

import polars as pl

from .core import IntoExpr, parse_expr


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
