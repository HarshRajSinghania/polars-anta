"""Volume-based indicators implemented natively for Polars."""

from __future__ import annotations

import polars as pl

from .core import IntoExpr, parse_expr


def obv(close: IntoExpr, volume: IntoExpr) -> pl.Expr:
    """On-Balance Volume (OBV).

    Measures buying and selling pressure as a cumulative indicator, adding
    volume on up days and subtracting it on down days.

    :param close: close price column name or expression
    :param volume: volume column name or expression
    :return: Polars expression
    """
    c = parse_expr(close)
    v = parse_expr(volume)
    diff = c.diff(1)

    signed_vol = pl.when(diff > 0).then(v).when(diff < 0).then(-v).otherwise(0)
    return signed_vol.cum_sum()


def vwap(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    period: int | None = None,
) -> pl.Expr:
    """Volume Weighted Average Price (VWAP).

    Calculates the average price weighted by volume. If `period` is None,
    computes cumulative session VWAP; otherwise computes rolling VWAP.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback window size, or None for cumulative VWAP
    :return: Polars expression
    """
    if period is not None and period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)

    typical_price = (hi + lo + c) / 3.0
    pv = typical_price * v

    if period is None:
        return pv.cum_sum() / v.cum_sum()
    return pv.rolling_sum(window_size=period) / v.rolling_sum(window_size=period)


def vwma(
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Volume-Weighted Moving Average (VWMA).

    Weights price by volume over a rolling window.

    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback period (default 20)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    c = parse_expr(close)
    v = parse_expr(volume)

    return (c * v).rolling_sum(window_size=period) / v.rolling_sum(window_size=period)


def adl(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
) -> pl.Expr:
    """Accumulation/Distribution Line (ADL).

    Measures cumulative flow of money into and out of a security.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param volume: volume column or expression
    :return: Polars expression
    """
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)

    hl_range = hi - lo
    mfm = pl.when(hl_range != 0.0).then(((c - lo) - (hi - c)) / hl_range).otherwise(0.0)
    mfv = mfm * v
    return mfv.cum_sum()


def cmf(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 20,
) -> pl.Expr:
    """Chaikin Money Flow (CMF).

    Measures money flow volume over a specified rolling period.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback period (default 20)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")

    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)

    hl_range = hi - lo
    mfm = pl.when(hl_range != 0.0).then(((c - lo) - (hi - c)) / hl_range).otherwise(0.0)
    mfv = mfm * v

    sum_mfv = mfv.rolling_sum(window_size=period)
    sum_vol = v.rolling_sum(window_size=period)

    return pl.when(sum_vol != 0.0).then(sum_mfv / sum_vol).otherwise(0.0)
