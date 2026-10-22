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


def mfi(
    high: IntoExpr,
    low: IntoExpr,
    close: IntoExpr,
    volume: IntoExpr,
    period: int = 14,
) -> pl.Expr:
    """Money Flow Index (MFI).

    A volume-weighted RSI oscillating between 0 and 100.

    :param high: high price column or expression
    :param low: low price column or expression
    :param close: close price column or expression
    :param volume: volume column or expression
    :param period: lookback period (default 14)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)
    c = parse_expr(close)
    v = parse_expr(volume)

    typical_price = (hi + lo + c) / 3.0
    raw_money_flow = typical_price * v
    diff = typical_price.diff(1)

    pos_flow = pl.when(diff > 0.0).then(raw_money_flow).otherwise(0.0)
    neg_flow = pl.when(diff < 0.0).then(raw_money_flow).otherwise(0.0)

    pos_sum = pos_flow.rolling_sum(window_size=period)
    neg_sum = neg_flow.rolling_sum(window_size=period)

    money_ratio = pos_sum / neg_sum

    return (
        pl.when(neg_sum == 0.0)
        .then(100.0)
        .when(pos_sum == 0.0)
        .then(0.0)
        .otherwise(100.0 - (100.0 / (1.0 + money_ratio)))
    )


def eom(
    high: IntoExpr,
    low: IntoExpr,
    volume: IntoExpr,
    period: int = 14,
    divisor: float = 100_000_000.0,
) -> pl.Expr:
    """Ease of Movement (EOM / EMV).

    Relates price change to trading volume over a lookback window.

    :param high: high price column or expression
    :param low: low price column or expression
    :param volume: volume column or expression
    :param period: moving average smoothing window (default 14)
    :param divisor: volume scaling divisor (default 100,000,000)
    :return: Polars expression
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    hi = parse_expr(high)
    lo = parse_expr(low)
    v = parse_expr(volume)

    hl_mid = (hi + lo) / 2.0
    dist_moved = hl_mid - hl_mid.shift(1)
    hl_range = hi - lo

    box_ratio = (v / float(divisor)) / hl_range
    emv_1 = (
        pl.when((hl_range > 0.0) & (box_ratio > 0.0)).then(dist_moved / box_ratio).otherwise(0.0)
    )

    return emv_1.rolling_mean(window_size=period)
