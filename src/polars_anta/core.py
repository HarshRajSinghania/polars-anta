"""Core utility functions and expression helpers for polars-anta."""

from __future__ import annotations

import polars as pl

IntoExpr = str | pl.Expr


def parse_expr(column: IntoExpr) -> pl.Expr:
    """Normalize a column name or Polars Expression into a pl.Expr."""
    return pl.col(column) if isinstance(column, str) else column


def wilder_smooth(expr: pl.Expr, period: int) -> pl.Expr:
    """Wilder's exponential smoothing (RMA).

    Uses alpha = 1 / period, mathematically identical to Wilder's Smoothing.
    """
    if period <= 0:
        raise ValueError("period must be a positive integer greater than zero")
    return expr.ewm_mean(alpha=1.0 / float(period), adjust=False)


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
