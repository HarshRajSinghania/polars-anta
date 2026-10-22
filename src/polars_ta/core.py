"""Core utility functions and expression helpers for polars-ta."""

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
