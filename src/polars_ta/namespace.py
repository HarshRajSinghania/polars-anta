"""Polars namespace registration for technical analysis expressions and frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from . import momentum, trend, volatility, volume

if TYPE_CHECKING:
    from .core import IntoExpr


@pl.api.register_expr_namespace("ta")
class TechnicalAnalysisExpr:
    """Namespace for technical analysis operations on Polars expressions.

    Allows chaining technical analysis indicators directly on `pl.col(...)`.

    Example:
    >>> import polars as pl
    >>> import polars_ta
    >>> df = pl.DataFrame({"close": [10.0, 11.0, 12.0, 11.5, 13.0]})
    >>> df.select(pl.col("close").ta.rsi(period=3))
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    # Trend Indicators
    def sma(self, period: int = 10) -> pl.Expr:
        """Simple Moving Average (SMA)."""
        return trend.sma(self._expr, period=period)

    def ema(self, period: int = 10) -> pl.Expr:
        """Exponential Moving Average (EMA)."""
        return trend.ema(self._expr, period=period)

    def wma(self, period: int = 10) -> pl.Expr:
        """Weighted Moving Average (WMA)."""
        return trend.wma(self._expr, period=period)

    def hma(self, period: int = 10) -> pl.Expr:
        """Hull Moving Average (HMA)."""
        return trend.hma(self._expr, period=period)

    # Momentum Indicators
    def rsi(self, period: int = 14) -> pl.Expr:
        """Relative Strength Index (RSI)."""
        return momentum.rsi(self._expr, period=period)

    def macd(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pl.Expr:
        """Moving Average Convergence Divergence (MACD)."""
        return momentum.macd(
            self._expr,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

    def stochastic(
        self,
        high: IntoExpr,
        low: IntoExpr,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> pl.Expr:
        """Stochastic Oscillator (%K, %D)."""
        return momentum.stochastic(
            high=high,
            low=low,
            close=self._expr,
            k_period=k_period,
            d_period=d_period,
            smooth_k=smooth_k,
        )

    def williams_r(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Williams %R."""
        return momentum.williams_r(
            high=high,
            low=low,
            close=self._expr,
            period=period,
        )

    # Volatility Indicators
    def true_range(
        self,
        high: IntoExpr,
        low: IntoExpr,
    ) -> pl.Expr:
        """True Range (TR)."""
        return volatility.true_range(high=high, low=low, close=self._expr)

    def atr(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Average True Range (ATR)."""
        return volatility.atr(high=high, low=low, close=self._expr, period=period)

    def bollinger_bands(
        self,
        period: int = 20,
        num_std: float = 2.0,
    ) -> pl.Expr:
        """Bollinger Bands."""
        return volatility.bollinger_bands(self._expr, period=period, num_std=num_std)

    def keltner_channels(
        self,
        high: IntoExpr,
        low: IntoExpr,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> pl.Expr:
        """Keltner Channels."""
        return volatility.keltner_channels(
            high=high,
            low=low,
            close=self._expr,
            ema_period=ema_period,
            atr_period=atr_period,
            multiplier=multiplier,
        )

    # Volume Indicators
    def obv(self, volume_col: IntoExpr) -> pl.Expr:
        """On-Balance Volume (OBV)."""
        return volume.obv(close=self._expr, volume=volume_col)

    def vwap(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
        period: int | None = None,
    ) -> pl.Expr:
        """Volume-Weighted Average Price (VWAP)."""
        return volume.vwap(
            high=high,
            low=low,
            close=self._expr,
            volume=volume_col,
            period=period,
        )

    def vwma(self, volume_col: IntoExpr, period: int = 20) -> pl.Expr:
        """Volume-Weighted Moving Average (VWMA)."""
        return volume.vwma(close=self._expr, volume=volume_col, period=period)

    def cmf(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
        period: int = 20,
    ) -> pl.Expr:
        """Chaikin Money Flow (CMF)."""
        return volume.cmf(
            high=high,
            low=low,
            close=self._expr,
            volume=volume_col,
            period=period,
        )

    def adl(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
    ) -> pl.Expr:
        """Accumulation/Distribution Line (ADL)."""
        return volume.adl(
            high=high,
            low=low,
            close=self._expr,
            volume=volume_col,
        )
