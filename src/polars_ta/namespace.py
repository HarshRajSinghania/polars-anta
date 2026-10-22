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
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    # =========================================================================
    # Trend Indicators (19)
    # =========================================================================

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

    def dema(self, period: int = 10) -> pl.Expr:
        """Double Exponential Moving Average (DEMA)."""
        return trend.dema(self._expr, period=period)

    def tema(self, period: int = 10) -> pl.Expr:
        """Triple Exponential Moving Average (TEMA)."""
        return trend.tema(self._expr, period=period)

    def zlema(self, period: int = 10) -> pl.Expr:
        """Zero-Lag Exponential Moving Average (ZLEMA)."""
        return trend.zlema(self._expr, period=period)

    def vortex(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Vortex Indicator (VI)."""
        return trend.vortex(high=high, low=low, close=self._expr, period=period)

    def adx(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Average Directional Index (ADX) and (+DI, -DI)."""
        return trend.adx(high=high, low=low, close=self._expr, period=period)

    def aroon(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 25,
    ) -> pl.Expr:
        """Aroon Indicator and Oscillator."""
        return trend.aroon(high=high, low=low, period=period)

    def chop(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Choppiness Index (CHOP)."""
        return trend.chop(high=high, low=low, close=self._expr, period=period)

    def t3(self, period: int = 5, volume_factor: float = 0.7) -> pl.Expr:
        """Tillson T3 Moving Average."""
        return trend.t3(self._expr, period=period, volume_factor=volume_factor)

    def alma(
        self,
        period: int = 9,
        offset: float = 0.85,
        sigma: float = 6.0,
    ) -> pl.Expr:
        """Arnaud Legoux Moving Average (ALMA)."""
        return trend.alma(self._expr, period=period, offset=offset, sigma=sigma)

    def mass_index(
        self,
        high: IntoExpr,
        low: IntoExpr,
        fast_period: int = 9,
        slow_period: int = 25,
    ) -> pl.Expr:
        """Mass Index."""
        return trend.mass_index(
            high=high, low=low, fast_period=fast_period, slow_period=slow_period
        )

    def supertrend(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pl.Expr:
        """SuperTrend Indicator."""
        return trend.supertrend(
            high=high, low=low, close=self._expr, period=period, multiplier=multiplier
        )

    def psar(
        self,
        high: IntoExpr,
        low: IntoExpr,
        af_step: float = 0.02,
        af_max: float = 0.2,
    ) -> pl.Expr:
        """Parabolic SAR (Stop and Reverse)."""
        return trend.psar(high=high, low=low, af_step=af_step, af_max=af_max)

    def kama(
        self,
        period: int = 10,
        fast_span: int = 2,
        slow_span: int = 30,
    ) -> pl.Expr:
        """Kaufman's Adaptive Moving Average (KAMA)."""
        return trend.kama(self._expr, period=period, fast_span=fast_span, slow_span=slow_span)

    def mcginley_dynamic(self, period: int = 10, k: float = 0.6) -> pl.Expr:
        """McGinley Dynamic tracking average."""
        return trend.mcginley_dynamic(self._expr, period=period, k=k)

    def vidya(self, period: int = 9, cmo_period: int = 9) -> pl.Expr:
        """Variable Index Dynamic Average (VIDYA)."""
        return trend.vidya(self._expr, period=period, cmo_period=cmo_period)

    # =========================================================================
    # Momentum Indicators (19)
    # =========================================================================

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

    def cci(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 20,
        constant: float = 0.015,
    ) -> pl.Expr:
        """Commodity Channel Index (CCI)."""
        return momentum.cci(
            high=high,
            low=low,
            close=self._expr,
            period=period,
            constant=constant,
        )

    def roc(self, period: int = 10) -> pl.Expr:
        """Rate of Change (ROC)."""
        return momentum.roc(self._expr, period=period)

    def stoch_rsi(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
    ) -> pl.Expr:
        """Stochastic RSI (StochRSI)."""
        return momentum.stoch_rsi(
            self._expr,
            rsi_period=rsi_period,
            stoch_period=stoch_period,
            k_period=k_period,
            d_period=d_period,
        )

    def awesome_oscillator(
        self,
        high: IntoExpr,
        low: IntoExpr,
        fast_period: int = 5,
        slow_period: int = 34,
    ) -> pl.Expr:
        """Awesome Oscillator (AO)."""
        return momentum.awesome_oscillator(
            high=high,
            low=low,
            fast_period=fast_period,
            slow_period=slow_period,
        )

    def tsi(
        self,
        long_period: int = 25,
        short_period: int = 13,
    ) -> pl.Expr:
        """True Strength Index (TSI)."""
        return momentum.tsi(
            self._expr,
            long_period=long_period,
            short_period=short_period,
        )

    def cmo(self, period: int = 14) -> pl.Expr:
        """Chande Momentum Oscillator (CMO)."""
        return momentum.cmo(self._expr, period=period)

    def ppo(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pl.Expr:
        """Percentage Price Oscillator (PPO)."""
        return momentum.ppo(
            self._expr,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

    def coppock(
        self,
        wma_period: int = 10,
        fast_roc: int = 11,
        slow_roc: int = 14,
    ) -> pl.Expr:
        """Coppock Curve."""
        return momentum.coppock(
            self._expr, wma_period=wma_period, fast_roc=fast_roc, slow_roc=slow_roc
        )

    def ultimate_oscillator(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
    ) -> pl.Expr:
        """Ultimate Oscillator (UO)."""
        return momentum.ultimate_oscillator(
            high=high,
            low=low,
            close=self._expr,
            period1=period1,
            period2=period2,
            period3=period3,
        )

    def fisher_transform(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 9,
    ) -> pl.Expr:
        """Fisher Transform."""
        return momentum.fisher_transform(high=high, low=low, period=period)

    def rvi(
        self,
        open: IntoExpr,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Relative Vigor Index (RVI)."""
        return momentum.rvi(open=open, high=high, low=low, close=self._expr, period=period)

    def kst(
        self,
        r1: int = 10,
        r2: int = 15,
        r3: int = 20,
        r4: int = 30,
        s1: int = 10,
        s2: int = 10,
        s3: int = 10,
        s4: int = 15,
        signal_period: int = 9,
    ) -> pl.Expr:
        """Know Sure Thing (KST)."""
        return momentum.kst(
            self._expr,
            r1=r1,
            r2=r2,
            r3=r3,
            r4=r4,
            s1=s1,
            s2=s2,
            s3=s3,
            s4=s4,
            signal_period=signal_period,
        )

    def dpo(self, period: int = 20) -> pl.Expr:
        """Detrended Price Oscillator (DPO)."""
        return momentum.dpo(self._expr, period=period)

    def bop(
        self,
        open: IntoExpr,
        high: IntoExpr,
        low: IntoExpr,
    ) -> pl.Expr:
        """Balance of Power (BOP)."""
        return momentum.bop(open=open, high=high, low=low, close=self._expr)

    def cfo(self, period: int = 14) -> pl.Expr:
        """Chande Forecast Oscillator (CFO)."""
        return momentum.cfo(self._expr, period=period)

    # =========================================================================
    # Volatility Indicators (14)
    # =========================================================================

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

    def donchian_channels(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 20,
    ) -> pl.Expr:
        """Donchian Channels."""
        return volatility.donchian_channels(high=high, low=low, period=period)

    def stddev(self, period: int = 20) -> pl.Expr:
        """Rolling Standard Deviation."""
        return volatility.stddev(self._expr, period=period)

    def ulcer_index(self, period: int = 14) -> pl.Expr:
        """Ulcer Index (UI)."""
        return volatility.ulcer_index(self._expr, period=period)

    def chandelier_exit(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 22,
        multiplier: float = 3.0,
    ) -> pl.Expr:
        """Chandelier Exit."""
        return volatility.chandelier_exit(
            high=high,
            low=low,
            close=self._expr,
            period=period,
            multiplier=multiplier,
        )

    def natr(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Normalized Average True Range (NATR)."""
        return volatility.natr(high=high, low=low, close=self._expr, period=period)

    def historical_volatility(
        self,
        period: int = 20,
        trading_periods: int = 252,
    ) -> pl.Expr:
        """Annualized Historical Volatility (HV)."""
        return volatility.historical_volatility(
            self._expr, period=period, trading_periods=trading_periods
        )

    def parkinson_volatility(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 20,
        trading_periods: int = 252,
    ) -> pl.Expr:
        """Parkinson Volatility."""
        return volatility.parkinson_volatility(
            high=high,
            low=low,
            period=period,
            trading_periods=trading_periods,
        )

    def garman_klass_volatility(
        self,
        open: IntoExpr,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 20,
        trading_periods: int = 252,
    ) -> pl.Expr:
        """Garman-Klass Volatility."""
        return volatility.garman_klass_volatility(
            open=open,
            high=high,
            low=low,
            close=self._expr,
            period=period,
            trading_periods=trading_periods,
        )

    def chaikin_volatility(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 10,
        roc_period: int = 10,
    ) -> pl.Expr:
        """Chaikin's Volatility."""
        return volatility.chaikin_volatility(
            high=high, low=low, period=period, roc_period=roc_period
        )

    def donchian_width(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 20,
    ) -> pl.Expr:
        """Donchian Channel Width."""
        return volatility.donchian_width(high=high, low=low, period=period)

    # =========================================================================
    # Volume Indicators (14)
    # =========================================================================

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

    def mfi(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Money Flow Index (MFI)."""
        return volume.mfi(
            high=high,
            low=low,
            close=self._expr,
            volume=volume_col,
            period=period,
        )

    def eom(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
        period: int = 14,
        divisor: float = 100_000_000.0,
    ) -> pl.Expr:
        """Ease of Movement (EOM)."""
        return volume.eom(
            high=high,
            low=low,
            volume=volume_col,
            period=period,
            divisor=divisor,
        )

    def klinger(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
        fast_period: int = 34,
        slow_period: int = 55,
        signal_period: int = 13,
    ) -> pl.Expr:
        """Klinger Volume Oscillator."""
        return volume.klinger(
            high=high,
            low=low,
            close=self._expr,
            volume=volume_col,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

    def nvi(
        self,
        volume_col: IntoExpr,
        initial: float = 1000.0,
    ) -> pl.Expr:
        """Negative Volume Index (NVI)."""
        return volume.nvi(close=self._expr, volume=volume_col, initial=initial)

    def pvi(
        self,
        volume_col: IntoExpr,
        initial: float = 1000.0,
    ) -> pl.Expr:
        """Positive Volume Index (PVI)."""
        return volume.pvi(close=self._expr, volume=volume_col, initial=initial)

    def pvt(self, volume_col: IntoExpr) -> pl.Expr:
        """Price Volume Trend (PVT)."""
        return volume.pvt(close=self._expr, volume=volume_col)

    def force_index(self, volume_col: IntoExpr, period: int = 13) -> pl.Expr:
        """Elder's Force Index (EFI)."""
        return volume.force_index(close=self._expr, volume=volume_col, period=period)

    def vpci(
        self,
        volume_col: IntoExpr,
        period_short: int = 5,
        period_long: int = 25,
    ) -> pl.Expr:
        """Volume Price Confirmation Indicator (VPCI)."""
        return volume.vpci(
            close=self._expr,
            volume=volume_col,
            period_short=period_short,
            period_long=period_long,
        )

    def vr(self, volume_col: IntoExpr, period: int = 26) -> pl.Expr:
        """Volume Ratio (VR)."""
        return volume.vr(close=self._expr, volume=volume_col, period=period)
