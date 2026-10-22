"""Tests for Polars .ta namespace integration."""

from __future__ import annotations

import polars as pl

# Importing polars_ta registers the `.ta` namespace on pl.Expr
import polars_ta  # noqa: F401


def test_namespace_trend(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.sma(5).alias("sma_5"),
            pl.col("close").ta.ema(5).alias("ema_5"),
            pl.col("close").ta.wma(5).alias("wma_5"),
            pl.col("close").ta.hma(9).alias("hma_9"),
            pl.col("close").ta.dema(5).alias("dema_5"),
            pl.col("close").ta.tema(5).alias("tema_5"),
            pl.col("close").ta.zlema(5).alias("zlema_5"),
            pl.col("close").ta.vortex(high="high", low="low", period=10).alias("vortex"),
            pl.col("close").ta.adx(high="high", low="low", period=10).alias("adx"),
            pl.col("close").ta.aroon(high="high", low="low", period=10).alias("aroon"),
            pl.col("close").ta.chop(high="high", low="low", period=10).alias("chop"),
            pl.col("close").ta.t3(period=5).alias("t3"),
            pl.col("close").ta.alma(period=5).alias("alma"),
            pl.col("close")
            .ta.mass_index(high="high", low="low", fast_period=3, slow_period=10)
            .alias("mi"),
            pl.col("close").ta.supertrend(high="high", low="low", period=5).alias("st"),
            pl.col("close").ta.psar(high="high", low="low").alias("psar"),
            pl.col("close").ta.kama(period=5).alias("kama"),
            pl.col("close").ta.mcginley_dynamic(period=5).alias("md"),
            pl.col("close").ta.vidya(period=5, cmo_period=5).alias("vidya"),
        ]
    )

    assert "sma_5" in df_res.columns
    assert "aroon" in df_res.columns
    assert "chop" in df_res.columns
    assert "t3" in df_res.columns
    assert "st" in df_res.columns
    assert "kama" in df_res.columns


def test_namespace_momentum(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.rsi(14).alias("rsi_14"),
            pl.col("close").ta.macd().alias("macd_struct"),
            pl.col("close").ta.stochastic(high="high", low="low").alias("stoch_struct"),
            pl.col("close").ta.williams_r(high="high", low="low").alias("wr_14"),
            pl.col("close").ta.cci(high="high", low="low", period=10).alias("cci"),
            pl.col("close").ta.roc(period=5).alias("roc"),
            pl.col("close").ta.stoch_rsi().alias("srsi"),
            pl.col("close").ta.awesome_oscillator(high="high", low="low").alias("ao"),
            pl.col("close").ta.tsi().alias("tsi"),
            pl.col("close").ta.cmo(period=10).alias("cmo"),
            pl.col("close").ta.ppo(fast_period=5, slow_period=10).alias("ppo"),
            pl.col("close").ta.coppock().alias("coppock"),
            pl.col("close")
            .ta.ultimate_oscillator(high="high", low="low", period1=3, period2=5, period3=8)
            .alias("uo"),
            pl.col("close").ta.fisher_transform(high="high", low="low", period=5).alias("ft"),
            pl.col("close").ta.rvi(open="open", high="high", low="low", period=5).alias("rvi"),
            pl.col("close").ta.kst().alias("kst"),
            pl.col("close").ta.dpo(period=10).alias("dpo"),
            pl.col("close").ta.bop(open="open", high="high", low="low").alias("bop"),
            pl.col("close").ta.cfo(period=5).alias("cfo"),
        ]
    )

    assert "rsi_14" in df_res.columns
    assert "cmo" in df_res.columns
    assert "ppo" in df_res.columns
    assert "uo" in df_res.columns
    assert "bop" in df_res.columns
    assert "cfo" in df_res.columns


def test_namespace_volatility(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.true_range(high="high", low="low").alias("tr"),
            pl.col("close").ta.atr(high="high", low="low", period=14).alias("atr_14"),
            pl.col("close").ta.bollinger_bands(period=20).alias("bb"),
            pl.col("close").ta.keltner_channels(high="high", low="low").alias("kc"),
            pl.col("close").ta.donchian_channels(high="high", low="low", period=10).alias("dc"),
            pl.col("close").ta.stddev(period=10).alias("std"),
            pl.col("close").ta.ulcer_index(period=10).alias("ui"),
            pl.col("close").ta.chandelier_exit(high="high", low="low", period=10).alias("ce"),
            pl.col("close").ta.natr(high="high", low="low", period=10).alias("natr"),
            pl.col("close").ta.historical_volatility(period=10).alias("hv"),
            pl.col("close").ta.parkinson_volatility(high="high", low="low", period=10).alias("pv"),
            pl.col("close")
            .ta.garman_klass_volatility(open="open", high="high", low="low", period=10)
            .alias("gk"),
            pl.col("close")
            .ta.chaikin_volatility(high="high", low="low", period=5, roc_period=5)
            .alias("cv"),
            pl.col("close").ta.donchian_width(high="high", low="low", period=10).alias("dw"),
        ]
    )

    assert "tr" in df_res.columns
    assert "ce" in df_res.columns
    assert "natr" in df_res.columns
    assert "pv" in df_res.columns
    assert "gk" in df_res.columns
    assert "dw" in df_res.columns


def test_namespace_volume(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.obv(volume_col="volume").alias("obv"),
            pl.col("close").ta.vwap(high="high", low="low", volume_col="volume").alias("vwap"),
            pl.col("close").ta.vwma(volume_col="volume", period=10).alias("vwma_10"),
            pl.col("close")
            .ta.cmf(high="high", low="low", volume_col="volume", period=10)
            .alias("cmf"),
            pl.col("close").ta.adl(high="high", low="low", volume_col="volume").alias("adl"),
            pl.col("close")
            .ta.mfi(high="high", low="low", volume_col="volume", period=10)
            .alias("mfi"),
            pl.col("close")
            .ta.eom(high="high", low="low", volume_col="volume", period=10)
            .alias("eom"),
            pl.col("close").ta.klinger(high="high", low="low", volume_col="volume").alias("kvo"),
            pl.col("close").ta.nvi(volume_col="volume").alias("nvi"),
            pl.col("close").ta.pvi(volume_col="volume").alias("pvi"),
            pl.col("close").ta.pvt(volume_col="volume").alias("pvt"),
            pl.col("close").ta.force_index(volume_col="volume", period=5).alias("fi"),
            pl.col("close")
            .ta.vpci(volume_col="volume", period_short=3, period_long=6)
            .alias("vpci"),
            pl.col("close").ta.vr(volume_col="volume", period=10).alias("vr"),
        ]
    )

    assert "obv" in df_res.columns
    assert "kvo" in df_res.columns
    assert "nvi" in df_res.columns
    assert "fi" in df_res.columns
    assert "vpci" in df_res.columns
    assert "vr" in df_res.columns


def test_namespace_multi_asset_over(multi_asset_ohlcv_df: pl.DataFrame):
    """The killer feature: parallel execution across assets using .over('ticker')."""
    res = multi_asset_ohlcv_df.with_columns(
        [
            pl.col("close").ta.sma(5).over("ticker").alias("sma_5"),
            pl.col("close").ta.rsi(10).over("ticker").alias("rsi_10"),
            pl.col("close").ta.obv("volume").over("ticker").alias("obv"),
            pl.col("close").ta.dema(5).over("ticker").alias("dema_5"),
            pl.col("close").ta.roc(5).over("ticker").alias("roc_5"),
        ]
    )

    assert "sma_5" in res.columns
    assert "dema_5" in res.columns
    assert "roc_5" in res.columns
