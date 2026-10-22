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
        ]
    )

    assert "sma_5" in df_res.columns
    assert "dema_5" in df_res.columns
    assert "tema_5" in df_res.columns
    assert "zlema_5" in df_res.columns
    assert "vortex" in df_res.columns
    assert "adx" in df_res.columns


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
        ]
    )

    assert "rsi_14" in df_res.columns
    assert "cci" in df_res.columns
    assert "roc" in df_res.columns
    assert "srsi" in df_res.columns
    assert "ao" in df_res.columns
    assert "tsi" in df_res.columns


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
        ]
    )

    assert "tr" in df_res.columns
    assert "bb" in df_res.columns
    assert "dc" in df_res.columns
    assert "std" in df_res.columns
    assert "ui" in df_res.columns


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
        ]
    )

    assert "obv" in df_res.columns
    assert "mfi" in df_res.columns
    assert "eom" in df_res.columns


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
