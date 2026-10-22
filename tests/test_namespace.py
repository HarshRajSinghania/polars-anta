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
        ]
    )

    assert "sma_5" in df_res.columns
    assert "ema_5" in df_res.columns
    assert "wma_5" in df_res.columns
    assert "hma_9" in df_res.columns
    assert df_res["sma_5"][4] is not None


def test_namespace_momentum(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.rsi(14).alias("rsi_14"),
            pl.col("close").ta.macd().alias("macd_struct"),
            pl.col("close").ta.stochastic(high="high", low="low").alias("stoch_struct"),
            pl.col("close").ta.williams_r(high="high", low="low").alias("wr_14"),
        ]
    )

    assert "rsi_14" in df_res.columns
    assert "macd_struct" in df_res.columns
    assert "stoch_struct" in df_res.columns
    assert "wr_14" in df_res.columns


def test_namespace_volatility(sample_ohlcv_df: pl.DataFrame):
    df_res = sample_ohlcv_df.select(
        [
            pl.col("close").ta.true_range(high="high", low="low").alias("tr"),
            pl.col("close").ta.atr(high="high", low="low", period=14).alias("atr_14"),
            pl.col("close").ta.bollinger_bands(period=20).alias("bb"),
            pl.col("close").ta.keltner_channels(high="high", low="low").alias("kc"),
        ]
    )

    assert "tr" in df_res.columns
    assert "atr_14" in df_res.columns
    assert "bb" in df_res.columns
    assert "kc" in df_res.columns


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
        ]
    )

    assert "obv" in df_res.columns
    assert "vwap" in df_res.columns
    assert "vwma_10" in df_res.columns
    assert "cmf" in df_res.columns
    assert "adl" in df_res.columns


def test_namespace_multi_asset_over(multi_asset_ohlcv_df: pl.DataFrame):
    """The killer feature: parallel execution across assets using .over('ticker')."""
    res = multi_asset_ohlcv_df.with_columns(
        [
            pl.col("close").ta.sma(5).over("ticker").alias("sma_5"),
            pl.col("close").ta.rsi(10).over("ticker").alias("rsi_10"),
            pl.col("close").ta.obv("volume").over("ticker").alias("obv"),
        ]
    )

    assert "sma_5" in res.columns
    assert "rsi_10" in res.columns
    assert "obv" in res.columns

    # Verify AAPL and MSFT each have their independent window calculation
    aapl_first_sma = res.filter(pl.col("ticker") == "AAPL")["sma_5"].to_list()
    assert aapl_first_sma[0] is None
    assert aapl_first_sma[4] is not None

    msft_first_sma = res.filter(pl.col("ticker") == "MSFT")["sma_5"].to_list()
    assert msft_first_sma[0] is None
    assert msft_first_sma[4] is not None
