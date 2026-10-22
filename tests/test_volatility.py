"""Tests for volatility indicators."""

from __future__ import annotations

import polars as pl

from polars_ta.volatility import (
    atr,
    bollinger_bands,
    donchian_channels,
    keltner_channels,
    stddev,
    true_range,
    ulcer_index,
)


def test_true_range(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(true_range("high", "low", "close").alias("tr"))
    tr_vals = result["tr"].to_list()

    assert len(tr_vals) == len(sample_ohlcv_df)
    for v in tr_vals[1:]:
        assert v > 0.0


def test_atr(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(atr("high", "low", "close", period=14).alias("atr"))
    atr_vals = [v for v in result["atr"].to_list() if v is not None]

    assert len(atr_vals) > 0
    for v in atr_vals:
        assert v > 0.0


def test_bollinger_bands(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        bollinger_bands("close", period=10, num_std=2.0).alias("bb")
    ).unnest("bb")

    assert set(result.columns) == {"upper", "middle", "lower", "bandwidth", "percent_b"}

    uppers = result["upper"].to_list()
    middles = result["middle"].to_list()
    lowers = result["lower"].to_list()

    for u, m, low_val in zip(uppers[9:], middles[9:], lowers[9:], strict=True):
        assert u > m > low_val


def test_keltner_channels(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        keltner_channels("high", "low", "close", ema_period=10, atr_period=5, multiplier=2.0).alias(
            "kc"
        )
    ).unnest("kc")

    assert set(result.columns) == {"upper", "middle", "lower"}

    uppers = result["upper"].to_list()
    middles = result["middle"].to_list()
    lowers = result["lower"].to_list()

    for u, m, low_val in zip(uppers[10:], middles[10:], lowers[10:], strict=True):
        assert u > m > low_val


def test_donchian_channels(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(donchian_channels("high", "low", period=10).alias("dc")).unnest(
        "dc"
    )

    assert set(result.columns) == {"upper", "middle", "lower"}
    uppers = result["upper"].to_list()
    middles = result["middle"].to_list()
    lowers = result["lower"].to_list()

    for u, m, low_val in zip(uppers[9:], middles[9:], lowers[9:], strict=True):
        assert u >= m >= low_val


def test_stddev(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(stddev("close", period=10).alias("std"))
    vals = [x for x in result["std"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v >= 0.0


def test_ulcer_index(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(ulcer_index("close", period=10).alias("ui"))
    vals = [x for x in result["ui"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert v >= 0.0
