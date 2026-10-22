"""Tests for trend indicators."""

from __future__ import annotations

import polars as pl
import pytest

from polars_ta.trend import ema, hma, sma, wma


def test_sma(sample_ohlcv_df: pl.DataFrame):
    period = 5
    result = sample_ohlcv_df.select(sma("close", period=period).alias("sma_5"))
    col = result["sma_5"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    # First (period - 1) values should be null
    for i in range(period - 1):
        assert col[i] is None

    # Check 5th element calculation
    expected_sma_5 = sum(sample_ohlcv_df["close"][:5]) / 5
    assert pytest.approx(col[4], rel=1e-5) == expected_sma_5


def test_sma_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        sma("close", period=0)


def test_ema(sample_ohlcv_df: pl.DataFrame):
    period = 10
    result = sample_ohlcv_df.select(ema("close", period=period).alias("ema_10"))
    col = result["ema_10"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    assert col[0] is not None
    # EMA follows trend
    assert col[-1] > col[0]


def test_ema_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        ema("close", period=-2)


def test_wma(sample_ohlcv_df: pl.DataFrame):
    period = 3
    result = sample_ohlcv_df.select(wma("close", period=period).alias("wma_3"))
    col = result["wma_3"].to_list()

    # For period 3, weights are 1, 2, 3 -> sum is 6
    p = sample_ohlcv_df["close"].to_list()
    expected = (p[0] * 1 + p[1] * 2 + p[2] * 3) / 6.0
    assert pytest.approx(col[2], rel=1e-5) == expected


def test_wma_period_1(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(wma("close", period=1).alias("wma_1"))
    assert result["wma_1"].to_list() == sample_ohlcv_df["close"].to_list()


def test_wma_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        wma("close", period=0)


def test_hma(sample_ohlcv_df: pl.DataFrame):
    period = 9
    result = sample_ohlcv_df.select(hma("close", period=period).alias("hma_9"))
    col = result["hma_9"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    # Later values should not be None and should be realistic price values
    assert col[-1] is not None
    assert 100.0 < col[-1] < 150.0


def test_hma_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        hma("close", period=-1)
