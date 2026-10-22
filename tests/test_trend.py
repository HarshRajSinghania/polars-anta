"""Tests for trend indicators."""

from __future__ import annotations

import polars as pl
import pytest

from polars_ta.trend import adx, dema, ema, hma, sma, tema, vortex, wma, zlema


def test_sma(sample_ohlcv_df: pl.DataFrame):
    period = 5
    result = sample_ohlcv_df.select(sma("close", period=period).alias("sma_5"))
    col = result["sma_5"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    for i in range(period - 1):
        assert col[i] is None

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
    assert col[-1] > col[0]


def test_ema_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        ema("close", period=-2)


def test_wma(sample_ohlcv_df: pl.DataFrame):
    period = 3
    result = sample_ohlcv_df.select(wma("close", period=period).alias("wma_3"))
    col = result["wma_3"].to_list()

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
    assert col[-1] is not None
    assert 100.0 < col[-1] < 150.0


def test_hma_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        hma("close", period=-1)


def test_dema(sample_ohlcv_df: pl.DataFrame):
    period = 10
    result = sample_ohlcv_df.select(dema("close", period=period).alias("dema_10"))
    col = result["dema_10"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    assert col[-1] is not None
    assert 100.0 < col[-1] < 150.0


def test_tema(sample_ohlcv_df: pl.DataFrame):
    period = 10
    result = sample_ohlcv_df.select(tema("close", period=period).alias("tema_10"))
    col = result["tema_10"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    assert col[-1] is not None
    assert 100.0 < col[-1] < 150.0


def test_zlema(sample_ohlcv_df: pl.DataFrame):
    period = 10
    result = sample_ohlcv_df.select(zlema("close", period=period).alias("zlema_10"))
    col = result["zlema_10"].to_list()

    assert len(col) == len(sample_ohlcv_df)
    assert col[-1] is not None
    assert 100.0 < col[-1] < 150.0


def test_vortex(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(vortex("high", "low", "close", period=14).alias("vi")).unnest(
        "vi"
    )

    assert "vi_plus" in result.columns
    assert "vi_minus" in result.columns

    vi_plus = [x for x in result["vi_plus"].to_list() if x is not None]
    vi_minus = [x for x in result["vi_minus"].to_list() if x is not None]

    assert len(vi_plus) > 0
    assert len(vi_minus) > 0
    for v in vi_plus:
        assert v > 0.0
    for v in vi_minus:
        assert v > 0.0


def test_adx(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        adx("high", "low", "close", period=14).alias("adx_struct")
    ).unnest("adx_struct")

    assert "adx" in result.columns
    assert "plus_di" in result.columns
    assert "minus_di" in result.columns

    adx_vals = [x for x in result["adx"].to_list() if x is not None]
    assert len(adx_vals) > 0
    for a in adx_vals:
        assert 0.0 <= a <= 100.0
