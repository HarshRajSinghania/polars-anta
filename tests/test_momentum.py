"""Tests for momentum indicators."""

from __future__ import annotations

import polars as pl
import pytest

from polars_ta.momentum import macd, rsi, stochastic, williams_r


def test_rsi(sample_ohlcv_df: pl.DataFrame):
    period = 14
    result = sample_ohlcv_df.select(rsi("close", period=period).alias("rsi"))
    vals = [v for v in result["rsi"].to_list() if v is not None]

    assert len(vals) > 0
    # RSI must be strictly between 0 and 100
    for v in vals:
        assert 0.0 <= v <= 100.0


def test_rsi_monotonic_up():
    df = pl.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]})
    res = df.select(rsi("close", period=3).alias("rsi"))
    # In pure monotonic uptrend, RSI must reach 100
    assert pytest.approx(res["rsi"][-1], rel=1e-3) == 100.0


def test_rsi_invalid_period():
    with pytest.raises(ValueError, match="period must be a positive integer"):
        rsi("close", period=0)


def test_macd(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(macd("close", 12, 26, 9).alias("macd_struct")).unnest(
        "macd_struct"
    )

    assert "macd" in result.columns
    assert "signal" in result.columns
    assert "histogram" in result.columns

    # Verify histogram = macd - signal
    macd_vals = result["macd"].to_list()
    signal_vals = result["signal"].to_list()
    hist_vals = result["histogram"].to_list()

    for m, s, h in zip(macd_vals[-5:], signal_vals[-5:], hist_vals[-5:], strict=True):
        assert pytest.approx(h, rel=1e-5) == (m - s)


def test_macd_invalid_periods():
    with pytest.raises(ValueError):
        macd("close", fast_period=0)
    with pytest.raises(ValueError, match="fast_period must be less than slow_period"):
        macd("close", fast_period=26, slow_period=12)


def test_stochastic(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        stochastic("high", "low", "close", k_period=14, d_period=3).alias("stoch")
    ).unnest("stoch")

    assert "k" in result.columns
    assert "d" in result.columns

    valid_k = [x for x in result["k"].to_list() if x is not None]
    valid_d = [x for x in result["d"].to_list() if x is not None]

    assert len(valid_k) > 0
    assert len(valid_d) > 0

    for val in valid_k:
        assert 0.0 <= val <= 100.0
    for val in valid_d:
        assert 0.0 <= val <= 100.0


def test_williams_r(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(williams_r("high", "low", "close", period=14).alias("wr"))
    valid_wr = [x for x in result["wr"].to_list() if x is not None]

    assert len(valid_wr) > 0
    # Williams %R is scaled between -100 and 0
    for val in valid_wr:
        assert -100.0 <= val <= 0.0
