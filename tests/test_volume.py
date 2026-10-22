"""Tests for volume indicators."""

from __future__ import annotations

import polars as pl

from polars_ta.volume import adl, cmf, eom, mfi, obv, vwap, vwma


def test_obv(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(obv("close", "volume").alias("obv"))
    obv_vals = result["obv"].to_list()

    assert len(obv_vals) == len(sample_ohlcv_df)
    assert all(x is not None for x in obv_vals)


def test_vwap_cumulative(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(vwap("high", "low", "close", "volume").alias("vwap"))
    vals = result["vwap"].to_list()

    assert len(vals) == len(sample_ohlcv_df)
    for v in vals:
        assert 90.0 < v < 140.0


def test_vwap_rolling(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(
        vwap("high", "low", "close", "volume", period=5).alias("vwap_5")
    )
    vals = result["vwap_5"].to_list()
    assert all(x is None for x in vals[:4])
    assert all(x is not None for x in vals[4:])


def test_vwma(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(vwma("close", "volume", period=10).alias("vwma"))
    vals = result["vwma"].to_list()

    assert all(x is None for x in vals[:9])
    assert all(x is not None for x in vals[9:])


def test_cmf(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(cmf("high", "low", "close", "volume", period=10).alias("cmf"))
    vals = [x for x in result["cmf"].to_list() if x is not None]

    assert len(vals) > 0
    for v in vals:
        assert -1.0 <= v <= 1.0


def test_adl(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(adl("high", "low", "close", "volume").alias("adl"))
    vals = result["adl"].to_list()

    assert len(vals) == len(sample_ohlcv_df)
    assert all(x is not None for x in vals)


def test_mfi(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(mfi("high", "low", "close", "volume", period=10).alias("mfi"))
    vals = [x for x in result["mfi"].to_list() if x is not None]
    assert len(vals) > 0
    for v in vals:
        assert 0.0 <= v <= 100.0


def test_eom(sample_ohlcv_df: pl.DataFrame):
    result = sample_ohlcv_df.select(eom("high", "low", "volume", period=10).alias("eom"))
    vals = [x for x in result["eom"].to_list() if x is not None]
    assert len(vals) > 0
