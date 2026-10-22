from __future__ import annotations

import polars as pl
import pytest

from polars_ta.alpha import (
    co_ratio,
    decay_linear,
    entropy,
    hl_ratio,
    relative_spread,
    returns_divergence,
    ts_argmax,
    ts_argmin,
    ts_rank,
    ts_zscore,
    twap,
    volume_weighted_momentum,
)


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
            "high": [10.5, 11.5, 12.8, 13.5, 14.8, 15.6, 16.9, 17.5, 18.8, 19.5, 20.8, 21.5],
            "low": [9.8, 10.7, 11.6, 12.8, 13.7, 14.8, 15.7, 16.8, 17.6, 18.7, 19.7, 20.8],
            "close": [10.2, 11.3, 12.1, 13.4, 14.2, 15.5, 16.2, 17.4, 18.2, 19.4, 20.2, 21.3],
            "volume": [100, 150, 120, 180, 200, 220, 190, 250, 230, 280, 300, 310],
        }
    )


def test_ts_rank_argmax_argmin(sample_ohlcv: pl.DataFrame):
    res = sample_ohlcv.select(
        [
            ts_rank("close", period=5).alias("rank"),
            ts_argmax("close", period=5).alias("argmax"),
            ts_argmin("close", period=5).alias("argmin"),
        ]
    )
    assert len(res) == 12
    # In strictly increasing sequence, last element is highest -> rank is 1.0, argmax is 0 bars ago, argmin is 4 bars ago
    assert res["rank"][-1] == 1.0
    assert res["argmax"][-1] == 0.0
    assert res["argmin"][-1] == 4.0


def test_decay_linear_and_zscore(sample_ohlcv: pl.DataFrame):
    res = sample_ohlcv.select(
        [
            decay_linear("close", period=5).alias("decay"),
            ts_zscore("close", period=5).alias("z"),
        ]
    )
    assert len(res) == 12
    assert res["decay"][-1] is not None
    assert res["z"][-1] is not None
    assert res["z"][-1] > 0.0


def test_ratios_and_divergence(sample_ohlcv: pl.DataFrame):
    res = sample_ohlcv.select(
        [
            hl_ratio("high", "low").alias("hl"),
            co_ratio("close", "open").alias("co"),
            returns_divergence("close", "volume", period=5).alias("div"),
            volume_weighted_momentum("close", "volume", period=5).alias("vwm"),
            relative_spread("high", "low", "close").alias("spread"),
            twap("high", "low", "close", period=5).alias("twap"),
            entropy("close", period=6, bins=3).alias("entropy"),
        ]
    )
    assert len(res) == 12
    assert res["hl"][-1] > 0.0
    assert res["spread"][-1] > 0.0
    assert res["twap"][-1] > 0.0
    assert res["entropy"][-1] is not None
