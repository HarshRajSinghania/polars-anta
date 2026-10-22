"""Pytest fixtures for polars-ta testing."""

from __future__ import annotations

import polars as pl
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pl.DataFrame:
    """Deterministic OHLCV sample DataFrame with 30 rows."""
    prices = [
        100.0,
        102.0,
        101.5,
        103.0,
        105.0,
        104.0,
        106.0,
        108.0,
        107.5,
        109.0,
        111.0,
        110.0,
        108.5,
        107.0,
        106.5,
        108.0,
        110.5,
        112.0,
        111.0,
        113.0,
        115.0,
        114.5,
        116.0,
        118.0,
        117.0,
        119.0,
        121.0,
        120.0,
        122.5,
        125.0,
    ]
    highs = [p + 1.5 for p in prices]
    lows = [p - 1.2 for p in prices]
    opens = [p - 0.5 for p in prices]
    volumes = [1000 + (i * 50) for i in range(len(prices))]

    return pl.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        }
    )


@pytest.fixture
def multi_asset_ohlcv_df() -> pl.DataFrame:
    """Multi-symbol dataset for testing grouping/over expressions."""
    rows = []
    for ticker in ["AAPL", "MSFT", "GOOG"]:
        base_price = 100.0 if ticker == "AAPL" else (200.0 if ticker == "MSFT" else 150.0)
        for i in range(25):
            price = base_price + i * 1.5
            rows.append(
                {
                    "ticker": ticker,
                    "open": price - 0.5,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 1000 + i * 100,
                }
            )
    return pl.DataFrame(rows)
