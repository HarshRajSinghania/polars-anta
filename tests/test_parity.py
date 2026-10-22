"""Mathematical parity and precision verification tests for polars-anta indicators.

Verifies that polars-anta indicators match textbook mathematical definitions
and golden reference implementations within machine epsilon / float precision (< 1e-6).
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_anta.stats import linear_regression_slope, zscore
from polars_anta.trend import dema, ema, sma, wma
from polars_anta.volatility import bollinger_bands


@pytest.fixture
def golden_price_series() -> list[float]:
    """Well-known golden closing price sequence (from Murphy's Technical Analysis)."""
    return [
        22.27,
        22.19,
        22.08,
        22.17,
        22.18,
        22.13,
        22.23,
        22.43,
        22.24,
        22.29,
        22.15,
        22.39,
        22.38,
        22.61,
        23.36,
        24.05,
        23.75,
        23.83,
        23.95,
        23.63,
        23.82,
        23.87,
        23.65,
        23.19,
        23.10,
        23.33,
        22.68,
        23.10,
        22.40,
        22.17,
    ]


def test_sma_mathematical_parity(golden_price_series: list[float]):
    period = 10
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(sma("close", period=period).alias("sma"))["sma"].to_list()

    # Manual textbook reference calculation
    for i in range(period - 1, len(golden_price_series)):
        window = golden_price_series[i - period + 1 : i + 1]
        expected_sma = sum(window) / float(period)
        assert pytest.approx(res[i], abs=1e-7) == expected_sma


def test_wma_mathematical_parity(golden_price_series: list[float]):
    period = 5
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(wma("close", period=period).alias("wma"))["wma"].to_list()

    weights = list(range(1, period + 1))
    weight_sum = sum(weights)

    for i in range(period - 1, len(golden_price_series)):
        window = golden_price_series[i - period + 1 : i + 1]
        expected_wma = sum(w * x for w, x in zip(weights, window, strict=False)) / float(weight_sum)
        assert pytest.approx(res[i], abs=1e-7) == expected_wma


def test_ema_mathematical_parity(golden_price_series: list[float]):
    period = 10
    alpha = 2.0 / (period + 1.0)
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(ema("close", period=period).alias("ema"))["ema"].to_list()

    # Recurrence relation: EMA[t] = alpha * Price[t] + (1 - alpha) * EMA[t-1]
    curr = golden_price_series[0]
    assert pytest.approx(res[0], abs=1e-7) == curr

    for i in range(1, len(golden_price_series)):
        curr = alpha * golden_price_series[i] + (1.0 - alpha) * curr
        assert pytest.approx(res[i], abs=1e-6) == curr


def test_dema_mathematical_parity(golden_price_series: list[float]):
    period = 10
    df = pl.DataFrame({"close": golden_price_series})
    # DEMA = 2 * EMA - EMA(EMA)
    res_dema = df.select(dema("close", period=period).alias("dema"))["dema"].to_list()

    e1 = df.select(ema("close", period=period).alias("e1"))
    e2 = e1.select(ema("e1", period=period).alias("e2"))
    expected_dema = (2.0 * e1["e1"] - e2["e2"]).to_list()

    for r, e in zip(res_dema, expected_dema, strict=False):
        assert pytest.approx(r, abs=1e-6) == e


def test_bollinger_bands_parity(golden_price_series: list[float]):
    period = 10
    num_std = 2.0
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(bollinger_bands("close", period=period, num_std=num_std).alias("bb")).unnest(
        "bb"
    )

    for i in range(period - 1, len(golden_price_series)):
        window = golden_price_series[i - period + 1 : i + 1]
        mean = sum(window) / float(period)
        # Sample standard deviation (ddof=1)
        variance = sum((x - mean) ** 2 for x in window) / float(period - 1)
        std = variance**0.5

        assert pytest.approx(res["middle"][i], abs=1e-7) == mean
        assert pytest.approx(res["upper"][i], abs=1e-6) == mean + num_std * std
        assert pytest.approx(res["lower"][i], abs=1e-6) == mean - num_std * std


def test_linear_regression_slope_parity(golden_price_series: list[float]):
    period = 10
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(linear_regression_slope("close", period=period).alias("slope"))[
        "slope"
    ].to_list()

    # OLS closed-form: sum((t - mean_t) * (y - mean_y)) / sum((t - mean_t)^2)
    t = list(range(period))
    mean_t = sum(t) / float(period)
    denom = sum((x - mean_t) ** 2 for x in t)

    for i in range(period - 1, len(golden_price_series)):
        window = golden_price_series[i - period + 1 : i + 1]
        mean_y = sum(window) / float(period)
        nom = sum((k - mean_t) * (y - mean_y) for k, y in zip(t, window, strict=False))
        expected_slope = nom / denom
        assert pytest.approx(res[i], abs=1e-6) == expected_slope


def test_zscore_parity(golden_price_series: list[float]):
    period = 10
    df = pl.DataFrame({"close": golden_price_series})
    res = df.select(zscore("close", period=period).alias("z"))["z"].to_list()

    for i in range(period - 1, len(golden_price_series)):
        window = golden_price_series[i - period + 1 : i + 1]
        mean = sum(window) / float(period)
        variance = sum((x - mean) ** 2 for x in window) / float(period - 1)
        std = variance**0.5
        expected_z = (golden_price_series[i] - mean) / std
        assert pytest.approx(res[i], abs=1e-6) == expected_z
