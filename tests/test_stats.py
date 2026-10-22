from __future__ import annotations

import polars as pl
import pytest

from polars_ta.stats import (
    kurtosis,
    linear_regression_angle,
    linear_regression_forecast,
    linear_regression_intercept,
    linear_regression_slope,
    percentile_rank,
    rolling_alpha,
    rolling_beta,
    rolling_corr,
    rolling_cov,
    skewness,
    zscore,
)


@pytest.fixture
def sample_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            "asset_ret": [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.01, -0.01, 0.02, 0.01],
            "bench_ret": [0.01, 0.01, -0.005, 0.02, 0.01, -0.01, 0.02, 0.005, -0.01, 0.01, 0.005],
        }
    )


def test_zscore(sample_data: pl.DataFrame):
    res = sample_data.select(zscore("close", period=5).alias("z")).get_column("z")
    assert len(res) == 11
    # For a strictly increasing series, the last element is greater than the rolling mean -> positive zscore
    assert res[-1] > 0.0


def test_percentile_rank(sample_data: pl.DataFrame):
    res = sample_data.select(percentile_rank("close", period=5).alias("pr")).get_column("pr")
    assert len(res) == 11
    # For strictly increasing sequence, current value is always highest in window -> 1.0
    assert res[-1] == 1.0


def test_skewness_kurtosis(sample_data: pl.DataFrame):
    res = sample_data.select(
        [
            skewness("close", period=5).alias("skew"),
            kurtosis("close", period=5).alias("kurt"),
        ]
    )
    assert len(res) == 11
    # Uniform linear progression has ~0 skewness
    assert abs(res["skew"][-1]) < 0.1
    assert res["kurt"][-1] is not None


def test_rolling_cov_corr(sample_data: pl.DataFrame):
    res = sample_data.select(
        [
            rolling_cov("asset_ret", "bench_ret", period=5).alias("cov"),
            rolling_corr("asset_ret", "bench_ret", period=5).alias("corr"),
        ]
    )
    assert len(res) == 11
    corr_val = res["corr"][-1]
    assert corr_val is not None
    assert -1.0 <= corr_val <= 1.0


def test_rolling_beta_alpha(sample_data: pl.DataFrame):
    res = sample_data.select(
        [
            rolling_beta("asset_ret", "bench_ret", period=5).alias("beta"),
            rolling_alpha("asset_ret", "bench_ret", rf=0.0, period=5).alias("alpha"),
        ]
    )
    assert len(res) == 11
    assert res["beta"][-1] is not None
    assert res["alpha"][-1] is not None


def test_linear_regression(sample_data: pl.DataFrame):
    res = sample_data.select(
        [
            linear_regression_slope("close", period=5).alias("slope"),
            linear_regression_intercept("close", period=5).alias("intercept"),
            linear_regression_angle("close", period=5).alias("angle"),
            linear_regression_forecast("close", period=5).alias("forecast"),
        ]
    )
    assert len(res) == 11
    # For series [10, 11, 12, ...], step is 1.0 so slope must be 1.0
    assert pytest.approx(res["slope"][-1], rel=1e-5) == 1.0
    # Current bar value is 20.0, forecast at current bar should be 20.0
    assert pytest.approx(res["forecast"][-1], rel=1e-5) == 20.0
    # Angle for slope 1.0 is 45 degrees
    assert pytest.approx(res["angle"][-1], rel=1e-5) == 45.0
