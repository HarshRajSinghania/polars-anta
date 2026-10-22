"""Quickstart example demonstrating polars-anta usage and multi-asset grouping."""

from __future__ import annotations

import sys

import polars as pl

# Importing polars_anta registers both `.anta` and `.ta` namespaces on expressions
import polars_anta  # noqa: F401

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("--- 1. Single Asset Example (using .anta namespace) ---")
    df = pl.DataFrame(
        {
            "open": [100.0, 101.5, 102.0, 101.0, 103.5, 105.0, 104.2, 106.0],
            "high": [102.0, 103.0, 103.5, 102.5, 105.0, 106.8, 105.5, 107.5],
            "low": [99.5, 100.8, 101.0, 100.2, 102.8, 103.9, 103.5, 105.0],
            "close": [101.5, 102.0, 101.0, 102.5, 104.5, 104.0, 105.8, 107.0],
            "volume": [1000, 1200, 950, 1300, 1500, 1100, 1400, 1600],
        }
    )

    # Compute SMA, RSI, Bollinger Bands, and OBV using .anta (or .ta)
    computed = df.with_columns(
        [
            pl.col("close").anta.sma(3).alias("sma_3"),
            pl.col("close").anta.rsi(3).alias("rsi_3"),
            pl.col("close").anta.bollinger_bands(period=3).alias("bb"),
            pl.col("close").anta.obv("volume").alias("obv"),
        ]
    )

    print("Computed Indicators:")
    print(computed.select(["close", "sma_3", "rsi_3", "obv"]))

    print("\n--- 2. Unnesting Structured Indicators (Bollinger Bands) ---")
    print(computed.select(["close", "bb"]).unnest("bb"))

    print("\n--- 3. Multi-Asset Scaling with .over('ticker') ---")
    multi_asset = pl.DataFrame(
        {
            "ticker": ["AAPL"] * 4 + ["NVDA"] * 4,
            "close": [150.0, 152.0, 151.0, 153.5, 450.0, 460.0, 455.0, 470.0],
        }
    )

    result_multi = multi_asset.with_columns(
        pl.col("close").anta.sma(2).over("ticker").alias("sma_2"),
        pl.col("close").anta.rsi(2).over("ticker").alias("rsi_2"),
    )
    print(result_multi)


if __name__ == "__main__":
    main()
