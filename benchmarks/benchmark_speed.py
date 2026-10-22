"""Speed benchmark comparing polars-ta execution speed across data scales."""

from __future__ import annotations

import time

import polars as pl

import polars_ta  # noqa: F401


def generate_benchmark_data(n_rows: int = 1_000_000, n_tickers: int = 100) -> pl.DataFrame:
    """Generate synthetic OHLCV data for benchmarking."""
    import numpy as np

    print(f"Generating synthetic dataset: {n_rows:,} rows across {n_tickers} tickers...")
    rows_per_ticker = n_rows // n_tickers

    tickers = [f"TICKER_{i:03d}" for i in range(n_tickers)]
    all_tickers = np.repeat(tickers, rows_per_ticker)

    # Random walk price generator
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.015, size=len(all_tickers))
    price_series = 100.0 * np.exp(np.cumsum(returns))

    highs = price_series * (1.0 + np.abs(np.random.normal(0, 0.005, size=len(all_tickers))))
    lows = price_series * (1.0 - np.abs(np.random.normal(0, 0.005, size=len(all_tickers))))
    volumes = np.random.randint(1000, 500_000, size=len(all_tickers))

    df = pl.DataFrame(
        {
            "ticker": all_tickers,
            "high": highs,
            "low": lows,
            "close": price_series,
            "volume": volumes,
        }
    )
    return df


def benchmark_indicators(df: pl.DataFrame):
    """Benchmark indicators across all rows."""
    print(f"\nBenchmarking polars-ta on {len(df):,} rows with multi-core Rust engine...")

    benchmarks = [
        ("SMA(20)", lambda d: d.select(pl.col("close").ta.sma(20))),
        ("EMA(20)", lambda d: d.select(pl.col("close").ta.ema(20))),
        ("RSI(14)", lambda d: d.select(pl.col("close").ta.rsi(14))),
        ("MACD(12,26,9)", lambda d: d.select(pl.col("close").ta.macd())),
        ("Bollinger Bands(20,2)", lambda d: d.select(pl.col("close").ta.bollinger_bands(20, 2.0))),
        ("ATR(14)", lambda d: d.select(pl.col("close").ta.atr("high", "low", 14))),
        ("OBV", lambda d: d.select(pl.col("close").ta.obv("volume"))),
        ("VWAP", lambda d: d.select(pl.col("close").ta.vwap("high", "low", "volume"))),
    ]

    results = []
    for name, fn in benchmarks:
        # Warmup
        _ = fn(df[:1000])

        start_time = time.perf_counter()
        _ = fn(df)
        elapsed = time.perf_counter() - start_time
        print(f"  ⚡ {name:25s}: {elapsed:.4f}s ({len(df) / elapsed:,.0f} rows/sec)")
        results.append((name, elapsed))

    # Benchmark multi-asset parallel execution (.over("ticker"))
    print("\nBenchmarking Parallel Multi-Asset Execution (.over('ticker'))...")
    start_over = time.perf_counter()
    _ = df.with_columns(
        [
            pl.col("close").ta.sma(20).over("ticker").alias("sma"),
            pl.col("close").ta.rsi(14).over("ticker").alias("rsi"),
            pl.col("close").ta.obv("volume").over("ticker").alias("obv"),
        ]
    )
    elapsed_over = time.perf_counter() - start_over
    print(f"  🚀 3 Indicators x {len(df):,} rows over tickers: {elapsed_over:.4f}s")

    return results


if __name__ == "__main__":
    df = generate_benchmark_data(n_rows=500_000, n_tickers=50)
    benchmark_indicators(df)
