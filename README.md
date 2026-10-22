<div align="center">

# polars-ta

**High-performance, zero-C-dependency technical analysis and alpha factor library powered natively by Polars Expressions.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org)
[![Polars Version](https://img.shields.io/badge/polars-%3E%3D1.0.0-orange)](https://pola.rs)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Zero C Dependencies](https://img.shields.io/badge/dependencies-pure%20polars-brightgreen)](#)

[Features](#key-features) • [Comparison](#why-polars-ta) • [Installation](#installation) • [Quickstart](#quickstart) • [Indicators](#supported-indicators) • [Development](#development--testing)

</div>

---

## Why `polars-ta`?

Quantitative researchers and financial engineers have historically faced a trade-off between installation complexity and computation speed:

| Capability | TA-Lib | pandas-ta | `polars-ta` |
| :--- | :---: | :---: | :---: |
| **Dependencies** | Requires external C library & MSVC / GCC | Pure Python / NumPy | **Zero C compilation** (pure Polars) |
| **Execution Engine** | Single-threaded C loops | Single-threaded Pandas / GIL | **Multi-threaded Rust** query engine |
| **Multi-Asset Scaling** | Manual sequential loops | Slow `.groupby().apply()` | **Native `.over("ticker")`** parallelization |
| **Lazy Evaluation** | Unsupported | Unsupported | **Full `pl.LazyFrame`** streaming support |
| **Memory Layout** | Array conversion overhead | High Pandas memory footprint | **Zero-copy Arrow memory format** |

`polars-ta` eliminates this compromise by implementing financial indicators directly as **Polars Expressions (`pl.Expr`)**. Computations are pushed into Polars' multi-threaded Rust engine, eliminating Python GIL bottlenecks without requiring any C compilers.

---

## Key Features

- **Expression-First Design**: Compose indicators directly within `.select()`, `.with_columns()`, and lazy query plans.
- **Parallel Multi-Asset Computation**: Compute indicators across thousands of tickers in parallel using Polars' native window context:
  ```python
  pl.col("close").ta.rsi(14).over("symbol")
  ```
- **Structured Outputs**: Multi-component indicators (MACD, Bollinger Bands, Stochastic, Keltner Channels) return typed `pl.Struct` expressions that maintain clean schemas.
- **Zero-C Compilation**: `pip install polars-ta` works out of the box across Linux, macOS, and Windows.

---

## Installation

```bash
pip install polars-ta
```

*Requirements: Python >= 3.10, Polars >= 1.0.0*

---

## Quickstart

### 1. Modern Chained Expression Syntax (`.ta`)

```python
import polars as pl
import polars_ta  # Registers the `.ta` expression namespace

df = pl.DataFrame(
    {
        "open": [100.0, 102.0, 101.5, 103.0, 105.0],
        "high": [101.5, 103.5, 103.0, 104.5, 106.5],
        "low": [98.8, 100.8, 100.3, 101.8, 103.8],
        "close": [100.0, 102.0, 101.5, 103.0, 105.0],
        "volume": [1000, 1050, 1100, 1150, 1200],
    }
)

# Compute indicators directly on expressions
result = df.with_columns(
    pl.col("close").ta.sma(period=3).alias("sma_3"),
    pl.col("close").ta.rsi(period=3).alias("rsi_3"),
    pl.col("close").ta.bollinger_bands(period=3).alias("bb"),
)

# Unnest structured outputs as needed
print(result.unnest("bb"))
```

### 2. Market-Wide Multi-Asset Parallel Execution

Running technical indicators across an entire investment universe is a common bottleneck. In `polars-ta`, Polars distributes evaluation across all CPU cores automatically:

```python
# Compute indicators across thousands of symbols simultaneously in Rust
universe.with_columns(
    [
        pl.col("close").ta.sma(20).over("ticker").alias("sma_20"),
        pl.col("close").ta.rsi(14).over("ticker").alias("rsi_14"),
        pl.col("close").ta.macd().over("ticker").alias("macd"),
        pl.col("close").ta.obv("volume").over("ticker").alias("obv"),
    ]
)
```

---

## Supported Indicators (32 Indicators)

### Trend
- **`sma(close, period=10)`**: Simple Moving Average
- **`ema(close, period=10)`**: Exponential Moving Average
- **`wma(close, period=10)`**: Linearly-Weighted Moving Average
- **`hma(close, period=10)`**: Hull Moving Average
- **`dema(close, period=10)`**: Double Exponential Moving Average
- **`tema(close, period=10)`**: Triple Exponential Moving Average
- **`zlema(close, period=10)`**: Zero-Lag Exponential Moving Average
- **`vortex(high, low, close, period=14)`**: Vortex Indicator (`vi_plus`, `vi_minus`)
- **`adx(high, low, close, period=14)`**: Average Directional Index (`adx`, `plus_di`, `minus_di`)

### Momentum
- **`rsi(close, period=14)`**: Relative Strength Index (Wilder's smoothing)
- **`macd(close, fast=12, slow=26, signal=9)`**: Moving Average Convergence Divergence
- **`stochastic(high, low, close, k=14, d=3)`**: Stochastic Oscillator (%K, %D)
- **`williams_r(high, low, close, period=14)`**: Williams %R
- **`cci(high, low, close, period=20)`**: Commodity Channel Index
- **`roc(close, period=10)`**: Rate of Change (Momentum %)
- **`stoch_rsi(close, rsi_period=14, stoch_period=14)`**: Stochastic RSI (%K, %D)
- **`awesome_oscillator(high, low, fast=5, slow=34)`**: Awesome Oscillator (AO)
- **`tsi(close, long_period=25, short_period=13)`**: True Strength Index (TSI)

### Volatility
- **`true_range(high, low, close)`**: True Range
- **`atr(high, low, close, period=14)`**: Average True Range
- **`bollinger_bands(close, period=20, num_std=2.0)`**: Bollinger Bands (Upper, Middle, Lower, Bandwidth, %B)
- **`keltner_channels(high, low, close, ema=20, atr=10, multiplier=2.0)`**: Keltner Channels
- **`donchian_channels(high, low, period=20)`**: Donchian Channels (Upper, Middle, Lower)
- **`stddev(close, period=20)`**: Rolling Standard Deviation
- **`ulcer_index(close, period=14)`**: Ulcer Index (Downside Drawdown Risk)

### Volume
- **`obv(close, volume)`**: On-Balance Volume
- **`vwap(high, low, close, volume, period=None)`**: Volume Weighted Average Price (Cumulative or Rolling)
- **`vwma(close, volume, period=20)`**: Volume Weighted Moving Average
- **`cmf(high, low, close, volume, period=20)`**: Chaikin Money Flow
- **`adl(high, low, close, volume)`**: Accumulation/Distribution Line
- **`mfi(high, low, close, volume, period=14)`**: Money Flow Index
- **`eom(high, low, volume, period=14)`**: Ease of Movement (EMV)

---

## Architecture & Performance

`polars-ta` is designed around three architectural pillars:

1. **Arrow Memory Layout**: Avoids memory copying between indicator transformations.
2. **SIMD-Accelerated Vectorization**: Executes through Polars' optimized analytical kernels.
3. **Query Optimization**: Integrates with Polars' expression engine for predicate pushdown and dead-code elimination.

---

## Development & Testing

```bash
# Clone the repository
git clone https://github.com/your-username/polars-ta.git
cd polars-ta

# Install in editable mode with development dependencies
pip install -e .[dev]

# Run the test suite
pytest -v

# Run linter and formatter checks
ruff check .
ruff format --check .
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
