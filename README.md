<div align="center">

# ⚡ polars-ta

**Lightning-fast, zero-C-dependency Technical Analysis & Factor Library powered natively by Polars Expressions.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org)
[![Polars Version](https://img.shields.io/badge/polars-%3E%3D1.0.0-orange)](https://pola.rs)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Zero C Dependencies](https://img.shields.io/badge/dependencies-pure%20polars-brightgreen)](#)

[English](README.md) • [Features](#key-features) • [Quickstart](#quickstart) • [Indicators](#supported-indicators) • [Benchmarks](#performance-philosophy)

</div>

---

## 🎯 Why `polars-ta`?

Quantitative researchers and financial engineers have long struggled with technical analysis tooling:

| Challenge | TA-Lib | pandas-ta | **`polars-ta`** |
| :--- | :---: | :---: | :---: |
| **Dependencies** | Requires external C library & MSVC / gcc compiler | Pure Python / NumPy | **Zero C compilation** (pure Polars) |
| **Execution Engine** | Single-threaded C loops | Single-threaded Pandas / Python GIL | **Multi-threaded Rust** query engine |
| **Multi-Asset Scaling** | Manual loops | Slow `.groupby().apply()` | **Native `.over("ticker")`** in parallel |
| **Lazy Evaluation** | ❌ No | ❌ No | ✅ Full **`pl.LazyFrame`** streaming support |
| **Memory Footprint** | Copies arrays back & forth | High Pandas memory overhead | **Zero-copy Arrow memory format** |

`polars-ta` solves this by expressing financial indicators directly as **Polars Expressions (`pl.Expr`)**. There is no overhead, no Python GIL bottleneck, and zero compilation headache.

---

## 🚀 Key Features

- **Expression-First Design**: Chain indicators seamlessly into `.select()`, `.with_columns()`, and lazy query plans.
- **Parallel Across 5,000+ Assets**: Process market-wide datasets effortlessly using native Polars grouping:
  ```python
  pl.col("close").ta.rsi(14).over("symbol")
  ```
- **Struct Output for Multi-Line Indicators**: Indicators like MACD, Bollinger Bands, and Stochastic return structured columns (`pl.Struct`) that preserve clean schemas.
- **Zero-C Compilation**: `pip install polars-ta` installs immediately on Linux, macOS, and Windows without requiring GCC, Clang, or Visual Studio Build Tools.

---

## 📦 Installation

```bash
pip install polars-ta
```

*(Requirements: Python >= 3.10, Polars >= 1.0.0)*

---

## ⚡ Quickstart

### 1. Modern Chained Expression Syntax (`.ta`)

```python
import polars as pl
import polars_ta  # Registers the `.ta` accessor

df = pl.DataFrame(
    {
        "open": [100.0, 102.0, 101.5, 103.0, 105.0],
        "high": [101.5, 103.5, 103.0, 104.5, 106.5],
        "low": [98.8, 100.8, 100.3, 101.8, 103.8],
        "close": [100.0, 102.0, 101.5, 103.0, 105.0],
        "volume": [1000, 1050, 1100, 1150, 1200],
    }
)

# Compute technical indicators effortlessly
res = df.with_columns(
    pl.col("close").ta.sma(period=3).alias("sma_3"),
    pl.col("close").ta.rsi(period=3).alias("rsi_3"),
    pl.col("close").ta.bollinger_bands(period=3).alias("bb"),
)

# Unnest structured outputs when needed
print(res.unnest("bb"))
```

### 2. Market-Wide Multi-Asset Parallel Execution

One of the biggest pain points in financial analysis is running indicators across thousands of tickers. With `polars-ta`, Rust handles thread scheduling across all CPU cores automatically:

```python
# Compute indicators across thousands of symbols simultaneously
dataset.with_columns(
    [
        pl.col("close").ta.sma(20).over("ticker").alias("sma_20"),
        pl.col("close").ta.rsi(14).over("ticker").alias("rsi_14"),
        pl.col("close").ta.macd().over("ticker").alias("macd"),
        pl.col("close").ta.obv("volume").over("ticker").alias("obv"),
    ]
)
```

---

## 📊 Supported Indicators (Phase 1)

### 📈 Trend
- **`sma(close, period=10)`**: Simple Moving Average
- **`ema(close, period=10)`**: Exponential Moving Average
- **`wma(close, period=10)`**: Linearly-Weighted Moving Average
- **`hma(close, period=10)`**: Hull Moving Average

### ⚡ Momentum
- **`rsi(close, period=14)`**: Relative Strength Index (Wilder's smoothing)
- **`macd(close, fast=12, slow=26, signal=9)`**: Moving Average Convergence Divergence
- **`stochastic(high, low, close, k=14, d=3)`**: Stochastic Oscillator (%K, %D)
- **`williams_r(high, low, close, period=14)`**: Williams %R

### 🌊 Volatility
- **`true_range(high, low, close)`**: True Range
- **`atr(high, low, close, period=14)`**: Average True Range
- **`bollinger_bands(close, period=20, num_std=2.0)`**: Bollinger Bands (Upper, Middle, Lower, Bandwidth, %B)
- **`keltner_channels(high, low, close, ema=20, atr=10, multiplier=2.0)`**: Keltner Channels

### 🔊 Volume
- **`obv(close, volume)`**: On-Balance Volume
- **`vwap(high, low, close, volume, period=None)`**: Volume Weighted Average Price (Cumulative or Rolling)
- **`vwma(close, volume, period=20)`**: Volume Weighted Moving Average
- **`cmf(high, low, close, volume, period=20)`**: Chaikin Money Flow
- **`adl(high, low, close, volume)`**: Accumulation/Distribution Line

---

## 🏎️ Performance Philosophy

Traditional libraries loop over rows or transfer arrays between C and Python memory space. `polars-ta` is designed from the ground up to:

1. **Leverage Arrow Memory**: Zero copy between indicator transformations.
2. **Exploit Vectorized Kernels**: Native SIMD instructions in Polars.
3. **Integrate with LazyFrame Optimizer**: Redundant computations are eliminated during query plan optimization.

---

## 🛠️ Development & Testing

```bash
# Clone the repository
git clone https://github.com/your-username/polars-ta.git
cd polars-ta

# Run tests
pytest -v

# Run code linter & formatter
ruff check .
ruff format .
```

---

## 📜 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
