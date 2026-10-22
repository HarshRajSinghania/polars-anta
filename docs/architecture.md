# Architecture & Design Principles

`polars-ta` is engineered around three fundamental architectural pillars:

---

## 1. Zero-C Dependency & Zero GIL Bottleneck

Existing technical analysis packages force quant researchers into an unpleasant dilemma:
- **TA-Lib**: Written in C. Fast, but notoriously difficult to compile on Windows/macOS, crashes frequently on ABI mismatches, and lacks native DataFrame integration.
- **pandas-ta**: Pure Python. Easy to install, but suffers from Python's Global Interpreter Lock (GIL), resulting in slow single-threaded loops on large datasets.

`polars-ta` circumvents this entirely:
- **Zero C Compilation**: Pure Polars expressions ensure installation always works out of the box (`pip install polars-ta`).
- **Native Rust Execution**: Polars pushes expression execution into its multi-threaded Rust analytical engine, bypassing Python's GIL.

---

## 2. Apache Arrow Memory Format & Zero-Copy Pipelines

Traditional Python pipelines waste significant CPU cycles converting data back and forth between NumPy arrays, Python lists, and C buffers.

In `polars-ta`:
- Data remains in contiguous Arrow memory blocks.
- Intermediate indicator steps share memory through Arrow buffers without memory replication.
- Streaming operations process chunked batches without out-of-memory (OOM) crashes on gigabyte-scale datasets.

---

## 3. Parallel Multi-Asset Windowing via `.over()`

In quantitative finance, features must be computed across a universe of assets (e.g. 500 S&P 500 stocks or thousands of crypto pairs).

In Pandas:
```python
# Slow: Single-threaded Python loop over groups
df.groupby("ticker").apply(lambda g: g["close"].rolling(20).mean())
```

In Polars:
```python
# Blazingly fast: Polars parallelizes groups across all CPU threads in Rust
df.with_columns(pl.col("close").ta.sma(20).over("ticker"))
```

By leveraging Polars' window context `.over("ticker")`, calculations for all tickers run concurrently across CPU cores with zero extra code.
