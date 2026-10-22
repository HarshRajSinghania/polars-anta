# Parallel Multi-Asset Scaling with `.over()`

A frequent challenge in quantitative backtesting and real-time scanning is computing indicators across an entire universe of tickers (e.g. all 500 S&P 500 stocks or thousands of crypto spot pairs).

---

## The Traditional Bottleneck (Pandas)

In Pandas, grouping operations execute sequentially in a single Python thread:

```python
# Slow & serial (runs on 1 CPU core)
df["sma_20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())
df["rsi_14"] = df.groupby("ticker")["close"].apply(calculate_rsi)
```

For 500 tickers $\times$ 2,000 bars (1,000,000 rows), this single-threaded approach takes seconds to execute and scales poorly.

---

## The Polars Advantage: Native Windowing

In Polars, any expression can be partitioned across groups using `.over("ticker")`:

```python
import polars as pl
import polars_anta

# Fast & parallelized across all CPU cores in Rust
universe_with_indicators = universe_df.with_columns(
    [
        pl.col("close").anta.sma(20).over("ticker").alias("sma_20"),
        pl.col("close").anta.rsi(14).over("ticker").alias("rsi_14"),
        pl.col("close").anta.macd().over("ticker").alias("macd"),
        pl.col("close").anta.obv("volume").over("ticker").alias("obv"),
        pl.col("close").anta.zscore(20).over("ticker").alias("zscore_20"),
    ]
)
```

### Why it is faster:
1. **Multi-Threaded Rayon Thread Pool**: Polars automatically allocates separate tickers to separate worker threads.
2. **Zero Serialization**: No Python function callback overhead per group.
3. **Cache Coherency**: Data stays in Arrow chunked arrays, maximizing CPU L1/L2 cache hit rates.
