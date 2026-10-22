# Getting Started

## Installation

Install `polars-ta` using pip:

```bash
pip install polars-ta
```

Requirements:
- Python >= 3.10
- Polars >= 1.0.0

No C compiler or MSVC / GCC tools are needed.

---

## Core Paradigm: Expression-First

In Polars, computations are expressed as expressions (`pl.Expr`) rather than eager array modifications. This allows Polars to optimize queries, combine multiple operations into a single execution pass, and parallelize calculations across all CPU threads.

Importing `polars_ta` automatically attaches the `.ta` namespace to Polars expressions:

```python
import polars as pl
import polars_ta
```

### 1. Basic Calculation

Calculate indicators using `.ta.<indicator>()` directly on columns:

```python
df = pl.DataFrame(
    {
        "open": [10.0, 10.5, 11.0, 10.8, 11.2],
        "high": [10.8, 11.2, 11.5, 11.1, 11.8],
        "low": [9.8, 10.2, 10.7, 10.5, 10.9],
        "close": [10.5, 11.0, 10.8, 11.2, 11.5],
        "volume": [1000, 1200, 950, 1100, 1500],
    }
)

result = df.with_columns(
    pl.col("close").ta.sma(3).alias("sma_3"),
    pl.col("close").ta.rsi(3).alias("rsi_3"),
    pl.col("close").ta.zscore(3).alias("zscore_3"),
)
print(result)
```

---

### 2. Multi-Component Struct Indicators

Indicators that output multiple components (like MACD, Bollinger Bands, Stochastic, Ichimoku) return a typed `pl.Struct` expression.

You can unpack all components into individual columns using `.unnest()`:

```python
# Compute Bollinger Bands
df_bands = df.with_columns(
    pl.col("close").ta.bollinger_bands(period=3, num_std=2.0).alias("bb")
).unnest("bb")

# Now columns `upper`, `middle`, `lower`, `bandwidth`, `percent_b` are directly available
print(df_bands.select(["close", "upper", "middle", "lower"]))
```

---

### 3. Functional Syntax

If you prefer functional import style, all indicators can also be imported directly from `polars_ta`:

```python
from polars_ta import sma, rsi, bollinger_bands

result = df.with_columns(
    sma("close", period=3).alias("sma_3"),
    rsi("close", period=3).alias("rsi_3"),
    bollinger_bands("close", period=3).alias("bb"),
)
```
