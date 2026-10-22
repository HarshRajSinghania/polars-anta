# Indicator Catalog (100 Indicators)

`polars-ta` provides 100 high-performance financial indicators categorized into six specialized domains:

| Category | Count | Key Features | Documentation |
| :--- | :---: | :--- | :--- |
| **Trend & Overlays** | **25** | Moving averages, adaptive tracking, SuperTrend, PSAR, Ichimoku, Pivots | [View Trend Docs](trend.md) |
| **Momentum Oscillators** | **19** | RSI, MACD, Stochastic, CMO, PPO, Coppock, Fisher, KST, UO | [View Momentum Docs](momentum.md) |
| **Volatility & Channels** | **18** | Bollinger, Keltner, Donchian, ATR, Parkinson, Garman-Klass, Envelopes | [View Volatility Docs](volatility.md) |
| **Volume Dynamics** | **14** | OBV, VWAP, VWMA, CMF, Klinger, NVI, PVI, Elder's Force Index | [View Volume Docs](volume.md) |
| **Quantitative Statistics** | **12** | Z-score, Skewness, Kurtosis, Rolling Cov/Corr, Rolling Beta/Alpha, OLS | [View Statistics Docs](stats.md) |
| **Alpha Factor Primitives** | **12** | WorldQuant 101 Alphas, ts_rank, ts_argmax, Decay Linear, Entropy | [View Alpha Docs](alpha.md) |
| **Total Catalog** | **100** | Pure Polars Expressions (`pl.Expr`) with `.over("ticker")` support | |

---

## Unified Usage Syntax

All indicators can be invoked either via the chained expression syntax `.ta.<name>()` or through functional imports:

```python
import polars as pl
import polars_ta

# Method 1: Chained syntax
df.select(pl.col("close").ta.rsi(14))

# Method 2: Functional syntax
from polars_ta import rsi

df.select(rsi("close", 14))
```
