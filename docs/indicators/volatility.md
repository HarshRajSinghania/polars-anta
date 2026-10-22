# Volatility Indicators & Channels (18 Indicators)

Volatility indicators measure the dispersion of returns, price range expansion, and channel boundaries.

---

### `true_range(high, low, close)`
True Range: $\max(H - L, |H - C_{prev}|, |L - C_{prev}|)$.

### `atr(high, low, close, period=14)`
Average True Range using Wilder's smoothing. Measures absolute market volatility.

### `natr(high, low, close, period=14)`
Normalized Average True Range: $\text{ATR} / \text{Close} \times 100$. Allows cross-asset volatility comparison.

### `bollinger_bands(close, period=20, num_std=2.0)`
Bollinger Bands. Returns Struct: `upper`, `middle`, `lower`, `bandwidth`, `percent_b`.

### `bollinger_delta(close, period=20, num_std=2.0)`
Bollinger Band Spread: $2 \cdot k \cdot \sigma$.

### `keltner_channels(high, low, close, ema=20, atr=10, multiplier=2.0)`
Keltner Channels. Combines an EMA baseline with ATR-based bands. Returns Struct: `upper`, `middle`, `lower`.

### `keltner_width(high, low, close, ema=20, atr=10, multiplier=2.0)`
Normalized Keltner Channel Width (%) for Bollinger/Keltner volatility squeeze detection.

### `donchian_channels(high, low, period=20)`
Donchian Channels. High/Low range breakout channel. Returns Struct: `upper`, `middle`, `lower`.

### `donchian_width(high, low, period=20)`
Donchian Channel Bandwidth (percentage spread relative to midline).

### `envelopes(close, period=20, percentage=0.025)`
Moving Average Envelopes. Fixed-percentage bands above and below an SMA midline. Returns Struct: `upper`, `middle`, `lower`.

### `starc_bands(high, low, close, period=20, atr_period=15, multiplier=2.0)`
STARC Bands (Stoller Average Range Channels). SMA midline + ATR bands. Returns Struct: `upper`, `middle`, `lower`.

### `stddev(close, period=20)`
Rolling sample standard deviation.

### `ulcer_index(close, period=14)`
Ulcer Index. Quantifies downside risk and retracement depth.

### `chandelier_exit(high, low, close, period=22, multiplier=3.0)`
Chandelier Exit. Trailing stop based on highest high/lowest low minus ATR multiple. Returns Struct: `long_stop`, `short_stop`.

### `historical_volatility(close, period=20, trading_days=252)`
Annualized Log Historical Volatility ($\sigma_{annual}$).

### `parkinson_volatility(high, low, period=20, trading_days=252)`
Parkinson Extreme-Value Volatility. 5x more efficient than close-to-close volatility by incorporating intraday High/Low.

### `garman_klass_volatility(open, high, low, close, period=20, trading_days=252)`
Garman-Klass Efficient Volatility. Incorporates Open, High, Low, and Close for maximum estimator efficiency (8x better than close-to-close).

### `chaikin_volatility(high, low, period=10, roc_period=10)`
Chaikin Volatility. Percentage rate of change of the exponential moving average of the High-Low spread.
