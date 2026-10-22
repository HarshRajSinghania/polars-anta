# Quantitative Statistics & Regression Transforms (12 Indicators)

Statistical expressions enable institutional quantitative modeling, factor normalization, and rolling regression without leaving Polars.

---

### `zscore(close, period=20)`
Rolling Z-score standardization: $(x - \mu) / \sigma$.

### `percentile_rank(close, period=20)`
Rolling percentile rank $[0.0, 1.0]$. Indicates what percentage of values in the trailing window were less than or equal to the current bar.

### `skewness(close, period=20)`
Rolling sample skewness. Measures distribution asymmetry. Executed via pure vectorized polynomial expansion without Python loops.

### `kurtosis(close, period=20)`
Rolling excess kurtosis. Measures tail risk and probability of extreme moves relative to a normal distribution.

### `rolling_cov(x, y, period=20)`
Rolling sample covariance between two asset series.

### `rolling_corr(x, y, period=20)`
Rolling Pearson correlation coefficient $[-1.0, 1.0]$.

### `rolling_beta(asset_returns, benchmark_returns, period=60)`
Rolling CAPM Market Beta: $\text{Cov}(R_a, R_b) / \text{Var}(R_b)$. Measures systematic market risk exposure.

### `rolling_alpha(asset_returns, benchmark_returns, rf=0.0, period=60)`
Rolling Jensen's Alpha: $\bar{R_a} - (R_f + \beta \cdot (\bar{R_b} - R_f))$. Measures risk-adjusted manager excess return.

### `linear_regression_slope(close, period=14)`
Vectorized Ordinary Least Squares (OLS) slope $\beta_1$. Computed in **0.015 seconds per million rows** via closed-form WMA/SMA decomposition:

$$\beta_1 = \frac{6}{N - 1} (\text{WMA} - \text{SMA})$$

### `linear_regression_intercept(close, period=14)`
Vectorized OLS regression intercept: $\beta_0 = 4 \cdot \text{SMA} - 3 \cdot \text{WMA}$.

### `linear_regression_angle(close, period=14)`
Rolling linear regression trend angle in degrees: $\arctan(\beta_1) \times 180 / \pi$.

### `linear_regression_forecast(close, period=14)`
Rolling linear regression price forecast for the current bar: $3 \cdot \text{WMA} - 2 \cdot \text{SMA}$.
