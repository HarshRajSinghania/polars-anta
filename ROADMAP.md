# Roadmap

This document outlines the development roadmap and strategic milestones for **polars-anta**.

---

## Release Milestones

### Phase 1: Core Foundation & Indicator Expansion (v0.1.0 - Completed)
- [x] Implementation of 100 core indicators across 6 financial domains:
  - 25 Trend & Moving Average Overlays
  - 19 Momentum Oscillators
  - 18 Volatility & Channel Bands
  - 14 Volume Dynamics
  - 12 Quantitative Statistics & Rolling Regression
  - 12 Alpha Factor Primitives (WorldQuant Alpha 101 formulas)
- [x] Dual expression namespace support: both `pl.col(...).anta` and `pl.col(...).ta`
- [x] Multi-asset window parallelization via `.over("ticker")`
- [x] Comprehensive test suite (106 unit tests, 90%+ code coverage)
- [x] Modern documentation website built with Material for MkDocs
- [x] Automated CI/CD with GitHub Actions across Ubuntu, Windows, and macOS

### Phase 2: Performance Parity & Advanced Optimization (v0.2.0 - Planned)
- [ ] Benchmarking against TA-Lib native C binaries across multi-million bar datasets
- [ ] Streaming execution optimizations for out-of-core datasets exceeding physical RAM
- [ ] Extended alpha factor library (expanding Alpha 101 primitives)
- [ ] Automated parity verification suite against textbook references

### Phase 3: Ecosystem & Advanced Acceleration (v0.3.0+)
- [ ] Polars GPU engine integration (GPU-accelerated expression execution)
- [ ] Interactive charting hooks with lightweight candlestick visualizers
- [ ] Integration recipes for backtesting frameworks (VectorBT, NautilusTrader)

---

## Community Feedback & Contributions

We actively welcome community feedback and suggestions. If there is an indicator, mathematical formula, or performance improvement you would like to see:
- Open a feature request via [GitHub Issues](https://github.com/huuan26/polars-anta/issues)
- Join the discussion on [GitHub Discussions](https://github.com/huuan26/polars-anta/discussions)
