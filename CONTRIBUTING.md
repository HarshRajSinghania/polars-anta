# Contributing to polars-ta

Thank you for your interest in contributing to **polars-ta**! We are building the fastest, most ergonomic technical analysis and factor library for the Polars ecosystem.

## 🌟 How Can You Contribute?

1. **Implement New Indicators**: Add indicators from TA-Lib, WorldQuant 101 Alphas, or academic literature.
2. **Performance Optimizations**: Benchmark and refine Polars expressions for maximum SIMD efficiency.
3. **Bug Reports & Feature Requests**: Open detailed issues with minimal reproducible examples.
4. **Documentation & Examples**: Improve tutorials, use cases, and notebooks.

---

## 🛠️ Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/polars-ta.git
   cd polars-ta
   ```

2. Create a virtual environment and install dev dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .[dev]
   ```

3. Run the test suite:
   ```bash
   pytest
   ```

4. Format and lint your code:
   ```bash
   ruff check --fix .
   ruff format .
   ```

---

## 📐 Design Principles for New Indicators

When implementing indicators:
- **Zero Python loops**: All computations must be expressed as pure `pl.Expr` expressions.
- **Support Grouping (`.over()`)**: Avoid stateful multi-step dependencies that break Polars window evaluation.
- **Standard Formulas**: Match standard financial literature (Wilder, Kaufman, Bollinger, Murphy).
- **Comprehensive Unit Tests**: Add tests in `tests/` checking correctness, edge cases (periods <= 0, flat prices), and namespace integration.
