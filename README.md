# 📈 Hybrid Regime-Switching SDE Framework for Bitcoin Futures

This repository implements a sophisticated quantitative trading framework that combines **Stochastic Differential Equations (SDE)** with **Hidden Regime-Switching** logic. Designed for the high-volatility environment of Bitcoin futures, the model dynamically estimates drift and diffusion parameters across latent market states.

---

## 🌟 Core Methodology

### 1. Asymmetric SDE Modeling
The framework estimates state-dependent parameters to capture the inherent asymmetry in cryptocurrency markets. The return dynamics are modeled as:

$$dr_t = \left[ (1 - \tilde{w}_t)(-\kappa r_t) + \tilde{w}_t (\alpha_{s_t} D_t) \right] dt + \left[ (1 - \tilde{w}_t)\sigma_{low} + \tilde{w}_t \sigma_{high} \right] dW_t$$

### 2. Parameter Definitions
* **$r_t$**: The logarithmic return of the asset at time $t$.
* **$\tilde{w}_t$**: The regime indicator weight (0 to 1), representing the probability of being in the high-volatility/trend regime at time $t$.
* **$\kappa$ (Kappa)**: The speed of mean-reversion in the **Normal (Low Volatility) Regime**. It represents the force pulling the price back to the equilibrium (zero return).
* **$\alpha_{s_t}$ (Alpha)**: The drift intensity coefficient during the **Regime-Switching (High Volatility) State**. This parameter is bifurcated to account for market asymmetry:
    * **$\alpha_{long}$**: The strength of the upward trend during bullish regime shifts.
    * **$\alpha_{short}$**: The strength of the downward momentum during bearish regime shifts.
* **$D_t$**: The directional trend component (typically derived from technical indicators or momentum).
* **$\sigma_{low}, \sigma_{high}$**: The diffusion coefficients (volatility) representing the noise level in the normal and trend regimes, respectively.
* **$dW_t$**: The standard Brownian motion (Wiener process) representing idiosyncratic market shocks.

---

## 🛠️ Key Framework Enhancements

* **Asymmetric Alpha Bifurcation**: By explicitly separating $\alpha_{long}$ and $\alpha_{short}$, the model captures the fundamental asymmetry of the crypto market (e.g., difference in velocity between rallies and liquidations).
* **Strict Out-of-Sample Integrity**: Implements a Walk-Forward Analysis with a 3-month training and 1-month testing window to eliminate lookahead bias.
* **Signal Persistence (Sticky Filter)**: Uses $k$ (sensitivity) and $\gamma$ (threshold) to ensure that only sustained regime shifts trigger trade signals, filtering out high-frequency market noise.
* **Dynamic Risk Management**: Exit strategies (TP/SL) and position sizes are dynamically adjusted based on the real-time **Signal-to-Noise Ratio (SNR)**.

---



## 🏗️ Project Structure
```text
.
├── benchmarks/              # Comparative baseline strategies
│   ├── buy_and_hold.py      # Simple Buy-and-Hold baseline for performance comparison
│   └── garch.py             # Asymmetric GARCH(1,1) volatility-based strategy benchmark
│
├── configs/                 # Strategy & environment configurations (TOML format)
│   ├── backtest_settings.toml    # Walk-Forward parameters, trading rules, risk management
│   ├── data_settings.toml        # Event detection thresholds, rolling windows, feature engineering
│   ├── model_parameters.toml     # SDE priors: drift, diffusion, regime-switching parameters
│   └── training_settings.toml    # MCMC sampling config: chains, draws, NUTS settings
│
├── data/                    # Market datasets and event definitions
│   ├── btcusdt_future.csv   # Raw OHLCV + Volume data (5-minute bars) from Binance API
│   └── events.toml          # Pre-detected breakout events with S/R levels and validation flags
│
├── figures/                 # Auto-generated visual analysis outputs
│   ├── equity_curve.png          # Cumulative PnL over time
│   ├── drawdown_profile.png      # Maximum drawdown evolution
│   ├── diag_pair_*.png           # MCMC trace diagnostics (pair plots)
│   ├── trace_*.png               # Parameter convergence traces
│   ├── rank_*.png                # Rank plots for MCMC quality check
│   ├── ridge_*.png               # Posterior distribution ridge plots
│   ├── drift_*.png               # Drift parameter evolution across windows
│   └── garch/                    # GARCH-specific performance plots
│
├── results/                 # Standardized experimental outputs (CSV + pickle)
│   ├── garch_trade_results.csv   # Trade-by-trade results from GARCH benchmark
│   ├── sde_trade_results.csv     # Trade-by-trade results from SDE strategy
│   ├── all_windows_summary.csv   # Aggregated SDE parameter statistics per WFA window
│   └── all_windows_traces.pkl    # Serialized PyMC InferenceData objects (MCMC traces)
│
├── src/                     # Core library modules (modular Python package)
│   ├── __init__.py
│   │
│   ├── data/                # Data acquisition, preprocessing, and event management
│   │   ├── data_collector.py     # Binance API wrapper for historical data download
│   │   ├── dataset_builder.py    # Maps detected events to training data segments
│   │   ├── event_detector.py     # Identifies breakout events using hybrid Z-scores + S/R
│   │   └── preprocessor.py       # Feature engineering: Z-scores, ADX, Donchian channels
│   │
│   ├── engines/             # Simulation & execution logic
│   │   └── backtester.py         # Walk-Forward backtest engine with SNR-based parameter scaling
│   │
│   ├── models/              # Statistical & mathematical modeling
│   │   └── sde_model.py          # Bayesian SDE estimation using PyMC NUTS sampler
│   │
│   ├── utils/               # Helper utilities and analyzers
│   │   ├── configuration_loader.py    # TOML config loader with path management
│   │   ├── performance_analyzer.py    # Sharpe ratio, win rate, drawdown calculations
│   │   └── signal_processing.py      # Sticky breakout filter, persistence validation
│   │
│   └── visualization/       # Automated plotting and reporting
│       └── plotter.py            # Strategy performance visualization (equity, drawdown, etc.)
│
├── run_analyze_results.py   # [Executable] Post-backtest analysis: parameter stability, diagnostics
├── run_backtester.py        # [Executable] Main SDE Walk-Forward Analysis pipeline
├── run_data_collector.py    # [Executable] Download raw market data from Binance API
├── run_event_detector.py    # [Executable] Detect and export breakout events to events.toml
│
├── notebook.ipynb           # Interactive exploration and visualization (Jupyter)
├── pyproject.toml           # Poetry project metadata and dependency specifications
├── poetry.lock              # Locked dependency versions for reproducible environments
├── LICENSE                  # MIT License
└── README.md                # Project overview and usage instructions
```

---


## 📊 Data Flow

```text
Raw Data (Binance)
    ↓
data_collector.py → btcusdt_future.csv
    ↓
preprocessor.py → Feature Engineering
    ↓
event_detector.py → events.toml
    ↓
dataset_builder.py → Tagged DataFrame
    ↓
Walk-Forward Loop:
    ├─ SDE Training (PyMC) → Parameter Estimates
    ├─ Signal Generation → regime_prob, sticky_signal
    ├─ Backtest Execution → Trade Results
    └─ Performance Analysis → Metrics + Plots
```



## 🔄 Typical Workflow

```bash
# 1. Download data
poetry run python run_data_collector.py

# 2. Detect events
poetry run python run_event_detector.py     # data/events.toml

# 3. Run backtest (parallelized Walk-Forward Analysis)
poetry run python run_backtester.py

# 4. Analyze results and generate plots
poetry run python run_analyze_results.py
```




## 🚀 Execution Guide

This project is managed using **Poetry**.

### 1. Environment Setup
```bash
# Install all dependencies
poetry install
```

### 2. Run SDE Pipeline
Executes the main WFA training and backtesting session.
```bash
poetry run python run_backtester.py
```

### 3. Run Benchmarks
Evaluate the SDE model against traditional baselines.
```bash
# Run Buy-and-Hold
poetry run python benchmarks/benchmark_buy_and_hold.py

# Run GARCH(1,1)
poetry run python benchmarks/garch.py
```

### 4. Performance Analysis
Generate diagnostic plots and statistical reports. Default paths are pre-configured to fetch data from the `results/` directory.
```bash
# Analyze specific window results
poetry run python run_analyze_results.py --window 2024-01-01
```

---



## ⚖️ Robust Benchmark Suite
To validate the SDE model's alpha, the framework provides a rigorous comparison against industry-standard baselines:

* **Buy-and-Hold (B&H)**: A time-series based benchmark calculating daily risk-adjusted metrics (Sharpe, MDD, T-stat) to evaluate long-term holding risk.
* **GARCH(1,1)**: An advanced volatility-clustering baseline, enhanced with **Asymmetric Alpha Estimation** and **Stable Multi-period Forecasting** to ensure a fair, high-level comparison.

---



## 📊 Performance Summary

The STRS-SDE framework was evaluated under a rigorous **15 bps per-side (30 bps round-trip)** transaction cost hurdle. The results empirically resolve the "IC-Alpha Paradox" by demonstrating that lower predictive IC can yield superior net alpha through noise-efficient filtering.

---

### 1. Benchmark Comparison (2024.01 – 2026.01)
Compared to the Buy-and-Hold (B&H) strategy and the GARCH(1,1) model, our framework achieves the highest risk-adjusted returns while maintaining a stable recovery profile.

| Metric | Buy-and-Hold | GARCH(1,1) | **STRS-SDE (Ours)** |
| :--- | :---: | :---: | :---: |
| **Total Return (%)** | **77.95%** | -6.78% | 71.18% |
| **Annualized Return (%)** | **31.84%** | -3.42% | 29.95% |
| **Max Drawdown (%)** | -36.85% | **-8.10%** | **-13.62%** |
| **Sharpe Ratio** | 0.82 | -1.46 | **1.52** |
| **Sortino Ratio** | 1.26 | -2.65 | **4.54** |
| **T-statistic** | 1.18 | -2.07* | **2.18*** |
| **Avg. Recovery Time** | **29.46 Days** | 210.57 Days | 30.78 Days |

> **Note:** GARCH(1,1) exhibited a prohibitive recovery time of **210.57 days**, indicating that while it limits drawdown, it fails to recover from losses in high-frequency regimes.

---

### 2. Rank Information Coefficient (IC) Analysis
The results demonstrate the **IC-Alpha Paradox**: while GARCH provides higher raw predictive power (IC), the STRS-SDE signals are more "tradable" due to the effective filtering of microstructure noise.

| Signal Source | Horizon (1-Tick) | Horizon (50-Ticks) | Significance |
| :--- | :---: | :---: | :---: |
| **GARCH (Regime Prob)** | 0.2331 | 0.1368 | p < 0.001 |
| **STRS-SDE (Ours)** | 0.1665 | 0.0939 | p < 0.001 |

---

### 3. Ablation Study: Filtering Efficiency
The combination of the **Sticky Breakout Filter** and **ADX Filter** is the primary driver of capital efficiency, reducing noise-driven signals by **80.59%**.

| Strategy Variant | Signal Reduction | Trades | Sharpe Ratio |
| :--- | :---: | :---: | :---: |
| **Full (SDE + Sticky + ADX)** | **80.59%** | **73** | **1.52** |
| SDE + Sticky (No ADX) | 77.66% | 84 | 1.37 |
| SDE + ADX (No Sticky) | 27.66% | 272 | 1.10 |
| Pure SDE (No Filters) | 0.0% | 376 | -0.15 |

---

### 4. Transaction Cost Sensitivity
The framework remains robust even under extreme liquidity constraints (25 bps per-side), proving its suitability for institutional-scale execution.

| Friction (Per-side) | 7 bps | 10 bps | **15 bps (Base)** | 25 bps |
| :--- | :---: | :---: | :---: | :---: |
| **Annualized Return** | 37.50% | 34.62% | **29.95%** | 21.07% |
| **Sharpe Ratio** | 1.83 | 1.71 | **1.52** | 1.13 |
| **Sortino Ratio** | 5.47 | 5.12 | **4.54%** | 2.91 |
| **T-statistic** | 2.62* | 2.45* | **2.18*** | 1.62 |

---

### 🧐 Key Takeaways
* **Filtering Power:** The Sticky Filter reduced trades by **80.59%**, effectively converting low-IC raw signals into high-Alpha executable trades.
* **Risk Control:** STRS-SDE maintained a **4.54 Sortino Ratio**, nearly 4x that of Buy-and-Hold, confirming superior downside protection.
* **Scalability:** Profitability is maintained up to **50 bps round-trip** (25 bps per-side), demonstrating significant institutional scalability.