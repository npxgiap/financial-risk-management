# Quantitative Portfolio Analysis, Optimization, Out-of-Sample Backtesting, and Post-OOS Rebalancing: VCB & DCM

## Overview
This project implements a full-scale, reproducible quantitative portfolio management workflow for a two-asset Vietnamese equity portfolio consisting of **VCB** (Joint Stock Commercial Bank for Foreign Trade of Vietnam - Vietcombank) and **DCM** (PetroVietnam Ca Mau Fertilizer JSC), benchmarked against the **VN-Index** (`VNINDEX`) and a **7.00% p.a. Bank Term Deposit** benchmark.

## Project Structure
```text
portfolio_analysis/
│
├── README.md
├── requirements.txt
├── config.py
├── portfolio_analysis.ipynb
│
├── data/
│   ├── raw/
│   │   ├── stock_prices_raw.csv
│   │   └── market_benchmark_raw.csv
│   └── processed/
│       └── aligned_returns.csv
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── validation.py
│   ├── returns.py
│   ├── risk.py
│   ├── optimization.py
│   ├── simulation.py
│   ├── capm.py
│   ├── backtest.py
│   ├── rebalance.py
│   ├── reporting.py
│   └── verification.py
│
└── outputs/
    ├── tables/
    ├── figures/
    └── reports/
        └── Portfolio_Analysis_VCB_DCM_Report.xlsx
```

## Quick Start
From the workspace root directory:
```bash
pip install -r portfolio_analysis/requirements.txt
python run_quant_analysis.py
```

## Key Features
- **Corporate-Action Audit & Dual Price Series**: Combines Corporate-Action Total-Return Adjusted Close prices (`Adj Close` from DNSE Entrade API) for return/risk estimation with Unadjusted Tradable Close prices (`Raw Close` from Yahoo Finance) for realistic HOSE 100-share board-lot backtesting.
- **Distribution Fitting & Downside Tail Risk**: Normal vs. Student's $t$ MLE fitting, Jarque-Bera normality tests, and 1-Day / 10-Day Value-at-Risk (VaR) and Expected Shortfall (ES / CVaR) at 95% and 99% confidence levels across Historical, Parametric Normal, Parametric Student-$t$, Cornish-Fisher, and Monte Carlo methods.
- **Markowitz Mean-Variance & Utility Optimization**: 10,001-point weight grid, Global Minimum Variance (GMV), Target Return Portfolio $P$ ($12.79\%$ p.a.), Constrained Maximum Sharpe Portfolio, Risky-Only Utility Optimization ($A = 2.0$), and Complete Portfolio Optimization with a Risk-Free Asset ($r_f = 4.50\%$ p.a.).
- **3-Year Joint Block Bootstrap Simulation**: 10,000 paths over 756 trading days comparing Buy-and-Hold vs. Annually Rebalanced wealth distributions.
- **CAPM & Rolling Beta Estimation**: OLS regression with Newey-West HAC standard errors (`maxlags=5`) and 126-day / 252-day rolling betas.
- **Out-of-Sample Backtesting**: Both Theoretical (frictionless fractional shares) and Practical (HOSE 100-share board lots, 15 bps brokerage fee, 10 bps sell tax, idle cash ledger) backtests over `2026-08-03` to `2026-10-30`.
- **Post-OOS Rebalancing & Sensitivity Analysis**: Evaluates Hold vs. Rebalance-to-Original vs. Re-Optimize options on `2026-10-30`, plus multi-dimensional parameter sensitivity sweeps.

