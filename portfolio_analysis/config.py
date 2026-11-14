"""
Centralized configuration module for the VCB & DCM Quantitative Portfolio Analysis project.
All parameters, assumptions, and conventions are explicitly defined here for reproducibility.
"""

from pathlib import Path
from typing import Dict, List, Any

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
LEGACY_OUTPUT_DIR = WORKSPACE_ROOT / "output"

for directory in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    OUTPUT_TABLES_DIR,
    OUTPUT_FIGURES_DIR,
    OUTPUT_REPORTS_DIR,
    LEGACY_OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Core mandatory configuration parameters
ASSETS: List[str] = ["VCB", "DCM"]
ASSET_NAMES: Dict[str, str] = {
    "VCB": "Joint Stock Commercial Bank for Foreign Trade of Vietnam (Vietcombank)",
    "DCM": "PetroVietnam Ca Mau Fertilizer JSC (Dam Ca Mau)",
    "VNINDEX": "Ho Chi Minh Stock Exchange Index (VN-Index)",
}
MARKET_BENCHMARK: str = "VNINDEX"

IN_SAMPLE_START: str = "2023-07-31"
IN_SAMPLE_END: str = "2026-07-31"
OUT_OF_SAMPLE_START: str = "2026-08-03"
OUT_OF_SAMPLE_END: str = "2026-10-30"

INVESTMENT_HORIZON_YEARS: int = 3
TARGET_ANNUAL_RETURN: float = 0.1279
RISK_FREE_ANNUAL_RATE: float = 0.045
DEPOSIT_ANNUAL_RATE: float = 0.07
RISK_AVERSION: float = 2.0
INITIAL_CAPITAL_VND: float = 50_000_000.0
TRADING_DAYS_PER_YEAR: int = 252
WEIGHT_EPSILON: float = 0.0001
CONFIDENCE_LEVELS: List[float] = [0.95, 0.99]
RANDOM_SEED: int = 42

# Extended execution, data provider, and econometric parameters
CONFIG: Dict[str, Any] = {
    "ASSETS": ASSETS,
    "ASSET_NAMES": ASSET_NAMES,
    "MARKET_BENCHMARK": MARKET_BENCHMARK,
    "IN_SAMPLE_START": IN_SAMPLE_START,
    "IN_SAMPLE_END": IN_SAMPLE_END,
    "OUT_OF_SAMPLE_START": OUT_OF_SAMPLE_START,
    "OUT_OF_SAMPLE_END": OUT_OF_SAMPLE_END,
    "INVESTMENT_HORIZON_YEARS": INVESTMENT_HORIZON_YEARS,
    "TARGET_ANNUAL_RETURN": TARGET_ANNUAL_RETURN,
    "RISK_FREE_ANNUAL_RATE": RISK_FREE_ANNUAL_RATE,
    "DEPOSIT_ANNUAL_RATE": DEPOSIT_ANNUAL_RATE,
    "RISK_AVERSION": RISK_AVERSION,
    "INITIAL_CAPITAL_VND": INITIAL_CAPITAL_VND,
    "TRADING_DAYS_PER_YEAR": TRADING_DAYS_PER_YEAR,
    "WEIGHT_EPSILON": WEIGHT_EPSILON,
    "CONFIDENCE_LEVELS": CONFIDENCE_LEVELS,
    "RANDOM_SEED": RANDOM_SEED,
    # Data provider and unit settings
    "PRIMARY_ADJUSTED_PROVIDER": "DNSE Entrade OHLC API (Vietnam Exchange Official Adjusted Feed)",
    "SECONDARY_UNADJUSTED_PROVIDER": "Yahoo Finance (yfinance VCB.VN, DCM.VN Unadjusted Close & Volume)",
    "DNSE_PRICE_MULTIPLIER_VND": 1000.0,  # DNSE quotes stocks in 1,000 VND units; multiplied by 1,000 to VND
    "INDEX_PRICE_MULTIPLIER": 1.0,        # VN-Index is quoted in index points
    "TIMEZONE": "Asia/Ho_Chi_Minh",
    # Trading and execution conventions (configurable parameters)
    "ENTRY_PRICE_CONVENTION": "Closing price on actual entry session (baseline NAV established at entry close)",
    "BUY_COMMISSION_RATE": 0.0015,        # Configurable parameter: 0.15% brokerage fee on buy orders
    "SELL_COMMISSION_RATE": 0.0015,       # Configurable parameter: 0.15% brokerage fee on sell orders
    "SELL_TAX_RATE": 0.0010,              # Configurable parameter: 0.10% personal income tax on gross sell proceeds
    "BOARD_LOT_SIZE": 100,                # Standard HOSE board lot size: 100 shares
    "ALLOW_ODD_LOT_TRADING": False,       # False = strict 100-share board lots in practical execution
    "ALLOW_FRACTIONAL_SHARES": False,     # False in practical backtest; True only in theoretical backtest
    "RESIDUAL_CASH_INTEREST_RATE": 0.0,   # Uninvested residual brokerage cash earns 0.0% p.a. by default
    "DIVIDEND_REINVESTMENT_POLICY": "Cash dividends credited to cash ledger in practical backtest; implicitly reinvested in theoretical adjusted-price backtest",
    "REBALANCING_THRESHOLD_WEIGHT": 0.02, # Minimum one-way turnover (2.0%) or positive net utility gain required
    # Econometric & simulation settings
    "GRID_POINTS": 10_001,                # 10,001 points on [epsilon, 1 - epsilon]
    "HAC_MAX_LAGS": 5,                    # Newey-West HAC standard errors lag truncation (5 trading days = 1 week)
    "ROLLING_BETA_WINDOWS": [126, 252],   # 6-month (126-day) and 1-year (252-day) rolling beta windows
    "BOOTSTRAP_PATHS": 10_000,            # 10,000 simulated 3-year paths
    "BOOTSTRAP_HORIZON_DAYS": 756,        # 3 years * 252 trading sessions = 756 sessions
    "BOOTSTRAP_BLOCK_LENGTH": 10,         # Moving block bootstrap length: 10 trading sessions
    # Visualization styling
    "COLOR_MAP": {
        "VCB": "#1f77b4",                 # Consistent deep blue for VCB
        "DCM": "#2ca02c",                 # Consistent forest green for DCM
        "VNINDEX": "#ff7f0e",             # Orange for VN-Index benchmark
        "PORTFOLIO_P": "#d62728",         # Crimson red for Portfolio P
        "EQUAL_WEIGHT": "#9467bd",        # Purple for Equal Weight
        "GMV": "#e377c2",                 # Magenta/Gold for GMV
        "MAX_SHARPE": "#8c564b",          # Brown/Dark gold for Max Sharpe
        "DEPOSIT": "#7f7f7f",             # Gray for Bank Deposit
    },
}

