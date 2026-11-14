"""
Return calculation and descriptive asset statistics module.
Implements simple returns, logarithmic returns, exact annualization conventions,
CAGR using elapsed calendar days (365.25/Elapsed_Days), Maximum Drawdown,
and both Optimization Sharpe and Realized Daily-Excess Sharpe conventions.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def calculate_returns(
    price_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate daily simple returns and logarithmic returns from a validated price DataFrame.

    Formulas
    --------
    Simple return: r_t = P_t / P_{t-1} - 1
    Log return:    g_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    price_df : pd.DataFrame
        Validated price DataFrame indexed by trading date.
    columns : Optional[List[str]]
        Subset of asset columns to compute returns for. Defaults to all numeric columns.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (simple_returns_df, log_returns_df)
    """
    target_df = price_df[columns].copy() if columns is not None else price_df.copy()
    simple_returns = target_df.pct_change().dropna(how="all")
    log_returns = np.log(target_df / target_df.shift(1)).dropna(how="all")
    return simple_returns, log_returns


def compute_drawdown_series(wealth_or_price: pd.Series) -> pd.Series:
    """
    Compute the running percentage drawdown series from a price or wealth index.

    Formula
    -------
    Running Peak_t = max(NAV_0, ..., NAV_t)
    Drawdown_t = NAV_t / Running Peak_t - 1 (negative or zero)
    """
    running_peak = wealth_or_price.cummax()
    return (wealth_or_price / running_peak) - 1.0


def calculate_asset_statistics(
    price_df: pd.DataFrame,
    simple_returns: pd.DataFrame,
    config: Dict[str, Any],
    assets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute comprehensive descriptive, risk, and return statistics for each asset.

    Parameters
    ----------
    price_df : pd.DataFrame
        Price panel (including initial base session at index[0]).
    simple_returns : pd.DataFrame
        Daily simple returns DataFrame.
    config : Dict[str, Any]
        Configuration dictionary.
    assets : Optional[List[str]]
        List of tickers to evaluate. Defaults to ['VCB', 'DCM', 'VNINDEX'].

    Returns
    -------
    pd.DataFrame
        Asset Statistics table with all mandatory metrics.
    """
    if assets is None:
        assets = config["ASSETS"] + [config["MARKET_BENCHMARK"]]

    ann_days = float(config["TRADING_DAYS_PER_YEAR"])
    rf_annual = float(config["RISK_FREE_ANNUAL_RATE"])
    rf_daily = (1.0 + rf_annual) ** (1.0 / ann_days) - 1.0

    elapsed_calendar_days = float((price_df.index[-1] - price_df.index[0]).days)
    if elapsed_calendar_days <= 0:
        raise ValueError("Elapsed calendar days must be positive to compute CAGR.")

    rows = []
    for ticker in assets:
        r = simple_returns[ticker].dropna()
        p = price_df[ticker].dropna()
        n_obs = int(len(r))

        mean_daily = float(r.mean())
        median_daily = float(r.median())
        daily_vol = float(r.std(ddof=1))

        ann_arith_mean = mean_daily * ann_days
        ann_vol = daily_vol * np.sqrt(ann_days)
        cum_return = float(np.prod(1.0 + r.values) - 1.0)
        cagr = float((p.iloc[-1] / p.iloc[0]) ** (365.25 / elapsed_calendar_days) - 1.0)

        min_daily = float(r.min())
        max_daily = float(r.max())
        skewness = float(stats.skew(r.values, bias=False))
        excess_kurt = float(stats.kurtosis(r.values, fisher=True, bias=False))

        dd_series = compute_drawdown_series(p)
        max_dd = float(dd_series.min())

        # Convention 1: Optimization Sharpe = (Annualized Arithmetic Mean - Rf_annual) / Annualized Volatility
        opt_sharpe = (ann_arith_mean - rf_annual) / ann_vol if ann_vol > 1e-12 else np.nan

        # Convention 2: Realized Daily-Excess Sharpe = sqrt(252) * Mean(r_t - rf_daily) / Std(r_t - rf_daily, ddof=1)
        excess_r = r - rf_daily
        realized_sharpe = (
            np.sqrt(ann_days) * float(excess_r.mean()) / float(excess_r.std(ddof=1))
            if float(excess_r.std(ddof=1)) > 1e-12
            else np.nan
        )

        rows.append(
            {
                "Asset": ticker,
                "Observation Count": n_obs,
                "Mean Daily Return": mean_daily,
                "Median Daily Return": median_daily,
                "Daily Volatility": daily_vol,
                "Annualized Arithmetic Mean Return": ann_arith_mean,
                "Annualized Volatility": ann_vol,
                "Cumulative Return": cum_return,
                "CAGR": cagr,
                "Minimum Daily Return": min_daily,
                "Maximum Daily Return": max_daily,
                "Skewness": skewness,
                "Excess Kurtosis": excess_kurt,
                "Maximum Drawdown": max_dd,
                "Annualized Sharpe Ratio (Optimization)": opt_sharpe,
                "Realized Daily-Excess Sharpe Ratio": realized_sharpe,
            }
        )

    return pd.DataFrame(rows)

