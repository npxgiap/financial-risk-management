"""
Three-year horizon wealth simulation module using Joint Moving Block Bootstrap.
Preserves cross-sectional correlation between VCB and DCM as well as short-horizon
serial dependence (volatility clustering) across 756 trading sessions (3 years).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def simulate_three_year_wealth(
    simple_returns: pd.DataFrame,
    portfolios_to_simulate: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Perform a 10,000-path Joint Moving Block Bootstrap over 756 trading sessions (3 years)
    comparing Annually Rebalanced and Buy-and-Hold policies.

    Parameters
    ----------
    simple_returns : pd.DataFrame
        In-sample daily simple returns for VCB and DCM.
    portfolios_to_simulate : Dict[str, Dict[str, float]]
        Mapping of portfolio name to {'w_vcb': float, 'w_dcm': float, 'w_rf': float}.
    config : Dict[str, Any]
        Centralized configuration dictionary.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, np.ndarray]]
        - summary_df: Three-Year Scenario Summary table.
        - terminal_wealth_distributions: Dictionary of simulated terminal wealth arrays (VND).
    """
    rng = np.random.default_rng(int(config["RANDOM_SEED"]))
    n_paths = int(config["BOOTSTRAP_PATHS"])
    horizon_days = int(config["BOOTSTRAP_HORIZON_DAYS"])  # 756 days
    block_len = int(config["BOOTSTRAP_BLOCK_LENGTH"])     # 10 days
    days_per_year = int(config["TRADING_DAYS_PER_YEAR"])  # 252 days
    init_cap = float(config["INITIAL_CAPITAL_VND"])
    target_ann = float(config["TARGET_ANNUAL_RETURN"])
    rf_ann = float(config["RISK_FREE_ANNUAL_RATE"])
    rf_daily = (1.0 + rf_ann) ** (1.0 / days_per_year) - 1.0

    compounded_target_wealth = init_cap * ((1.0 + target_ann) ** float(config["INVESTMENT_HORIZON_YEARS"]))

    ret_mat = simple_returns[["VCB", "DCM"]].dropna().values
    n_obs = ret_mat.shape[0]
    n_blocks = int(np.ceil(horizon_days / block_len))

    # Draw block start indices jointly for all paths: shape (n_paths, n_blocks)
    max_start_idx = n_obs - block_len
    block_starts = rng.integers(0, max_start_idx + 1, size=(n_paths, n_blocks))
    offsets = np.arange(block_len)
    # Build full index array of shape (n_paths, n_blocks * block_len) and truncate to horizon_days
    time_indices = (block_starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)[:, :horizon_days]

    # Jointly sampled daily returns: shape (n_paths, horizon_days, 2)
    sim_vcb = ret_mat[:, 0][time_indices]
    sim_dcm = ret_mat[:, 1][time_indices]

    summary_rows = []
    terminal_distributions: Dict[str, np.ndarray] = {}

    for port_name, weights in portfolios_to_simulate.items():
        w_v = float(weights["w_vcb"])
        w_d = float(weights["w_dcm"])
        w_f = float(weights.get("w_rf", 0.0))

        # Policy 1: Buy-and-Hold over 756 trading sessions (no intermediate rebalancing)
        cum_vcb_3y = np.prod(1.0 + sim_vcb, axis=1)
        cum_dcm_3y = np.prod(1.0 + sim_dcm, axis=1)
        cum_rf_3y = (1.0 + rf_daily) ** horizon_days
        tw_bh = init_cap * (w_v * cum_vcb_3y + w_d * cum_dcm_3y + w_f * cum_rf_3y)

        # Policy 2: Annually Rebalanced (rebalanced at end of Year 1 and Year 2 back to target weights)
        wealth_ar = np.full(n_paths, init_cap, dtype=float)
        for yr in range(3):
            s_idx = yr * days_per_year
            e_idx = (yr + 1) * days_per_year
            yr_vcb = np.prod(1.0 + sim_vcb[:, s_idx:e_idx], axis=1)
            yr_dcm = np.prod(1.0 + sim_dcm[:, s_idx:e_idx], axis=1)
            yr_rf = (1.0 + rf_daily) ** days_per_year
            yr_growth = w_v * yr_vcb + w_d * yr_dcm + w_f * yr_rf
            wealth_ar = wealth_ar * yr_growth

        for policy_label, tw_arr in [
            ("Buy-and-Hold (3 Years)", tw_bh),
            ("Annually Rebalanced (3 Years)", wealth_ar),
        ]:
            key = f"{port_name} | {policy_label}"
            terminal_distributions[key] = tw_arr

            summary_rows.append(
                {
                    "Portfolio": port_name,
                    "Rebalancing_Policy": policy_label,
                    "Initial_Capital_VND": init_cap,
                    "Compounded_Target_Wealth_VND": compounded_target_wealth,
                    "Median_Terminal_Wealth_VND": float(np.median(tw_arr)),
                    "Mean_Terminal_Wealth_VND": float(np.mean(tw_arr)),
                    "Percentile_5th_Wealth_VND": float(np.percentile(tw_arr, 5)),
                    "Percentile_95th_Wealth_VND": float(np.percentile(tw_arr, 95)),
                    "Prob_Capital_Loss": float(np.mean(tw_arr < init_cap)),
                    "Prob_Reaching_Compounded_Target": float(np.mean(tw_arr >= compounded_target_wealth)),
                    "Implied_Median_3Y_CAGR": float((np.median(tw_arr) / init_cap) ** (1.0 / 3.0) - 1.0),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, terminal_distributions

