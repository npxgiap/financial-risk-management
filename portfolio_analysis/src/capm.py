"""
CAPM, Security Market Line (SML), and Beta Regression Diagnostics module.
Uses exact compound daily risk-free rate rf_daily = (1 + 0.045)^(1/252) - 1,
OLS estimation with Newey-West HAC standard errors, residual diagnostics,
and rolling 126-day and 252-day beta series.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm


def estimate_capm_regression(
    simple_returns: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Estimate CAPM and OLS Beta regression with Newey-West HAC standard errors
    for VCB and DCM against VN-Index.

    Parameters
    ----------
    simple_returns : pd.DataFrame
        In-sample daily simple returns containing VCB, DCM, and VNINDEX.
    config : Dict[str, Any]
        Centralized configuration dictionary.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]
        - capm_table_df: Combined CAPM and Regression Results table.
        - diagnostics_dict: Detailed regression objects, residuals, confidence bands, and rolling betas.
    """
    ann_days = float(config["TRADING_DAYS_PER_YEAR"])
    rf_annual = float(config["RISK_FREE_ANNUAL_RATE"])
    rf_daily = (1.0 + rf_annual) ** (1.0 / ann_days) - 1.0
    hac_lags = int(config["HAC_MAX_LAGS"])

    df_ret = simple_returns[config["ASSETS"] + [config["MARKET_BENCHMARK"]]].dropna().copy()
    mkt_ret = df_ret[config["MARKET_BENCHMARK"]]
    mkt_excess = mkt_ret - rf_daily

    est_annual_mkt_return = float(mkt_ret.mean() * ann_days)
    mkt_risk_premium = est_annual_mkt_return - rf_annual

    rows = []
    diagnostics: Dict[str, Dict[str, Any]] = {
        "rf_annual": rf_annual,
        "rf_daily": rf_daily,
        "estimated_annual_market_return": est_annual_mkt_return,
        "market_risk_premium": mkt_risk_premium,
        "hac_max_lags": hac_lags,
        "stocks": {},
    }

    X = sm.add_constant(mkt_excess.values)

    for ticker in config["ASSETS"]:
        stock_ret = df_ret[ticker]
        stock_excess = stock_ret - rf_daily
        hist_ann_mean = float(stock_ret.mean() * ann_days)

        # Fit OLS with Newey-West HAC standard errors
        ols_fit = sm.OLS(stock_excess.values, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags, "use_correction": True}
        )

        alpha_daily = float(ols_fit.params[0])
        beta_val = float(ols_fit.params[1])
        se_alpha_hac = float(ols_fit.bse[0])
        se_beta_hac = float(ols_fit.bse[1])
        t_alpha_hac = float(ols_fit.tvalues[0])
        t_beta_hac = float(ols_fit.tvalues[1])
        p_alpha_hac = float(ols_fit.pvalues[0])
        p_beta_hac = float(ols_fit.pvalues[1])
        ci_mat = ols_fit.conf_int(alpha=0.05)
        ci_alpha_low, ci_alpha_high = float(ci_mat[0, 0]), float(ci_mat[0, 1])
        ci_beta_low, ci_beta_high = float(ci_mat[1, 0]), float(ci_mat[1, 1])

        r2 = float(ols_fit.rsquared)
        adj_r2 = float(ols_fit.rsquared_adj)
        n_obs = int(ols_fit.nobs)

        # CAPM annual expected return and gaps
        capm_implied_ann = rf_annual + beta_val * mkt_risk_premium
        hist_capm_gap = hist_ann_mean - capm_implied_ann
        ann_reg_alpha = alpha_daily * ann_days

        # Prediction / confidence band across sorted market excess returns
        x_grid = np.linspace(float(mkt_excess.min()), float(mkt_excess.max()), 200)
        X_grid = sm.add_constant(x_grid)
        pred_summary = ols_fit.get_prediction(X_grid).summary_frame(alpha=0.05)

        # Rolling 126-session and 252-session betas (strictly requiring min_periods = window)
        cov_126 = stock_excess.rolling(window=126, min_periods=126).cov(mkt_excess)
        var_126 = mkt_excess.rolling(window=126, min_periods=126).var()
        rolling_beta_126 = (cov_126 / var_126).dropna()

        cov_252 = stock_excess.rolling(window=252, min_periods=252).cov(mkt_excess)
        var_252 = mkt_excess.rolling(window=252, min_periods=252).var()
        rolling_beta_252 = (cov_252 / var_252).dropna()

        residuals_series = pd.Series(ols_fit.resid, index=df_ret.index, name=f"{ticker}_Residual")

        diagnostics["stocks"][ticker] = {
            "market_excess": mkt_excess,
            "stock_excess": stock_excess,
            "residuals": residuals_series,
            "x_grid": x_grid,
            "y_hat": pred_summary["mean"].values,
            "ci_lower": pred_summary["mean_ci_lower"].values,
            "ci_upper": pred_summary["mean_ci_upper"].values,
            "rolling_beta_126": rolling_beta_126,
            "rolling_beta_252": rolling_beta_252,
            "beta": beta_val,
            "alpha_daily": alpha_daily,
            "r2": r2,
        }

        rows.append(
            {
                "Asset": ticker,
                "Beta": beta_val,
                "Beta_HAC_SE": se_beta_hac,
                "Beta_t_stat": t_beta_hac,
                "Beta_p_value": p_beta_hac,
                "Beta_95_CI_Lower": ci_beta_low,
                "Beta_95_CI_Upper": ci_beta_high,
                "Historical_Annualized_Mean_Return": hist_ann_mean,
                "CAPM_Implied_Annual_Return": capm_implied_ann,
                "Historical_CAPM_Gap": hist_capm_gap,
                "Daily_Regression_Alpha": alpha_daily,
                "Annualized_Regression_Alpha": ann_reg_alpha,
                "Alpha_HAC_SE": se_alpha_hac,
                "Alpha_t_stat": t_alpha_hac,
                "Alpha_p_value": p_alpha_hac,
                "Alpha_95_CI_Lower": ci_alpha_low,
                "Alpha_95_CI_Upper": ci_alpha_high,
                "R_squared": r2,
                "Adjusted_R_squared": adj_r2,
                "Observations": n_obs,
                "HAC_Max_Lags": hac_lags,
            }
        )

    return pd.DataFrame(rows), diagnostics

