"""
Risk modeling module: parametric and non-parametric return distribution fitting
(Normal vs. Student's t, AIC/BIC, Jarque-Bera), 1-day and 10-day Value at Risk (VaR)
and Expected Shortfall (ES), and covariance/correlation matrix diagnostics.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def fit_return_distributions(
    simple_returns: pd.DataFrame,
    assets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fit Normal and Student's t distributions to daily simple returns via Maximum Likelihood,
    compute Log-Likelihood, AIC, BIC, and perform the Jarque-Bera test.

    Parameters
    ----------
    simple_returns : pd.DataFrame
        Daily simple returns DataFrame.
    assets : Optional[List[str]]
        List of tickers to evaluate. Defaults to ['VCB', 'DCM'].

    Returns
    -------
    pd.DataFrame
        Distribution Fit Results table.
    """
    if assets is None:
        assets = ["VCB", "DCM"]

    rows = []
    for ticker in assets:
        r = simple_returns[ticker].dropna().values
        n = len(r)

        # 1. Normal MLE fit
        norm_mu, norm_sigma = stats.norm.fit(r)
        ll_norm = float(np.sum(stats.norm.logpdf(r, loc=norm_mu, scale=norm_sigma)))
        k_norm = 2
        aic_norm = 2 * k_norm - 2 * ll_norm
        bic_norm = k_norm * np.log(n) - 2 * ll_norm

        # 2. Student's t MLE fit
        t_df, t_loc, t_scale = stats.t.fit(r)
        ll_t = float(np.sum(stats.t.logpdf(r, df=t_df, loc=t_loc, scale=t_scale)))
        k_t = 3
        aic_t = 2 * k_t - 2 * ll_t
        bic_t = k_t * np.log(n) - 2 * ll_t

        if t_df > 4.0:
            moment_note = f"df={t_df:.2f} > 4: Mean, variance, skewness, and kurtosis are finite."
        elif t_df > 2.0:
            moment_note = f"2 < df={t_df:.2f} <= 4: Mean and variance exist, but theoretical 4th moment (kurtosis) is infinite."
        elif t_df > 1.0:
            moment_note = f"1 < df={t_df:.2f} <= 2: Mean exists, but theoretical variance and higher moments are infinite."
        else:
            moment_note = f"df={t_df:.2f} <= 1: Theoretical mean and variance do not exist."

        # 3. Jarque-Bera test
        jb_stat, jb_pval = stats.jarque_bera(r)
        skew_val = float(stats.skew(r, bias=False))
        kurt_val = float(stats.kurtosis(r, bias=False))
        bic_diff = float(bic_norm - bic_t)

        rows.append(
            {
                "Asset": ticker,
                "Observations": int(n),
                "Normal_Mean": float(norm_mu),
                "Normal_Std_MLE": float(norm_sigma),
                "Normal_LogLikelihood": ll_norm,
                "Normal_AIC": float(aic_norm),
                "Normal_BIC": float(bic_norm),
                "StudentT_df": float(t_df),
                "StudentT_Loc": float(t_loc),
                "StudentT_Scale": float(t_scale),
                "StudentT_LogLikelihood": ll_t,
                "StudentT_AIC": float(aic_t),
                "StudentT_BIC": float(bic_t),
                "BIC_Difference_Normal_minus_t": bic_diff,
                "Preferred_InSample_Fit_BIC": "Student's t" if bic_t < bic_norm else "Normal",
                "Skewness": skew_val,
                "Empirical_Excess_Kurtosis": kurt_val,
                "StudentT_Moment_Existence": moment_note,
                "Jarque_Bera_Statistic": float(jb_stat),
                "Jarque_Bera_pvalue": float(jb_pval),
                "Jarque_Bera_Conclusion": (
                    "Reject Normal Null at 1% Level (Fat tails / skewness present)"
                    if jb_pval < 0.01
                    else "Do Not Reject Normal Null"
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_var_es_for_series(
    return_series: pd.Series,
    asset_name: str,
    confidence_levels: List[float],
    capital_vnd: float,
    include_10day: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compute 1-day and 10-day Historical and Gaussian VaR and Expected Shortfall (ES)
    using the positive loss convention L_t = -r_t.

    Quantile & Tail Convention
    --------------------------
    - Loss is defined as L_t = -r_t.
    - Historical VaR at confidence c is the linear-interpolated c-quantile of L_t.
    - Historical ES is the arithmetic average of all observed losses in the tail L_t >= VaR_c.
    """
    r = return_series.dropna().values
    losses_1d = -r
    mu_1d = float(np.mean(r))
    sigma_1d = float(np.std(r, ddof=1))

    records: List[Dict[str, Any]] = []

    for c in confidence_levels:
        alpha_tail = 1.0 - c
        z_c = float(stats.norm.ppf(c))

        # 1. Historical 1-Day VaR & ES
        hist_var_1d = float(np.quantile(losses_1d, c, method="linear"))
        tail_losses_1d = losses_1d[losses_1d >= hist_var_1d - 1e-12]
        hist_es_1d = float(np.mean(tail_losses_1d))

        records.append(
            {
                "Asset or Portfolio": asset_name,
                "Method": "Historical Simulation (Linear Quantile)",
                "Confidence Level": f"{c*100:.0f}%",
                "Horizon": "1 Trading Day",
                "VaR Percentage": hist_var_1d,
                "ES Percentage": hist_es_1d,
                "VaR VND": hist_var_1d * capital_vnd,
                "ES VND": hist_es_1d * capital_vnd,
                "Tail Observation Count": int(len(tail_losses_1d)),
            }
        )

        # 2. Gaussian 1-Day VaR & ES
        gauss_var_1d = float(-mu_1d + z_c * sigma_1d)
        gauss_es_1d = float(-mu_1d + sigma_1d * (stats.norm.pdf(z_c) / alpha_tail))

        records.append(
            {
                "Asset or Portfolio": asset_name,
                "Method": "Parametric Gaussian",
                "Confidence Level": f"{c*100:.0f}%",
                "Horizon": "1 Trading Day",
                "VaR Percentage": gauss_var_1d,
                "ES Percentage": gauss_es_1d,
                "VaR VND": gauss_var_1d * capital_vnd,
                "ES VND": gauss_es_1d * capital_vnd,
                "Tail Observation Count": int(np.ceil(len(losses_1d) * alpha_tail)),
            }
        )

        if include_10day and len(r) >= 20:
            # 3. Historical 10-Day Compounded Overlapping Returns
            comp_10d = np.array([np.prod(1.0 + r[i : i + 10]) - 1.0 for i in range(len(r) - 9)])
            losses_10d = -comp_10d
            hist_var_10d = float(np.quantile(losses_10d, c, method="linear"))
            tail_10d = losses_10d[losses_10d >= hist_var_10d - 1e-12]
            hist_es_10d = float(np.mean(tail_10d))

            records.append(
                {
                    "Asset or Portfolio": asset_name,
                    "Method": "Historical 10-Day Compounded (Overlapping Windows)",
                    "Confidence Level": f"{c*100:.0f}%",
                    "Horizon": "10 Trading Days",
                    "VaR Percentage": hist_var_10d,
                    "ES Percentage": hist_es_10d,
                    "VaR VND": hist_var_10d * capital_vnd,
                    "ES VND": hist_es_10d * capital_vnd,
                    "Tail Observation Count": int(len(tail_10d)),
                }
            )

            # 4. Gaussian 10-Day Square-Root-of-Time Approximation (i.i.d. Normal assumption)
            mu_10d = 10.0 * mu_1d
            sigma_10d = np.sqrt(10.0) * sigma_1d
            gauss_var_10d = float(-mu_10d + z_c * sigma_10d)
            gauss_es_10d = float(-mu_10d + sigma_10d * (stats.norm.pdf(z_c) / alpha_tail))

            records.append(
                {
                    "Asset or Portfolio": asset_name,
                    "Method": "Parametric Gaussian (Square-Root-of-Time i.i.d.)",
                    "Confidence Level": f"{c*100:.0f}%",
                    "Horizon": "10 Trading Days",
                    "VaR Percentage": gauss_var_10d,
                    "ES Percentage": gauss_es_10d,
                    "VaR VND": gauss_var_10d * capital_vnd,
                    "ES VND": gauss_es_10d * capital_vnd,
                    "Tail Observation Count": int(np.ceil(len(losses_10d) * alpha_tail)),
                }
            )

    return records


def calculate_var_es(
    simple_returns: pd.DataFrame,
    config: Dict[str, Any],
    extra_portfolios: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    """
    Calculate VaR and Expected Shortfall across assets and specified constant-weight portfolios.
    """
    all_records: List[Dict[str, Any]] = []
    cap = float(config["INITIAL_CAPITAL_VND"])
    conf_levels = config["CONFIDENCE_LEVELS"]

    for ticker in config["ASSETS"]:
        all_records.extend(
            calculate_var_es_for_series(simple_returns[ticker], ticker, conf_levels, cap, include_10day=True)
        )

    if extra_portfolios:
        for port_name, port_ret_series in extra_portfolios.items():
            all_records.extend(
                calculate_var_es_for_series(port_ret_series, port_name, conf_levels, cap, include_10day=True)
            )

    var_es_df = pd.DataFrame(all_records)
    # Validate ES >= VaR under identical method, confidence level, and horizon
    if np.any(var_es_df["ES Percentage"].values < var_es_df["VaR Percentage"].values - 1e-9):
        raise AssertionError("Sub-additivity / tail coherence violation: ES < VaR detected.")

    return var_es_df


def calculate_covariance_correlation(
    simple_returns: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Calculate Daily Covariance, Annualized Covariance, and Pearson Correlation matrices
    for VCB and DCM, and verify all mathematical matrix properties.
    """
    assets = config["ASSETS"]
    ann_factor = float(config["TRADING_DAYS_PER_YEAR"])
    sub_ret = simple_returns[assets].dropna()

    daily_cov = sub_ret.cov(ddof=1)
    ann_cov = daily_cov * ann_factor
    corr_mat = sub_ret.corr(method="pearson")

    # Mathematical verification checks
    cov_arr = ann_cov.values
    corr_arr = corr_mat.values
    eigenvalues = np.linalg.eigvalsh(cov_arr)

    is_symmetric = bool(np.allclose(cov_arr, cov_arr.T, atol=1e-12))
    non_negative_diag = bool(np.all(np.diag(cov_arr) >= 0.0))
    psd_eigenvalues = bool(np.all(eigenvalues >= -1e-12))
    corr_in_bounds = bool(np.all((corr_arr >= -1.0 - 1e-12) & (corr_arr <= 1.0 + 1e-12)))
    unit_corr_diag = bool(np.allclose(np.diag(corr_arr), 1.0, atol=1e-12))

    if not (is_symmetric and non_negative_diag and psd_eigenvalues and corr_in_bounds and unit_corr_diag):
        raise ValueError("Covariance or correlation matrix failed mathematical validity checks.")

    # Compute constant-correlation / diagonal shrinkage covariance matrix (alpha = 0.20) for sensitivity testing
    shrinkage_intensity = 0.20
    diag_target = np.diag(np.diag(cov_arr))
    shrunk_ann_cov = pd.DataFrame(
        (1.0 - shrinkage_intensity) * cov_arr + shrinkage_intensity * diag_target,
        index=assets,
        columns=assets,
    )

    diagnostics = {
        "is_symmetric": is_symmetric,
        "non_negative_diagonal": non_negative_diag,
        "positive_semidefinite": psd_eigenvalues,
        "eigenvalues_annualized": eigenvalues.tolist(),
        "correlation_in_bounds": corr_in_bounds,
        "unit_correlation_diagonal": unit_corr_diag,
        "pearson_correlation_vcb_dcm": float(corr_mat.loc["VCB", "DCM"]),
        "shrunk_annualized_covariance": shrunk_ann_cov,
    }
    return daily_cov, ann_cov, corr_mat, diagnostics

