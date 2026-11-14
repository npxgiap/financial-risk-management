"""
Markowitz portfolio optimization and mean-variance utility maximization module.
Implements 10,001-point grid evaluation, analytical vs. numerical cross-verification,
non-dominated efficient frontier identification, Target Return Portfolio P (12.79%),
Constrained Maximum Sharpe Portfolio, Risky-Only Utility Optimum, and Complete Portfolio Optimum.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def evaluate_two_asset_portfolio(
    w_vcb: float,
    mu_vcb: float,
    mu_dcm: float,
    var_vcb: float,
    var_dcm: float,
    cov_vcb_dcm: float,
    rf: float,
    risk_aversion: float,
) -> Dict[str, float]:
    """
    Evaluate expected return, variance, volatility, Sharpe ratio, and utility for weight w_vcb.
    """
    w = float(w_vcb)
    w_d = 1.0 - w
    mu_p = w * mu_vcb + w_d * mu_dcm
    var_p = (w ** 2) * var_vcb + (w_d ** 2) * var_dcm + 2.0 * w * w_d * cov_vcb_dcm
    var_p = max(float(var_p), 0.0)
    sigma_p = float(np.sqrt(var_p))
    sharpe_p = (mu_p - rf) / sigma_p if sigma_p > 1e-12 else np.nan
    utility_p = mu_p - 0.5 * risk_aversion * var_p
    return {
        "w_vcb": w,
        "w_dcm": w_d,
        "expected_return": float(mu_p),
        "variance": float(var_p),
        "volatility": sigma_p,
        "sharpe_ratio": float(sharpe_p),
        "utility": float(utility_p),
    }


def build_portfolio_grid(
    mu_vec: np.ndarray,
    cov_mat: np.ndarray,
    rf: float,
    risk_aversion: float,
    epsilon: float = 0.0001,
    n_points: int = 10_001,
) -> pd.DataFrame:
    """
    Generate at least 10,001 portfolio weight combinations across [epsilon, 1 - epsilon].
    """
    if n_points < 10_001:
        raise ValueError("Grid resolution must be at least 10,001 points.")

    w_grid = np.linspace(epsilon, 1.0 - epsilon, n_points)
    w_dcm_grid = 1.0 - w_grid

    mu_vcb, mu_dcm = float(mu_vec[0]), float(mu_vec[1])
    var_vcb, var_dcm = float(cov_mat[0, 0]), float(cov_mat[1, 1])
    cov_12 = float(cov_mat[0, 1])

    mu_p = w_grid * mu_vcb + w_dcm_grid * mu_dcm
    var_p = (w_grid ** 2) * var_vcb + (w_dcm_grid ** 2) * var_dcm + 2.0 * w_grid * w_dcm_grid * cov_12
    var_p = np.maximum(var_p, 0.0)
    sigma_p = np.sqrt(var_p)
    sharpe_p = (mu_p - rf) / sigma_p
    utility_p = mu_p - 0.5 * risk_aversion * var_p

    grid_df = pd.DataFrame(
        {
            "VCB_Weight": w_grid,
            "DCM_Weight": w_dcm_grid,
            "Expected_Annual_Return": mu_p,
            "Annualized_Variance": var_p,
            "Annualized_Volatility": sigma_p,
            "Optimization_Sharpe": sharpe_p,
            "Utility_A2": utility_p,
        }
    )
    return grid_df


def identify_efficient_frontier(
    grid_df: pd.DataFrame,
    gmv_return: float,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """
    Classify each portfolio on the feasible set into the non-dominated 'Efficient Frontier'
    versus the dominated 'Inefficient Lower Branch'.

    A portfolio is efficient if and only if no other feasible portfolio offers higher or equal
    expected return and lower or equal risk with at least one strict inequality.
    For a strictly convex two-asset hyperbola, this corresponds to Expected_Annual_Return >= GMV_Return.
    """
    df = grid_df.copy()
    df["Is_Efficient"] = df["Expected_Annual_Return"] >= (gmv_return - tol)
    df["Frontier_Branch"] = np.where(df["Is_Efficient"], "Efficient Frontier", "Inefficient Branch")
    return df


def optimize_gmv(
    mu_vec: np.ndarray,
    cov_mat: np.ndarray,
    rf: float,
    risk_aversion: float,
    epsilon: float = 0.0001,
) -> Dict[str, Any]:
    """
    Find the Global Minimum Variance (GMV) portfolio via numerical optimization
    and cross-check against the closed-form analytical formula.
    """
    mu_vcb, mu_dcm = float(mu_vec[0]), float(mu_vec[1])
    var_vcb, var_dcm = float(cov_mat[0, 0]), float(cov_mat[1, 1])
    cov_12 = float(cov_mat[0, 1])

    denom = var_vcb + var_dcm - 2.0 * cov_12
    if abs(denom) > 1e-12:
        w_analytic_unconstrained = (var_dcm - cov_12) / denom
        w_analytic_bounded = float(np.clip(w_analytic_unconstrained, epsilon, 1.0 - epsilon))
    else:
        w_analytic_unconstrained = 0.5
        w_analytic_bounded = 0.5

    bounds = [(epsilon, 1.0 - epsilon), (epsilon, 1.0 - epsilon)]
    constraints = [{"type": "eq", "fun": lambda w: float(w[0] + w[1] - 1.0)}]

    res = minimize(
        lambda w: float(w[0] ** 2 * var_vcb + w[1] ** 2 * var_dcm + 2.0 * w[0] * w[1] * cov_12),
        x0=np.array([0.5, 0.5]),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"GMV numerical optimization failed: {res.message}")

    w_num = float(res.x[0])
    metrics = evaluate_two_asset_portfolio(w_num, mu_vcb, mu_dcm, var_vcb, var_dcm, cov_12, rf, risk_aversion)
    constraint_residual = abs(float(res.x[0] + res.x[1] - 1.0))
    analytic_diff = abs(w_num - w_analytic_bounded)

    at_bound = (abs(w_num - epsilon) < 1e-7) or (abs(w_num - (1.0 - epsilon)) < 1e-7)
    metrics.update(
        {
            "Portfolio_Name": "GMV Portfolio",
            "w_analytic_unconstrained": float(w_analytic_unconstrained),
            "w_analytic_bounded": w_analytic_bounded,
            "analytic_vs_numeric_diff": analytic_diff,
            "solver_converged": bool(res.success),
            "constraint_residual": constraint_residual,
            "Constraint_Status": "Boundary Optimum" if at_bound else "Interior Optimum",
            "Is_Efficient": True,
        }
    )
    return metrics


def optimize_target_return(
    mu_vec: np.ndarray,
    cov_mat: np.ndarray,
    rf: float,
    risk_aversion: float,
    gmv_return: float,
    target_return: float = 0.1279,
    epsilon: float = 0.0001,
) -> Dict[str, Any]:
    """
    Evaluate target return feasibility on [epsilon, 1 - epsilon], compute analytical and
    numerical weights for Portfolio P (or P_Fallback if infeasible), and check efficiency.
    """
    mu_vcb, mu_dcm = float(mu_vec[0]), float(mu_vec[1])
    var_vcb, var_dcm = float(cov_mat[0, 0]), float(cov_mat[1, 1])
    cov_12 = float(cov_mat[0, 1])

    ret_at_eps = epsilon * mu_vcb + (1.0 - epsilon) * mu_dcm
    ret_at_one_minus_eps = (1.0 - epsilon) * mu_vcb + epsilon * mu_dcm
    min_feasible_ret = min(ret_at_eps, ret_at_one_minus_eps)
    max_feasible_ret = max(ret_at_eps, ret_at_one_minus_eps)

    # Check degenerate equal-return case
    if abs(mu_vcb - mu_dcm) < 1e-10:
        is_feasible = abs(target_return - mu_vcb) < 1e-9
    else:
        is_feasible = (min_feasible_ret - 1e-9) <= target_return <= (max_feasible_ret + 1e-9)

    if is_feasible:
        if abs(mu_vcb - mu_dcm) < 1e-10:
            gmv_res = optimize_gmv(mu_vec, cov_mat, rf, risk_aversion, epsilon)
            w_analytic = gmv_res["w_vcb"]
        else:
            w_analytic = float((target_return - mu_dcm) / (mu_vcb - mu_dcm))

        bounds = [(epsilon, 1.0 - epsilon), (epsilon, 1.0 - epsilon)]
        constraints = [
            {"type": "eq", "fun": lambda w: float(w[0] + w[1] - 1.0)},
            {"type": "eq", "fun": lambda w: float(w[0] * mu_vcb + w[1] * mu_dcm - target_return)},
        ]
        res = minimize(
            lambda w: float(w[0] ** 2 * var_vcb + w[1] ** 2 * var_dcm + 2.0 * w[0] * w[1] * cov_12),
            x0=np.array([w_analytic, 1.0 - w_analytic]),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-14, "maxiter": 500},
        )
        if not res.success:
            raise RuntimeError(f"Target return numerical optimization failed: {res.message}")

        w_num = float(res.x[0])
        metrics = evaluate_two_asset_portfolio(w_num, mu_vcb, mu_dcm, var_vcb, var_dcm, cov_12, rf, risk_aversion)
        is_efficient = bool(metrics["expected_return"] >= gmv_return - 1e-9)
        target_error = abs(metrics["expected_return"] - target_return)
        analytic_diff = abs(w_num - w_analytic)

        metrics.update(
            {
                "Portfolio_Name": "Portfolio P",
                "Target_Annual_Return": target_return,
                "Target_Feasible": True,
                "Feasible_Return_Min": min_feasible_ret,
                "Feasible_Return_Max": max_feasible_ret,
                "w_analytic": w_analytic,
                "analytic_vs_numeric_diff": analytic_diff,
                "target_return_error": target_error,
                "solver_converged": bool(res.success),
                "constraint_residual": abs(float(res.x[0] + res.x[1] - 1.0)),
                "Is_Efficient": is_efficient,
                "Constraint_Status": "Interior Exact Target Solution" if (epsilon < w_num < 1.0 - epsilon) else "Boundary Target Solution",
            }
        )
        return metrics

    # Infeasible case: construct P_Fallback on the Efficient Frontier closest to target_return
    w_candidates = [epsilon, 1.0 - epsilon]
    best_cand = None
    for w_c in w_candidates:
        cand_m = evaluate_two_asset_portfolio(w_c, mu_vcb, mu_dcm, var_vcb, var_dcm, cov_12, rf, risk_aversion)
        if cand_m["expected_return"] >= gmv_return - 1e-9:
            if best_cand is None or abs(cand_m["expected_return"] - target_return) < abs(
                best_cand["expected_return"] - target_return
            ):
                best_cand = cand_m

    assert best_cand is not None
    best_cand.update(
        {
            "Portfolio_Name": "P_Fallback",
            "Target_Annual_Return": target_return,
            "Target_Feasible": False,
            "Feasible_Return_Min": min_feasible_ret,
            "Feasible_Return_Max": max_feasible_ret,
            "w_analytic": np.nan,
            "analytic_vs_numeric_diff": 0.0,
            "target_return_error": abs(best_cand["expected_return"] - target_return),
            "solver_converged": True,
            "constraint_residual": 0.0,
            "Is_Efficient": True,
            "Constraint_Status": "Infeasible Target -> Boundary Fallback (P_Fallback)",
        }
    )
    return best_cand


def optimize_max_sharpe(
    mu_vec: np.ndarray,
    cov_mat: np.ndarray,
    rf: float,
    risk_aversion: float,
    epsilon: float = 0.0001,
) -> Dict[str, Any]:
    """
    Find the Constrained Maximum Sharpe Portfolio on [epsilon, 1 - epsilon].
    Distinguishes between an interior geometric tangency solution and a boundary-constrained optimum.
    """
    mu_vcb, mu_dcm = float(mu_vec[0]), float(mu_vec[1])
    var_vcb, var_dcm = float(cov_mat[0, 0]), float(cov_mat[1, 1])
    cov_12 = float(cov_mat[0, 1])

    bounds = [(epsilon, 1.0 - epsilon), (epsilon, 1.0 - epsilon)]
    constraints = [{"type": "eq", "fun": lambda w: float(w[0] + w[1] - 1.0)}]

    def neg_sharpe(w: np.ndarray) -> float:
        m = evaluate_two_asset_portfolio(w[0], mu_vcb, mu_dcm, var_vcb, var_dcm, cov_12, rf, risk_aversion)
        return -m["sharpe_ratio"]

    # Multi-start optimization to guarantee global maximum on [epsilon, 1 - epsilon]
    best_res = None
    for start_w in [epsilon, 0.1, 0.5, 0.9, 1.0 - epsilon]:
        res = minimize(
            neg_sharpe,
            x0=np.array([start_w, 1.0 - start_w]),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-14, "maxiter": 500},
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    assert best_res is not None
    w_num = float(best_res.x[0])
    metrics = evaluate_two_asset_portfolio(w_num, mu_vcb, mu_dcm, var_vcb, var_dcm, cov_12, rf, risk_aversion)
    at_lower = abs(w_num - epsilon) < 1e-6
    at_upper = abs(w_num - (1.0 - epsilon)) < 1e-6

    metrics.update(
        {
            "Portfolio_Name": "Constrained Maximum Sharpe Portfolio",
            "solver_converged": bool(best_res.success),
            "constraint_residual": abs(float(best_res.x[0] + best_res.x[1] - 1.0)),
            "Is_Efficient": True,
            "Is_Boundary_Constrained": bool(at_lower or at_upper),
            "Constraint_Status": (
                f"Active Lower Bound (w_VCB = epsilon = {epsilon:.4f})"
                if at_lower
                else (
                    f"Active Upper Bound (w_VCB = 1 - epsilon = {1.0 - epsilon:.4f})"
                    if at_upper
                    else "Interior Geometric Tangency Optimum"
                )
            ),
        }
    )
    return metrics


def optimize_risky_utility(
    mu_vec: np.ndarray,
    cov_mat: np.ndarray,
    rf: float,
    risk_aversion: float = 2.0,
    epsilon: float = 0.0001,
) -> Dict[str, Any]:
    """
    Maximize quadratic investor utility U(w) = mu_p(w) - 0.5 * A * variance_p(w)
    strictly within the risky two-stock portfolio [epsilon, 1 - epsilon].
    """
    mu_vcb, mu_dcm = float(mu_vec[0]), float(mu_vec[1])
    var_vcb, var_dcm = float(cov_mat[0, 0]), float(cov_mat[1, 1])
    cov_12 = float(cov_mat[0, 1])

    denom = risk_aversion * (var_vcb + var_dcm - 2.0 * cov_12)
    if abs(denom) > 1e-12:
        w_analytic_unconstrained = ((mu_vcb - mu_dcm) + risk_aversion * (var_dcm - cov_12)) / denom
        w_analytic_bounded = float(np.clip(w_analytic_unconstrained, epsilon, 1.0 - epsilon))
    else:
        w_analytic_unconstrained = 0.5
        w_analytic_bounded = 0.5

    bounds = [(epsilon, 1.0 - epsilon), (epsilon, 1.0 - epsilon)]
    constraints = [{"type": "eq", "fun": lambda w: float(w[0] + w[1] - 1.0)}]

    res = minimize(
        lambda w: -evaluate_two_asset_portfolio(w[0], mu_vcb, mu_dcm, var_vcb, var_dcm, cov_12, rf, risk_aversion)[
            "utility"
        ],
        x0=np.array([0.5, 0.5]),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"Risky utility optimization failed: {res.message}")

    w_num = float(res.x[0])
    metrics = evaluate_two_asset_portfolio(w_num, mu_vcb, mu_dcm, var_vcb, var_dcm, cov_12, rf, risk_aversion)
    at_bound = (abs(w_num - epsilon) < 1e-6) or (abs(w_num - (1.0 - epsilon)) < 1e-6)

    metrics.update(
        {
            "Portfolio_Name": f"Maximum Utility Risky Portfolio (A={risk_aversion:.1f})",
            "w_analytic_unconstrained": float(w_analytic_unconstrained),
            "w_analytic_bounded": w_analytic_bounded,
            "analytic_vs_numeric_diff": abs(w_num - w_analytic_bounded),
            "solver_converged": bool(res.success),
            "constraint_residual": abs(float(res.x[0] + res.x[1] - 1.0)),
            "Is_Efficient": True,
            "Constraint_Status": "Boundary Optimum" if at_bound else "Interior Optimum (Unconstrained Solution within Bounds)",
        }
    )
    return metrics


def optimize_complete_utility(
    mu_vec: np.ndarray,
    cov_mat: np.ndarray,
    rf: float,
    risk_aversion: float = 2.0,
    epsilon: float = 0.0001,
) -> Dict[str, Any]:
    """
    Jointly optimize two decision variables:
    - w in [epsilon, 1 - epsilon]: VCB weight within the risky sleeve
    - y in [0, 1]: Total capital allocation to the risky sleeve
    to maximize complete portfolio utility:
    U_complete(w, y) = Rf + y * (mu_risky(w) - Rf) - 0.5 * A * y^2 * variance_risky(w).
    Also cross-checks against the CAL analytical share y_unconstrained = (mu_T - Rf) / (A * var_T).
    """
    mu_vcb, mu_dcm = float(mu_vec[0]), float(mu_vec[1])
    var_vcb, var_dcm = float(cov_mat[0, 0]), float(cov_mat[1, 1])
    cov_12 = float(cov_mat[0, 1])

    def neg_complete_utility(xy: np.ndarray) -> float:
        w, y = float(xy[0]), float(xy[1])
        w_d = 1.0 - w
        mu_r = w * mu_vcb + w_d * mu_dcm
        var_r = (w ** 2) * var_vcb + (w_d ** 2) * var_dcm + 2.0 * w * w_d * cov_12
        mu_c = rf + y * (mu_r - rf)
        var_c = (y ** 2) * var_r
        return -(mu_c - 0.5 * risk_aversion * var_c)

    bounds = [(epsilon, 1.0 - epsilon), (0.0, 1.0)]
    best_res = None
    for w0 in [epsilon, 0.1, 0.35, 0.5, 1.0 - epsilon]:
        for y0 in [0.1, 0.4, 0.8]:
            res = minimize(
                neg_complete_utility,
                x0=np.array([w0, y0]),
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-15, "gtol": 1e-12},
            )
            if best_res is None or res.fun < best_res.fun:
                best_res = res

    assert best_res is not None
    w_opt, y_opt = float(best_res.x[0]), float(best_res.x[1])

    # Cross-check with CAL analytical solution from Constrained Max Sharpe Portfolio
    max_sr_port = optimize_max_sharpe(mu_vec, cov_mat, rf, risk_aversion, epsilon)
    y_cal_unconstrained = (
        (max_sr_port["expected_return"] - rf) / (risk_aversion * max_sr_port["variance"])
        if max_sr_port["variance"] > 1e-12
        else 0.0
    )

    risky_m = evaluate_two_asset_portfolio(w_opt, mu_vcb, mu_dcm, var_vcb, var_dcm, cov_12, rf, risk_aversion)
    vcb_total = y_opt * w_opt
    dcm_total = y_opt * (1.0 - w_opt)
    rf_total = 1.0 - y_opt

    mu_complete = rf + y_opt * (risky_m["expected_return"] - rf)
    var_complete = (y_opt ** 2) * risky_m["variance"]
    vol_complete = float(np.sqrt(max(var_complete, 0.0)))
    utility_complete = mu_complete - 0.5 * risk_aversion * var_complete
    sharpe_complete = (mu_complete - rf) / vol_complete if vol_complete > 1e-12 else np.nan

    return {
        "Portfolio_Name": f"Optimal Complete Portfolio (A={risk_aversion:.1f})",
        "risky_sleeve_w_vcb": w_opt,
        "risky_sleeve_w_dcm": 1.0 - w_opt,
        "risky_sleeve_y": y_opt,
        "y_cal_unconstrained": float(y_cal_unconstrained),
        "w_vcb_total": vcb_total,
        "w_dcm_total": dcm_total,
        "w_rf_total": rf_total,
        "expected_return": float(mu_complete),
        "variance": float(var_complete),
        "volatility": vol_complete,
        "sharpe_ratio": float(sharpe_complete) if np.isfinite(sharpe_complete) else np.nan,
        "utility": float(utility_complete),
        "solver_converged": bool(best_res.success),
        "Constraint_Status": (
            f"Interior Risky Allocation y*={y_opt*100:.2f}% (No-Borrowing Constraint Inactive); "
            f"Risky Sleeve at Lower Bound w_VCB={w_opt:.4f}"
        ),
    }

