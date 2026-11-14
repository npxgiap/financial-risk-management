"""
Post-out-of-sample portfolio rebalancing (as of 30 October 2026) and
multi-dimensional sensitivity & robustness analysis module.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from .returns import calculate_returns
from .optimization import (
    evaluate_two_asset_portfolio,
    optimize_gmv,
    optimize_target_return,
    optimize_max_sharpe,
    optimize_risky_utility,
    optimize_complete_utility,
)


def propose_rebalance(
    full_aligned_panel: pd.DataFrame,
    backtest_series_dict: Dict[str, pd.DataFrame],
    backtest_meta_dict: Dict[str, Any],
    initial_p_weights: Dict[str, float],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate post-OOS rebalancing options using data available through 30 October 2026.
    Signal is generated after market close on 2026-10-30 for earliest execution on the
    next trading session (2026-11-02). No backdating is permitted.

    Compares three mandatory alternatives under updated forward moments:
    1. Hold Current Positions (drifted buy-and-hold weights as of 2026-10-30).
    2. Rebalance to Original Target Weights (initial Portfolio P weights from 2026-07-31).
    3. Re-optimize with Updated Data (updated Portfolio P achieving 12.79% target return,
       plus updated Risky-Only Utility A=2 and Complete Portfolio Utility A=2 benchmarks).
    """
    ann_days = float(config["TRADING_DAYS_PER_YEAR"])
    rf = float(config["RISK_FREE_ANNUAL_RATE"])
    A = float(config["RISK_AVERSION"])
    eps = float(config["WEIGHT_EPSILON"])
    target_ret = float(config["TARGET_ANNUAL_RETURN"])
    buy_fee = float(config["BUY_COMMISSION_RATE"])
    sell_fee = float(config["SELL_COMMISSION_RATE"])
    sell_tax = float(config["SELL_TAX_RATE"])
    lot_size = int(config["BOARD_LOT_SIZE"])

    # Rolling 3-year window ending 2026-10-30 (2023-10-30 to 2026-10-30)
    rolling_3y_panel = full_aligned_panel.loc["2023-10-30" : config["OUT_OF_SAMPLE_END"]].copy()
    ret_3y, _ = calculate_returns(rolling_3y_panel, columns=["VCB", "DCM"])

    mu_upd = ret_3y[["VCB", "DCM"]].mean().values * ann_days
    cov_upd = ret_3y[["VCB", "DCM"]].cov(ddof=1).values * ann_days

    gmv_upd = optimize_gmv(mu_upd, cov_upd, rf, A, eps)
    target_upd = optimize_target_return(mu_upd, cov_upd, rf, A, gmv_upd["expected_return"], target_ret, eps)
    risky_u_upd = optimize_risky_utility(mu_upd, cov_upd, rf, A, eps)
    comp_u_upd = optimize_complete_utility(mu_upd, cov_upd, rf, A, eps)

    # Current drifted state from Theoretical Buy-and-Hold Portfolio P at 2026-10-30 close
    theo_p_df = backtest_series_dict["theo_p"]
    cur_nav_theo = float(theo_p_df["Portfolio_NAV_VND"].iloc[-1])
    cur_w_vcb_theo = float(theo_p_df["VCB_Weight"].iloc[-1])
    cur_w_dcm_theo = float(theo_p_df["DCM_Weight"].iloc[-1])

    orig_w_vcb = float(initial_p_weights["VCB"])
    orig_w_dcm = float(initial_p_weights["DCM"])

    upd_w_vcb = float(target_upd["w_vcb"])
    upd_w_dcm = float(target_upd["w_dcm"])

    def _eval_option(name: str, w_v: float, w_d: float, w_cash: float, nav_base: float, cur_weights_vec: np.ndarray, feasible_flag: str):
        # Evaluate risky performance under updated moments (2023-10-30 to 2026-10-30)
        m = evaluate_two_asset_portfolio(
            w_v / (w_v + w_d) if (w_v + w_d) > 0 else 0.5,
            float(mu_upd[0]),
            float(mu_upd[1]),
            float(cov_upd[0, 0]),
            float(cov_upd[1, 1]),
            float(cov_upd[0, 1]),
            rf,
            A,
        )
        y_risky = w_v + w_d
        exp_ret = rf * w_cash + y_risky * m["expected_return"]
        exp_var = (y_risky ** 2) * m["variance"]
        exp_vol = float(np.sqrt(max(exp_var, 0.0)))
        sharpe = (exp_ret - rf) / exp_vol if exp_vol > 1e-12 else np.nan
        utility = exp_ret - 0.5 * A * exp_var

        new_vec = np.array([w_v, w_d, w_cash])
        # One-Way Turnover including cash: 0.5 * sum(|w_new - w_cur|)
        one_way_turnover = 0.5 * float(np.sum(np.abs(new_vec - cur_weights_vec)))
        # Estimate transaction costs on equity buys and sells
        weight_diff_equities = np.array([w_v - cur_weights_vec[0], w_d - cur_weights_vec[1]])
        buy_turnover = float(np.sum(np.maximum(weight_diff_equities, 0.0)))
        sell_turnover = float(np.sum(np.maximum(-weight_diff_equities, 0.0)))
        est_cost_rate = buy_turnover * buy_fee + sell_turnover * (sell_fee + sell_tax)
        est_cost_vnd = est_cost_rate * nav_base

        return {
            "Alternative": name,
            "Proposed_VCB_Weight": w_v,
            "Proposed_DCM_Weight": w_d,
            "Proposed_Cash_Rf_Weight": w_cash,
            "Updated_Expected_Annual_Return": exp_ret,
            "Updated_Expected_Volatility": exp_vol,
            "Updated_Optimization_Sharpe": sharpe,
            "Updated_Investor_Utility_A2": utility,
            "One_Way_Turnover_Pct": one_way_turnover,
            "Estimated_Transaction_Costs_VND": est_cost_vnd,
            "Net_First_Year_Utility_After_Costs": utility - est_cost_rate,
            "Target_12_79_Feasibility": feasible_flag,
        }

    cur_vec_theo = np.array([cur_w_vcb_theo, cur_w_dcm_theo, 0.0])
    rebalance_comp_rows = [
        _eval_option(
            "1. Hold Current Drifted Positions (No Trade)",
            cur_w_vcb_theo,
            cur_w_dcm_theo,
            0.0,
            cur_nav_theo,
            cur_vec_theo,
            f"Drifted Return = {(cur_w_vcb_theo*mu_upd[0] + cur_w_dcm_theo*mu_upd[1])*100:.2f}% (> 12.79% Target)",
        ),
        _eval_option(
            "2. Rebalance to Original Target Weights (31 Jul 2026 Policy)",
            orig_w_vcb,
            orig_w_dcm,
            0.0,
            cur_nav_theo,
            cur_vec_theo,
            f"Restores Original Weights; Updated Return = {(orig_w_vcb*mu_upd[0] + orig_w_dcm*mu_upd[1])*100:.2f}%",
        ),
        _eval_option(
            "3A. Re-optimize Target Return Portfolio P (12.79% Policy)",
            upd_w_vcb,
            upd_w_dcm,
            0.0,
            cur_nav_theo,
            cur_vec_theo,
            "Feasible Exact 12.79% Target" if target_upd["Target_Feasible"] else "Infeasible -> Fallback",
        ),
        _eval_option(
            "3B. Re-optimize Risky-Only Maximum Utility (A=2 Policy)",
            float(risky_u_upd["w_vcb"]),
            float(risky_u_upd["w_dcm"]),
            0.0,
            cur_nav_theo,
            cur_vec_theo,
            "Maximizes Risky Utility U(w) on [epsilon, 1-epsilon]",
        ),
        _eval_option(
            "3C. Re-optimize Complete Portfolio Utility (A=2 with Risk-Free Asset)",
            float(comp_u_upd["w_vcb_total"]),
            float(comp_u_upd["w_dcm_total"]),
            float(comp_u_upd["w_rf_total"]),
            cur_nav_theo,
            cur_vec_theo,
            "Global Utility Optimum Combining Risky Sleeve & Rf",
        ),
    ]
    rebalance_comp_df = pd.DataFrame(rebalance_comp_rows)

    # Build Proposed Trades Table (Section 17.3) across ALL Rebalancing Alternatives (both Practical Board-Lot & Theoretical)
    prac_meta = backtest_meta_dict["prac_p_meta"]
    prac_df = backtest_series_dict["prac_p"]
    cur_nav_prac = float(prac_df["Portfolio_NAV_VND"].iloc[-1])
    cur_qty_vcb = int(prac_meta["qty_vcb"])
    cur_qty_dcm = int(prac_meta["qty_dcm"])
    cur_cash_prac = float(prac_meta["ending_cash_vnd"])
    ref_p_vcb = float(prac_meta["exit_price_vcb"])
    ref_p_dcm = float(prac_meta["exit_price_dcm"])

    cur_val_vcb = cur_qty_vcb * ref_p_vcb
    cur_val_dcm = cur_qty_dcm * ref_p_dcm
    cur_w_vcb_prac = cur_val_vcb / cur_nav_prac
    cur_w_dcm_prac = cur_val_dcm / cur_nav_prac
    cur_w_cash_prac = cur_cash_prac / cur_nav_prac

    # Ensure post-trade cash is non-negative after buy commissions and sell commissions/taxes
    def _calc_trade_cash(q_v: int, q_d: int) -> Tuple[float, float, float]:
        dv = (q_v - cur_qty_vcb) * ref_p_vcb
        dd = (q_d - cur_qty_dcm) * ref_p_dcm
        cost_v = dv * buy_fee if dv > 0 else abs(dv) * (sell_fee + sell_tax)
        cost_d = dd * buy_fee if dd > 0 else abs(dd) * (sell_fee + sell_tax)
        post_c = cur_cash_prac - dv - dd - cost_v - cost_d
        return cost_v, cost_d, post_c

    def _solve_practical_lots_for_weights(target_wv: float, target_wd: float, is_hold: bool = False):
        if is_hold:
            q_v, q_d = cur_qty_vcb, cur_qty_dcm
        else:
            q_v = int((cur_nav_prac * target_wv) // (ref_p_vcb * lot_size * (1.0 + buy_fee))) * lot_size
            q_d = int((cur_nav_prac * target_wd) // (ref_p_dcm * lot_size * (1.0 + buy_fee))) * lot_size
        c_v, c_d, post_c = _calc_trade_cash(q_v, q_d)
        while post_c < 0.0 and q_d >= lot_size:
            q_d -= lot_size
            c_v, c_d, post_c = _calc_trade_cash(q_v, q_d)
        return q_v, q_d, c_v, c_d, post_c

    trade_plan_specs = [
        ("Option 1: Hold Current Drifted Positions", cur_w_vcb_theo, cur_w_dcm_theo, 0.0, True),
        ("Option 2: Rebalance to Original Portfolio P (10.63% / 89.37%)", orig_w_vcb, orig_w_dcm, 0.0, False),
        ("Option 3A: Re-optimize Target Return Portfolio P (12.79% Target)", upd_w_vcb, upd_w_dcm, 0.0, False),
        ("Option 3B: Re-optimize Risky-Only Maximum Utility (A=2.0)", float(risky_u_upd["w_vcb"]), float(risky_u_upd["w_dcm"]), 0.0, False),
        ("Option 3C: Re-optimize Complete Portfolio Utility (A=2.0 + Rf)", float(comp_u_upd["w_vcb_total"]), float(comp_u_upd["w_dcm_total"]), float(comp_u_upd["w_rf_total"]), False),
    ]

    trade_rows = []
    for opt_label, tw_v, tw_d, tw_rf, is_hold_flag in trade_plan_specs:
        q_v, q_d, c_v, c_d, post_c = _solve_practical_lots_for_weights(tw_v, tw_d, is_hold=is_hold_flag)
        diff_v = q_v - cur_qty_vcb
        diff_d = q_d - cur_qty_dcm
        tval_v = diff_v * ref_p_vcb
        tval_d = diff_d * ref_p_dcm

        # Practical rows (VCB, DCM, Cash/Rf)
        trade_rows.extend(
            [
                {
                    "Plan_Type": f"Practical Board-Lot Plan ({opt_label})",
                    "Asset": "VCB",
                    "Current Quantity": cur_qty_vcb,
                    "Reference Price": ref_p_vcb,
                    "Current Value": cur_val_vcb,
                    "Current Weight": cur_w_vcb_prac,
                    "Proposed Weight": (q_v * ref_p_vcb) / cur_nav_prac,
                    "Model Target Weight": tw_v,
                    "Proposed Quantity": q_v,
                    "Quantity to Buy or Sell": diff_v,
                    "Estimated Trade Value": tval_v,
                    "Estimated Costs": c_v,
                    "Post-Trade Cash": post_c,
                },
                {
                    "Plan_Type": f"Practical Board-Lot Plan ({opt_label})",
                    "Asset": "DCM",
                    "Current Quantity": cur_qty_dcm,
                    "Reference Price": ref_p_dcm,
                    "Current Value": cur_val_dcm,
                    "Current Weight": cur_w_dcm_prac,
                    "Proposed Weight": (q_d * ref_p_dcm) / cur_nav_prac,
                    "Model Target Weight": tw_d,
                    "Proposed Quantity": q_d,
                    "Quantity to Buy or Sell": diff_d,
                    "Estimated Trade Value": tval_d,
                    "Estimated Costs": c_d,
                    "Post-Trade Cash": post_c,
                },
                {
                    "Plan_Type": f"Practical Board-Lot Plan ({opt_label})",
                    "Asset": "Cash / Risk-Free Ledger (VND)",
                    "Current Quantity": 1,
                    "Reference Price": 1.0,
                    "Current Value": cur_cash_prac,
                    "Current Weight": cur_w_cash_prac,
                    "Proposed Weight": post_c / cur_nav_prac,
                    "Model Target Weight": tw_rf,
                    "Proposed Quantity": 1,
                    "Quantity to Buy or Sell": 0,
                    "Estimated Trade Value": post_c - cur_cash_prac,
                    "Estimated Costs": 0.0,
                    "Post-Trade Cash": post_c,
                },
            ]
        )

        # Theoretical rows (VCB, DCM, Risk-Free)
        t_dv = cur_nav_theo * (tw_v - cur_w_vcb_theo)
        t_dd = cur_nav_theo * (tw_d - cur_w_dcm_theo)
        t_cost_v = t_dv * buy_fee if t_dv > 0 else abs(t_dv) * (sell_fee + sell_tax)
        t_cost_d = t_dd * buy_fee if t_dd > 0 else abs(t_dd) * (sell_fee + sell_tax)
        trade_rows.extend(
            [
                {
                    "Plan_Type": f"Theoretical Fractional Plan ({opt_label})",
                    "Asset": "VCB",
                    "Current Quantity": float(theo_p_df["VCB_Value_VND"].iloc[-1] / full_aligned_panel["VCB"].iloc[-1]),
                    "Reference Price": float(full_aligned_panel["VCB"].iloc[-1]),
                    "Current Value": float(theo_p_df["VCB_Value_VND"].iloc[-1]),
                    "Current Weight": cur_w_vcb_theo,
                    "Proposed Weight": tw_v,
                    "Model Target Weight": tw_v,
                    "Proposed Quantity": float((cur_nav_theo * tw_v) / full_aligned_panel["VCB"].iloc[-1]),
                    "Quantity to Buy or Sell": float(t_dv / full_aligned_panel["VCB"].iloc[-1]),
                    "Estimated Trade Value": t_dv,
                    "Estimated Costs": t_cost_v,
                    "Post-Trade Cash": cur_nav_theo * tw_rf,
                },
                {
                    "Plan_Type": f"Theoretical Fractional Plan ({opt_label})",
                    "Asset": "DCM",
                    "Current Quantity": float(theo_p_df["DCM_Value_VND"].iloc[-1] / full_aligned_panel["DCM"].iloc[-1]),
                    "Reference Price": float(full_aligned_panel["DCM"].iloc[-1]),
                    "Current Value": float(theo_p_df["DCM_Value_VND"].iloc[-1]),
                    "Current Weight": cur_w_dcm_theo,
                    "Proposed Weight": tw_d,
                    "Model Target Weight": tw_d,
                    "Proposed Quantity": float((cur_nav_theo * tw_d) / full_aligned_panel["DCM"].iloc[-1]),
                    "Quantity to Buy or Sell": float(t_dd / full_aligned_panel["DCM"].iloc[-1]),
                    "Estimated Trade Value": t_dd,
                    "Estimated Costs": t_cost_d,
                    "Post-Trade Cash": cur_nav_theo * tw_rf,
                },
            ]
        )

    proposed_trades_df = pd.DataFrame(trade_rows)

    rebalance_meta = {
        "signal_date": "2026-10-30 (Post-Close)",
        "earliest_execution_date": "2026-11-02 (Next Trading Session)",
        "rolling_window_start": rolling_3y_panel.index.min().strftime("%Y-%m-%d"),
        "rolling_window_end": rolling_3y_panel.index.max().strftime("%Y-%m-%d"),
        "rolling_observations": int(len(ret_3y)),
        "mu_updated_vcb": float(mu_upd[0]),
        "mu_updated_dcm": float(mu_upd[1]),
        "vol_updated_vcb": float(np.sqrt(cov_upd[0, 0])),
        "vol_updated_dcm": float(np.sqrt(cov_upd[1, 1])),
        "corr_updated": float(cov_upd[0, 1] / np.sqrt(cov_upd[0, 0] * cov_upd[1, 1])),
        "updated_target_p": target_upd,
        "updated_risky_u": risky_u_upd,
        "updated_complete_u": comp_u_upd,
    }
    return rebalance_comp_df, proposed_trades_df, rebalance_meta


def run_sensitivity_analysis(
    in_sample_panel: pd.DataFrame,
    oos_panel: pd.DataFrame,
    baseline_target_p: Dict[str, Any],
    shrunk_ann_cov: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Execute systematic sensitivity and robustness checks across all 6 required dimensions:
    1. Weight epsilon: 0.0001 (Baseline), 0.001, 0.01
    2. Estimation window: 1-Year (252d), 2-Year (504d), 3-Year (Baseline)
    3. Expected return perturbation: +/-1.0% and +/-2.0% p.a.
    4. Covariance estimator: Sample Covariance vs. Shrinkage Covariance (Alpha = 0.20)
    5. Transaction-cost scenarios on Practical OOS Backtest
    6. Risk aversion A: 1.0, 2.0 (Baseline), 3.0, 5.0
    """
    ann_days = float(config["TRADING_DAYS_PER_YEAR"])
    rf = float(config["RISK_FREE_ANNUAL_RATE"])
    base_A = float(config["RISK_AVERSION"])
    base_eps = float(config["WEIGHT_EPSILON"])
    target_ret = float(config["TARGET_ANNUAL_RETURN"])

    ret_full, _ = calculate_returns(in_sample_panel, columns=["VCB", "DCM"])
    mu_base = ret_full.mean().values * ann_days
    cov_base = ret_full.cov(ddof=1).values * ann_days
    base_w_vcb = float(baseline_target_p["w_vcb"])

    rows = []

    def _add_target_scenario(label: str, mu_vec: np.ndarray, cov_mat: np.ndarray, eps_val: float, A_val: float):
        gmv = optimize_gmv(mu_vec, cov_mat, rf, A_val, eps_val)
        tp = optimize_target_return(mu_vec, cov_mat, rf, A_val, gmv["expected_return"], target_ret, eps_val)
        rows.append(
            {
                "Scenario": label,
                "VCB Weight": tp["w_vcb"],
                "DCM Weight": tp["w_dcm"],
                "Risk-Free Weight": 0.0,
                "Expected Return": tp["expected_return"],
                "Volatility": tp["volatility"],
                "Sharpe": tp["sharpe_ratio"],
                "Target Feasibility": "Feasible (Portfolio P)" if tp["Target_Feasible"] else "Infeasible (P_Fallback)",
                "Weight Change versus Baseline": tp["w_vcb"] - base_w_vcb,
            }
        )

    # 0. Baseline
    _add_target_scenario("0. Baseline (3Y Window, eps=0.0001, Sample Cov)", mu_base, cov_base, base_eps, base_A)

    # 1. Weight epsilon scenarios (impact on Target Portfolio P and Constrained Max Sharpe)
    for eps_v in [0.0001, 0.001, 0.01]:
        _add_target_scenario(f"1A. Target Portfolio P | epsilon = {eps_v}", mu_base, cov_base, eps_v, base_A)
        ms = optimize_max_sharpe(mu_base, cov_base, rf, base_A, eps_v)
        rows.append(
            {
                "Scenario": f"1B. Constrained Max Sharpe | epsilon = {eps_v}",
                "VCB Weight": ms["w_vcb"],
                "DCM Weight": ms["w_dcm"],
                "Risk-Free Weight": 0.0,
                "Expected Return": ms["expected_return"],
                "Volatility": ms["volatility"],
                "Sharpe": ms["sharpe_ratio"],
                "Target Feasibility": "Boundary Optimum (w_VCB = epsilon)",
                "Weight Change versus Baseline": ms["w_vcb"] - 0.0001,
            }
        )

    # 2. Estimation window scenarios (1Y = last 252 obs, 2Y = last 504 obs, 3Y = all obs)
    for win_label, n_tail in [("2A. Estimation Window: 1 Year (Last 252 Sessions)", 252), ("2B. Estimation Window: 2 Years (Last 504 Sessions)", 504)]:
        sub_r = ret_full.iloc[-n_tail:]
        mu_w = sub_r.mean().values * ann_days
        cov_w = sub_r.cov(ddof=1).values * ann_days
        _add_target_scenario(win_label, mu_w, cov_w, base_eps, base_A)

    # 3. Expected return perturbation scenarios (+/- 1% and +/- 2% p.a.)
    perturbations = [
        ("3A. DCM Return Perturbation +1.0% p.a.", np.array([mu_base[0], mu_base[1] + 0.01])),
        ("3B. DCM Return Perturbation -1.0% p.a.", np.array([mu_base[0], mu_base[1] - 0.01])),
        ("3C. DCM Return Perturbation +2.0% p.a.", np.array([mu_base[0], mu_base[1] + 0.02])),
        ("3D. DCM Return Perturbation -2.0% p.a.", np.array([mu_base[0], mu_base[1] - 0.02])),
        ("3E. VCB Return Perturbation +2.0% p.a.", np.array([mu_base[0] + 0.02, mu_base[1]])),
        ("3F. VCB Return Perturbation -2.0% p.a.", np.array([mu_base[0] - 0.02, mu_base[1]])),
    ]
    for p_label, mu_p_vec in perturbations:
        _add_target_scenario(p_label, mu_p_vec, cov_base, base_eps, base_A)

    # 4. Shrinkage covariance estimator scenario
    _add_target_scenario(
        "4A. Shrinkage Covariance (Alpha=0.20) | Target Portfolio P",
        mu_base,
        shrunk_ann_cov.values,
        base_eps,
        base_A,
    )
    gmv_sh = optimize_gmv(mu_base, shrunk_ann_cov.values, rf, base_A, base_eps)
    gmv_bs = optimize_gmv(mu_base, cov_base, rf, base_A, base_eps)
    rows.append(
        {
            "Scenario": "4B. Shrinkage Covariance (Alpha=0.20) | GMV Portfolio",
            "VCB Weight": gmv_sh["w_vcb"],
            "DCM Weight": gmv_sh["w_dcm"],
            "Risk-Free Weight": 0.0,
            "Expected Return": gmv_sh["expected_return"],
            "Volatility": gmv_sh["volatility"],
            "Sharpe": gmv_sh["sharpe_ratio"],
            "Target Feasibility": "GMV Portfolio",
            "Weight Change versus Baseline": gmv_sh["w_vcb"] - gmv_bs["w_vcb"],
        }
    )

    # 5. Risk aversion A scenarios (A = 1, 2, 3, 5) for Risky-Only and Complete Portfolio
    base_ru = optimize_risky_utility(mu_base, cov_base, rf, base_A, base_eps)
    base_cu = optimize_complete_utility(mu_base, cov_base, rf, base_A, base_eps)
    for a_val in [1.0, 2.0, 3.0, 5.0]:
        ru = optimize_risky_utility(mu_base, cov_base, rf, a_val, base_eps)
        rows.append(
            {
                "Scenario": f"5A. Risky-Only Max Utility | Risk Aversion A = {a_val:.1f}",
                "VCB Weight": ru["w_vcb"],
                "DCM Weight": ru["w_dcm"],
                "Risk-Free Weight": 0.0,
                "Expected Return": ru["expected_return"],
                "Volatility": ru["volatility"],
                "Sharpe": ru["sharpe_ratio"],
                "Target Feasibility": f"Utility U = {ru['utility']*100:.2f}%",
                "Weight Change versus Baseline": ru["w_vcb"] - base_ru["w_vcb"],
            }
        )
        cu = optimize_complete_utility(mu_base, cov_base, rf, a_val, base_eps)
        rows.append(
            {
                "Scenario": f"5B. Complete Portfolio Max Utility | Risk Aversion A = {a_val:.1f}",
                "VCB Weight": cu["w_vcb_total"],
                "DCM Weight": cu["w_dcm_total"],
                "Risk-Free Weight": cu["w_rf_total"],
                "Expected Return": cu["expected_return"],
                "Volatility": cu["volatility"],
                "Sharpe": cu["sharpe_ratio"],
                "Target Feasibility": f"Utility U = {cu['utility']*100:.2f}% (y*={cu['risky_sleeve_y']*100:.1f}%)",
                "Weight Change versus Baseline": cu["w_dcm_total"] - base_cu["w_dcm_total"],
            }
        )

    # 6. Transaction-cost scenarios on Practical OOS Backtest
    from .backtest import run_practical_backtest

    init_w = {"VCB": baseline_target_p["w_vcb"], "DCM": baseline_target_p["w_dcm"]}
    for tc_label, b_fee, s_fee, s_tax in [
        ("6A. Zero Transaction Cost (0 bps buy/sell, 0 bps tax)", 0.0, 0.0, 0.0),
        ("6B. Baseline Retail Friction (15 bps buy/sell + 10 bps sell tax)", 0.0015, 0.0015, 0.0010),
        ("6C. High Friction Scenario (35 bps buy/sell + 10 bps sell tax)", 0.0035, 0.0035, 0.0010),
    ]:
        sim_df, meta = run_practical_backtest(oos_panel, init_w, config, tc_label, b_fee, s_fee, s_tax)
        hpr_mtm = (meta["ending_mtm_nav_vnd"] / config["INITIAL_CAPITAL_VND"]) - 1.0
        hpr_liq = (meta["net_liquidation_value_vnd"] / config["INITIAL_CAPITAL_VND"]) - 1.0
        rows.append(
            {
                "Scenario": tc_label,
                "VCB Weight": float(sim_df["VCB_Weight"].iloc[0]),
                "DCM Weight": float(sim_df["DCM_Weight"].iloc[0]),
                "Risk-Free Weight": float(sim_df["Cash_Weight"].iloc[0]),
                "Expected Return": hpr_mtm,
                "Volatility": hpr_liq,
                "Sharpe": meta["entry_buy_costs_vnd"],
                "Target Feasibility": f"MTM NAV={meta['ending_mtm_nav_vnd']:,.0f} | Net Liq={meta['net_liquidation_value_vnd']:,.0f} VND",
                "Weight Change versus Baseline": 0.0,
            }
        )

    return pd.DataFrame(rows)

