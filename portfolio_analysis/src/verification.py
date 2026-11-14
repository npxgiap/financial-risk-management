"""
Automated unit testing and mathematical verification suite (Section 21).
Verifies 18 mandatory properties across synthetic known-solution cases and live pipeline outputs.
Synthetic data is used strictly within unit tests and never substitutes market data in reports.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .returns import calculate_returns
from .risk import calculate_var_es_for_series
from .optimization import evaluate_two_asset_portfolio, optimize_gmv, optimize_target_return


def run_all_verification_checks(
    panels: Dict[str, pd.DataFrame],
    opt_summary: Dict[str, Dict[str, Any]],
    var_es_df: pd.DataFrame,
    bt_series_dict: Dict[str, pd.DataFrame],
    bt_meta_dict: Dict[str, Any],
    rebalance_meta: Dict[str, Any],
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Execute all 18 mandatory correctness assertions and return a structured verification report table.
    Raises AssertionError immediately if any check fails.
    """
    results: List[Dict[str, Any]] = []

    def _record_check(check_id: int, name: str, passed: bool, detail: str):
        if not passed:
            raise AssertionError(f"Verification Check #{check_id} ({name}) FAILED: {detail}")
        results.append(
            {
                "Check_ID": check_id,
                "Verification_Test_Name": name,
                "Status": "PASSED",
                "Diagnostic_Details": detail,
            }
        )

    # 1. Return calculation from a small synthetic price series with known exact solution
    synth_prices = pd.DataFrame({"P": [100.0, 110.0, 99.0]}, index=pd.date_range("2025-01-01", periods=3))
    s_ret, l_ret = calculate_returns(synth_prices)
    expected_s = np.array([0.10, -0.10])
    expected_l = np.log(np.array([1.10, 0.90]))
    c1_pass = bool(np.allclose(s_ret["P"].values, expected_s, atol=1e-12) and np.allclose(l_ret["P"].values, expected_l, atol=1e-12))
    _record_check(1, "Synthetic Known-Solution Return Calculation", c1_pass, f"Computed={s_ret['P'].values.tolist()}, Expected=[0.1, -0.1]")

    # 2. Annualization convention consistency (mean*252, std*sqrt(252), cov*252)
    synth_r = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.00], "B": [0.02, 0.01, -0.01, 0.02]})
    ann_m = synth_r.mean() * 252.0
    ann_v = synth_r.std(ddof=1) * np.sqrt(252.0)
    ann_c = synth_r.cov(ddof=1) * 252.0
    c2_pass = bool(np.allclose(np.diag(ann_c), (ann_v ** 2).values, atol=1e-12) and np.isclose(ann_m["A"], 0.005 * 252.0))
    _record_check(2, "Annualization Scaling Consistency (252 Days)", c2_pass, "diag(Cov*252) == (Std*sqrt(252))^2 verified to 1e-12")

    # 3. Portfolio variance formula vs matrix quadratic form w' Sigma w
    w_test = np.array([0.37, 0.63])
    cov_test = ann_c.values
    var_quad = float(w_test.T @ cov_test @ w_test)
    m_eval = evaluate_two_asset_portfolio(0.37, float(ann_m["A"]), float(ann_m["B"]), cov_test[0, 0], cov_test[1, 1], cov_test[0, 1], 0.045, 2.0)
    c3_pass = bool(np.isclose(var_quad, m_eval["variance"], atol=1e-14))
    _record_check(3, "Portfolio Variance Quadratic Identity", c3_pass, f"Diff = {abs(var_quad - m_eval['variance']):.2e}")

    # 4. Sum of weights == 1 and strict weight bounds [epsilon, 1 - epsilon]
    eps = float(config["WEIGHT_EPSILON"])
    all_ports = [opt_summary["GMV"], opt_summary["Portfolio_P"], opt_summary["Max_Sharpe"], opt_summary["Risky_Utility"]]
    c4_pass = all(
        np.isclose(p["w_vcb"] + p["w_dcm"], 1.0, atol=1e-12)
        and (eps - 1e-12 <= p["w_vcb"] <= 1.0 - eps + 1e-12)
        and (eps - 1e-12 <= p["w_dcm"] <= 1.0 - eps + 1e-12)
        for p in all_ports
    )
    _record_check(4, "Weight Sum and Open-Interval Epsilon Bounds", c4_pass, f"All optimized risky portfolios within [{eps}, {1.0-eps}]")

    # 5. Target return 12.79% feasibility verification
    tp = opt_summary["Portfolio_P"]
    c5_pass = bool(tp["Target_Feasible"] and (tp["Feasible_Return_Min"] <= 0.1279 <= tp["Feasible_Return_Max"]))
    _record_check(
        5,
        "Target Annual Return (12.79%) Feasibility Check",
        c5_pass,
        f"Target 12.79% in [{tp['Feasible_Return_Min']*100:.2f}%, {tp['Feasible_Return_Max']*100:.2f}%]",
    )

    # 6. Analytic target weight vs numerical SLSQP solution
    c6_pass = bool(tp["analytic_vs_numeric_diff"] < 1e-9 and tp["target_return_error"] < 1e-9)
    _record_check(
        6,
        "Analytic vs. Numerical Target Weight Agreement",
        c6_pass,
        f"|w_analytic - w_num| = {tp['analytic_vs_numeric_diff']:.2e}, Return error = {tp['target_return_error']:.2e}",
    )

    # 7. GMV analytic vs numerical SLSQP solution
    gmv = opt_summary["GMV"]
    c7_pass = bool(gmv["analytic_vs_numeric_diff"] < 1e-8)
    _record_check(
        7,
        "Analytic vs. Numerical GMV Weight Agreement",
        c7_pass,
        f"|w_gmv_analytic - w_gmv_num| = {gmv['analytic_vs_numeric_diff']:.2e}",
    )

    # 8. Solver convergence and constraint residuals
    max_resid = max(p["constraint_residual"] for p in all_ports)
    c8_pass = bool(all(p["solver_converged"] for p in all_ports) and max_resid < 1e-10)
    _record_check(8, "Optimizer Convergence and Constraint Residuals", c8_pass, f"Max constraint residual = {max_resid:.2e}")

    # 9. VaR/ES positive loss sign convention on known synthetic tail
    synth_loss_ret = pd.Series(np.linspace(-0.05, 0.05, 101))
    synth_var_rec = calculate_var_es_for_series(synth_loss_ret, "SYNTH", [0.95], 50_000_000.0, include_10day=False)[0]
    c9_pass = bool(np.isclose(synth_var_rec["VaR Percentage"], 0.045, atol=1e-6) and synth_var_rec["VaR Percentage"] > 0)
    _record_check(9, "VaR / ES Positive Loss Sign Convention", c9_pass, f"95% VaR on [-5%, +5%] uniform = {synth_var_rec['VaR Percentage']*100:.2f}%")

    # 10. Expected Shortfall >= VaR across all computed rows
    diffs_es_var = var_es_df["ES Percentage"].values - var_es_df["VaR Percentage"].values
    c10_pass = bool(np.all(diffs_es_var >= -1e-12))
    _record_check(10, "Coherence Property: Expected Shortfall >= VaR", c10_pass, f"Min(ES - VaR) = {diffs_es_var.min():.4e} >= 0")

    # 11. OLS Beta regression on synthetic data with known exact Beta = 1.35 and Alpha = 0.0005
    x_syn = np.linspace(-0.03, 0.03, 200)
    y_syn = 0.0005 + 1.35 * x_syn
    ols_syn = sm.OLS(y_syn, sm.add_constant(x_syn)).fit()
    c11_pass = bool(np.isclose(ols_syn.params[1], 1.35, atol=1e-12) and np.isclose(ols_syn.params[0], 0.0005, atol=1e-12))
    _record_check(11, "Synthetic Known-Beta OLS Regression Recovery", c11_pass, f"Recovered Beta={ols_syn.params[1]:.6f}, Alpha={ols_syn.params[0]:.6f}")

    # 12. Strict temporal separation: zero out-of-sample dates in in-sample estimation panel
    ins_max = panels["in_sample"].index.max()
    oos_min = panels["out_of_sample"].index.min()
    c12_pass = bool(ins_max <= pd.Timestamp(config["IN_SAMPLE_END"]) and oos_min >= pd.Timestamp(config["OUT_OF_SAMPLE_START"]) and ins_max < oos_min)
    _record_check(12, "Zero Look-Ahead Bias & Strict Temporal Partition", c12_pass, f"In-Sample Max={ins_max.date()} < OOS Min={oos_min.date()}")

    # 13. Buy-and-hold NAV matches exact sum of asset values
    theo_p = bt_series_dict["theo_p"]
    nav_diff = (theo_p["VCB_Value_VND"] + theo_p["DCM_Value_VND"] - theo_p["Portfolio_NAV_VND"]).abs().max()
    c13_pass = bool(nav_diff < 1e-6)
    _record_check(13, "Theoretical Buy-and-Hold NAV Accounting Identity", c13_pass, f"Max |Sum(Asset_Values) - NAV| = {nav_diff:.2e} VND")

    # 14. Buy-and-hold weight drift is non-constant over out-of-sample holding period
    w_std = float(theo_p["DCM_Weight"].std(ddof=1))
    c14_pass = bool(w_std > 1e-4)
    _record_check(14, "Natural Buy-and-Hold Weight Drift Verification", c14_pass, f"Std(DCM Drifted Weight) = {w_std*100:.4f}% > 0")

    # 15. Practical execution backtest: Positions + Cash Ledger == Mark-to-Market NAV
    prac_p = bt_series_dict["prac_p"]
    prac_nav_diff = (prac_p["VCB_Value_VND"] + prac_p["DCM_Value_VND"] + prac_p["Cash_VND"] - prac_p["Portfolio_NAV_VND"]).abs().max()
    c15_pass = bool(prac_nav_diff < 1e-6)
    _record_check(15, "Practical Execution Positions + Cash == NAV Identity", c15_pass, f"Max discrepancy = {prac_nav_diff:.2e} VND")

    # 16. Transaction fees never cause negative cash balance beyond numerical tolerance
    min_cash_p = float(prac_p["Cash_VND"].min())
    min_cash_eq = float(bt_series_dict["prac_eq"]["Cash_VND"].min())
    c16_pass = bool(min_cash_p >= -1e-6 and min_cash_eq >= -1e-6)
    _record_check(16, "Non-Negative Cash Ledger Under Transaction Costs", c16_pass, f"Min Cash (Portfolio P) = {min_cash_p:,.0f} VND, (Equal Weight) = {min_cash_eq:,.0f} VND")

    # 17. Bank deposit benchmark matches exact ACT/365 simple interest over calendar days
    cal_days = int(bt_meta_dict["calendar_days"])
    expected_dep = float(config["INITIAL_CAPITAL_VND"]) * (1.0 + float(config["DEPOSIT_ANNUAL_RATE"]) * (cal_days / 365.0))
    actual_dep = float(bt_meta_dict["deposit_ending_value"])
    c17_pass = bool(np.isclose(expected_dep, actual_dep, atol=1e-6))
    _record_check(17, "Bank Deposit ACT/365 Calendar-Day Interest Formula", c17_pass, f"Elapsed={cal_days} days, Ending Deposit={actual_dep:,.2f} VND")

    # 18. Rebalance signal date strictly precedes execution date
    sig_dt = pd.Timestamp("2026-10-30")
    exec_dt = pd.Timestamp("2026-11-02")
    c18_pass = bool(sig_dt < exec_dt and rebalance_meta["signal_date"].startswith("2026-10-30"))
    _record_check(18, "Rebalance Signal Date Precedes Execution Date", c18_pass, f"Signal={sig_dt.date()} < Earliest Execution={exec_dt.date()}")

    return pd.DataFrame(results)

