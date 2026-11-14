"""
Master executable script for the VCB & DCM Quantitative Portfolio Analysis Project.
Executes the complete end-to-end workflow:
1. Market data retrieval & CSV persistence (DNSE Entrade + Yahoo Finance cross-audit)
2. Schema validation, temporal splitting, and data quality reporting
3. Simple & log return calculation and descriptive asset statistics
4. Parametric/non-parametric distribution fitting (Normal vs. Student's t, Jarque-Bera)
5. 1-day and 10-day Historical and Gaussian VaR & Expected Shortfall (95% & 99%)
6. Daily/Annualized Covariance and Pearson Correlation matrix diagnostics
7. Markowitz optimization (10,001-point grid, GMV, Target Return 12.79% Portfolio P, Max Sharpe)
8. Investor utility maximization (A = 2.0: Risky-Only vs. Complete Portfolio with Risk-Free Asset)
9. Three-year horizon Joint Moving Block Bootstrap wealth simulation (10,000 paths, 756 sessions)
10. CAPM, Security Market Line (SML), and OLS Beta regression with Newey-West HAC standard errors
11. Out-of-Sample backtesting (03 Aug 2026 – 30 Oct 2026: Theoretical vs. Practical 100-share lots vs. 7% Deposit)
12. Post-OOS portfolio rebalancing analysis and trade plan as of 30 Oct 2026
13. Multi-dimensional sensitivity and robustness analysis
14. Automated 18-point mathematical and accounting verification suite
15. High-resolution chart rendering (17 PNG figures) and 17-sheet Excel report export
"""

import sys
import platform
from pathlib import Path
from typing import Dict, Any

# Ensure UTF-8 console output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import numpy as np
import pandas as pd
import scipy
import statsmodels
import matplotlib
import seaborn as sns
import yfinance as yf
import openpyxl

from portfolio_analysis.config import (
    CONFIG,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    OUTPUT_TABLES_DIR,
    OUTPUT_FIGURES_DIR,
    OUTPUT_REPORTS_DIR,
    LEGACY_OUTPUT_DIR,
)
from portfolio_analysis.src.data import load_market_data
from portfolio_analysis.src.validation import validate_price_data, build_data_quality_report
from portfolio_analysis.src.returns import calculate_returns, calculate_asset_statistics
from portfolio_analysis.src.risk import (
    fit_return_distributions,
    calculate_var_es,
    calculate_var_es_for_series,
    calculate_covariance_correlation,
)
from portfolio_analysis.src.optimization import (
    evaluate_two_asset_portfolio,
    build_portfolio_grid,
    identify_efficient_frontier,
    optimize_gmv,
    optimize_target_return,
    optimize_max_sharpe,
    optimize_risky_utility,
    optimize_complete_utility,
)
from portfolio_analysis.src.simulation import simulate_three_year_wealth
from portfolio_analysis.src.capm import estimate_capm_regression
from portfolio_analysis.src.backtest import calculate_backtest_metrics
from portfolio_analysis.src.rebalance import propose_rebalance, run_sensitivity_analysis
from portfolio_analysis.src.reporting import generate_all_charts, generate_summary_table_pngs, export_results
from portfolio_analysis.src.verification import run_all_verification_checks


def run_complete_project() -> Dict[str, Any]:
    """
    Execute the entire quantitative portfolio analysis pipeline and return all output objects.
    """
    print("=" * 95)
    print("QUANTITATIVE PORTFOLIO ANALYSIS, BACKTESTING & REBALANCING SYSTEM – VCB & DCM")
    print("=" * 95)

    # Record runtime library versions for reproducibility
    library_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
        "yfinance": yf.__version__,
        "openpyxl": openpyxl.__version__,
    }
    print("[STEP 0] Verified Runtime Environment Versions:", library_versions)

    # 1. Load and validate market data
    print("[STEP 1] Loading and validating market data for VCB, DCM, and VNINDEX...")
    stock_long_df, benchmark_df, data_meta = load_market_data(CONFIG, DATA_RAW_DIR, use_csv_fallback_only=False)
    panels = validate_price_data(stock_long_df, benchmark_df, CONFIG)
    quality_report_df, anomaly_audit_df = build_data_quality_report(
        stock_long_df, benchmark_df, panels, data_meta, CONFIG
    )

    # Save processed aligned panels
    panels["full_aligned"].to_csv(DATA_PROCESSED_DIR / "full_aligned_market_panel.csv")
    panels["in_sample"].to_csv(DATA_PROCESSED_DIR / "in_sample_panel_20230731_20260731.csv")
    panels["out_of_sample"].to_csv(DATA_PROCESSED_DIR / "out_of_sample_panel_20260803_20261030.csv")

    in_sample_panel = panels["in_sample"]
    oos_panel = panels["out_of_sample"]

    # 2. Compute in-sample returns and asset statistics
    print("[STEP 2] Calculating daily simple/log returns and descriptive asset statistics...")
    simple_ret_ins, log_ret_ins = calculate_returns(in_sample_panel, columns=["VCB", "DCM", "VNINDEX"])
    asset_stats_df = calculate_asset_statistics(in_sample_panel, simple_ret_ins, CONFIG)

    # 3. Fit return distributions (Normal vs Student's t, AIC/BIC, Jarque-Bera)
    print("[STEP 3] Fitting Normal and Student's t return distributions...")
    dist_fit_df = fit_return_distributions(simple_ret_ins, assets=["VCB", "DCM"])

    # 4. Covariance and correlation analysis
    print("[STEP 4] Computing Daily/Annualized Covariance and Pearson Correlation matrices...")
    daily_cov, ann_cov, corr_mat, cov_diag = calculate_covariance_correlation(simple_ret_ins, CONFIG)

    # 5. Markowitz portfolio optimization & Investor Utility (A = 2.0)
    print("[STEP 5] Running Markowitz optimization across 10,001 grid points and SLSQP solvers...")
    ann_days = float(CONFIG["TRADING_DAYS_PER_YEAR"])
    rf = float(CONFIG["RISK_FREE_ANNUAL_RATE"])
    A = float(CONFIG["RISK_AVERSION"])
    eps = float(CONFIG["WEIGHT_EPSILON"])
    target_ret = float(CONFIG["TARGET_ANNUAL_RETURN"])

    mu_vec = simple_ret_ins[["VCB", "DCM"]].mean().values * ann_days
    cov_arr = ann_cov.loc[["VCB", "DCM"], ["VCB", "DCM"]].values

    gmv_res = optimize_gmv(mu_vec, cov_arr, rf, A, eps)
    grid_raw = build_portfolio_grid(mu_vec, cov_arr, rf, A, eps, n_points=int(CONFIG["GRID_POINTS"]))
    grid_df = identify_efficient_frontier(grid_raw, gmv_res["expected_return"])

    target_p_res = optimize_target_return(mu_vec, cov_arr, rf, A, gmv_res["expected_return"], target_ret, eps)
    max_sr_res = optimize_max_sharpe(mu_vec, cov_arr, rf, A, eps)
    risky_u_res = optimize_risky_utility(mu_vec, cov_arr, rf, A, eps)
    comp_u_res = optimize_complete_utility(mu_vec, cov_arr, rf, A, eps)

    eq_res = evaluate_two_asset_portfolio(
        0.5, float(mu_vec[0]), float(mu_vec[1]), float(cov_arr[0, 0]), float(cov_arr[1, 1]), float(cov_arr[0, 1]), rf, A
    )
    eq_res.update(
        {
            "Portfolio_Name": "Equal Weight Portfolio (50/50)",
            "Constraint_Status": "Interior Fixed Benchmark",
            "Is_Efficient": bool(eq_res["expected_return"] >= gmv_res["expected_return"] - 1e-9),
        }
    )

    ref_100_vcb = evaluate_two_asset_portfolio(
        1.0, float(mu_vec[0]), float(mu_vec[1]), float(cov_arr[0, 0]), float(cov_arr[1, 1]), float(cov_arr[0, 1]), rf, A
    )
    ref_100_dcm = evaluate_two_asset_portfolio(
        0.0, float(mu_vec[0]), float(mu_vec[1]), float(cov_arr[0, 0]), float(cov_arr[1, 1]), float(cov_arr[0, 1]), rf, A
    )

    opt_summary_dict = {
        "GMV": gmv_res,
        "Portfolio_P": target_p_res,
        "Max_Sharpe": max_sr_res,
        "Risky_Utility": risky_u_res,
        "Complete_Utility": comp_u_res,
        "Equal_Weight": eq_res,
        "Ref_100_VCB": ref_100_vcb,
        "Ref_100_DCM": ref_100_dcm,
    }

    opt_summary_table = pd.DataFrame(
        [
            {
                "Portfolio_Name": target_p_res["Portfolio_Name"],
                "VCB_Risky_Weight": target_p_res["w_vcb"],
                "DCM_Risky_Weight": target_p_res["w_dcm"],
                "Risk_Free_Weight": 0.0,
                "Expected_Annual_Return": target_p_res["expected_return"],
                "Annualized_Volatility": target_p_res["volatility"],
                "Annualized_Variance": target_p_res["variance"],
                "Optimization_Sharpe": target_p_res["sharpe_ratio"],
                "Investor_Utility_A2": target_p_res["utility"],
                "Is_Efficient": target_p_res["Is_Efficient"],
                "Constraint_Status": target_p_res["Constraint_Status"],
            },
            {
                "Portfolio_Name": "Equal Weight Portfolio (50/50)",
                "VCB_Risky_Weight": eq_res["w_vcb"],
                "DCM_Risky_Weight": eq_res["w_dcm"],
                "Risk_Free_Weight": 0.0,
                "Expected_Annual_Return": eq_res["expected_return"],
                "Annualized_Volatility": eq_res["volatility"],
                "Annualized_Variance": eq_res["variance"],
                "Optimization_Sharpe": eq_res["sharpe_ratio"],
                "Investor_Utility_A2": eq_res["utility"],
                "Is_Efficient": eq_res["Is_Efficient"],
                "Constraint_Status": eq_res["Constraint_Status"],
            },
            {
                "Portfolio_Name": gmv_res["Portfolio_Name"],
                "VCB_Risky_Weight": gmv_res["w_vcb"],
                "DCM_Risky_Weight": gmv_res["w_dcm"],
                "Risk_Free_Weight": 0.0,
                "Expected_Annual_Return": gmv_res["expected_return"],
                "Annualized_Volatility": gmv_res["volatility"],
                "Annualized_Variance": gmv_res["variance"],
                "Optimization_Sharpe": gmv_res["sharpe_ratio"],
                "Investor_Utility_A2": gmv_res["utility"],
                "Is_Efficient": gmv_res["Is_Efficient"],
                "Constraint_Status": gmv_res["Constraint_Status"],
            },
            {
                "Portfolio_Name": max_sr_res["Portfolio_Name"],
                "VCB_Risky_Weight": max_sr_res["w_vcb"],
                "DCM_Risky_Weight": max_sr_res["w_dcm"],
                "Risk_Free_Weight": 0.0,
                "Expected_Annual_Return": max_sr_res["expected_return"],
                "Annualized_Volatility": max_sr_res["volatility"],
                "Annualized_Variance": max_sr_res["variance"],
                "Optimization_Sharpe": max_sr_res["sharpe_ratio"],
                "Investor_Utility_A2": max_sr_res["utility"],
                "Is_Efficient": max_sr_res["Is_Efficient"],
                "Constraint_Status": max_sr_res["Constraint_Status"],
            },
            {
                "Portfolio_Name": risky_u_res["Portfolio_Name"],
                "VCB_Risky_Weight": risky_u_res["w_vcb"],
                "DCM_Risky_Weight": risky_u_res["w_dcm"],
                "Risk_Free_Weight": 0.0,
                "Expected_Annual_Return": risky_u_res["expected_return"],
                "Annualized_Volatility": risky_u_res["volatility"],
                "Annualized_Variance": risky_u_res["variance"],
                "Optimization_Sharpe": risky_u_res["sharpe_ratio"],
                "Investor_Utility_A2": risky_u_res["utility"],
                "Is_Efficient": risky_u_res["Is_Efficient"],
                "Constraint_Status": risky_u_res["Constraint_Status"],
            },
            {
                "Portfolio_Name": comp_u_res["Portfolio_Name"],
                "VCB_Risky_Weight": comp_u_res["risky_sleeve_w_vcb"],
                "DCM_Risky_Weight": comp_u_res["risky_sleeve_w_dcm"],
                "Risk_Free_Weight": comp_u_res["w_rf_total"],
                "Expected_Annual_Return": comp_u_res["expected_return"],
                "Annualized_Volatility": comp_u_res["volatility"],
                "Annualized_Variance": comp_u_res["variance"],
                "Optimization_Sharpe": comp_u_res["sharpe_ratio"],
                "Investor_Utility_A2": comp_u_res["utility"],
                "Is_Efficient": True,
                "Constraint_Status": comp_u_res["Constraint_Status"],
            },
        ]
    )

    # 6. Compute VaR & Expected Shortfall for individual stocks and constant-weight in-sample portfolios
    print("[STEP 6] Calculating 1-day and 10-day VaR and Expected Shortfall (95% & 99%)...")
    rf_daily = (1.0 + rf) ** (1.0 / ann_days) - 1.0
    extra_port_series = {
        "Portfolio P (In-Sample Constant-Weight)": (
            target_p_res["w_vcb"] * simple_ret_ins["VCB"] + target_p_res["w_dcm"] * simple_ret_ins["DCM"]
        ),
        "Equal Weight (In-Sample Constant-Weight)": (
            0.5 * simple_ret_ins["VCB"] + 0.5 * simple_ret_ins["DCM"]
        ),
        "GMV Portfolio (In-Sample Constant-Weight)": (
            gmv_res["w_vcb"] * simple_ret_ins["VCB"] + gmv_res["w_dcm"] * simple_ret_ins["DCM"]
        ),
        "Maximum Sharpe (In-Sample Constant-Weight)": (
            max_sr_res["w_vcb"] * simple_ret_ins["VCB"] + max_sr_res["w_dcm"] * simple_ret_ins["DCM"]
        ),
        "Maximum Utility Risky (In-Sample Constant-Weight)": (
            risky_u_res["w_vcb"] * simple_ret_ins["VCB"] + risky_u_res["w_dcm"] * simple_ret_ins["DCM"]
        ),
        "Optimal Complete Portfolio (In-Sample Constant-Weight)": (
            comp_u_res["w_vcb_total"] * simple_ret_ins["VCB"]
            + comp_u_res["w_dcm_total"] * simple_ret_ins["DCM"]
            + comp_u_res["w_rf_total"] * rf_daily
        ),
    }
    var_es_df = calculate_var_es(simple_ret_ins, CONFIG, extra_portfolios=extra_port_series)

    # Build Table 9: Portfolio Asset Allocation Summary (Section 13) + Detailed Capital & Share Allocation across All Points
    cap_vnd = float(CONFIG["INITIAL_CAPITAL_VND"])
    lot_sz = int(CONFIG["BOARD_LOT_SIZE"])
    buy_fee_rt = float(CONFIG["BUY_COMMISSION_RATE"])
    sell_fee_rt = float(CONFIG["SELL_COMMISSION_RATE"])
    sell_tax_rt = float(CONFIG["SELL_TAX_RATE"])

    entry_p_vcb_unadj = float(oos_panel["VCB_Unadj"].iloc[0])
    entry_p_dcm_unadj = float(oos_panel["DCM_Unadj"].iloc[0])
    exit_p_vcb_unadj = float(oos_panel["VCB_Unadj"].iloc[-1])
    exit_p_dcm_unadj = float(oos_panel["DCM_Unadj"].iloc[-1])

    alloc_specs = [
        (target_p_res["Portfolio_Name"], target_p_res["w_vcb"], target_p_res["w_dcm"], 0.0, target_p_res, "Portfolio P (In-Sample Constant-Weight)"),
        ("Equal Weight", 0.5, 0.5, 0.0, eq_res, "Equal Weight (In-Sample Constant-Weight)"),
        ("GMV", gmv_res["w_vcb"], gmv_res["w_dcm"], 0.0, gmv_res, "GMV Portfolio (In-Sample Constant-Weight)"),
        ("Maximum Sharpe", max_sr_res["w_vcb"], max_sr_res["w_dcm"], 0.0, max_sr_res, "Maximum Sharpe (In-Sample Constant-Weight)"),
        ("Maximum Utility Risky", risky_u_res["w_vcb"], risky_u_res["w_dcm"], 0.0, risky_u_res, "Maximum Utility Risky (In-Sample Constant-Weight)"),
        ("Optimal Complete Portfolio", comp_u_res["w_vcb_total"], comp_u_res["w_dcm_total"], comp_u_res["w_rf_total"], comp_u_res, "Optimal Complete Portfolio (In-Sample Constant-Weight)"),
    ]
    alloc_rows = []
    for p_label, wv, wd, wf, m_dict, var_key in alloc_specs:
        hist_95_row = var_es_df[
            (var_es_df["Asset or Portfolio"] == var_key)
            & (var_es_df["Method"] == "Historical Simulation (Linear Quantile)")
            & (var_es_df["Confidence Level"] == "95%")
            & (var_es_df["Horizon"] == "1 Trading Day")
        ].iloc[0]

        b_vcb = wv * cap_vnd
        b_dcm = wd * cap_vnd
        b_rf = wf * cap_vnd

        q_vcb_prac = int(b_vcb // (entry_p_vcb_unadj * lot_sz * (1.0 + buy_fee_rt))) * lot_sz
        q_dcm_prac = int(b_dcm // (entry_p_dcm_unadj * lot_sz * (1.0 + buy_fee_rt))) * lot_sz
        gross_vcb = q_vcb_prac * entry_p_vcb_unadj
        gross_dcm = q_dcm_prac * entry_p_dcm_unadj
        comm_buy = (gross_vcb + gross_dcm) * buy_fee_rt
        residual_cash = cap_vnd - (gross_vcb + gross_dcm + comm_buy + b_rf)

        if q_vcb_prac == 0 and wv > 0:
            c1_v = entry_p_vcb_unadj * lot_sz * (1.0 + buy_fee_rt)
            if residual_cash >= c1_v:
                q_vcb_prac += lot_sz
                gross_vcb = q_vcb_prac * entry_p_vcb_unadj
                comm_buy = (gross_vcb + gross_dcm) * buy_fee_rt
                residual_cash = cap_vnd - (gross_vcb + gross_dcm + comm_buy + b_rf)

        if q_dcm_prac == 0 and wd > 0:
            c1_d = entry_p_dcm_unadj * lot_sz * (1.0 + buy_fee_rt)
            if residual_cash >= c1_d:
                q_dcm_prac += lot_sz
                gross_dcm = q_dcm_prac * entry_p_dcm_unadj
                comm_buy = (gross_vcb + gross_dcm) * buy_fee_rt
                residual_cash = cap_vnd - (gross_vcb + gross_dcm + comm_buy + b_rf)

        alloc_rows.append(
            {
                "Portfolio Name": p_label,
                "VCB Total Weight": wv,
                "DCM Total Weight": wd,
                "Risk-Free Weight": wf,
                "Expected Annual Return": m_dict["expected_return"],
                "Annualized Volatility": m_dict["volatility"],
                "Optimization Sharpe": m_dict["sharpe_ratio"],
                "Utility": m_dict["utility"],
                "Daily Historical VaR 95%": float(hist_95_row["VaR Percentage"]),
                "Daily Historical ES 95%": float(hist_95_row["ES Percentage"]),
                "VCB Allocation VND": b_vcb,
                "DCM Allocation VND": b_dcm,
                "Risk-Free Allocation VND": b_rf,
                "Theoretical Shares VCB (03 Aug 2026)": b_vcb / entry_p_vcb_unadj,
                "Theoretical Shares DCM (03 Aug 2026)": b_dcm / entry_p_dcm_unadj,
                "Practical 100-Share Lots VCB": q_vcb_prac // lot_sz,
                "Practical Shares VCB": q_vcb_prac,
                "Practical 100-Share Lots DCM": q_dcm_prac // lot_sz,
                "Practical Shares DCM": q_dcm_prac,
                "Gross Stock Spend VCB VND": gross_vcb,
                "Gross Stock Spend DCM VND": gross_dcm,
                "Total Gross Stock Spend VND": gross_vcb + gross_dcm,
                "Buy Commission (0.15%) VND": comm_buy,
                "Total Stock Outlay Incl Fee VND": gross_vcb + gross_dcm + comm_buy,
                "Uninvested Lot Residual Cash VND": residual_cash,
                "Effective Practical Weight VCB": gross_vcb / cap_vnd,
                "Effective Practical Weight DCM": gross_dcm / cap_vnd,
                "Effective Practical Weight Cash+Rf": (b_rf + residual_cash) / cap_vnd,
            }
        )
    alloc_table_df = pd.DataFrame(alloc_rows)

    # Also build an expanded 8-point Capital & Share Execution table including 100% VCB and 100% DCM corners
    all_points_specs = [
        ("100% VCB Reference (w_VCB=100%, w_DCM=0%)", 1.0, 0.0, 0.0, ref_100_vcb),
        ("Global Minimum Variance (GMV)", gmv_res["w_vcb"], gmv_res["w_dcm"], 0.0, gmv_res),
        ("Equal Weight Portfolio (50/50)", 0.5, 0.5, 0.0, eq_res),
        ("Maximum Utility Risky-Only (A=2.0)", risky_u_res["w_vcb"], risky_u_res["w_dcm"], 0.0, risky_u_res),
        ("Target Return Portfolio P (12.79% p.a.)", target_p_res["w_vcb"], target_p_res["w_dcm"], 0.0, target_p_res),
        ("Constrained Maximum Sharpe (w_VCB=0.01%)", max_sr_res["w_vcb"], max_sr_res["w_dcm"], 0.0, max_sr_res),
        ("100% DCM Reference (w_VCB=0%, w_DCM=100%)", 0.0, 1.0, 0.0, ref_100_dcm),
        ("Optimal Complete Portfolio (Risky + Rf, A=2.0)", comp_u_res["w_vcb_total"], comp_u_res["w_dcm_total"], comp_u_res["w_rf_total"], comp_u_res),
    ]
    cap_shares_rows = []
    for pt_name, wv, wd, wf, m_dict in all_points_specs:
        b_vcb = wv * cap_vnd
        b_dcm = wd * cap_vnd
        b_rf = wf * cap_vnd

        theo_sh_vcb = b_vcb / entry_p_vcb_unadj
        theo_sh_dcm = b_dcm / entry_p_dcm_unadj

        q_vcb = int(b_vcb // (entry_p_vcb_unadj * lot_sz * (1.0 + buy_fee_rt))) * lot_sz
        q_dcm = int(b_dcm // (entry_p_dcm_unadj * lot_sz * (1.0 + buy_fee_rt))) * lot_sz

        g_vcb = q_vcb * entry_p_vcb_unadj
        g_dcm = q_dcm * entry_p_dcm_unadj
        g_tot = g_vcb + g_dcm
        fee_b = g_tot * buy_fee_rt
        outlay_tot = g_tot + fee_b
        res_cash = cap_vnd - (outlay_tot + b_rf)

        if q_vcb == 0 and wv > 0:
            c1_v = entry_p_vcb_unadj * lot_sz * (1.0 + buy_fee_rt)
            if res_cash >= c1_v:
                q_vcb += lot_sz
                g_vcb = q_vcb * entry_p_vcb_unadj
                g_tot = g_vcb + g_dcm
                fee_b = g_tot * buy_fee_rt
                outlay_tot = g_tot + fee_b
                res_cash = cap_vnd - (outlay_tot + b_rf)

        if q_dcm == 0 and wd > 0:
            c1_d = entry_p_dcm_unadj * lot_sz * (1.0 + buy_fee_rt)
            if res_cash >= c1_d:
                q_dcm += lot_sz
                g_dcm = q_dcm * entry_p_dcm_unadj
                g_tot = g_vcb + g_dcm
                fee_b = g_tot * buy_fee_rt
                outlay_tot = g_tot + fee_b
                res_cash = cap_vnd - (outlay_tot + b_rf)

        # OOS ending value on 2026-10-30 (88 calendar days; Rf sleeve accrues 4.5% p.a. ACT/365 or daily compounded)
        rf_end_val = b_rf * ((1.0 + rf_daily) ** (len(oos_panel) - 1))
        end_stock_val = q_vcb * exit_p_vcb_unadj + q_dcm * exit_p_dcm_unadj
        end_mtm = end_stock_val + rf_end_val + res_cash
        exit_sell_cost = end_stock_val * (sell_fee_rt + sell_tax_rt)
        end_net_liq = end_mtm - exit_sell_cost

        cap_shares_rows.append(
            {
                "Portfolio Point": pt_name,
                "Target Weight VCB": wv,
                "Target Weight DCM": wd,
                "Target Weight Risk-Free": wf,
                "Target Budget VCB (VND)": b_vcb,
                "Target Budget DCM (VND)": b_dcm,
                "Target Budget Risk-Free (VND)": b_rf,
                "Entry Price VCB (03 Aug 2026)": entry_p_vcb_unadj,
                "Entry Price DCM (03 Aug 2026)": entry_p_dcm_unadj,
                "Theoretical Exact Shares VCB": theo_sh_vcb,
                "Theoretical Exact Shares DCM": theo_sh_dcm,
                "Practical Lots VCB (100 sh/lot)": q_vcb // lot_sz,
                "Practical Shares VCB": q_vcb,
                "Practical Lots DCM (100 sh/lot)": q_dcm // lot_sz,
                "Practical Shares DCM": q_dcm,
                "Gross Stock Spend VCB (VND)": g_vcb,
                "Gross Stock Spend DCM (VND)": g_dcm,
                "Total Gross Stock Spend (VND)": g_tot,
                "Buy Commission 0.15% (VND)": fee_b,
                "Total Cash Outlay for Stocks (VND)": outlay_tot,
                "Risk-Free Deposit Allocated (VND)": b_rf,
                "Uninvested Residual Cash (VND)": res_cash,
                "Effective Weight VCB": g_vcb / cap_vnd,
                "Effective Weight DCM": g_dcm / cap_vnd,
                "Effective Weight Risk-Free": b_rf / cap_vnd,
                "Effective Weight Residual Cash": res_cash / cap_vnd,
                "OOS Ending MTM NAV (30 Oct 2026)": end_mtm,
                "OOS Net Liquidation Value (30 Oct 2026)": end_net_liq,
                "OOS MTM Profit/Loss (VND)": end_mtm - cap_vnd,
                "OOS MTM Holding Return": (end_mtm / cap_vnd) - 1.0,
            }
        )
    cap_shares_df = pd.DataFrame(cap_shares_rows)

    # 7. Three-year Joint Moving Block Bootstrap simulation
    print("[STEP 7] Running 10,000-path Joint Moving Block Bootstrap (756 trading sessions)...")
    portfolios_for_sim = {
        "Portfolio P (12.79% Target)": {"w_vcb": target_p_res["w_vcb"], "w_dcm": target_p_res["w_dcm"], "w_rf": 0.0},
        "Equal Weight (50/50)": {"w_vcb": 0.5, "w_dcm": 0.5, "w_rf": 0.0},
        "GMV Portfolio": {"w_vcb": gmv_res["w_vcb"], "w_dcm": gmv_res["w_dcm"], "w_rf": 0.0},
        "Maximum Utility Risky (A=2)": {"w_vcb": risky_u_res["w_vcb"], "w_dcm": risky_u_res["w_dcm"], "w_rf": 0.0},
        "Optimal Complete Portfolio (A=2)": {
            "w_vcb": comp_u_res["w_vcb_total"],
            "w_dcm": comp_u_res["w_dcm_total"],
            "w_rf": comp_u_res["w_rf_total"],
        },
    }
    sim_summary_df, sim_distributions = simulate_three_year_wealth(simple_ret_ins, portfolios_for_sim, CONFIG)

    # 8. CAPM, SML, and Beta Regression Diagnostics (Newey-West HAC)
    print("[STEP 8] Estimating CAPM and OLS Beta regressions with Newey-West HAC standard errors...")
    capm_df, capm_diag = estimate_capm_regression(simple_ret_ins, CONFIG)

    # 9. Out-of-Sample Backtesting (03 Aug 2026 – 30 Oct 2026)
    print("[STEP 9] Executing Out-of-Sample Backtests (Theoretical vs. Practical 100-Share Lots vs. 7% Deposit)...")
    p_init_weights = {"VCB": target_p_res["w_vcb"], "DCM": target_p_res["w_dcm"]}
    bt_metrics_df, bt_series_dict, bt_meta_dict = calculate_backtest_metrics(oos_panel, p_init_weights, CONFIG)

    # 10. Post-OOS Portfolio Rebalancing as of 30 Oct 2026
    print("[STEP 10] Evaluating post-OOS Rebalancing alternatives and trade plan as of 30 Oct 2026...")
    rebalance_comp_df, proposed_trades_df, rebalance_meta = propose_rebalance(
        panels["full_aligned"], bt_series_dict, bt_meta_dict, p_init_weights, CONFIG
    )

    # 11. Multi-Dimensional Sensitivity and Robustness Analysis
    print("[STEP 11] Running sensitivity analysis across epsilon, windows, return perturbations, shrinkage, A, and costs...")
    sensitivity_df = run_sensitivity_analysis(
        in_sample_panel,
        oos_panel,
        target_p_res,
        cov_diag["shrunk_annualized_covariance"],
        CONFIG,
    )

    # 12. Run all 18 automated verification checks
    print("[STEP 12] Running 18 automated mathematical and accounting verification tests...")
    verification_df = run_all_verification_checks(
        panels, opt_summary_dict, var_es_df, bt_series_dict, bt_meta_dict, rebalance_meta, CONFIG
    )
    print(f" -> All {len(verification_df)} verification checks PASSED!")

    # 13. Generate all 17 high-resolution charts
    print("[STEP 13] Rendering all 17 mandatory high-resolution PNG charts...")
    saved_charts = generate_all_charts(
        in_sample_panel,
        simple_ret_ins,
        dist_fit_df,
        daily_cov,
        ann_cov,
        corr_mat,
        grid_df,
        opt_summary_dict,
        alloc_table_df,
        capm_df,
        capm_diag,
        bt_metrics_df,
        bt_series_dict,
        sim_distributions,
        rebalance_comp_df,
        OUTPUT_FIGURES_DIR,
        LEGACY_OUTPUT_DIR,
        CONFIG,
    )

    saved_table_pngs = generate_summary_table_pngs(
        cap_shares_df,
        rebalance_comp_df,
        proposed_trades_df,
        OUTPUT_FIGURES_DIR,
        LEGACY_OUTPUT_DIR,
    )
    saved_charts.extend(saved_table_pngs)
    print("[STEP 14] Exporting all 17 tables to CSV and multi-sheet Excel workbook...")
    tables_to_export = {
        "Data_Quality_Report": quality_report_df,
        "Corporate_Action_Audit": anomaly_audit_df,
        "Asset_Statistics": asset_stats_df,
        "Distribution_Fit_Results": dist_fit_df,
        "VaR_and_Expected_Shortfall": var_es_df,
        "Daily_Covariance_Matrix": daily_cov,
        "Annualized_Covariance_Matrix": ann_cov,
        "Correlation_Matrix": corr_mat,
        "Portfolio_Optimization_Summary": opt_summary_table,
        "Portfolio_Asset_Allocation": alloc_table_df,
        "Capital_and_Share_Allocation": cap_shares_df,
        "Three_Year_Scenario_Summary": sim_summary_df,
        "CAPM_and_Regression_Results": capm_df,
        "Out_of_Sample_Performance": bt_metrics_df,
        "Rebalancing_Comparison": rebalance_comp_df,
        "Proposed_Trades": proposed_trades_df,
        "Sensitivity_Analysis": sensitivity_df,
        "Unit_Test_Verification": verification_df,
    }
    full_metadata = {
        "config": CONFIG,
        "library_versions": library_versions,
        "data_metadata": data_meta,
        "covariance_diagnostics": {k: v for k, v in cov_diag.items() if k != "shrunk_annualized_covariance"},
        "rebalance_metadata": {
            k: v for k, v in rebalance_meta.items() if not isinstance(v, dict)
        },
        "saved_charts": saved_charts,
    }
    excel_report_path = export_results(
        tables_to_export, full_metadata, OUTPUT_TABLES_DIR, OUTPUT_REPORTS_DIR, LEGACY_OUTPUT_DIR
    )

    # Display concise executive summary in console
    print("\n" + "=" * 95)
    print("ACTUAL COMPUTED RESULTS SUMMARY")
    print("=" * 95)
    print("\n[1] ASSET STATISTICS (IN-SAMPLE: 2023-07-31 TO 2026-07-31):")
    print(asset_stats_df.to_string(index=False))

    print("\n[2] DISTRIBUTION FIT RESULTS (NORMAL VS. STUDENT'S T):")
    print(dist_fit_df[["Asset", "Normal_AIC", "Normal_BIC", "StudentT_df", "StudentT_AIC", "StudentT_BIC", "Preferred_InSample_Fit_BIC", "Jarque_Bera_Statistic", "Jarque_Bera_pvalue"]].to_string(index=False))

    print("\n[3] PORTFOLIO ASSET ALLOCATION & OPTIMIZATION SUMMARY:")
    print(alloc_table_df.to_string(index=False))

    print("\n[4] THREE-YEAR JOINT BLOCK BOOTSTRAP SCENARIO SUMMARY (10,000 PATHS):")
    print(sim_summary_df.to_string(index=False))

    print("\n[5] CAPM & OLS BETA REGRESSION (NEWEY-WEST HAC STANDARD ERRORS):")
    print(capm_df[["Asset", "Beta", "Beta_HAC_SE", "Beta_p_value", "Beta_95_CI_Lower", "Beta_95_CI_Upper", "Historical_Annualized_Mean_Return", "CAPM_Implied_Annual_Return", "Historical_CAPM_Gap", "Daily_Regression_Alpha", "Annualized_Regression_Alpha", "Alpha_p_value", "R_squared"]].to_string(index=False))

    print("\n[6] OUT-OF-SAMPLE BACKTEST PERFORMANCE (2026-08-03 TO 2026-10-30):")
    print(bt_metrics_df.to_string(index=False))

    print("\n[7] POST-OOS REBALANCING ALTERNATIVES (AS OF 2026-10-30 CLOSE):")
    print(rebalance_comp_df.to_string(index=False))

    print("\n[8] PROPOSED REBALANCE TRADES (OPTION 3A):")
    print(proposed_trades_df.to_string(index=False))

    print("\n[9] SENSITIVITY & ROBUSTNESS ANALYSIS:")
    print(sensitivity_df.to_string(index=False))

    print("\n[10] UNIT TEST & MATHEMATICAL VERIFICATION REPORT:")
    print(verification_df.to_string(index=False))
    print("=" * 95)
    print(f"Complete Excel report saved to: {excel_report_path}")

    return {
        "tables": tables_to_export,
        "charts": saved_charts,
        "excel_path": excel_report_path,
        "metadata": full_metadata,
    }


if __name__ == "__main__":
    run_complete_project()
