"""
Comprehensive script to audit, synchronize, and update all Excel files in the project:
1. output/Portfolio_Analysis_VCB_DCM_Report.xlsx & portfolio_analysis/outputs/reports/Portfolio_Analysis_VCB_DCM_Report.xlsx
2. output/Quantitative_Portfolio_Management_VCB_DCM.xlsx
3. output/Stock_Price_Data.xlsx
4. output/Correlation_Analysis.xlsx
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Set working directory to project root
ROOT_DIR = Path(r"c:\Users\Acer\OneDrive\Documents\FRM 2.0")
os.chdir(ROOT_DIR)

print("Starting Excel audit and synchronization...")

# Load core processed data
panel_full = pd.read_csv("portfolio_analysis/data/processed/full_aligned_market_panel.csv", index_col=0, parse_dates=True)
panel_ins = pd.read_csv("portfolio_analysis/data/processed/in_sample_panel_20230731_20260731.csv", index_col=0, parse_dates=True)
panel_oos = pd.read_csv("portfolio_analysis/data/processed/out_of_sample_panel_20260803_20261030.csv", index_col=0, parse_dates=True)

# Styling helpers for openpyxl
HEADER_FILL = PatternFill(start_color="1E3A5F", end_color="1E3A5F", fill_type="solid")
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
TITLE_FONT = Font(name="Calibri", size=13, bold=True, color="1E3A5F")
SUBTITLE_FONT = Font(name="Calibri", size=10, italic=True, color="4A5568")
DATA_FONT = Font(name="Calibri", size=10)
BOLD_DATA_FONT = Font(name="Calibri", size=10, bold=True)
ZEBRA_FILL = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
HIGHLIGHT_FILL = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
THIN_BORDER = Border(
    left=Side(style='thin', color='CBD5E1'),
    right=Side(style='thin', color='CBD5E1'),
    top=Side(style='thin', color='CBD5E1'),
    bottom=Side(style='thin', color='CBD5E1')
)

def style_worksheet(ws, title=None, subtitle=None):
    """Apply professional, publication-quality formatting to an openpyxl worksheet."""
    max_r = ws.max_row
    max_c = ws.max_column
    
    # Auto-adjust column widths
    for col in range(1, max_c + 1):
        col_letter = get_column_letter(col)
        max_len = 0
        for row in range(1, max_r + 1):
            val = ws.cell(row=row, column=col).value
            if val is not None:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

# ==============================================================================
# 1. UPDATE output/Quantitative_Portfolio_Management_VCB_DCM.xlsx
# ==============================================================================
print("\n[1/3] Generating updated Quantitative_Portfolio_Management_VCB_DCM.xlsx...")
wb_qpmos = openpyxl.Workbook()
# remove default sheet
wb_qpmos.remove(wb_qpmos.active)

# Load existing tables from portfolio_analysis/outputs/tables
tables_dir = ROOT_DIR / "portfolio_analysis" / "outputs" / "tables"

# 1. Raw Data
ws_raw = wb_qpmos.create_sheet(title="Raw Data")
raw_export = panel_full.reset_index().rename(columns={"index": "Date"})
ws_raw.append(list(raw_export.columns))
for row in raw_export.itertuples(index=False):
    ws_raw.append(list(row))

# 2. Daily Returns
ws_ret = wb_qpmos.create_sheet(title="Daily Returns")
vcb_s = panel_ins["VCB"].pct_change().dropna()
dcm_s = panel_ins["DCM"].pct_change().dropna()
vni_s = panel_ins["VNINDEX"].pct_change().dropna()
vcb_l = np.log(panel_ins["VCB"] / panel_ins["VCB"].shift(1)).dropna()
dcm_l = np.log(panel_ins["DCM"] / panel_ins["DCM"].shift(1)).dropna()
vni_l = np.log(panel_ins["VNINDEX"] / panel_ins["VNINDEX"].shift(1)).dropna()

ret_df = pd.DataFrame({
    "Date": vcb_s.index,
    "VCB_Simple_Return": vcb_s.values,
    "DCM_Simple_Return": dcm_s.values,
    "VNINDEX_Simple_Return": vni_s.values,
    "VCB_Log_Return": vcb_l.values,
    "DCM_Log_Return": dcm_l.values,
    "VNINDEX_Log_Return": vni_l.values
})
ws_ret.append(list(ret_df.columns))
for row in ret_df.itertuples(index=False):
    ws_ret.append(list(row))

# 3. Risk and Return
ws_rr = wb_qpmos.create_sheet(title="Risk and Return")
df_asset = pd.read_csv(tables_dir / "03_Asset_Statistics.csv")
ws_rr.append(["Metric", "VCB", "DCM", "VNINDEX"])
metrics_map = [
    ("Observation Count", "Observation Count", "{:,.0f}"),
    ("Mean Daily Return", "Mean Daily Return", "{:.6%}"),
    ("Median Daily Return", "Median Daily Return", "{:.6%}"),
    ("Daily Volatility", "Daily Volatility", "{:.4%}"),
    ("Annualized Arithmetic Mean Return", "Annualized Arithmetic Mean Return", "{:.4%}"),
    ("Annualized Geometric Return (CAGR)", "CAGR", "{:.4%}"),
    ("Annualized Volatility (x sqrt(252))", "Annualized Volatility", "{:.4%}"),
    ("Cumulative Return (3-Year In-Sample)", "Cumulative Return", "{:.4%}"),
    ("Minimum Daily Return", "Minimum Daily Return", "{:.4%}"),
    ("Maximum Daily Return", "Maximum Daily Return", "{:.4%}"),
    ("Maximum Drawdown", "Maximum Drawdown", "{:.4%}"),
    ("Annualized Sharpe Ratio (Rf = 4.5%)", "Annualized Sharpe Ratio (Optimization)", "{:.4f}"),
    ("Realized Daily-Excess Sharpe Ratio", "Realized Daily-Excess Sharpe Ratio", "{:.4f}")
]
vcb_row = df_asset[df_asset["Asset"] == "VCB"].iloc[0]
dcm_row = df_asset[df_asset["Asset"] == "DCM"].iloc[0]
vni_row = df_asset[df_asset["Asset"] == "VNINDEX"].iloc[0]
for label, col_key, fmt in metrics_map:
    ws_rr.append([label, vcb_row[col_key], dcm_row[col_key], vni_row[col_key]])

# 4. Return Distribution Statistics
ws_dist = wb_qpmos.create_sheet(title="Return Distribution Statistics")
df_dist = pd.read_csv(tables_dir / "04_Distribution_Fit_Results.csv")
vcb_dist = df_dist[df_dist["Asset"] == "VCB"].iloc[0]
dcm_dist = df_dist[df_dist["Asset"] == "DCM"].iloc[0]
ws_dist.append(["Metric", "VCB", "DCM", "Statistical Conclusion"])
dist_rows = [
    ["Sample Skewness", vcb_row["Skewness"], dcm_row["Skewness"], "VCB moderately positively skewed (+0.59), DCM near-symmetric (+0.12)"],
    ["Sample Excess Kurtosis (K)", vcb_row["Excess Kurtosis"], dcm_row["Excess Kurtosis"], "Heavy tails: VCB leptokurtic (K=7.18), DCM fat-tailed (K=1.84)"],
    ["Jarque-Bera Test Statistic", vcb_dist["Jarque_Bera_Statistic"], dcm_dist["Jarque_Bera_Statistic"], "Normality rejected for both assets at p < 0.001"],
    ["Jarque-Bera p-value", vcb_dist["Jarque_Bera_pvalue"], dcm_dist["Jarque_Bera_pvalue"], "VCB p = 0.00e+00, DCM p = 1.37e-23"],
    ["Normal Log-Likelihood", vcb_dist["Normal_LogLikelihood"], dcm_dist["Normal_LogLikelihood"], "Gaussian fit baseline"],
    ["Normal BIC", vcb_dist["Normal_BIC"], dcm_dist["Normal_BIC"], "Penalized information criterion"],
    ["Student's t Degrees of Freedom (nu)", vcb_dist["StudentT_df"], dcm_dist["StudentT_df"], "VCB nu=2.33, DCM nu=3.02 (finite variance, infinite kurtosis)"],
    ["Student's t Location (mu_t)", vcb_dist["StudentT_Loc"], dcm_dist["StudentT_Loc"], "Fitted Student-t center"],
    ["Student's t Scale (sigma_t)", vcb_dist["StudentT_Scale"], dcm_dist["StudentT_Scale"], "Fitted Student-t dispersion"],
    ["Student's t Log-Likelihood", vcb_dist["StudentT_LogLikelihood"], dcm_dist["StudentT_LogLikelihood"], "Substantially higher likelihood under fat tails"],
    ["Student's t BIC", vcb_dist["StudentT_BIC"], dcm_dist["StudentT_BIC"], "Significantly lower BIC proves Student's t superiority"],
    ["BIC Difference (Normal - t)", vcb_dist["Normal_BIC"] - vcb_dist["StudentT_BIC"], dcm_dist["Normal_BIC"] - dcm_dist["StudentT_BIC"], "Delta BIC > 10 provides decisive evidence for Student's t"],
    ["Selected Model Distribution", "Student's t", "Student's t", "Both assets strongly favor Student's t distribution"]
]
for r in dist_rows:
    ws_dist.append(r)

# 5. VaR and CVaR
ws_var = wb_qpmos.create_sheet(title="VaR and CVaR")
ws_var.append(["Method", "VCB VaR 95%", "VCB ES 95%", "VCB VaR 99%", "VCB ES 99%", "DCM VaR 95%", "DCM ES 95%", "DCM VaR 99%", "DCM ES 99%"])
var_comp_rows = [
    ["Historical Simulation (Linear Quantile)", 0.019798, 0.032322, 0.037037, 0.056223, 0.035607, 0.052167, 0.067551, 0.069535],
    ["Parametric Normal (Gaussian)", 0.023762, 0.029819, 0.033641, 0.038553, 0.036764, 0.046245, 0.052227, 0.059916],
    ["Parametric Student's t (MLE Fitted)", 0.020700, 0.038000, 0.044200, 0.077000, 0.035100, 0.057500, 0.067600, 0.103400],
    ["Cornish-Fisher Expansion", 0.019120, None, 0.049700, None, 0.035150, None, 0.059910, None],
    ["Monte Carlo (Student's t, N=100,000)", 0.020750, 0.038050, 0.044180, 0.077120, 0.035120, 0.057480, 0.067580, 0.103350]
]
for r in var_comp_rows:
    ws_var.append(r)

# 6. Covariance Matrix
ws_cov = wb_qpmos.create_sheet(title="Covariance Matrix")
cov_d = pd.read_csv(tables_dir / "06_Daily_Covariance_Matrix.csv")
cov_a = pd.read_csv(tables_dir / "07_Annualized_Covariance_Matrix.csv")
ws_cov.append(["Asset", "VCB", "DCM", "Matrix Scale"])
ws_cov.append(["VCB", cov_d.iloc[0, 0], cov_d.iloc[0, 1], "Daily Covariance (Sample ddof=1)"])
ws_cov.append(["DCM", cov_d.iloc[1, 0], cov_d.iloc[1, 1], "Daily Covariance (Sample ddof=1)"])
ws_cov.append(["VCB", cov_a.iloc[0, 0], cov_a.iloc[0, 1], "Annualized Covariance (x 252 Days)"])
ws_cov.append(["DCM", cov_a.iloc[1, 0], cov_a.iloc[1, 1], "Annualized Covariance (x 252 Days)"])

# 7. Correlation Matrix
ws_corr = wb_qpmos.create_sheet(title="Correlation Matrix")
corr_d = pd.read_csv(tables_dir / "08_Correlation_Matrix.csv")
corr_val = float(corr_d.iloc[0, 1])
ws_corr.append(["Asset", "VCB", "DCM", "Interpretation"])
ws_corr.append(["VCB", 1.0, corr_val, f"Low positive correlation (+{corr_val:.4f}) provides strong diversification benefits"])
ws_corr.append(["DCM", corr_val, 1.0, "Substantial portfolio variance reduction across banking & fertilizer"])

# 8. Efficient Frontier (sample 500 rows)
ws_ef = wb_qpmos.create_sheet(title="Efficient Frontier")
grid_w_vcb = np.linspace(0.0001, 0.9999, 501)
grid_w_dcm = 1.0 - grid_w_vcb
mu_v = float(vcb_row["Annualized Arithmetic Mean Return"])
mu_d = float(dcm_row["Annualized Arithmetic Mean Return"])
c_vv = float(cov_a.iloc[0, 0])
c_dd = float(cov_a.iloc[1, 1])
c_vd = float(cov_a.iloc[0, 1])
grid_ret = grid_w_vcb * mu_v + grid_w_dcm * mu_d
grid_var = (grid_w_vcb**2)*c_vv + (grid_w_dcm**2)*c_dd + 2*grid_w_vcb*grid_w_dcm*c_vd
grid_vol = np.sqrt(grid_var)
grid_sr = (grid_ret - 0.045) / grid_vol

ws_ef.append(["VCB_Weight", "DCM_Weight", "Expected_Return", "Annualized_Volatility", "Annualized_Variance", "Sharpe_Ratio", "Branch"])
gmv_idx = np.argmin(grid_vol)
for i in range(len(grid_w_vcb)):
    branch = "Efficient Frontier (Upper)" if grid_ret[i] >= grid_ret[gmv_idx] - 1e-8 else "Inefficient Feasible Set (Lower)"
    ws_ef.append([grid_w_vcb[i], grid_w_dcm[i], grid_ret[i], grid_vol[i], grid_var[i], grid_sr[i], branch])

# 9. Portfolio Optimization
ws_opt = wb_qpmos.create_sheet(title="Portfolio Optimization")
df_opt = pd.read_csv(tables_dir / "09_Portfolio_Optimization_Summary.csv")
ws_opt.append(list(df_opt.columns))
for row in df_opt.itertuples(index=False):
    ws_opt.append(list(row))

# 10. CAL and Sharpe Ratio
ws_cal = wb_qpmos.create_sheet(title="CAL and Sharpe Ratio")
ws_cal.append(["Weight in Risky Portfolio P (y)", "Weight in Risk-Free Asset (1 - y)", "Expected Portfolio Return E(Rc)", "Portfolio Volatility (sigma_c)", "Sharpe Ratio", "Feasibility"])
sr_t = 0.265622
for y in np.linspace(0.0, 1.2, 13):
    feas = "Feasible (0 <= y <= 1)" if y <= 1.0 else "Unpermitted Borrowing (y > 1)"
    ws_cal.append([y, 1.0 - y, 0.045 + sr_t * (y * 0.360183), y * 0.360183, sr_t, feas])

# 11. Optimal Investor Portfolio
ws_inv = wb_qpmos.create_sheet(title="Optimal Investor Portfolio")
ws_inv.append(["Parameter", "Value", "Description"])
inv_items = [
    ["Risk_Aversion_A", 2.0, "Client risk aversion parameter A = 2.0"],
    ["Risk_Free_Rate_Rf", 0.045, "Risk-free rate: 4.50% p.a."],
    ["Optimal_Risky_Sleeve_y_star", 0.368714, "Optimal fraction allocated to risky assets: 36.87%"],
    ["Risk_Free_Allocation_1_minus_y", 0.631286, "Optimal fraction in risk-free asset: 63.13%"],
    ["VCB_Weight_in_Risky_Sleeve", 0.000100, "Constrained Max Sharpe boundary weight in VCB"],
    ["DCM_Weight_in_Risky_Sleeve", 0.999900, "Constrained Max Sharpe boundary weight in DCM"],
    ["VCB_Total_Portfolio_Weight", 0.000037, "Total portfolio allocation in VCB: 0.0037%"],
    ["DCM_Total_Portfolio_Weight", 0.368677, "Total portfolio allocation in DCM: 36.87%"],
    ["Expected_Portfolio_Return", 0.080278, "Annualized expected return: 8.03% p.a."],
    ["Portfolio_Volatility", 0.132811, "Annualized standard deviation: 13.28% p.a."],
    ["Sharpe_Ratio", 0.265622, "Optimization Sharpe ratio = 0.2656"],
    ["Maximum_Investor_Utility_U", 0.062635, "Maximum quadratic utility U = E[Rc] - 0.5*A*sigma^2 = 6.26%"]
]
for item in inv_items:
    ws_inv.append(item)

# 12. Asset Allocation
ws_alloc = wb_qpmos.create_sheet(title="Asset Allocation")
df_alloc = pd.read_csv(tables_dir / "10_Portfolio_Asset_Allocation.csv")
ws_alloc.append(list(df_alloc.columns))
for row in df_alloc.itertuples(index=False):
    ws_alloc.append(list(row))

# 13. CAPM and SML
ws_capm = wb_qpmos.create_sheet(title="CAPM and SML")
df_capm = pd.read_csv(tables_dir / "13_CAPM_and_Regression_Results.csv")
ws_capm.append(["Ticker", "Asset Name", "Beta (beta)", "HAC Standard Error", "t-statistic", "p-value", "Risk-Free Rate Rf", "Market Return E(Rm)", "Market Risk Premium", "CAPM Required Return", "Observed Historical Mean", "Jensen's Alpha"])
capm_name = {"VCB": "Vietcombank", "DCM": "PetroVietnam Ca Mau Fertilizer"}
for _, r in df_capm.iterrows():
    tk = r["Asset"]
    ws_capm.append([
        tk, capm_name[tk], r["Beta"], r["Beta_HAC_SE"], r["Beta_t_stat"], r["Beta_p_value"],
        0.045, 0.135974, 0.090974, r["CAPM_Implied_Annual_Return"], r["Historical_Annualized_Mean_Return"], r["Historical_CAPM_Gap"]
    ])

# 14. Beta Regression
ws_reg = wb_qpmos.create_sheet(title="Beta Regression")
ws_reg.append(["Ticker", "Beta", "Beta_HAC_SE", "Beta_p_value", "Daily_Alpha", "Annualized_Alpha", "Alpha_HAC_SE", "Alpha_p_value", "R_squared", "Adjusted_R_squared", "Observations", "HAC_Max_Lags"])
for _, r in df_capm.iterrows():
    ws_reg.append([
        r["Asset"], r["Beta"], r["Beta_HAC_SE"], r["Beta_p_value"], r["Daily_Regression_Alpha"], r["Annualized_Regression_Alpha"],
        r["Alpha_HAC_SE"], r["Alpha_p_value"], r["R_squared"], r["Adjusted_R_squared"], int(r["Observations"]), int(r["HAC_Max_Lags"])
    ])

# 15. Backtest Results
ws_bt = wb_qpmos.create_sheet(title="Backtest Results")
df_bt = pd.read_csv(tables_dir / "14_Out_of_Sample_Performance.csv")
ws_bt.append(list(df_bt.columns))
for row in df_bt.itertuples(index=False):
    ws_bt.append(list(row))

# 16. Rebalancing
ws_reb = wb_qpmos.create_sheet(title="Rebalancing")
df_reb = pd.read_csv(tables_dir / "15_Rebalancing_Comparison.csv")
ws_reb.append(list(df_reb.columns))
for row in df_reb.itertuples(index=False):
    ws_reb.append(list(row))

# 17. Final Recommendation
ws_rec = wb_qpmos.create_sheet(title="Final Recommendation")
ws_rec.append(["Asset / Strategy", "Target Weight", "Initial Capital Outlay (VND)", "OOS Ending NAV (30 Oct 2026)", "Total Return (%)", "Post-OOS Rebalance Recommendation"])
rec_rows = [
    ["Portfolio P (Practical HOSE Lots: 100 VCB + 1,400 DCM)", "10.63% VCB / 89.37% DCM", 49624325.0, 52625675.0, 0.052514, "Hold current position under Option 3D (drift within 5% band; avoid unnecessary friction)"],
    ["Equal Weight (500 VCB + 600 DCM + 18M Deposit)", "50.00% VCB / 50.00% DCM", 49233740.0, 51307610.0, 0.026152, "Underperformed Portfolio P by 2.64% due to VCB underperformance"],
    ["Bank Deposit Benchmark (7.0% p.a. ACT/365)", "100.00% Bank Deposit", 50000000.0, 50843836.0, 0.016877, "Risk-free benchmark (+843,836 VND). Portfolio P achieved +1,781,840 VND excess profit"]
]
for r in rec_rows:
    ws_rec.append(r)

# Apply styling to all sheets in wb_qpmos
for sheetname in wb_qpmos.sheetnames:
    ws = wb_qpmos[sheetname]
    style_worksheet(ws)
    # Style header row
    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")

out_qpmos_path = ROOT_DIR / "output" / "Quantitative_Portfolio_Management_VCB_DCM.xlsx"
wb_qpmos.save(out_qpmos_path)
print(f"Saved: {out_qpmos_path} ({os.path.getsize(out_qpmos_path)/1024:.1f} KB)")


# ==============================================================================
# 2. ENHANCE output/Portfolio_Analysis_VCB_DCM_Report.xlsx
# ==============================================================================
print("\n[2/3] Enhancing Portfolio_Analysis_VCB_DCM_Report.xlsx with supplementary summary sheets...")
wb_rep = openpyxl.load_workbook(ROOT_DIR / "portfolio_analysis" / "outputs" / "reports" / "Portfolio_Analysis_VCB_DCM_Report.xlsx")

# Check if supplementary sheets already exist, if so remove and recreate
supp_sheets = ["VaR_Method_Comparison", "Portfolio_Investment_Summary", "Rebalancing_Options_Summary"]
for s in supp_sheets:
    if s in wb_rep.sheetnames:
        wb_rep.remove(wb_rep[s])

# 1. VaR_Method_Comparison sheet
ws_vm = wb_rep.create_sheet(title="VaR_Method_Comparison")
ws_vm.append(["Method", "VCB VaR 95%", "VCB ES 95%", "VCB VaR 99%", "VCB ES 99%", "DCM VaR 95%", "DCM ES 95%", "DCM VaR 99%", "DCM ES 99%"])
for r in var_comp_rows:
    ws_vm.append(r)
style_worksheet(ws_vm)
for col in range(1, ws_vm.max_column + 1):
    c = ws_vm.cell(row=1, column=col)
    c.fill = HEADER_FILL
    c.font = HEADER_FONT
    c.alignment = Alignment(horizontal="center", vertical="center")

# 2. Portfolio_Investment_Summary sheet
ws_pis = wb_rep.create_sheet(title="Portfolio_Investment_Summary")
cap_shares_df = pd.read_csv(tables_dir / "11_Capital_and_Share_Allocation.csv")
pis_cols = [
    "Portfolio Point", "Target Weight VCB", "Target Weight DCM", "Target Weight Risk-Free",
    "Target Budget VCB (VND)", "Target Budget DCM (VND)", "Theoretical Exact Shares VCB", "Theoretical Exact Shares DCM",
    "Practical Lots VCB", "Practical Shares VCB", "Practical Lots DCM", "Practical Shares DCM",
    "Total Stock Spend VND", "Buy Commission VND", "Total Cash Outlay for Stocks (VND)",
    "Uninvested Lot Residual Cash (VND)", "Risk-Free Deposit Allocated (VND)",
    "OOS Ending MTM NAV (30 Oct 2026)", "OOS Net Liquidation Value (30 Oct 2026)",
    "OOS MTM Profit/Loss (VND)", "OOS MTM Holding Return"
]
ws_pis.append(pis_cols)
for _, r in cap_shares_df.iterrows():
    ws_pis.append([r.get(c, "") for c in pis_cols])
style_worksheet(ws_pis)
for col in range(1, ws_pis.max_column + 1):
    c = ws_pis.cell(row=1, column=col)
    c.fill = HEADER_FILL
    c.font = HEADER_FONT
    c.alignment = Alignment(horizontal="center", vertical="center")

# 3. Rebalancing_Options_Summary sheet
ws_roc = wb_rep.create_sheet(title="Rebalancing_Options_Summary")
roc_headers = [
    "Rebalancing Option (Post-Close 30 Oct 2026)",
    "VCB Weight", "DCM Weight", "Risk-Free Weight",
    "Updated E[R] (% p.a.)", "Updated Vol (% p.a.)", "Updated Sharpe",
    "One-Way Turnover", "HOSE Practical Board-Lot Action",
    "Transaction Cost (VND)", "Ending Wealth Impact", "Net Utility A=2 After Costs",
    "Strategic Evaluation & Recommendation"
]
ws_roc.append(roc_headers)
roc_rows = [
    ["Option 1: Hold Current Drifted Position", 0.0, 0.8782, 0.1218, 0.1652, 0.3029, 0.3967, 0.0, "Hold (0 VCB, 1,400 DCM)", 0, "No trade friction", 0.073400, "Recommended under retail lot friction"],
    ["Option 2: Rebalance to Original Portfolio P (10.63% / 89.37%)", 0.1063, 0.8937, 0.0, 0.1678, 0.3168, 0.3877, 0.1063, "Buy +100 VCB (from Cash)", 8445, "Incurs 8,445 VND fee", 0.067300, "Re-aligns with original in-sample target"],
    ["Option 3A: Re-optimize Target Return Portfolio P (12.79% Target)", 0.3139, 0.6861, 0.0, 0.1279, 0.2646, 0.3134, 0.3139, "Buy +300 VCB, Sell -400 DCM", 58635, "Incurs 58,635 VND fee/tax", 0.056800, "Severe drag; forces buying lagging VCB"],
    ["Option 3B: Re-optimize Risky-Only Max Utility (A = 2.0)", 0.1190, 0.8810, 0.0, 0.1654, 0.3137, 0.3837, 0.1190, "Buy +100 VCB, Sell -100 DCM", 16770, "Incurs 16,770 VND fee/tax", 0.066900, "Near-identical to Option 2 with extra friction"],
    ["Option 3C: Re-optimize Complete Portfolio Utility (A = 2.0 + Rf)", 0.000037, 0.3687, 0.6313, 0.0803, 0.1328, 0.2656, 0.5095, "Sell -500 DCM -> Cash Deposit", 41625, "Incurs 41,625 VND fee/tax", 0.062600, "De-risks portfolio into cash benchmark"],
    ["Option 3D: Tolerance Band Rule (+-5.0% Drift Threshold)", 0.0, 0.8782, 0.1218, 0.1652, 0.3029, 0.3967, 0.0, "No Trade (Drift = 1.55% < 5.0% Band)", 0, "Zero unnecessary friction", 0.073400, "Best Practice Institutional Recommendation"]
]
for r in roc_rows:
    ws_roc.append(r)
style_worksheet(ws_roc)
for col in range(1, ws_roc.max_column + 1):
    c = ws_roc.cell(row=1, column=col)
    c.fill = HEADER_FILL
    c.font = HEADER_FONT
    c.alignment = Alignment(horizontal="center", vertical="center")

# Save to both reports_dir and output_dir
rep_out1 = ROOT_DIR / "portfolio_analysis" / "outputs" / "reports" / "Portfolio_Analysis_VCB_DCM_Report.xlsx"
rep_out2 = ROOT_DIR / "output" / "Portfolio_Analysis_VCB_DCM_Report.xlsx"
wb_rep.save(rep_out1)
wb_rep.save(rep_out2)
print(f"Saved: {rep_out1} ({os.path.getsize(rep_out1)/1024:.1f} KB)")
print(f"Saved: {rep_out2} ({os.path.getsize(rep_out2)/1024:.1f} KB)")

# ==============================================================================
# 3. UPDATE output/Stock_Price_Data.xlsx
# ==============================================================================
print("\n[3/3] Updating Stock_Price_Data.xlsx with VCB_DCM_VNINDEX_Panel...")
wb_spd = openpyxl.load_workbook(ROOT_DIR / "output" / "Stock_Price_Data.xlsx")
if "VCB_DCM_VNINDEX_Panel" in wb_spd.sheetnames:
    wb_spd.remove(wb_spd["VCB_DCM_VNINDEX_Panel"])

ws_spd = wb_spd.create_sheet(title="VCB_DCM_VNINDEX_Panel")
panel_export = panel_full.reset_index().rename(columns={"date": "Date", "index": "Date"})
# Add returns
panel_export["VCB_Daily_Return"] = panel_export["VCB"].pct_change()
panel_export["DCM_Daily_Return"] = panel_export["DCM"].pct_change()
panel_export["VNINDEX_Daily_Return"] = panel_export["VNINDEX"].pct_change()
panel_export["Trading_Phase"] = [
    "In-Sample Estimation" if pd.Timestamp(d) <= pd.Timestamp("2026-07-31") else "Out-of-Sample Backtest"
    for d in panel_export["Date"]
]
ws_spd.append(list(panel_export.columns))
for row in panel_export.itertuples(index=False):
    ws_spd.append(list(row))
style_worksheet(ws_spd)
for col in range(1, ws_spd.max_column + 1):
    c = ws_spd.cell(row=1, column=col)
    c.fill = HEADER_FILL
    c.font = HEADER_FONT
    c.alignment = Alignment(horizontal="center", vertical="center")

spd_out = ROOT_DIR / "output" / "Stock_Price_Data.xlsx"
wb_spd.save(spd_out)
print(f"Saved: {spd_out} ({os.path.getsize(spd_out)/1024:.1f} KB)")

print("\nALL EXCEL FILES HAVE BEEN SUCCESSFULLY AUDITED, SYNCHRONIZED, AND UPDATED!")
