"""
Script to build a C-suite, publication-grade Master Summary Excel Report:
'Master_Portfolio_Report_VCB_DCM.xlsx'
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT_DIR = Path(r"c:\Users\Acer\OneDrive\Documents\FRM 2.0")
os.chdir(ROOT_DIR)

tables_dir = ROOT_DIR / "portfolio_analysis" / "outputs" / "tables"
data_dir = ROOT_DIR / "portfolio_analysis" / "data" / "processed"

# Load core processed tables
df_asset = pd.read_csv(tables_dir / "03_Asset_Statistics.csv")
df_dist = pd.read_csv(tables_dir / "04_Distribution_Fit_Results.csv")
df_var = pd.read_csv(tables_dir / "05_VaR_and_Expected_Shortfall.csv")
cov_d = pd.read_csv(tables_dir / "06_Daily_Covariance_Matrix.csv")
cov_a = pd.read_csv(tables_dir / "07_Annualized_Covariance_Matrix.csv")
corr_d = pd.read_csv(tables_dir / "08_Correlation_Matrix.csv")
df_opt = pd.read_csv(tables_dir / "09_Portfolio_Optimization_Summary.csv")
df_alloc = pd.read_csv(tables_dir / "10_Portfolio_Asset_Allocation.csv")
df_cap = pd.read_csv(tables_dir / "11_Capital_and_Share_Allocation.csv")
df_capm = pd.read_csv(tables_dir / "13_CAPM_and_Regression_Results.csv")
df_bt = pd.read_csv(tables_dir / "14_Out_of_Sample_Performance.csv")
df_reb = pd.read_csv(tables_dir / "15_Rebalancing_Comparison.csv")
df_trades = pd.read_csv(tables_dir / "16_Proposed_Trades.csv")
df_full_panel = pd.read_csv(data_dir / "full_aligned_market_panel.csv", index_col=0, parse_dates=True)

# Create workbook
wb = openpyxl.Workbook()
wb.remove(wb.active) # Remove default sheet

# Professional Color Palette & Fonts
FONT_FAMILY = "Segoe UI"
COLOR_PRIMARY_NAVY = "1E3A5F"
COLOR_SECONDARY_BLUE = "2C5282"
COLOR_ACCENT_GOLD = "FEF3C7"
COLOR_ZEBRA = "F8FAFC"
COLOR_BORDER = "CBD5E1"
COLOR_TEXT_MAIN = "1A252F"
COLOR_TEXT_MUTED = "64748B"

FONT_TITLE = Font(name=FONT_FAMILY, size=14, bold=True, color=COLOR_PRIMARY_NAVY)
FONT_SUBTITLE = Font(name=FONT_FAMILY, size=9.5, italic=True, color=COLOR_TEXT_MUTED)
FONT_SECTION = Font(name=FONT_FAMILY, size=11, bold=True, color=COLOR_SECONDARY_BLUE)
FONT_HEADER = Font(name=FONT_FAMILY, size=10, bold=True, color="FFFFFF")
FONT_DATA = Font(name=FONT_FAMILY, size=9.5, color=COLOR_TEXT_MAIN)
FONT_DATA_BOLD = Font(name=FONT_FAMILY, size=9.5, bold=True, color=COLOR_TEXT_MAIN)
FONT_KEY_METRIC = Font(name=FONT_FAMILY, size=10, bold=True, color=COLOR_PRIMARY_NAVY)

FILL_HEADER = PatternFill(start_color=COLOR_PRIMARY_NAVY, end_color=COLOR_PRIMARY_NAVY, fill_type="solid")
FILL_SUBHEADER = PatternFill(start_color="334E68", end_color="334E68", fill_type="solid")
FILL_ZEBRA = PatternFill(start_color=COLOR_ZEBRA, end_color=COLOR_ZEBRA, fill_type="solid")
FILL_HIGHLIGHT = PatternFill(start_color=COLOR_ACCENT_GOLD, end_color=COLOR_ACCENT_GOLD, fill_type="solid")
FILL_CARD = PatternFill(start_color="EDF2F7", end_color="EDF2F7", fill_type="solid")

BORDER_THIN = Border(
    left=Side(style='thin', color=COLOR_BORDER),
    right=Side(style='thin', color=COLOR_BORDER),
    top=Side(style='thin', color=COLOR_BORDER),
    bottom=Side(style='thin', color=COLOR_BORDER)
)
BORDER_TOTAL = Border(
    top=Side(style='thin', color=COLOR_PRIMARY_NAVY),
    bottom=Side(style='double', color=COLOR_PRIMARY_NAVY)
)

ALIGN_LEFT = Alignment(horizontal="left", vertical="center")
ALIGN_CENTER = Alignment(horizontal="center", vertical="center")
ALIGN_RIGHT = Alignment(horizontal="right", vertical="center")
ALIGN_HEADER = Alignment(horizontal="center", vertical="center", wrap_text=True)

def apply_table_formatting(ws, start_row, end_row, start_col, end_col, highlight_rows=None, num_formats=None):
    """Format a tabular range with header fills, alternating stripes, and number formats."""
    highlight_set = set(highlight_rows or [])
    
    # Header row
    ws.row_dimensions[start_row].height = 26
    for c in range(start_col, end_col + 1):
        cell = ws.cell(row=start_row, column=c)
        cell.fill = FILL_HEADER
        cell.font = FONT_HEADER
        cell.alignment = ALIGN_HEADER
        cell.border = BORDER_THIN

    # Data rows
    for r in range(start_row + 1, end_row + 1):
        ws.row_dimensions[r].height = 19
        is_highlight = (r - start_row) in highlight_set
        is_even = ((r - start_row) % 2 == 0)
        
        row_fill = FILL_HIGHLIGHT if is_highlight else (FILL_ZEBRA if is_even else None)
        row_font = FONT_DATA_BOLD if is_highlight else FONT_DATA
        
        for c in range(start_col, end_col + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = row_font
            cell.border = BORDER_THIN
            if row_fill:
                cell.fill = row_fill
            
            # Formatting
            if c == start_col:
                cell.alignment = ALIGN_LEFT
            else:
                cell.alignment = ALIGN_RIGHT if (isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool)) else ALIGN_CENTER
            
            if num_formats and c in num_formats and isinstance(cell.value, (int, float)):
                cell.number_format = num_formats[c]

def set_sheet_views(ws, freeze_cell="A5"):
    """Enable visible grid lines and set freeze panes."""
    ws.views.sheetView[0].showGridLines = True
    ws.freeze_panes = freeze_cell

def auto_fit_columns(ws, min_col=1, max_col=None, padding=3, min_width=12):
    """Auto-adjust column widths based on content."""
    max_c = max_col or ws.max_column
    for col in range(min_col, max_c + 1):
        col_letter = get_column_letter(col)
        max_len = 0
        for row in range(1, ws.max_row + 1):
            val = ws.cell(row=row, column=col).value
            if val is not None:
                str_val = str(val)
                # Ignore very long text in title rows
                if row <= 3 and len(str_val) > 40:
                    continue
                max_len = max(max_len, len(str_val.split('\n')[0]))
        ws.column_dimensions[col_letter].width = max(max_len + padding, min_width)

print("Building professional master sheets...")

# ==============================================================================
# SHEET 1: EXECUTIVE DASHBOARD & NAVIGATION
# ==============================================================================
ws1 = wb.create_sheet(title="Executive_Dashboard")
ws1.append(["QUANTITATIVE PORTFOLIO MANAGEMENT MASTER REPORT"])
ws1.append(["Vietcombank (VCB) & Ca Mau Fertilizer (DCM) | In-Sample: 2023-07-31 to 2026-07-31 | Out-of-Sample: 2026-08-03 to 2026-10-30"])
ws1.append([])

ws1.cell(1, 1).font = FONT_TITLE
ws1.cell(2, 1).font = FONT_SUBTITLE
ws1.row_dimensions[1].height = 24
ws1.row_dimensions[2].height = 16

# KPI Summary Cards
ws1.append(["EXECUTIVE KEY PERFORMANCE INDICATORS (INITIAL CAPITAL = 50,000,000 VND)"])
ws1.cell(4, 1).font = FONT_SECTION
ws1.append(["Metric / Parameter", "Portfolio P (Target Return 12.79%)", "Equal Weight (50/50)", "7.0% Bank Deposit Benchmark", "Strategic Economic Assessment"])
kpi_data = [
    ["In-Sample Expected Return (% p.a.)", 0.1279, 0.0806, 0.0700, "Portfolio P satisfies exact 12.79% target with positive weights (10.63% VCB, 89.37% DCM)"],
    ["In-Sample Volatility (% p.a.)", 0.3289, 0.2370, 0.0000, "Low correlation (rho = 0.2529) yields substantial risk reduction on the Efficient Frontier"],
    ["In-Sample Sharpe Ratio (Rf = 4.5%)", 0.2520, 0.1502, 0.3571, "Portfolio P provides 68% higher Sharpe ratio than 50/50 Equal Weight"],
    ["Investor Utility (Risk Aversion A = 2.0)", 0.0197, 0.0244, 0.0700, "Optimal complete portfolio allocates y* = 36.87% to risky assets and 63.13% to risk-free"],
    ["Initial Stock Cash Outlay (03 Aug 2026)", 49624325.0, 49233740.0, 50000000.0, "Practical execution: 100 shares VCB (1 lot) + 1,400 shares DCM (14 lots) + 0.15% fee"],
    ["Initial Cash Reserve / Deposit (VND)", 375675.0, 766260.0, 0.0, "Uninvested lot residual cash held securely; no odd-lot fragmentation"],
    ["Ending Mark-to-Market NAV (30 Oct 2026)", 52625675.0, 49926260.0, 50843836.0, "Portfolio P generated +2,625,675 VND gross MTM profit (+5.25% holding period return)"],
    ["Ending Net Liquidation Value (VND)", 52495050.0, 49803360.0, 50843836.0, "Net of 0.15% sell commission and 0.10% personal income tax (130,625 VND total friction)"],
    ["Excess Profit vs. 7.0% Bank Deposit", 1781839.0, -917576.0, 0.0, "Portfolio P outperformed 7.0% deposit by +1,781,840 VND; Equal Weight lagged by -917,576 VND"],
    ["Systematic Market Risk (CAPM Beta)", 0.9402, 0.8396, 0.0000, "VCB Beta = 0.7119 (HAC p < 0.001); DCM Beta = 0.9674 (HAC p < 0.001)"],
    ["Jensen's Alpha vs. Market Benchmark", 0.0069, -0.0408, 0.0000, "VCB Alpha = -8.93% p.a.; DCM Alpha = +0.77% p.a. (Market efficiency confirmed: p > 0.40)"],
    ["Post-OOS Rebalance Recommendation", "Hold Drifted Position (Option 3D)", "Rebalance Required", "Benchmark Hold", "Drift (1.55%) is within 5.0% tolerance band. Best practice: avoid retail fee friction"]
]
for row in kpi_data:
    ws1.append(row)

apply_table_formatting(
    ws1, start_row=5, end_row=5 + len(kpi_data), start_col=1, end_col=5,
    highlight_rows=[1, 5, 7, 9, 12],
    num_formats={2: "0.00%", 3: "0.00%", 4: "0.00%"}
)
# Special formats for rows 5, 6, 7, 8, 9
for r in [10, 11, 12, 13, 14]:
    for c in [2, 3, 4]:
        if isinstance(ws1.cell(r, c).value, (int, float)):
            ws1.cell(r, c).number_format = '#,##0 "VND"'

# Table of Contents
toc_start = 5 + len(kpi_data) + 2
ws1.cell(toc_start, 1, "REPORT NAVIGATION & WORKBOOK STRUCTURE").font = FONT_SECTION
ws1.append(["Sheet Name", "Key Contents & Financial Topics", "Source Tables & Regulatory Alignment"])
toc_rows = [
    ["Executive_Dashboard", "High-level KPI scorecard, terminal wealth summary, and navigation index", "Executive Committee Synthesis"],
    ["Asset_Statistics_&_Risk", "In-sample returns, annualized volatility, CAGR, Sharpe, and distribution fits", "Table 03, Table 04 (MLE Student-t)"],
    ["VaR_&_Expected_Shortfall", "Comparative VaR & ES across 5 methods at 95% & 99% (1-day and 10-day)", "Table 05, Cornish-Fisher, Monte Carlo"],
    ["Markowitz_Asset_Allocation", "Efficient frontier optimization, CAL, GMV, Max Sharpe, Target Return P", "Table 09, Table 10 (SLSQP 10,001 Grid)"],
    ["Capital_&_Lot_Allocation", "HOSE 100-share lot allocation of 50M VND, trade execution, residual cash", "Table 11, Practical HOSE Lot Sizing"],
    ["CAPM_&_Beta_Regression", "Beta regressions with Newey-West HAC errors, Jensen's Alpha, SML gap", "Table 13, OLS HAC Regression"],
    ["Out_of_Sample_Backtest", "Daily NAV tracking, drawdowns, terminal wealth vs. 7.0% deposit", "Table 14, 64-Session OOS Walk"],
    ["Rebalancing_Strategy", "Decision matrix (Options 1 to 3D), tolerance band rule, trade order tickets", "Table 15, Table 16 (HOSE Practical Orders)"],
    ["Aligned_Market_Data", "Aligned 811-session price panel (raw, adjusted, simple & log returns)", "Full Data Panel (2023-2026)"]
]
for r in toc_rows:
    ws1.append(r)
apply_table_formatting(ws1, start_row=toc_start + 1, end_row=toc_start + 1 + len(toc_rows), start_col=1, end_col=3)
set_sheet_views(ws1, "A6")
auto_fit_columns(ws1)

# ==============================================================================
# SHEET 2: ASSET STATISTICS & DISTRIBUTION FITTING
# ==============================================================================
ws2 = wb.create_sheet(title="Asset_Statistics_&_Risk")
ws2.append(["ASSET RETURN & RISK CHARACTERISTICS (IN-SAMPLE: 31 JUL 2023 – 31 JUL 2026)"])
ws2.append(["Sample Size: 747 Daily Trading Sessions | Annualization Factor: 252 Days | Benchmark: VN-Index"])
ws2.append([])
ws2.cell(1, 1).font = FONT_TITLE
ws2.cell(2, 1).font = FONT_SUBTITLE

# Section A: Asset Statistics
ws2.cell(4, 1, "PANEL A: DESCRIPTIVE RETURN AND RISK METRICS").font = FONT_SECTION
ws2.append(list(df_asset.columns))
for r in df_asset.itertuples(index=False):
    ws2.append(list(r))
apply_table_formatting(
    ws2, start_row=5, end_row=5 + len(df_asset), start_col=1, end_col=len(df_asset.columns),
    num_formats={
        2: "#,##0", 3: "0.0000%", 4: "0.0000%", 5: "0.00%", 6: "0.00%", 7: "0.00%", 8: "0.00%",
        9: "0.00%", 10: "0.00%", 11: "0.00%", 12: "0.00", 13: "0.00", 14: "0.00%", 15: "0.0000", 16: "0.0000"
    }
)

# Section B: Distribution Fit Results (Table 4.2)
r_dist_start = 5 + len(df_asset) + 2
ws2.cell(r_dist_start, 1, "PANEL B: RETURN DISTRIBUTION FITTING & NORMALITY TESTS (MLE STUDENT'S T VS. NORMAL)").font = FONT_SECTION
dist_display = pd.DataFrame({
    "Metric / Specification": [
        "BIC – Normal Distribution",
        "BIC – Student's t Distribution",
        "BIC Difference (Normal – Student's t)",
        "Estimated Degrees of Freedom (nu_hat)",
        "Empirical Excess Kurtosis (K)",
        "Sample Skewness (S)",
        "Jarque-Bera Test Statistic",
        "Jarque-Bera p-value",
        "Preferred Distribution (Model Selection)"
    ],
    "VCB (Vietcombank)": [-4193.3, -4470.0, 276.7, 2.33, 7.18, 0.59, 1621.7, 0.0000, "Student's t"],
    "DCM (PetroVietnam Ca Mau)": [-3523.9, -3608.7, 84.8, 3.02, 1.84, 0.12, 105.3, 1.37e-23, "Student's t"],
    "Financial Econometric Interpretation": [
        "Gaussian fit penalized likelihood criterion",
        "Penalized likelihood under fat-tailed distribution",
        "Delta BIC > 10 provides decisive evidence favoring Student's t",
        "2 < nu <= 4: Finite variance exists, theoretical 4th moment is infinite",
        "Heavy tails: VCB leptokurtic (7.18), DCM fat-tailed (1.84)",
        "VCB right-skewed (+0.59), DCM near-symmetric (+0.12)",
        "Normality decisively rejected for both equities (p < 0.001)",
        "Both p-values reject H0 of normality at any significance level",
        "Student's t mandatory for tail risk, VaR, and stress testing"
    ]
})
ws2.append(list(dist_display.columns))
for r in dist_display.itertuples(index=False):
    ws2.append(list(r))
apply_table_formatting(
    ws2, start_row=r_dist_start + 1, end_row=r_dist_start + 1 + len(dist_display), start_col=1, end_col=4,
    highlight_rows=[2, 3, 4, 5, 9]
)
# Format numbers in dist_display
for row_idx in [r_dist_start + 2, r_dist_start + 3, r_dist_start + 4]:
    ws2.cell(row_idx, 2).number_format = "#,##0.0"
    ws2.cell(row_idx, 3).number_format = "#,##0.0"
for row_idx in [r_dist_start + 5, r_dist_start + 6, r_dist_start + 7]:
    ws2.cell(row_idx, 2).number_format = "0.00"
    ws2.cell(row_idx, 3).number_format = "0.00"

# Section C: Covariance and Correlation
r_cov_start = r_dist_start + 1 + len(dist_display) + 2
ws2.cell(r_cov_start, 1, "PANEL C: COVARIANCE & CORRELATION ANALYSIS (DIVERSIFICATION ASSESSMENT)").font = FONT_SECTION
ws2.append(["Matrix Type", "VCB", "DCM", "Interpretation & Diversification Impact"])
ws2.append(["Daily Covariance (ddof = 1)", float(cov_d.iloc[0, 0]), float(cov_d.iloc[0, 1]), "Sample covariance matrix"])
ws2.append(["Daily Covariance (ddof = 1)", float(cov_d.iloc[1, 0]), float(cov_d.iloc[1, 1]), "Positive co-movement is modest across banking & fertilizer"])
ws2.append(["Annualized Covariance (x 252 Days)", float(cov_a.iloc[0, 0]), float(cov_a.iloc[0, 1]), "Scaled annual covariance matrix for Markowitz optimization"])
ws2.append(["Annualized Covariance (x 252 Days)", float(cov_a.iloc[1, 0]), float(cov_a.iloc[1, 1]), "Cross-term covariance = 0.0210 provides significant portfolio variance dampening"])
corr_val = float(corr_d.iloc[0, 1])
ws2.append(["Pearson Correlation Matrix", 1.0, corr_val, f"Low positive correlation (+{corr_val:.4f} < 0.30) confirms outstanding diversification potential"])
ws2.append(["Pearson Correlation Matrix", corr_val, 1.0, "Portfolio volatility drops from 36.02% (DCM) to 21.37% (GMV) - a 40.7% risk reduction"])

apply_table_formatting(
    ws2, start_row=r_cov_start + 1, end_row=r_cov_start + 7, start_col=1, end_col=4,
    highlight_rows=[5, 6],
    num_formats={2: "0.0000", 3: "0.0000"}
)
set_sheet_views(ws2, "A5")
auto_fit_columns(ws2)

# ==============================================================================
# SHEET 3: VALUE AT RISK (VaR) & EXPECTED SHORTFALL (ES)
# ==============================================================================
ws3 = wb.create_sheet(title="VaR_&_Expected_Shortfall")
ws3.append(["VALUE AT RISK (VaR) AND EXPECTED SHORTFALL (ES) COMPREHENSIVE BENCHMARK"])
ws3.append(["1-Day and 10-Day Horizons | Loss Convention: L_t = -r_t | Sub-additivity verified: ES >= VaR"])
ws3.append([])
ws3.cell(1, 1).font = FONT_TITLE
ws3.cell(2, 1).font = FONT_SUBTITLE

# 5-Method Comparison Table (User Image 2 Match)
ws3.cell(4, 1, "PANEL A: 1-DAY HORIZON METHOD COMPARISON TABLE (VCB VS. DCM)").font = FONT_SECTION
ws3.append(["Method", "VCB VaR 95%", "VCB ES 95%", "VCB VaR 99%", "VCB ES 99%", "DCM VaR 95%", "DCM ES 95%", "DCM VaR 99%", "DCM ES 99%"])
var_comp_rows = [
    ["Historical Simulation (Linear Quantile)", 0.019798, 0.032322, 0.037037, 0.056223, 0.035607, 0.052167, 0.067551, 0.069535],
    ["Parametric Normal (Gaussian)", 0.023762, 0.029819, 0.033641, 0.038553, 0.036764, 0.046245, 0.052227, 0.059916],
    ["Parametric Student's t (MLE Fitted)", 0.020700, 0.038000, 0.044200, 0.077000, 0.035100, 0.057500, 0.067600, 0.103400],
    ["Cornish-Fisher Expansion (Skew & Kurtosis)", 0.019120, None, 0.049700, None, 0.035150, None, 0.059910, None],
    ["Monte Carlo (Student's t, N = 100,000)", 0.020750, 0.038050, 0.044180, 0.077120, 0.035120, 0.057480, 0.067580, 0.103350]
]
for r in var_comp_rows:
    ws3.append(r)
apply_table_formatting(
    ws3, start_row=5, end_row=5 + len(var_comp_rows), start_col=1, end_col=9,
    highlight_rows=[1, 3],
    num_formats={c: "0.00%" for c in range(2, 10)}
)

# Detailed VaR Table for Portfolios & Horizons
r_var_port = 5 + len(var_comp_rows) + 2
ws3.cell(r_var_port, 1, "PANEL B: PORTFOLIO VaR & ES ACROSS HORIZONS (CAPITAL = 50,000,000 VND)").font = FONT_SECTION
sub_var_df = df_var[df_var["Asset or Portfolio"].isin([
    "Portfolio P (In-Sample Constant-Weight)", "Equal Weight (In-Sample Constant-Weight)", "GMV Portfolio (In-Sample Constant-Weight)"
])].copy()
ws3.append(list(sub_var_df.columns))
for r in sub_var_df.itertuples(index=False):
    ws3.append(list(r))

apply_table_formatting(
    ws3, start_row=r_var_port + 1, end_row=r_var_port + 1 + len(sub_var_df), start_col=1, end_col=len(sub_var_df.columns),
    num_formats={5: "0.00%", 6: "0.00%", 7: '#,##0 "VND"', 8: '#,##0 "VND"', 9: "#,##0"}
)
set_sheet_views(ws3, "A5")
auto_fit_columns(ws3)

# ==============================================================================
# SHEET 4: MARKOWITZ OPTIMIZATION & ASSET ALLOCATION
# ==============================================================================
ws4 = wb.create_sheet(title="Markowitz_Asset_Allocation")
ws4.append(["MARKOWITZ MEAN-VARIANCE OPTIMIZATION & ASSET ALLOCATION"])
ws4.append(["Universe: 0 < w < 1 (Strict Open Interval with epsilon = 0.0001) | Target Return = 12.79% p.a."])
ws4.append([])
ws4.cell(1, 1).font = FONT_TITLE
ws4.cell(2, 1).font = FONT_SUBTITLE

# Optimization Summary Table
ws4.cell(4, 1, "PANEL A: SUMMARY OF KEY PORTFOLIO POINTS ON EFFICIENT FRONTIER").font = FONT_SECTION
ws4.append(list(df_opt.columns))
for r in df_opt.itertuples(index=False):
    ws4.append(list(r))
apply_table_formatting(
    ws4, start_row=5, end_row=5 + len(df_opt), start_col=1, end_col=len(df_opt.columns),
    highlight_rows=[2, 4], # Highlight Target P and Complete Utility
    num_formats={
        2: "0.00%", 3: "0.00%", 4: "0.00%", 5: "0.00%", 6: "0.00%", 7: "0.0000",
        8: "0.0000", 9: "0.0000", 10: "0.0000", 11: "0.0000"
    }
)

# Detailed Asset Allocation Table
r_alloc = 5 + len(df_opt) + 2
ws4.cell(r_alloc, 1, "PANEL B: MODEL ASSET ALLOCATION COMPARISON (THEORETICAL WEIGHTS)").font = FONT_SECTION
alloc_subset = df_alloc[[
    "Portfolio Name", "VCB Total Weight", "DCM Total Weight", "Risk-Free Weight",
    "Expected Annual Return", "Annualized Volatility", "Optimization Sharpe", "Utility",
    "Daily Historical VaR 95%", "Daily Historical ES 95%"
]].copy()
ws4.append(list(alloc_subset.columns))
for r in alloc_subset.itertuples(index=False):
    ws4.append(list(r))
apply_table_formatting(
    ws4, start_row=r_alloc + 1, end_row=r_alloc + 1 + len(alloc_subset), start_col=1, end_col=len(alloc_subset.columns),
    highlight_rows=[1, 6],
    num_formats={2: "0.00%", 3: "0.00%", 4: "0.00%", 5: "0.00%", 6: "0.00%", 7: "0.0000", 8: "0.0000", 9: "0.00%", 10: "0.00%"}
)
set_sheet_views(ws4, "A5")
auto_fit_columns(ws4)

# ==============================================================================
# SHEET 5: CAPITAL & SHARE ALLOCATION (HOSE BOARD LOTS)
# ==============================================================================
ws5 = wb.create_sheet(title="Capital_&_Lot_Allocation")
ws5.append(["PRACTICAL CAPITAL OUTLAY & HOSE 100-SHARE LOT ALLOCATION (50,000,000 VND)"])
ws5.append(["Entry Date: 03 Aug 2026 | HOSE Board Lot: 100 Shares | Buy Commission: 0.15% | Entry Prices: VCB 60,800 VND, DCM 31,050 VND"])
ws5.append([])
ws5.cell(1, 1).font = FONT_TITLE
ws5.cell(2, 1).font = FONT_SUBTITLE

ws5.cell(4, 1, "PRACTICAL SHARE PURCHASE AND RESIDUAL CASH BREAKDOWN").font = FONT_SECTION
cap_disp_cols = [
    "Portfolio Point", "Target Weight VCB", "Target Weight DCM", "Target Weight Risk-Free",
    "Practical Lots VCB (100 sh/lot)", "Practical Shares VCB", "Practical Lots DCM (100 sh/lot)", "Practical Shares DCM",
    "Total Cash Outlay for Stocks (VND)", "Uninvested Residual Cash (VND)", "Risk-Free Deposit Allocated (VND)",
    "OOS Ending MTM NAV (30 Oct 2026)", "OOS MTM Profit/Loss (VND)", "OOS MTM Holding Return"
]
df_cap_disp = df_cap[cap_disp_cols].copy()
ws5.append(list(df_cap_disp.columns))
for r in df_cap_disp.itertuples(index=False):
    ws5.append(list(r))

apply_table_formatting(
    ws5, start_row=5, end_row=5 + len(df_cap_disp), start_col=1, end_col=len(cap_disp_cols),
    highlight_rows=[5, 8], # Highlight Portfolio P and Complete Portfolio
    num_formats={
        2: "0.00%", 3: "0.00%", 4: "0.00%", 5: "#,##0", 6: "#,##0", 7: "#,##0", 8: "#,##0",
        9: '#,##0 "VND"', 10: '#,##0 "VND"', 11: '#,##0 "VND"', 12: '#,##0 "VND"', 13: '#,##0 "VND"', 14: "0.00%"
    }
)
set_sheet_views(ws5, "A5")
auto_fit_columns(ws5)

# ==============================================================================
# SHEET 6: CAPM, SML & BETA REGRESSION
# ==============================================================================
ws6 = wb.create_sheet(title="CAPM_&_Beta_Regression")
ws6.append(["CAPM REGRESSION & SECURITY MARKET LINE (SML) CHOW/HAC DIAGNOSTICS"])
ws6.append(["Market Benchmark: VN-Index | Newey-West Heteroskedasticity & Autocorrelation Consistent (HAC) Errors | Max Lags = 5"])
ws6.append([])
ws6.cell(1, 1).font = FONT_TITLE
ws6.cell(2, 1).font = FONT_SUBTITLE

ws6.cell(4, 1, "CAPM REGRESSION RESULTS AND JENSEN'S ALPHA ESTIMATES").font = FONT_SECTION
capm_sub = df_capm[[
    "Asset", "Beta", "Beta_HAC_SE", "Beta_t_stat", "Beta_p_value",
    "Historical_Annualized_Mean_Return", "CAPM_Implied_Annual_Return", "Historical_CAPM_Gap",
    "Daily_Regression_Alpha", "Annualized_Regression_Alpha", "Alpha_HAC_SE", "Alpha_p_value",
    "R_squared", "Adjusted_R_squared", "Observations"
]].copy()
ws6.append(list(capm_sub.columns))
for r in capm_sub.itertuples(index=False):
    ws6.append(list(r))

apply_table_formatting(
    ws6, start_row=5, end_row=5 + len(capm_sub), start_col=1, end_col=len(capm_sub.columns),
    highlight_rows=[1, 2],
    num_formats={
        2: "0.0000", 3: "0.0000", 4: "0.00", 5: "0.0000e+00",
        6: "0.00%", 7: "0.00%", 8: "0.00%", 9: "0.0000%", 10: "0.00%", 11: "0.0000%", 12: "0.0000",
        13: "0.00%", 14: "0.00%", 15: "#,##0"
    }
)

# Economic commentary on CAPM
ws6.append([])
ws6.append(["CAPM Financial Econometric Conclusions:"])
ws6.append(["1. Systematic Risk (Beta): VCB has Beta = 0.7119 (defensive), DCM has Beta = 0.9674 (aggressive). Both are highly statistically significant (p < 0.0001)."])
ws6.append(["2. Jensen's Alpha: VCB Jensen's Alpha is -8.93% p.a. (underperformed CAPM SML); DCM Jensen's Alpha is +0.77% p.a. (in line with CAPM SML)."])
ws6.append(["3. Market Efficiency: Both alpha p-values (VCB p = 0.437, DCM p = 0.966) exceed 0.05, confirming market efficiency with no statistically significant abnormal return."])
for r_idx in range(len(capm_sub) + 7, len(capm_sub) + 11):
    ws6.cell(r_idx, 1).font = FONT_DATA_BOLD if r_idx == len(capm_sub) + 7 else FONT_DATA
set_sheet_views(ws6, "A5")
auto_fit_columns(ws6)

# ==============================================================================
# SHEET 7: OUT-OF-SAMPLE BACKTEST (64 TRADING SESSIONS)
# ==============================================================================
ws7 = wb.create_sheet(title="Out_of_Sample_Backtest")
ws7.append(["OUT-OF-SAMPLE BACKTEST PERFORMANCE (03 AUG 2026 – 30 OCT 2026)"])
ws7.append(["Actual Entry Date: 03 Aug 2026 | Exit Date: 30 Oct 2026 | Calendar Days: 88 | Trading Sessions: 64 | Benchmark: 7.0% p.a. ACT/365 Deposit"])
ws7.append([])
ws7.cell(1, 1).font = FONT_TITLE
ws7.cell(2, 1).font = FONT_SUBTITLE

ws7.cell(4, 1, "STRATEGY BACKTEST SUMMARY METRICS (INITIAL CAPITAL = 50,000,000 VND)").font = FONT_SECTION
ws7.append(list(df_bt.columns))
for r in df_bt.itertuples(index=False):
    ws7.append(list(r))

apply_table_formatting(
    ws7, start_row=5, end_row=5 + len(df_bt), start_col=1, end_col=len(df_bt.columns),
    highlight_rows=[3], # Highlight Practical Portfolio P
    num_formats={
        4: "#,##0", 5: "#,##0", 6: '#,##0 "VND"', 7: '#,##0 "VND"', 8: '#,##0 "VND"', 9: '#,##0 "VND"',
        10: "0.00%", 11: "0.00%", 12: "0.0000", 13: "0.00%", 14: "0.00%", 15: "0.00%",
        16: '#,##0 "VND"', 17: '#,##0 "VND"', 18: '#,##0 "VND"'
    }
)
set_sheet_views(ws7, "A5")
auto_fit_columns(ws7)

# ==============================================================================
# SHEET 8: REBALANCING STRATEGY & TRADE PLAN
# ==============================================================================
ws8 = wb.create_sheet(title="Rebalancing_Strategy")
ws8.append(["POST-OUT-OF-SAMPLE PORTFOLIO REBALANCING EVALUATION (AS OF 30 OCT 2026 CLOSE)"])
ws8.append(["Decision Date: Post-Close 30 Oct 2026 | Earliest Execution Date: 02 Nov 2026 | Total Practical NAV = 52,625,675 VND"])
ws8.append([])
ws8.cell(1, 1).font = FONT_TITLE
ws8.cell(2, 1).font = FONT_SUBTITLE

# Panel A: Comprehensive Rebalancing Decision Matrix (User Comparison Table)
ws8.cell(4, 1, "PANEL A: REBALANCING DECISION MATRIX (OPTIONS 1 TO 3D)").font = FONT_SECTION
roc_headers = [
    "Rebalancing Option", "Proposed VCB Weight", "Proposed DCM Weight", "Proposed Cash/Rf Weight",
    "Updated E[R] (% p.a.)", "Updated Vol (% p.a.)", "Updated Sharpe", "One-Way Turnover",
    "Practical HOSE Board-Lot Trade Action", "Estimated Friction (VND)", "Net Investor Utility (A=2)", "Strategic Evaluation & Recommendation"
]
ws8.append(roc_headers)
roc_rows = [
    ["Option 1: Hold Current Drifted Positions", 0.0, 0.8782, 0.1218, 0.1652, 0.3029, 0.3967, 0.0, "Hold (0 VCB, 1,400 DCM, Cash 6.46M)", 0, 0.073400, "Recommended under retail lot friction (highest utility)"],
    ["Option 2: Rebalance to Original Portfolio P", 0.1063, 0.8937, 0.0, 0.1678, 0.3168, 0.3877, 0.1063, "Buy +100 VCB (using Cash)", 8445, 0.067300, "Re-establishes original target; incurs minimal fee"],
    ["Option 3A: Re-optimize Target Return (12.79%)", 0.3139, 0.6861, 0.0, 0.1279, 0.2646, 0.3134, 0.3139, "Buy +300 VCB, Sell -400 DCM", 58635, 0.056800, "Severe drag; forces capital into lagging VCB"],
    ["Option 3B: Re-optimize Risky Utility (A = 2.0)", 0.1190, 0.8810, 0.0, 0.1654, 0.3137, 0.3837, 0.1190, "Buy +100 VCB, Sell -100 DCM", 16770, 0.066900, "Near-identical to Option 2 with extra friction"],
    ["Option 3C: Re-optimize Complete Portfolio (A = 2.0 + Rf)", 0.000037, 0.3687, 0.6313, 0.0803, 0.1328, 0.2656, 0.5095, "Sell -500 DCM -> Cash Deposit", 41625, 0.062600, "De-risks portfolio into cash deposit"],
    ["Option 3D: Tolerance Band Rule (+-5.0% Band)", 0.0, 0.8782, 0.1218, 0.1652, 0.3029, 0.3967, 0.0, "No Trade (Drift = 1.55% < 5.0% Band)", 0, 0.073400, "Institutional Best Practice Recommendation"]
]
for r in roc_rows:
    ws8.append(r)
apply_table_formatting(
    ws8, start_row=5, end_row=5 + len(roc_rows), start_col=1, end_col=len(roc_headers),
    highlight_rows=[1, 6],
    num_formats={
        2: "0.00%", 3: "0.00%", 4: "0.00%", 5: "0.00%", 6: "0.00%", 7: "0.0000",
        8: "0.00%", 10: '#,##0 "VND"', 11: "0.0000"
    }
)

# Panel B: Practical Board-Lot Trade Plan
r_trades = 5 + len(roc_rows) + 2
ws8.cell(r_trades, 1, "PANEL B: PRACTICAL HOSE 100-SHARE LOT TRADE EXECUTION ORDERS (T+2 SETTLEMENT)").font = FONT_SECTION
trades_sub = df_trades[df_trades["Plan_Type"].str.startswith("Practical Board-Lot Plan")].copy()
ws8.append(list(trades_sub.columns))
for r in trades_sub.itertuples(index=False):
    ws8.append(list(r))

apply_table_formatting(
    ws8, start_row=r_trades + 1, end_row=r_trades + 1 + len(trades_sub), start_col=1, end_col=len(trades_sub.columns),
    num_formats={
        3: "#,##0", 4: '#,##0 "VND"', 5: '#,##0 "VND"', 6: "0.00%", 7: "0.00%", 8: "0.00%",
        9: "#,##0", 10: "#,##0", 11: '#,##0 "VND"', 12: '#,##0 "VND"', 13: '#,##0 "VND"'
    }
)
set_sheet_views(ws8, "A5")
auto_fit_columns(ws8)

# ==============================================================================
# SHEET 9: ALIGNED MARKET DATA PANEL
# ==============================================================================
ws9 = wb.create_sheet(title="Aligned_Market_Data")
ws9.append(["ALIGNED MARKET PRICE & RETURN PANEL (2023-07-31 TO 2026-10-30)"])
ws9.append(["Total Observations: 811 Sessions (747 In-Sample + 64 Out-of-Sample) | Source: Yahoo Finance & DNSE"])
ws9.append([])
ws9.cell(1, 1).font = FONT_TITLE
ws9.cell(2, 1).font = FONT_SUBTITLE

raw_panel_disp = df_full_panel.reset_index().rename(columns={"date": "Date", "index": "Date"}).copy()
raw_panel_disp["VCB_Daily_Return"] = raw_panel_disp["VCB"].pct_change()
raw_panel_disp["DCM_Daily_Return"] = raw_panel_disp["DCM"].pct_change()
raw_panel_disp["VNINDEX_Daily_Return"] = raw_panel_disp["VNINDEX"].pct_change()
raw_panel_disp["Trading_Phase"] = [
    "In-Sample Estimation" if pd.Timestamp(d) <= pd.Timestamp("2026-07-31") else "Out-of-Sample Backtest"
    for d in raw_panel_disp["Date"]
]
ws9.append(list(raw_panel_disp.columns))
for r in raw_panel_disp.itertuples(index=False):
    ws9.append(list(r))

apply_table_formatting(
    ws9, start_row=4, end_row=4 + len(raw_panel_disp), start_col=1, end_col=len(raw_panel_disp.columns),
    num_formats={
        2: "#,##0.00", 3: "#,##0.00", 4: "#,##0.00", 5: "#,##0", 6: "#,##0", 7: "#,##0", 8: "#,##0",
        9: "#,##0.00", 10: "#,##0.00", 11: "0.0000%", 12: "0.0000%", 13: "0.0000%"
    }
)
set_sheet_views(ws9, "A5")
auto_fit_columns(ws9)

# Save Master Workbook
master_path = ROOT_DIR / "output" / "Master_Portfolio_Report_VCB_DCM.xlsx"
wb.save(master_path)
print(f"Successfully generated Master Report: {master_path} ({os.path.getsize(master_path)/1024:.1f} KB)")
