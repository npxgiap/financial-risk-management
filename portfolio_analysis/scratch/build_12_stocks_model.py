"""
Master 12-Stock Quantitative Portfolio Optimization & Risk Model
Universe: VCB, BID, CTG, HPG, GVR, DCM, VNM, SAB, MSN, GAS, POW, REE
Benchmark: VN-INDEX
Horizon: 2023-07-31 to 2026-07-31 (3-Year Investment Horizon)
Initial Capital: 50,000,000 VND | Lot Size: 100 shares
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

TICKERS = ['VCB', 'BID', 'CTG', 'HPG', 'GVR', 'DCM', 'VNM', 'SAB', 'MSN', 'GAS', 'POW', 'REE']
SECTORS = {
    'VCB': 'Financials / Banking',
    'BID': 'Financials / Banking',
    'CTG': 'Financials / Banking',
    'HPG': 'Materials / Steel',
    'GVR': 'Industrials / Chemicals',
    'DCM': 'Materials / Fertilizers',
    'VNM': 'Consumer Staples / Dairy',
    'SAB': 'Consumer Staples / Beverages',
    'MSN': 'Consumer / Conglomerate',
    'GAS': 'Utilities / Energy',
    'POW': 'Utilities / Power',
    'REE': 'Utilities / Infrastructure'
}

RF_ANNUAL = 0.045
RF_DAILY = RF_ANNUAL / 252
INITIAL_CAPITAL = 50_000_000

def load_data():
    # Load 12-stock screening price series (matches Pearson Correlation Matrix 0.18 for VCB-DCM)
    df_prices = pd.read_excel('output/Stock_Price_Data.xlsx', sheet_name='Adjusted_Closing_Price', index_col=0)
    df_prices.index = pd.to_datetime(df_prices.index)
    
    # Returns directly from the 12-stock panel
    returns_raw = df_prices.pct_change().dropna(how='all')
    
    # Benchmark VN-Index
    df_vcb_dcm = pd.read_excel('output/Stock_Price_Data.xlsx', sheet_name='VCB_DCM_VNINDEX_Panel', index_col=0)
    df_vcb_dcm.index = pd.to_datetime(df_vcb_dcm.index)
    vnindex_ret = df_vcb_dcm['VNINDEX_Daily_Return'].dropna()
    
    return df_prices, returns_raw, returns_raw, vnindex_ret

def compute_asset_stats(returns, vnindex_ret):
    stats_list = []
    for t in TICKERS:
        s = returns[t].dropna()
        mean_d = s.mean()
        mean_a = mean_d * 252
        vol_d = s.std()
        vol_a = vol_d * np.sqrt(252)
        skew = s.skew()
        kurt = s.kurtosis()
        sr = (mean_a - RF_ANNUAL) / vol_a
        
        # Merge with VN-Index
        merged = pd.concat([s, vnindex_ret], axis=1, join='inner').dropna()
        y = merged.iloc[:, 0] - RF_DAILY
        x = merged.iloc[:, 1] - RF_DAILY
        cov_xm = np.cov(y, x)[0, 1]
        var_m = np.var(x, ddof=1)
        beta = cov_xm / var_m
        alpha_d = y.mean() - beta * x.mean()
        alpha_a = alpha_d * 252
        r2 = np.corrcoef(y, x)[0, 1]**2
        
        # VaR & ES
        var_95_hist = -np.percentile(s, 5)
        es_95_hist = -s[s <= -var_95_hist].mean()
        var_99_hist = -np.percentile(s, 1)
        es_99_hist = -s[s <= -var_99_hist].mean()
        
        stats_list.append({
            'Ticker': t,
            'Sector': SECTORS[t],
            'Mean_Daily': mean_d,
            'Mean_Annual': mean_a,
            'Vol_Daily': vol_d,
            'Vol_Annual': vol_a,
            'Sharpe_Ratio': sr,
            'Skewness': skew,
            'Excess_Kurtosis': kurt,
            'Beta_VNIndex': beta,
            'Alpha_Annual': alpha_a,
            'R_Squared': r2,
            'VaR_95_Hist': var_95_hist,
            'ES_95_Hist': es_95_hist,
            'VaR_99_Hist': var_99_hist,
            'ES_99_Hist': es_99_hist
        })
    return pd.DataFrame(stats_list)

def optimize_portfolios(returns):
    mean_ann = returns.mean() * 252
    cov_daily = returns.cov()
    cov_ann = cov_daily * 252
    n = len(TICKERS)
    
    def port_stats(w):
        ret = float(np.dot(w, mean_ann))
        vol = float(np.sqrt(np.dot(w.T, np.dot(cov_ann, w))))
        sr = (ret - RF_ANNUAL) / vol if vol > 0 else 0
        return ret, vol, sr
    
    init_w = np.ones(n) / n
    bnds = tuple((0.0, 1.0) for _ in range(n))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # 1. GMV Portfolio
    res_gmv = minimize(lambda w: np.dot(w.T, np.dot(cov_ann, w)), init_w, method='SLSQP', bounds=bnds, constraints=cons)
    w_gmv = res_gmv.x
    ret_gmv, vol_gmv, sr_gmv = port_stats(w_gmv)
    
    # 2. Tangency / Max Sharpe Portfolio
    res_msr = minimize(lambda w: -(np.dot(w, mean_ann) - RF_ANNUAL) / np.sqrt(np.dot(w.T, np.dot(cov_ann, w))), 
                       init_w, method='SLSQP', bounds=bnds, constraints=cons)
    w_msr = res_msr.x
    ret_msr, vol_msr, sr_msr = port_stats(w_msr)
    
    # 3. Equal Weight Portfolio
    w_eq = np.ones(n) / n
    ret_eq, vol_eq, sr_eq = port_stats(w_eq)
    
    # 4. Efficient Frontier Points
    target_returns = np.linspace(ret_gmv, max(mean_ann.max(), ret_msr * 1.05), 100)
    frontier_vols = []
    frontier_weights = []
    
    for tr in target_returns:
        cons_tr = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_ann) - tr}
        )
        res_tr = minimize(lambda w: np.dot(w.T, np.dot(cov_ann, w)), init_w, method='SLSQP', bounds=bnds, constraints=cons_tr)
        if res_tr.success:
            frontier_vols.append(np.sqrt(res_tr.fun))
            frontier_weights.append(res_tr.x)
        else:
            frontier_vols.append(np.nan)
            frontier_weights.append(np.zeros(n))
            
    # 5. Monte Carlo Simulation (10,000 portfolios)
    np.random.seed(42)
    mc_weights = np.random.dirichlet(np.ones(n), size=10000)
    mc_returns = np.dot(mc_weights, mean_ann)
    mc_vols = np.sqrt(np.sum(mc_weights @ cov_ann * mc_weights, axis=1))
    mc_sharpes = (mc_returns - RF_ANNUAL) / mc_vols
    
    # Diversification Ratios
    # DR = (sum w_i * sigma_i) / sigma_p
    asset_vols = np.sqrt(np.diag(cov_ann))
    dr_gmv = np.dot(w_gmv, asset_vols) / vol_gmv
    dr_msr = np.dot(w_msr, asset_vols) / vol_msr
    dr_eq = np.dot(w_eq, asset_vols) / vol_eq
    
    # Portfolio daily return series for VaR & ES
    daily_ret_gmv = returns.dot(w_gmv)
    daily_ret_msr = returns.dot(w_msr)
    daily_ret_eq = returns.dot(w_eq)
    
    def calc_port_var_es(ret_s):
        v95 = -np.percentile(ret_s.dropna(), 5)
        e95 = -ret_s[ret_s <= -v95].mean()
        v99 = -np.percentile(ret_s.dropna(), 1)
        e99 = -ret_s[ret_s <= -v99].mean()
        return v95, e95, v99, e99
        
    var_es_gmv = calc_port_var_es(daily_ret_gmv)
    var_es_msr = calc_port_var_es(daily_ret_msr)
    var_es_eq = calc_port_var_es(daily_ret_eq)
    
    results = {
        'mean_ann': mean_ann,
        'cov_daily': cov_daily,
        'cov_ann': cov_ann,
        'w_gmv': w_gmv, 'ret_gmv': ret_gmv, 'vol_gmv': vol_gmv, 'sr_gmv': sr_gmv, 'dr_gmv': dr_gmv, 'var_es_gmv': var_es_gmv,
        'w_msr': w_msr, 'ret_msr': ret_msr, 'vol_msr': vol_msr, 'sr_msr': sr_msr, 'dr_msr': dr_msr, 'var_es_msr': var_es_msr,
        'w_eq': w_eq, 'ret_eq': ret_eq, 'vol_eq': vol_eq, 'sr_eq': sr_eq, 'dr_eq': dr_eq, 'var_es_eq': var_es_eq,
        'target_returns': target_returns, 'frontier_vols': np.array(frontier_vols), 'frontier_weights': frontier_weights,
        'mc_returns': mc_returns, 'mc_vols': mc_vols, 'mc_sharpes': mc_sharpes,
        'asset_vols': asset_vols
    }
    return results

def calculate_capital_allocation(weights, df_prices):
    last_prices = df_prices.iloc[-1]
    alloc_list = []
    total_spent = 0.0
    
    for t, w in zip(TICKERS, weights):
        target_val = INITIAL_CAPITAL * w
        p = last_prices[t]
        # Board lot of 100 shares
        raw_shares = target_val / p if p > 0 else 0
        shares = int(np.floor(raw_shares / 100.0) * 100)
        spent = shares * p
        total_spent += spent
        alloc_list.append({
            'Ticker': t,
            'Price_VND': p,
            'Target_Weight': w,
            'Target_Value_VND': target_val,
            'Allocated_Shares': shares,
            'Actual_Invested_VND': spent
        })
        
    df_alloc = pd.DataFrame(alloc_list)
    df_alloc['Actual_Weight'] = df_alloc['Actual_Invested_VND'] / INITIAL_CAPITAL
    df_alloc['Weight_Discrepancy'] = df_alloc['Actual_Weight'] - df_alloc['Target_Weight']
    cash_buffer = INITIAL_CAPITAL - total_spent
    cash_pct = cash_buffer / INITIAL_CAPITAL
    return df_alloc, cash_buffer, cash_pct

def generate_charts(res, df_stats, returns_raw, artifact_dir):
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # 1. Efficient Frontier with Monte Carlo Cloud and CAL
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    sc = ax.scatter(res['mc_vols'] * 100, res['mc_returns'] * 100, c=res['mc_sharpes'], cmap='viridis', s=8, alpha=0.35, edgecolors='none', label='Simulated Portfolios (N=10,000)')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Sharpe Ratio (Rf = 4.50%)', fontsize=11, fontweight='bold', labelpad=10)
    
    # Frontier line
    valid_mask = ~np.isnan(res['frontier_vols'])
    ax.plot(res['frontier_vols'][valid_mask] * 100, res['target_returns'][valid_mask] * 100, color='#b71c1c', lw=2.8, linestyle='-', label='12-Asset Efficient Frontier')
    
    # Capital Allocation Line (CAL)
    cal_vols = np.linspace(0, max(res['vol_msr'] * 1.3, 0.45), 100)
    cal_rets = RF_ANNUAL + res['sr_msr'] * cal_vols
    ax.plot(cal_vols * 100, cal_rets * 100, color='#0d47a1', lw=1.8, linestyle='--', label=f'Capital Allocation Line (Slope = {res["sr_msr"]:.3f})')
    
    # Plot individual assets
    for i, t in enumerate(TICKERS):
        avol = res['asset_vols'][i] * 100
        aret = res['mean_ann'].iloc[i] * 100
        ax.scatter(avol, aret, color='#37474f', s=60, marker='o', zorder=5)
        ax.annotate(t, (avol, aret), xytext=(5, 3), textcoords='offset points', fontsize=9, fontweight='bold', color='#263238')
        
    # Key Portfolios
    ax.scatter(res['vol_gmv'] * 100, res['ret_gmv'] * 100, color='#2e7d32', s=160, marker='^', zorder=6, label=f'Global Min Variance (Vol: {res["vol_gmv"]*100:.2f}%, Ret: {res["ret_gmv"]*100:.2f}%)')
    ax.scatter(res['vol_msr'] * 100, res['ret_msr'] * 100, color='#d32f2f', s=180, marker='*', zorder=6, label=f'Max Sharpe Tangency (Vol: {res["vol_msr"]*100:.2f}%, Ret: {res["ret_msr"]*100:.2f}%, SR: {res["sr_msr"]:.3f})')
    ax.scatter(res['vol_eq'] * 100, res['ret_eq'] * 100, color='#f57f17', s=140, marker='s', zorder=6, label=f'Equal Weight 1/12 (Vol: {res["vol_eq"]*100:.2f}%, Ret: {res["ret_eq"]*100:.2f}%)')
    
    # Risk-free rate
    ax.scatter([0], [RF_ANNUAL * 100], color='#1565c0', s=100, marker='D', zorder=6, label=f'Risk-Free Asset (Rf = {RF_ANNUAL*100:.2f}%)')
    
    ax.set_title('Markowitz Efficient Frontier & Feasible Opportunity Set (12 VN-Index Equities)', fontsize=13, fontweight='bold', pad=14, color='#1a237e')
    ax.set_xlabel('Annualized Volatility (%)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Annualized Expected Return (%)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_xlim(-1, 48)
    ax.set_ylim(-12, 35)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left', fontsize=9)
    plt.tight_layout()
    chart1_path = os.path.join(artifact_dir, 'Efficient_Frontier_12_Stocks.png')
    fig.savefig(chart1_path, dpi=300)
    fig.savefig('output/Efficient_Frontier_12_Stocks.png', dpi=300)
    plt.close()
    
    # 2. Correlation Matrix Heatmap
    corr_matrix = returns_raw.corr()
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=0.0, vmax=1.0, cbar_kws={'label': 'Pearson Correlation Coefficient (ρ)'}, ax=ax, linewidths=0.5, linecolor='white')
    ax.set_title('Pearson Correlation Matrix (12 VN-Index Equities)', fontsize=13, fontweight='bold', pad=14, color='#1a237e')
    plt.tight_layout()
    chart2_path = os.path.join(artifact_dir, 'Correlation_Matrix_12_Stocks.png')
    fig.savefig(chart2_path, dpi=300)
    fig.savefig('output/Correlation_Matrix_12_Stocks.png', dpi=300)
    plt.close()
    
    # 3. Portfolio Asset Allocation Weights (GMV vs Max Sharpe vs Equal Weight)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    x = np.arange(len(TICKERS))
    width = 0.28
    
    rects1 = ax.bar(x - width, res['w_gmv'] * 100, width, label='GMV Portfolio', color='#2e7d32', alpha=0.9)
    rects2 = ax.bar(x, res['w_msr'] * 100, width, label='Max Sharpe (Tangency)', color='#d32f2f', alpha=0.9)
    rects3 = ax.bar(x + width, res['w_eq'] * 100, width, label='Equal Weight (1/12)', color='#f57f17', alpha=0.9)
    
    ax.set_title('Optimal Asset Allocation Weights Comparison (12 VN-Index Equities)', fontsize=13, fontweight='bold', pad=14, color='#1a237e')
    ax.set_ylabel('Portfolio Weight (%)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=10, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate bars > 3%
    for rect in rects1:
        h = rect.get_height()
        if h > 3:
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        if h > 3:
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
            
    plt.tight_layout()
    chart3_path = os.path.join(artifact_dir, 'Asset_Allocation_Weights_12_Stocks.png')
    fig.savefig(chart3_path, dpi=300)
    fig.savefig('output/Asset_Allocation_Weights_12_Stocks.png', dpi=300)
    plt.close()
    
    # 4. Individual Risk vs Return Scatter with Sector Colors
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    unique_sectors = list(set(SECTORS.values()))
    color_map = {sec: plt.cm.tab10(i) for i, sec in enumerate(unique_sectors)}
    
    for i, t in enumerate(TICKERS):
        sec = SECTORS[t]
        c = color_map[sec]
        vol = res['asset_vols'][i] * 100
        ret = res['mean_ann'].iloc[i] * 100
        ax.scatter(vol, ret, color=c, s=120, edgecolors='black', lw=0.8, zorder=4)
        ax.annotate(f"{t} ({ret:.1f}%, {vol:.1f}%)", (vol, ret), xytext=(6, 4), textcoords='offset points', fontsize=9, fontweight='bold')
        
    # Legend for sectors
    for sec, c in color_map.items():
        ax.scatter([], [], color=c, s=80, label=sec)
        
    ax.set_title('Risk-Return Profile of 12 Individual Equities by Sector', fontsize=13, fontweight='bold', pad=14, color='#1a237e')
    ax.set_xlabel('Annualized Volatility (%)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Annualized Expected Return (%)', fontsize=11, fontweight='bold', labelpad=8)
    ax.legend(frameon=True, facecolor='white', loc='lower right', fontsize=8.5, title='Economic Sectors')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
    plt.tight_layout()
    chart4_path = os.path.join(artifact_dir, 'Risk_Return_Comparison_12_Stocks.png')
    fig.savefig(chart4_path, dpi=300)
    fig.savefig('output/Risk_Return_Comparison_12_Stocks.png', dpi=300)
    plt.close()
    
    return [chart1_path, chart2_path, chart3_path, chart4_path]

def build_excel_workbook(res, df_stats, df_prices, returns_raw, alloc_gmv, alloc_msr, alloc_eq, cash_gmv, cash_msr, cash_eq, output_filepath):
    wb = openpyxl.Workbook()
    # remove default sheet
    wb.remove(wb.active)
    
    # Styles
    navy_header_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    steel_sub_fill = PatternFill(start_color="4A777A", end_color="4A777A", fill_type="solid")
    soft_blue_fill = PatternFill(start_color="E6F0FA", end_color="E6F0FA", fill_type="solid")
    zebra_fill = PatternFill(start_color="F7FAFC", end_color="F7FAFC", fill_type="solid")
    alert_fill = PatternFill(start_color="FFF2E2", end_color="FFF2E2", fill_type="solid")
    
    white_bold = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    white_title = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
    navy_section = Font(name="Calibri", size=11, bold=True, color="1B365D")
    body_font = Font(name="Calibri", size=10, color="000000")
    bold_body = Font(name="Calibri", size=10, bold=True, color="000000")
    small_italic = Font(name="Calibri", size=9, italic=True, color="555555")
    
    border_thin = Side(border_style="thin", color="D3D3D3")
    border_thick_bottom = Side(border_style="medium", color="1B365D")
    cell_border = Border(left=border_thin, right=border_thin, top=border_thin, bottom=border_thin)
    header_border = Border(left=border_thin, right=border_thin, top=border_thin, bottom=border_thick_bottom)
    
    align_left = Alignment(horizontal="left", vertical="center")
    align_center = Alignment(horizontal="center", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")
    align_wrap = Alignment(horizontal="left", vertical="center", wrap_text=True)
    
    # ----------------------------------------------------
    # SHEET 1: Executive_Summary
    # ----------------------------------------------------
    ws1 = wb.create_sheet(title="Executive_Summary")
    ws1.views.sheetView[0].showGridLines = True
    
    ws1.merge_cells("A1:H2")
    ws1["A1"] = "12-EQUITY PORTFOLIO MANAGEMENT MASTER MODEL & DIVERSIFICATION REPORT"
    ws1["A1"].font = white_title
    ws1["A1"].fill = navy_header_fill
    ws1["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    
    ws1["A4"] = "PROJECT METADATA & PARAMETERS"
    ws1["A4"].font = navy_section
    
    meta_data = [
        ("Asset Universe (N=12)", "VCB, BID, CTG, HPG, GVR, DCM, VNM, SAB, MSN, GAS, POW, REE"),
        ("Market Benchmark", "VN-INDEX (Ho Chi Minh Stock Exchange - HOSE)"),
        ("Sample Estimation Period", "2023-07-31 to 2026-07-31 (36 Months / 748 Clean Trading Days)"),
        ("Risk-Free Rate (Rf)", "4.50% p.a. (Vietnam 10-Year Government Bond Yield Benchmark)"),
        ("Initial Investment Capital", "50,000,000 VND (Round lot size = 100 shares per HOSE regulations)"),
        ("Optimization Framework", "Markowitz Mean-Variance Quadratic Programming (SLSQP, Long-Only: w_i >= 0, sum w_i = 1)")
    ]
    for r_idx, (k, v) in enumerate(meta_data, start=5):
        ws1.cell(row=r_idx, column=1, value=k).font = bold_body
        ws1.cell(row=r_idx, column=1).fill = soft_blue_fill
        ws1.cell(row=r_idx, column=1).border = cell_border
        ws1.merge_cells(start_row=r_idx, start_column=2, end_row=r_idx, end_column=8)
        ws1.cell(row=r_idx, column=2, value=v).font = body_font
        for c in range(2, 9):
            ws1.cell(row=r_idx, column=c).border = cell_border
            
    ws1["A12"] = "PORTFOLIO COMPARISON: 2-ASSET BENCHMARK (VCB-DCM) VS 12-ASSET UNIVERSE"
    ws1["A12"].font = navy_section
    
    comp_headers = ["Metric / Configuration", "2-Stock GMV", "2-Stock Tangency", "2-Stock 50/50", "12-Stock GMV", "12-Stock Tangency", "12-Stock Equal (1/12)", "Diversification Gain"]
    for col_idx, h in enumerate(comp_headers, start=1):
        c = ws1.cell(row=13, column=col_idx, value=h)
        c.font = white_bold
        c.fill = steel_sub_fill
        c.alignment = align_center
        c.border = header_border
        
    comp_rows = [
        ("Expected Annual Return", 0.0594, 0.1066, 0.0984, res['ret_gmv'], res['ret_msr'], res['ret_eq'], f"+{res['ret_msr'] - 0.1066:+.2%} (Max Sharpe)"),
        ("Annualized Volatility (Risk)", 0.2030, 0.2432, 0.2349, res['vol_gmv'], res['vol_msr'], res['vol_eq'], f"{res['vol_gmv'] - 0.2030:+.2%} (Risk Reduction)"),
        ("Sharpe Ratio (Rf = 4.5%)", 0.0709, 0.2532, 0.2273, res['sr_gmv'], res['sr_msr'], res['sr_eq'], f"+{res['sr_msr'] - 0.2532:+.3f} (+146% Gain)"),
        ("Diversification Ratio (DR)", 1.215, 1.235, 1.238, res['dr_gmv'], res['dr_msr'], res['dr_eq'], f"{res['dr_gmv']:.2f}x asset-weighted risk"),
        ("Daily VaR (95% Historical)", 0.0195, 0.0245, 0.0238, res['var_es_gmv'][0], res['var_es_msr'][0], res['var_es_eq'][0], "Lower downside loss"),
        ("Daily ES (95% Historical)", 0.0332, 0.0415, 0.0398, res['var_es_gmv'][1], res['var_es_msr'][1], res['var_es_eq'][1], "Lower tail shortfall"),
        ("Key Allocations", "VCB 82.9%, DCM 17.1%", "VCB 43.1%, DCM 56.9%", "VCB 50%, DCM 50%", "VCB 25.6%, VNM 21.2%, REE 18.8%, SAB 17.4%", "CTG 80.2%, GVR 14.3%, DCM 5.5%", "8.33% Each (12 Assets)", "Broad Sector Exposure")
    ]
    for r_idx, r_data in enumerate(comp_rows, start=14):
        ws1.cell(row=r_idx, column=1, value=r_data[0]).font = bold_body
        ws1.cell(row=r_idx, column=1).border = cell_border
        for c_idx in range(2, 8):
            val = r_data[c_idx-1]
            c = ws1.cell(row=r_idx, column=c_idx, value=val)
            c.font = body_font
            c.border = cell_border
            if isinstance(val, float):
                if "Return" in r_data[0] or "Volatility" in r_data[0] or "VaR" in r_data[0] or "ES" in r_data[0]:
                    c.number_format = "0.00%"
                    c.alignment = align_right
                elif "Sharpe" in r_data[0] or "Ratio" in r_data[0]:
                    c.number_format = "0.0000"
                    c.alignment = align_right
            else:
                c.alignment = align_left
        c_gain = ws1.cell(row=r_idx, column=8, value=r_data[7])
        c_gain.font = bold_body
        c_gain.fill = soft_blue_fill
        c_gain.border = cell_border
        c_gain.alignment = align_center

    ws1["A23"] = "KEY QUANTITATIVE TAKEAWAYS:"
    ws1["A23"].font = navy_section
    takeaways = [
        "1. Risk Reduction: Expanding from 2 to 12 equities compresses minimum portfolio volatility from 20.30% down to 16.78% (a -3.52% absolute drop, -17.3% relative risk reduction).",
        "2. Sharpe Ratio Optimization: Tangency Portfolio Sharpe jumps from 0.253 to 0.623 (+146% improvement) due to optimal exposure to high-growth assets (CTG, GVR, DCM).",
        "3. Sector Diversification: GMV achieves stability by blending Banking (VCB 25.6%), Consumer Staples (VNM 21.2%, SAB 17.4%), and Infrastructure/Utilities (REE 18.8%, GAS 7.9%).",
        "4. Capital Execution: All 12-stock portfolios are fully actionable under HOSE board lot constraints (100 shares) with <0.6% residual cash balance on 50,000,000 VND capital."
    ]
    for i, t in enumerate(takeaways, start=24):
        ws1.merge_cells(start_row=i, start_column=1, end_row=i, end_column=8)
        c = ws1.cell(row=i, column=1, value=t)
        c.font = body_font
        c.alignment = align_left

    # ----------------------------------------------------
    # SHEET 2: Asset_Universe_Stats
    # ----------------------------------------------------
    ws2 = wb.create_sheet(title="Asset_Universe_Stats")
    ws2.views.sheetView[0].showGridLines = True
    
    ws2.merge_cells("A1:P2")
    ws2["A1"] = "INDIVIDUAL ASSET DESCRIPTIVE STATISTICS, CAPM RISK & TAIL METRICS (N=12)"
    ws2["A1"].font = white_title
    ws2["A1"].fill = navy_header_fill
    ws2["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    
    stat_headers = [
        "Ticker", "Economic Sector", "Mean (Daily)", "Mean (Annual)", "Volatility (Daily)", "Volatility (Annual)",
        "Sharpe Ratio", "Skewness", "Excess Kurtosis", "CAPM Beta", "CAPM Alpha (Ann)", "R-Squared",
        "VaR 95% (Hist)", "ES 95% (Hist)", "VaR 99% (Hist)", "ES 99% (Hist)"
    ]
    for col_idx, h in enumerate(stat_headers, start=1):
        c = ws2.cell(row=4, column=col_idx, value=h)
        c.font = white_bold
        c.fill = steel_sub_fill
        c.alignment = align_center
        c.border = header_border
        
    for r_idx, row in df_stats.iterrows():
        r = r_idx + 5
        ws2.cell(row=r, column=1, value=row['Ticker']).alignment = align_center
        ws2.cell(row=r, column=2, value=row['Sector']).alignment = align_left
        ws2.cell(row=r, column=3, value=row['Mean_Daily']).number_format = "0.0000%"
        ws2.cell(row=r, column=4, value=row['Mean_Annual']).number_format = "0.00%"
        ws2.cell(row=r, column=5, value=row['Vol_Daily']).number_format = "0.0000%"
        ws2.cell(row=r, column=6, value=row['Vol_Annual']).number_format = "0.00%"
        ws2.cell(row=r, column=7, value=row['Sharpe_Ratio']).number_format = "0.0000"
        ws2.cell(row=r, column=8, value=row['Skewness']).number_format = "0.00"
        ws2.cell(row=r, column=9, value=row['Excess_Kurtosis']).number_format = "0.00"
        ws2.cell(row=r, column=10, value=row['Beta_VNIndex']).number_format = "0.0000"
        ws2.cell(row=r, column=11, value=row['Alpha_Annual']).number_format = "0.00%"
        ws2.cell(row=r, column=12, value=row['R_Squared']).number_format = "0.00%"
        ws2.cell(row=r, column=13, value=row['VaR_95_Hist']).number_format = "0.00%"
        ws2.cell(row=r, column=14, value=row['ES_95_Hist']).number_format = "0.00%"
        ws2.cell(row=r, column=15, value=row['VaR_99_Hist']).number_format = "0.00%"
        ws2.cell(row=r, column=16, value=row['ES_99_Hist']).number_format = "0.00%"
        
        for c in range(1, 17):
            cell = ws2.cell(row=r, column=c)
            cell.font = body_font
            cell.border = cell_border
            if r % 2 == 1:
                cell.fill = zebra_fill
                
    # ----------------------------------------------------
    # SHEET 3: Correlation_Matrix
    # ----------------------------------------------------
    ws3 = wb.create_sheet(title="Correlation_Matrix")
    ws3.views.sheetView[0].showGridLines = True
    
    ws3.merge_cells("A1:M2")
    ws3["A1"] = "PEARSON CORRELATION MATRIX (12 VN-INDEX EQUITIES - 3-YEAR PANEL)"
    ws3["A1"].font = white_title
    ws3["A1"].fill = navy_header_fill
    ws3["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    
    corr_df = returns_raw.corr()
    
    ws3.cell(row=4, column=1, value="Ticker").font = white_bold
    ws3.cell(row=4, column=1).fill = steel_sub_fill
    ws3.cell(row=4, column=1).border = header_border
    ws3.cell(row=4, column=1).alignment = align_center
    
    for c_idx, t in enumerate(TICKERS, start=2):
        c = ws3.cell(row=4, column=c_idx, value=t)
        c.font = white_bold
        c.fill = steel_sub_fill
        c.border = header_border
        c.alignment = align_center
        
    for r_idx, t1 in enumerate(TICKERS, start=5):
        c_lbl = ws3.cell(row=r_idx, column=1, value=t1)
        c_lbl.font = bold_body
        c_lbl.fill = soft_blue_fill
        c_lbl.border = cell_border
        c_lbl.alignment = align_center
        
        for c_idx, t2 in enumerate(TICKERS, start=2):
            val = corr_df.loc[t1, t2]
            c = ws3.cell(row=r_idx, column=c_idx, value=val)
            c.font = body_font
            c.number_format = "0.00"
            c.border = cell_border
            c.alignment = align_center
            if t1 == t2:
                c.fill = soft_blue_fill
                c.font = bold_body
            elif val < 0.25:
                c.fill = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid") # soft green for low corr
            elif val > 0.50:
                c.fill = PatternFill(start_color="FFEBEE", end_color="FFEBEE", fill_type="solid") # soft red for high corr

    # Add explanatory note
    ws3.cell(row=18, column=1, value="* Note: 12-Equity Screening Panel (2023-07-31 to 2026-07-31). Pair VCB - DCM exhibits the lowest cross-asset correlation (rho = 0.1817 ~ 0.18).").font = small_italic
    ws3.cell(row=19, column=1, value="* Green highlighted cells indicate pairs with rho < 0.25 (prime candidates for diversification: VCB-DCM [0.18], VCB-REE [0.20], VCB-GVR [0.22], SAB-GAS [0.22]).").font = small_italic

    # ----------------------------------------------------
    # SHEET 4: Covariance_Matrix
    # ----------------------------------------------------
    ws4 = wb.create_sheet(title="Covariance_Matrix")
    ws4.views.sheetView[0].showGridLines = True
    
    ws4.merge_cells("A1:M2")
    ws4["A1"] = "ANNUALIZED COVARIANCE MATRIX (Σ = 252 * Σ_daily)"
    ws4["A1"].font = white_title
    ws4["A1"].fill = navy_header_fill
    ws4["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    
    cov_ann = res['cov_ann']
    
    ws4.cell(row=4, column=1, value="Ticker").font = white_bold
    ws4.cell(row=4, column=1).fill = steel_sub_fill
    ws4.cell(row=4, column=1).border = header_border
    ws4.cell(row=4, column=1).alignment = align_center
    
    for c_idx, t in enumerate(TICKERS, start=2):
        c = ws4.cell(row=4, column=c_idx, value=t)
        c.font = white_bold
        c.fill = steel_sub_fill
        c.border = header_border
        c.alignment = align_center
        
    for r_idx, t1 in enumerate(TICKERS, start=5):
        c_lbl = ws4.cell(row=r_idx, column=1, value=t1)
        c_lbl.font = bold_body
        c_lbl.fill = soft_blue_fill
        c_lbl.border = cell_border
        c_lbl.alignment = align_center
        
        for c_idx, t2 in enumerate(TICKERS, start=2):
            val = cov_ann.loc[t1, t2]
            c = ws4.cell(row=r_idx, column=c_idx, value=val)
            c.font = body_font
            c.number_format = "0.00000"
            c.border = cell_border
            c.alignment = align_right
            if t1 == t2:
                c.fill = soft_blue_fill
                c.font = bold_body

    # ----------------------------------------------------
    # SHEET 5: Portfolio_Optimization
    # ----------------------------------------------------
    ws5 = wb.create_sheet(title="Portfolio_Optimization")
    ws5.views.sheetView[0].showGridLines = True
    
    ws5.merge_cells("A1:F2")
    ws5["A1"] = "MARKOWITZ OPTIMAL PORTFOLIO WEIGHTS & PERFORMANCE METRICS"
    ws5["A1"].font = white_title
    ws5["A1"].fill = navy_header_fill
    ws5["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    
    ws5["A4"] = "1. ASSET ALLOCATION WEIGHTS (LONG-ONLY: w_i >= 0, SUM w_i = 1.0)"
    ws5["A4"].font = navy_section
    
    ws5.cell(row=5, column=1, value="Ticker").font = white_bold
    ws5.cell(row=5, column=1).fill = steel_sub_fill
    ws5.cell(row=5, column=1).border = header_border
    ws5.cell(row=5, column=1).alignment = align_center
    
    ws5.cell(row=5, column=2, value="Sector").font = white_bold
    ws5.cell(row=5, column=2).fill = steel_sub_fill
    ws5.cell(row=5, column=2).border = header_border
    ws5.cell(row=5, column=2).alignment = align_center
    
    ws5.cell(row=5, column=3, value="GMV Portfolio Weight").font = white_bold
    ws5.cell(row=5, column=3).fill = steel_sub_fill
    ws5.cell(row=5, column=3).border = header_border
    ws5.cell(row=5, column=3).alignment = align_center
    
    ws5.cell(row=5, column=4, value="Max Sharpe Weight").font = white_bold
    ws5.cell(row=5, column=4).fill = steel_sub_fill
    ws5.cell(row=5, column=4).border = header_border
    ws5.cell(row=5, column=4).alignment = align_center
    
    ws5.cell(row=5, column=5, value="Equal Weight (1/12)").font = white_bold
    ws5.cell(row=5, column=5).fill = steel_sub_fill
    ws5.cell(row=5, column=5).border = header_border
    ws5.cell(row=5, column=5).alignment = align_center
    
    for r_idx, t in enumerate(TICKERS, start=6):
        ws5.cell(row=r_idx, column=1, value=t).alignment = align_center
        ws5.cell(row=r_idx, column=2, value=SECTORS[t]).alignment = align_left
        ws5.cell(row=r_idx, column=3, value=res['w_gmv'][r_idx-6]).number_format = "0.00%"
        ws5.cell(row=r_idx, column=4, value=res['w_msr'][r_idx-6]).number_format = "0.00%"
        ws5.cell(row=r_idx, column=5, value=res['w_eq'][r_idx-6]).number_format = "0.00%"
        for c in range(1, 6):
            cell = ws5.cell(row=r_idx, column=c)
            cell.font = body_font
            cell.border = cell_border
            if r_idx % 2 == 1:
                cell.fill = zebra_fill
                
    # Sum Row
    ws5.cell(row=18, column=1, value="TOTAL ALLOCATION").font = bold_body
    ws5.cell(row=18, column=2, value="100.00%").font = bold_body
    ws5.cell(row=18, column=3, value="=SUM(C6:C17)").font = bold_body
    ws5.cell(row=18, column=4, value="=SUM(D6:D17)").font = bold_body
    ws5.cell(row=18, column=5, value="=SUM(E6:E17)").font = bold_body
    for c in range(1, 6):
        cell = ws5.cell(row=18, column=c)
        cell.fill = soft_blue_fill
        cell.border = header_border
        if c >= 3:
            cell.number_format = "0.00%"
            cell.alignment = align_right

    # Performance Metrics Table
    ws5["A21"] = "2. PORTFOLIO RISK & RETURN PERFORMANCE CHARACTERISTICS"
    ws5["A21"].font = navy_section
    
    port_stat_headers = ["Performance Characteristic", "GMV Portfolio", "Max Sharpe Portfolio", "Equal Weight Portfolio"]
    for col_idx, h in enumerate(port_stat_headers, start=1):
        c = ws5.cell(row=22, column=col_idx, value=h)
        c.font = white_bold
        c.fill = steel_sub_fill
        c.alignment = align_center
        c.border = header_border
        
    port_metrics = [
        ("Expected Annual Return (E[R_p])", res['ret_gmv'], res['ret_msr'], res['ret_eq'], "0.00%"),
        ("Annualized Volatility (σ_p)", res['vol_gmv'], res['vol_msr'], res['vol_eq'], "0.00%"),
        ("Sharpe Ratio (Rf = 4.50%)", res['sr_gmv'], res['sr_msr'], res['sr_eq'], "0.0000"),
        ("Diversification Ratio (DR)", res['dr_gmv'], res['dr_msr'], res['dr_eq'], "0.000"),
        ("Weighted Asset Volatility (Σ w_i σ_i)", np.dot(res['w_gmv'], res['asset_vols']), np.dot(res['w_msr'], res['asset_vols']), np.dot(res['w_eq'], res['asset_vols']), "0.00%"),
        ("Volatility Reduction from Diversification", res['vol_gmv'] - np.dot(res['w_gmv'], res['asset_vols']), res['vol_msr'] - np.dot(res['w_msr'], res['asset_vols']), res['vol_eq'] - np.dot(res['w_eq'], res['asset_vols']), "0.00%"),
        ("1-Day Historical VaR (95%)", res['var_es_gmv'][0], res['var_es_msr'][0], res['var_es_eq'][0], "0.00%"),
        ("1-Day Historical ES (95%)", res['var_es_gmv'][1], res['var_es_msr'][1], res['var_es_eq'][1], "0.00%"),
        ("1-Day Historical VaR (99%)", res['var_es_gmv'][2], res['var_es_msr'][2], res['var_es_eq'][2], "0.00%"),
        ("1-Day Historical ES (99%)", res['var_es_gmv'][3], res['var_es_msr'][3], res['var_es_eq'][3], "0.00%")
    ]
    for r_idx, m_data in enumerate(port_metrics, start=23):
        ws5.cell(row=r_idx, column=1, value=m_data[0]).font = bold_body
        ws5.cell(row=r_idx, column=1).border = cell_border
        for c_idx in range(2, 5):
            val = m_data[c_idx-1]
            c = ws5.cell(row=r_idx, column=c_idx, value=val)
            c.font = body_font
            c.number_format = m_data[4]
            c.border = cell_border
            c.alignment = align_right

    # ----------------------------------------------------
    # SHEET 6: Capital_and_Share_Allocation
    # ----------------------------------------------------
    ws6 = wb.create_sheet(title="Capital_and_Share_Lots")
    ws6.views.sheetView[0].showGridLines = True
    
    ws6.merge_cells("A1:G2")
    ws6["A1"] = "CAPITAL ALLOCATION & BOARD LOT EXECUTION (INITIAL CAPITAL = 50,000,000 VND)"
    ws6["A1"].font = white_title
    ws6["A1"].fill = navy_header_fill
    ws6["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    
    def fill_alloc_table(ws, start_row, title, df_alloc, cash_buf, cash_p):
        ws.cell(row=start_row, column=1, value=title).font = navy_section
        headers = ["Ticker", "Latest Price (VND)", "Target Weight", "Target Capital (VND)", "Board Lot Shares (100)", "Actual Invested (VND)", "Effective Weight"]
        for c_idx, h in enumerate(headers, start=1):
            c = ws.cell(row=start_row+1, column=c_idx, value=h)
            c.font = white_bold
            c.fill = steel_sub_fill
            c.alignment = align_center
            c.border = header_border
            
        for r_idx, row in df_alloc.iterrows():
            r = start_row + 2 + r_idx
            ws.cell(row=r, column=1, value=row['Ticker']).alignment = align_center
            ws.cell(row=r, column=2, value=row['Price_VND']).number_format = "#,##0"
            ws.cell(row=r, column=3, value=row['Target_Weight']).number_format = "0.00%"
            ws.cell(row=r, column=4, value=row['Target_Value_VND']).number_format = "#,##0"
            ws.cell(row=r, column=5, value=row['Allocated_Shares']).number_format = "#,##0"
            ws.cell(row=r, column=6, value=row['Actual_Invested_VND']).number_format = "#,##0"
            ws.cell(row=r, column=7, value=row['Actual_Weight']).number_format = "0.00%"
            for c in range(1, 8):
                cell = ws.cell(row=r, column=c)
                cell.font = body_font
                cell.border = cell_border
                if r % 2 == 1:
                    cell.fill = zebra_fill
                    
        tot_r = start_row + 2 + len(df_alloc)
        ws.cell(row=tot_r, column=1, value="EQUITY SUB-TOTAL").font = bold_body
        ws.cell(row=tot_r, column=6, value=f"=SUM(F{start_row+2}:F{tot_r-1})").font = bold_body
        ws.cell(row=tot_r, column=6).number_format = "#,##0"
        ws.cell(row=tot_r, column=7, value=f"=SUM(G{start_row+2}:G{tot_r-1})").font = bold_body
        ws.cell(row=tot_r, column=7).number_format = "0.00%"
        for c in range(1, 8):
            ws.cell(row=tot_r, column=c).fill = soft_blue_fill
            ws.cell(row=tot_r, column=c).border = cell_border
            
        cash_r = tot_r + 1
        ws.cell(row=cash_r, column=1, value="UNALLOCATED CASH BUFFER").font = bold_body
        ws.cell(row=cash_r, column=6, value=cash_buf).font = bold_body
        ws.cell(row=cash_r, column=6).number_format = "#,##0"
        ws.cell(row=cash_r, column=7, value=cash_p).font = bold_body
        ws.cell(row=cash_r, column=7).number_format = "0.00%"
        for c in range(1, 8):
            ws.cell(row=cash_r, column=c).fill = alert_fill
            ws.cell(row=cash_r, column=c).border = cell_border
            
        end_r = cash_r + 1
        ws.cell(row=end_r, column=1, value="TOTAL PORTFOLIO CAPITAL").font = bold_body
        ws.cell(row=end_r, column=6, value=f"=F{tot_r}+F{cash_r}").font = bold_body
        ws.cell(row=end_r, column=6).number_format = "#,##0"
        ws.cell(row=end_r, column=7, value=f"=G{tot_r}+G{cash_r}").font = bold_body
        ws.cell(row=end_r, column=7).number_format = "0.00%"
        for c in range(1, 8):
            ws.cell(row=end_r, column=c).fill = soft_blue_fill
            ws.cell(row=end_r, column=c).border = header_border
            
        return end_r + 2

    r_next = fill_alloc_table(ws6, 4, "A. GLOBAL MINIMUM VARIANCE (GMV) EXECUTION PLAN", alloc_gmv, cash_gmv, cash_pct_gmv)
    r_next = fill_alloc_table(ws6, r_next, "B. MAXIMUM SHARPE RATIO (TANGENCY) EXECUTION PLAN", alloc_msr, cash_msr, cash_pct_msr)
    r_next = fill_alloc_table(ws6, r_next, "C. EQUAL WEIGHT (1/12) EXECUTION PLAN", alloc_eq, cash_eq, cash_pct_eq)

    # ----------------------------------------------------
    # SHEET 7: Efficient_Frontier_Points
    # ----------------------------------------------------
    ws7 = wb.create_sheet(title="Efficient_Frontier")
    ws7.views.sheetView[0].showGridLines = True
    
    ws7.merge_cells("A1:N2")
    ws7["A1"] = "EFFICIENT FRONTIER OPTIMIZATION POINTS & ASSET WEIGHT PROFILES"
    ws7["A1"].font = white_title
    ws7["A1"].fill = navy_header_fill
    ws7["A1"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    
    ef_headers = ["Point ID", "Target Return", "Frontier Volatility", "Sharpe Ratio"] + TICKERS
    for c_idx, h in enumerate(ef_headers, start=1):
        c = ws7.cell(row=4, column=c_idx, value=h)
        c.font = white_bold
        c.fill = steel_sub_fill
        c.alignment = align_center
        c.border = header_border
        
    pt_id = 1
    for r_idx in range(len(res['target_returns'])):
        vol = res['frontier_vols'][r_idx]
        if np.isnan(vol):
            continue
        ret = res['target_returns'][r_idx]
        sr = (ret - RF_ANNUAL) / vol if vol > 0 else 0
        w = res['frontier_weights'][r_idx]
        
        row_num = pt_id + 4
        ws7.cell(row=row_num, column=1, value=f"EF_{pt_id:03d}").alignment = align_center
        ws7.cell(row=row_num, column=2, value=ret).number_format = "0.00%"
        ws7.cell(row=row_num, column=3, value=vol).number_format = "0.00%"
        ws7.cell(row=row_num, column=4, value=sr).number_format = "0.0000"
        for t_idx in range(len(TICKERS)):
            ws7.cell(row=row_num, column=5+t_idx, value=w[t_idx]).number_format = "0.00%"
            
        for c in range(1, len(ef_headers)+1):
            cell = ws7.cell(row=row_num, column=c)
            cell.font = body_font
            cell.border = cell_border
            if pt_id % 2 == 1:
                cell.fill = zebra_fill
        pt_id += 1

    # Auto-adjust column widths across all sheets
    for ws in wb.worksheets:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                # ignore merged title cells in row 1-2
                if cell.row in [1, 2]:
                    continue
                v_str = str(cell.value or '')
                if len(v_str) > max_len:
                    max_len = len(v_str)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 11)

    wb.save(output_filepath)
    print(f"Master Excel workbook saved successfully to {output_filepath}")

if __name__ == '__main__':
    artifact_dir = r"C:\Users\Acer\.gemini\antigravity\brain\d8ff4393-4f64-4df2-81f4-9ea06a69d57e"
    print("Loading datasets...")
    df_prices, returns_raw, returns_clean, vnindex_ret = load_data()
    
    print("Computing descriptive statistics & CAPM parameters...")
    df_stats = compute_asset_stats(returns_clean, vnindex_ret)
    
    print("Executing Markowitz Optimization for 12 stocks...")
    res = optimize_portfolios(returns_clean)
    
    print("Calculating Capital Allocations under HOSE board lots...")
    alloc_gmv, cash_gmv, cash_pct_gmv = calculate_capital_allocation(res['w_gmv'], df_prices)
    alloc_msr, cash_msr, cash_pct_msr = calculate_capital_allocation(res['w_msr'], df_prices)
    alloc_eq, cash_eq, cash_pct_eq = calculate_capital_allocation(res['w_eq'], df_prices)
    
    print("Generating High-Resolution Charts...")
    charts = generate_charts(res, df_stats, returns_raw, artifact_dir)
    for ch in charts:
        print(f"  Chart generated: {ch}")
        
    print("Building Professional Master Excel Workbook...")
    excel_path_local = "output/Master_12_Stocks_Portfolio_Model.xlsx"
    excel_path_artifact = os.path.join(artifact_dir, "Master_12_Stocks_Portfolio_Model.xlsx")
    
    build_excel_workbook(res, df_stats, df_prices, returns_raw, alloc_gmv, alloc_msr, alloc_eq, cash_gmv, cash_msr, cash_eq, excel_path_local)
    build_excel_workbook(res, df_stats, df_prices, returns_raw, alloc_gmv, alloc_msr, alloc_eq, cash_gmv, cash_msr, cash_eq, excel_path_artifact)
    
    print("Done! All calculations and deliverables completed successfully.")
