"""
Visualization and reporting engine.
Generates all 17 mandatory high-resolution English-labeled charts (without chart numbering in titles
or filenames, no 3D charts, no dual-axis traps, consistent color coding for VCB and DCM) and exports
all tables to CSV and Excel workbooks.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300


def generate_all_charts(
    in_sample_panel: pd.DataFrame,
    simple_returns: pd.DataFrame,
    dist_fit_df: pd.DataFrame,
    daily_cov: pd.DataFrame,
    ann_cov: pd.DataFrame,
    corr_mat: pd.DataFrame,
    grid_df: pd.DataFrame,
    opt_summary_dict: Dict[str, Dict[str, Any]],
    alloc_table_df: pd.DataFrame,
    capm_df: pd.DataFrame,
    capm_diag: Dict[str, Any],
    bt_metrics_df: pd.DataFrame,
    bt_series_dict: Dict[str, pd.DataFrame],
    sim_distributions: Dict[str, np.ndarray],
    rebalance_comp_df: pd.DataFrame,
    figures_dir: Path,
    legacy_dir: Path,
    config: Dict[str, Any],
) -> List[str]:
    """
    Render and save all 17 required figures as high-resolution PNG files without chart numbering.
    """
    cmap = config["COLOR_MAP"]
    saved_files: List[str] = []

    # Clean up any previously numbered PNG files in figures_dir and legacy_dir
    for folder in [figures_dir, legacy_dir]:
        if folder.exists():
            for old_png in folder.glob("*.png"):
                try:
                    old_png.unlink()
                except OSError:
                    pass

    def _save_fig(filename: str):
        out_path = figures_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        shutil.copyfile(out_path, legacy_dir / filename)
        saved_files.append(str(out_path))

    # ==========================================================================
    # Adjusted Closing Prices (Separate Subplots - Avoid Dual-Axis)
    # ==========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True)
    axes[0].plot(in_sample_panel.index, in_sample_panel["VCB"], color=cmap["VCB"], linewidth=1.8, label="VCB Adjusted Close (VND)")
    axes[0].set_title("In-Sample Historical Adjusted Closing Prices (31 Jul 2023 – 31 Jul 2026)", fontsize=13, weight="bold")
    axes[0].set_ylabel("VCB Price (VND)", fontsize=10.5, weight="bold")
    axes[0].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].plot(in_sample_panel.index, in_sample_panel["DCM"], color=cmap["DCM"], linewidth=1.8, label="DCM Adjusted Close (VND)")
    axes[1].set_ylabel("DCM Price (VND)", fontsize=10.5, weight="bold")
    axes[1].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    axes[1].legend(loc="upper left", frameon=True)

    axes[2].plot(in_sample_panel.index, in_sample_panel["VNINDEX"], color=cmap["VNINDEX"], linewidth=1.8, label="VN-Index Close (Points)")
    axes[2].set_xlabel("Trading Date", fontsize=11, weight="bold")
    axes[2].set_ylabel("VN-Index (Points)", fontsize=10.5, weight="bold")
    axes[2].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    axes[2].legend(loc="upper left", frameon=True)
    plt.tight_layout()
    _save_fig("Adjusted_Closing_Prices.png")

    # ==========================================================================
    # Normalized Total Return Performance & Daily Simple Returns
    # ==========================================================================
    norm_df = (in_sample_panel[["VCB", "DCM", "VNINDEX"]] / in_sample_panel[["VCB", "DCM", "VNINDEX"]].iloc[0]) * 100.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), gridspec_kw={"height_ratios": [1.3, 1.0]})

    axes[0].plot(norm_df.index, norm_df["VCB"], color=cmap["VCB"], linewidth=2.0, label="VCB (Base = 100)")
    axes[0].plot(norm_df.index, norm_df["DCM"], color=cmap["DCM"], linewidth=2.0, label="DCM (Base = 100)")
    axes[0].plot(norm_df.index, norm_df["VNINDEX"], color=cmap["VNINDEX"], linestyle="--", linewidth=1.7, label="VN-Index (Base = 100)")
    axes[0].axhline(100.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_title("Normalized Total Return Performance (Base = 100 on 31 Jul 2023)", fontsize=13, weight="bold")
    axes[0].set_ylabel("Normalized Wealth Index", fontsize=10.5, weight="bold")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot(simple_returns.index, simple_returns["VCB"] * 100.0, color=cmap["VCB"], alpha=0.65, linewidth=0.9, label="VCB Daily Simple Return (%)")
    axes[1].plot(simple_returns.index, simple_returns["DCM"] * 100.0, color=cmap["DCM"], alpha=0.55, linewidth=0.9, label="DCM Daily Simple Return (%)")
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_title("In-Sample Daily Simple Returns (%)", fontsize=12.5, weight="bold")
    axes[1].set_xlabel("Trading Date", fontsize=11, weight="bold")
    axes[1].set_ylabel("Daily Return (%)", fontsize=10.5, weight="bold")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend(loc="upper right", frameon=True)
    plt.tight_layout()
    _save_fig("Normalized_Performance.png")

    # ==========================================================================
    # Return Distribution (Histogram + KDE + Normal Fit + Student's t Fit)
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for idx, ticker in enumerate(["VCB", "DCM"]):
        ax = axes[idx]
        r = simple_returns[ticker].dropna().values
        fit_row = dist_fit_df[dist_fit_df["Asset"] == ticker].iloc[0]

        sns.histplot(r, bins=45, stat="density", color=cmap[ticker], alpha=0.30, edgecolor="black", ax=ax, label="Empirical Histogram")
        sns.kdeplot(r, color=cmap[ticker], linewidth=2.0, ax=ax, label="Empirical KDE")

        x_vals = np.linspace(r.min() * 1.1, r.max() * 1.1, 400)
        norm_pdf = stats.norm.pdf(x_vals, loc=fit_row["Normal_Mean"], scale=fit_row["Normal_Std_MLE"])
        t_pdf = stats.t.pdf(x_vals, df=fit_row["StudentT_df"], loc=fit_row["StudentT_Loc"], scale=fit_row["StudentT_Scale"])

        ax.plot(x_vals, norm_pdf, color="crimson", linestyle="--", linewidth=2.0,
                label=f"Normal Fit (BIC={fit_row['Normal_BIC']:.0f})")
        ax.plot(x_vals, t_pdf, color="darkorange", linestyle="-.", linewidth=2.2,
                label=f"Student's t Fit (df={fit_row['StudentT_df']:.2f}, BIC={fit_row['StudentT_BIC']:.0f})")

        ax.set_title(f"Daily Simple Return Distribution – {ticker}", fontsize=12.5, weight="bold")
        ax.set_xlabel("Daily Simple Return (Decimal)", fontsize=10.5, weight="bold")
        ax.set_ylabel("Probability Density", fontsize=10.5, weight="bold")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    _save_fig("Return_Distribution.png")

    # ==========================================================================
    # Normal Q-Q Plots
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for idx, ticker in enumerate(["VCB", "DCM"]):
        ax = axes[idx]
        r = simple_returns[ticker].dropna().values
        (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
        ax.scatter(osm, osr, color=cmap[ticker], alpha=0.6, s=22, label=f"{ticker} Sample Quantiles")
        x_line = np.array([osm.min(), osm.max()])
        ax.plot(x_line, intercept + slope * x_line, color="crimson", linewidth=2.0, label="Theoretical Normal Reference Line")
        ax.set_title(f"Normal Q-Q Plot – {ticker} Daily Simple Returns", fontsize=12.5, weight="bold")
        ax.set_xlabel("Theoretical Normal Quantiles (z-score)", fontsize=10.5, weight="bold")
        ax.set_ylabel("Observed Daily Return Quantiles", fontsize=10.5, weight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    _save_fig("Normal_QQ_Plots.png")

    # ==========================================================================
    # Covariance & Correlation Heatmaps
    # ==========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    sns.heatmap(daily_cov, annot=True, fmt=".6f", cmap="Blues", square=True, linewidths=1.0, ax=axes[0],
                cbar_kws={"label": "Daily Covariance"})
    axes[0].set_title("Daily Covariance Matrix", fontsize=11.5, weight="bold")

    sns.heatmap(ann_cov, annot=True, fmt=".4f", cmap="Greens", square=True, linewidths=1.0, ax=axes[1],
                cbar_kws={"label": "Annualized Covariance (×252)"})
    axes[1].set_title("Annualized Covariance Matrix", fontsize=11.5, weight="bold")

    sns.heatmap(corr_mat, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1.0, vmax=1.0, center=0.0,
                square=True, linewidths=1.0, ax=axes[2], cbar_kws={"label": "Pearson Correlation (ρ)"})
    axes[2].set_title("Pearson Correlation Matrix", fontsize=11.5, weight="bold")
    plt.tight_layout()
    _save_fig("Correlation_Heatmap.png")

    # ==========================================================================
    # Markowitz Efficient Frontier (10,001 Points)
    # ==========================================================================
    eff_branch = grid_df[grid_df["Is_Efficient"]]
    ineff_branch = grid_df[~grid_df["Is_Efficient"]]
    gmv = opt_summary_dict["GMV"]
    port_p = opt_summary_dict["Portfolio_P"]
    max_sr = opt_summary_dict["Max_Sharpe"]
    eq_w = opt_summary_dict["Equal_Weight"]
    risky_u = opt_summary_dict["Risky_Utility"]
    ref_vcb = opt_summary_dict["Ref_100_VCB"]
    ref_dcm = opt_summary_dict["Ref_100_DCM"]

    plt.figure(figsize=(11, 6.8))
    plt.plot(eff_branch["Annualized_Volatility"], eff_branch["Expected_Annual_Return"],
             color="#1f77b4", linewidth=2.8, label="Efficient Frontier (Non-Dominated Branch)")
    plt.plot(ineff_branch["Annualized_Volatility"], ineff_branch["Expected_Annual_Return"],
             color="#aec7e8", linestyle="--", linewidth=2.0, label="Inefficient Feasible Set Branch")

    plt.scatter(gmv["volatility"], gmv["expected_return"], color="gold", marker="D", s=130, edgecolors="black", zorder=6,
                label=f"GMV ({gmv['w_vcb']*100:.2f}% VCB, {gmv['w_dcm']*100:.2f}% DCM)")
    plt.scatter(eq_w["volatility"], eq_w["expected_return"], color=cmap["EQUAL_WEIGHT"], marker="s", s=115, edgecolors="black", zorder=6,
                label="Equal Weight (50.00% VCB, 50.00% DCM)")
    plt.scatter(risky_u["volatility"], risky_u["expected_return"], color="darkorange", marker="^", s=135, edgecolors="black", zorder=6,
                label=f"Max Utility Risky A=2 ({risky_u['w_vcb']*100:.2f}% VCB, {risky_u['w_dcm']*100:.2f}% DCM)")
    plt.scatter(port_p["volatility"], port_p["expected_return"], color=cmap["PORTFOLIO_P"], marker="*", s=250, edgecolors="black", zorder=8,
                label=f"Target Return Portfolio P (12.79%): {port_p['w_vcb']*100:.2f}% VCB, {port_p['w_dcm']*100:.2f}% DCM")
    plt.scatter(max_sr["volatility"], max_sr["expected_return"], color="maroon", marker="P", s=140, edgecolors="black", zorder=7,
                label=f"Constrained Max Sharpe (w_VCB=0.01%, SR={max_sr['sharpe_ratio']:.4f})")

    # Reference boundary points (w=0 and w=1 outside strict open interval)
    plt.scatter(ref_vcb["volatility"], ref_vcb["expected_return"], facecolors="none", edgecolors=cmap["VCB"], s=110, linewidth=2.0, zorder=5,
                label="100% VCB Reference (Outside 0 < w < 1)")
    plt.scatter(ref_dcm["volatility"], ref_dcm["expected_return"], facecolors="none", edgecolors=cmap["DCM"], s=110, linewidth=2.0, zorder=5,
                label="100% DCM Reference (Outside 0 < w < 1)")

    plt.title("Markowitz Feasible Set & Efficient Frontier (10,001 Grid Points, ε = 0.0001)", fontsize=13, weight="bold")
    plt.xlabel("Annualized Volatility (σ, % p.a.)", fontsize=11, weight="bold")
    plt.ylabel("Annualized Expected Arithmetic Return (% p.a.)", fontsize=11, weight="bold")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(loc="lower right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    _save_fig("Markowitz_Efficient_Frontier.png")

    # ==========================================================================
    # Capital Allocation Line (CAL) and Efficient Frontier
    # ==========================================================================
    rf = float(config["RISK_FREE_ANNUAL_RATE"])
    sr_t = max_sr["sharpe_ratio"]
    vol_t = max_sr["volatility"]
    ret_t = max_sr["expected_return"]

    plt.figure(figsize=(11, 6.5))
    plt.plot(eff_branch["Annualized_Volatility"], eff_branch["Expected_Annual_Return"],
             color="#1f77b4", linewidth=2.6, label="Efficient Frontier")
    plt.plot(ineff_branch["Annualized_Volatility"], ineff_branch["Expected_Annual_Return"],
             color="#aec7e8", linestyle="--", linewidth=1.8, label="Inefficient Feasible Set")

    # Feasible CAL segment: 0 <= y <= 1
    sig_feasible = np.linspace(0.0, vol_t, 100)
    plt.plot(sig_feasible, rf + sr_t * sig_feasible, color="crimson", linewidth=2.5,
             label=f"Feasible CAL Segment (0 ≤ y ≤ 1, Slope = {sr_t:.4f})")

    # Unpermitted borrowing segment: y > 1 (dashed for illustration only)
    sig_borrow = np.linspace(vol_t, vol_t * 1.25, 60)
    plt.plot(sig_borrow, rf + sr_t * sig_borrow, color="crimson", linestyle=":", linewidth=1.8,
             label="Hypothetical Borrowing CAL (y > 1, Not Permitted)")

    plt.scatter(0.0, rf, color="forestgreen", marker="s", s=115, edgecolors="black", zorder=6,
                label=f"Risk-Free Asset (Rf = {rf*100:.2f}% p.a.)")
    plt.scatter(vol_t, ret_t, color="maroon", marker="P", s=150, edgecolors="black", zorder=7,
                label="Constrained Max Sharpe Portfolio (Boundary Optimum)")
    plt.scatter(port_p["volatility"], port_p["expected_return"], color=cmap["PORTFOLIO_P"], marker="*", s=220, edgecolors="black", zorder=8,
                label="Portfolio P (Target 12.79% p.a.)")

    plt.title("Capital Allocation Line (CAL) and Markowitz Efficient Frontier", fontsize=13, weight="bold")
    plt.xlabel("Annualized Volatility (σ, % p.a.)", fontsize=11, weight="bold")
    plt.ylabel("Annualized Expected Return (% p.a.)", fontsize=11, weight="bold")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlim(-0.01, vol_t * 1.28)
    plt.legend(loc="lower right", frameon=True, fontsize=8.8)
    plt.tight_layout()
    _save_fig("CAL_and_Efficient_Frontier.png")

    # ==========================================================================
    # Investor Indifference Curves and Optimal Portfolio (A = 2)
    # ==========================================================================
    A = float(config["RISK_AVERSION"])
    comp_u = opt_summary_dict["Complete_Utility"]
    u_complete_val = comp_u["utility"]
    u_risky_val = risky_u["utility"]

    plt.figure(figsize=(11, 6.8))
    plt.plot(eff_branch["Annualized_Volatility"], eff_branch["Expected_Annual_Return"],
             color="#1f77b4", linewidth=2.5, label="Efficient Frontier")
    plt.plot(sig_feasible, rf + sr_t * sig_feasible, color="crimson", linewidth=2.3, label="Feasible CAL (0 ≤ y ≤ 1)")

    sig_ic = np.linspace(0.0, vol_t * 1.15, 250)
    u_curves = [
        (u_risky_val, f"Risky-Only Utility Curve (U_risky = {u_risky_val*100:.2f}%)", "darkorange", "--"),
        (u_complete_val, f"Optimal Complete Utility Curve (U* = {u_complete_val*100:.2f}%)", "forestgreen", "-"),
        (u_complete_val + 0.03, f"Higher Unattainable Utility Curve (U_high = {(u_complete_val+0.03)*100:.2f}%)", "purple", ":"),
    ]
    for u_lvl, u_lbl, u_col, u_ls in u_curves:
        plt.plot(sig_ic, u_lvl + 0.5 * A * (sig_ic ** 2), color=u_col, linestyle=u_ls, linewidth=1.9, label=u_lbl)

    plt.scatter(risky_u["volatility"], risky_u["expected_return"], color="darkorange", marker="^", s=150, edgecolors="black", zorder=7,
                label=f"Optimal Risky-Only Portfolio (σ={risky_u['volatility']*100:.2f}%, E[R]={risky_u['expected_return']*100:.2f}%)")
    plt.scatter(comp_u["volatility"], comp_u["expected_return"], color="forestgreen", marker="o", s=160, edgecolors="black", zorder=8,
                label=f"Optimal Complete Portfolio (y*={comp_u['risky_sleeve_y']*100:.2f}%, Rf={comp_u['w_rf_total']*100:.2f}%)")
    plt.scatter(vol_t, ret_t, color="maroon", marker="P", s=140, edgecolors="black", zorder=7,
                label="Constrained Max Sharpe Sleeve")

    plt.title(f"Investor Indifference Curves (A = {A:.1f}) and Optimal Portfolios", fontsize=13, weight="bold")
    plt.xlabel("Annualized Volatility (σ, % p.a.)", fontsize=11, weight="bold")
    plt.ylabel("Annualized Expected Return (% p.a.)", fontsize=11, weight="bold")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlim(-0.01, vol_t * 1.18)
    plt.ylim(0.0, 0.22)
    plt.legend(loc="lower right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    _save_fig("Indifference_Curves_and_Optimal_Portfolio.png")

    # ==========================================================================
    # Asset Allocation by Portfolio (Stacked Bar Chart)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(alloc_table_df))
    w_v = alloc_table_df["VCB Total Weight"].values * 100.0
    w_d = alloc_table_df["DCM Total Weight"].values * 100.0
    w_f = alloc_table_df["Risk-Free Weight"].values * 100.0

    ax.bar(x, w_v, width=0.55, color=cmap["VCB"], edgecolor="black", label="VCB Weight (%)")
    ax.bar(x, w_d, width=0.55, bottom=w_v, color=cmap["DCM"], edgecolor="black", label="DCM Weight (%)")
    ax.bar(x, w_f, width=0.55, bottom=w_v + w_d, color="#ffbb78", edgecolor="black", label="Risk-Free Asset (%)")

    for i in range(len(x)):
        if w_v[i] > 4.0:
            ax.text(x[i], w_v[i] / 2.0, f"{w_v[i]:.1f}%", ha="center", va="center", color="white", weight="bold", fontsize=9)
        if w_d[i] > 4.0:
            ax.text(x[i], w_v[i] + w_d[i] / 2.0, f"{w_d[i]:.1f}%", ha="center", va="center", color="white", weight="bold", fontsize=9)
        if w_f[i] > 4.0:
            ax.text(x[i], w_v[i] + w_d[i] + w_f[i] / 2.0, f"{w_f[i]:.1f}%", ha="center", va="center", color="black", weight="bold", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(alloc_table_df["Portfolio Name"], rotation=15, ha="right", fontsize=9.5, weight="bold")
    ax.set_ylabel("Total Portfolio Allocation (%)", fontsize=11, weight="bold")
    ax.set_ylim(0, 112)
    ax.set_title("Asset Allocation Comparison Across Model Portfolios", fontsize=13, weight="bold")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    _save_fig("Asset_Allocation_by_Portfolio.png")

    # ==========================================================================
    # Security Market Line (SML)
    # ==========================================================================
    est_mkt_ret = capm_diag["estimated_annual_market_return"]
    mrp = capm_diag["market_risk_premium"]
    beta_grid = np.linspace(0.0, 1.35, 200)
    sml_vals = rf + beta_grid * mrp

    plt.figure(figsize=(10.8, 6.4))
    plt.plot(beta_grid, sml_vals, color="crimson", linewidth=2.4, label="Security Market Line (SML: E[R] = Rf + β·MRP)")
    plt.scatter(0.0, rf, color="forestgreen", marker="s", s=115, edgecolors="black", zorder=6,
                label=f"Risk-Free Asset (β = 0.00, Rf = {rf*100:.2f}% p.a.)")
    plt.scatter(1.0, est_mkt_ret, color=cmap["VNINDEX"], marker="D", s=125, edgecolors="black", zorder=6,
                label=f"VN-Index Market Proxy (β = 1.00, E[Rm] = {est_mkt_ret*100:.2f}% p.a.)")

    for _, row in capm_df.iterrows():
        tk = row["Asset"]
        b_val = row["Beta"]
        capm_r = row["CAPM_Implied_Annual_Return"]
        hist_r = row["Historical_Annualized_Mean_Return"]
        gap_r = row["Historical_CAPM_Gap"]

        plt.scatter(b_val, capm_r, facecolors="none", edgecolors=cmap[tk], s=115, linewidth=2.2, zorder=6,
                    label=f"{tk} CAPM-Implied Return ({capm_r*100:.2f}% p.a.)")
        plt.scatter(b_val, hist_r, color=cmap[tk], s=135, edgecolors="black", zorder=7,
                    label=f"{tk} Observed Historical Mean ({hist_r*100:.2f}% p.a.)")
        plt.plot([b_val, b_val], [capm_r, hist_r], color=cmap[tk], linestyle=":", linewidth=2.0,
                 label=f"{tk} Jensen's Alpha α ({gap_r*100:+.2f}% p.a.)")

    # Add visual callout annotations for Jensen's Alpha
    vcb_row = capm_df[capm_df["Asset"] == "VCB"].iloc[0]
    dcm_row = capm_df[capm_df["Asset"] == "DCM"].iloc[0]

    plt.annotate(
        f"VCB Jensen's α = {vcb_row['Historical_CAPM_Gap']*100:+.2f}% p.a.\n(β = {vcb_row['Beta']:.4f}, Underperformed CAPM)",
        xy=(vcb_row["Beta"], (vcb_row["CAPM_Implied_Annual_Return"] + vcb_row["Historical_Annualized_Mean_Return"]) / 2),
        xytext=(vcb_row["Beta"] - 0.38, 0.040),
        arrowprops=dict(arrowstyle="->", color=cmap["VCB"], lw=1.6),
        fontsize=9, weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor=cmap["VCB"], alpha=0.92),
    )

    plt.annotate(
        f"DCM Jensen's α = {dcm_row['Historical_CAPM_Gap']*100:+.2f}% p.a.\n(β = {dcm_row['Beta']:.4f}, In Line with CAPM)",
        xy=(dcm_row["Beta"], (dcm_row["CAPM_Implied_Annual_Return"] + dcm_row["Historical_Annualized_Mean_Return"]) / 2),
        xytext=(dcm_row["Beta"] - 0.30, 0.165),
        arrowprops=dict(arrowstyle="->", color=cmap["DCM"], lw=1.6),
        fontsize=9, weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor=cmap["DCM"], alpha=0.92),
    )

    plt.title("Security Market Line (SML) and Jensen's Alpha (α)", fontsize=13, weight="bold")
    plt.xlabel("Systematic Risk (Beta β vs. VN-Index)", fontsize=11, weight="bold")
    plt.ylabel("Annualized Return (% p.a.)", fontsize=11, weight="bold")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlim(-0.05, 1.30)
    plt.legend(loc="lower right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    _save_fig("SML.png")

    # ==========================================================================
    # Beta Regression Diagnostics for VCB and DCM
    # ==========================================================================
    for ticker in ["VCB", "DCM"]:
        st_diag = capm_diag["stocks"][ticker]
        row_c = capm_df[capm_df["Asset"] == ticker].iloc[0]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
        # Panel A: Scatter + OLS Line + 95% Confidence Band
        ax0 = axes[0, 0]
        ax0.scatter(st_diag["market_excess"], st_diag["stock_excess"], color=cmap[ticker], alpha=0.45, s=22, label="Daily Excess Returns")
        
        ols_fit_label = (
            f"OLS Fit: Ri - rf = α + β(Rm - rf)\n"
            f"  • Daily α = {row_c['Daily_Regression_Alpha']*100:+.4f}% (HAC p = {row_c['Alpha_p_value']:.3f})\n"
            f"  • Annualized α = {row_c['Annualized_Regression_Alpha']*100:+.2f}% p.a.\n"
            f"  • Systematic β = {row_c['Beta']:.4f} (HAC SE = {row_c['Beta_HAC_SE']:.4f})\n"
            f"  • R² = {row_c['R_squared']*100:.2f}% (Adj R² = {row_c['Adjusted_R_squared']*100:.2f}%)"
        )
        ax0.plot(st_diag["x_grid"], st_diag["y_hat"], color="crimson", linewidth=2.2, label=ols_fit_label)
        ax0.fill_between(st_diag["x_grid"], st_diag["ci_lower"], st_diag["ci_upper"], color="crimson", alpha=0.22,
                         label="95% Mean Confidence Band")
        ax0.set_title(f"{ticker} Excess Return vs. Market (Alpha α = {row_c['Annualized_Regression_Alpha']*100:+.2f}% p.a., Beta β = {row_c['Beta']:.4f})", fontsize=11.5, weight="bold")
        ax0.set_xlabel("VN-Index Daily Excess Return (Rm - rf_daily)", fontsize=10.5, weight="bold")
        ax0.set_ylabel(f"{ticker} Daily Excess Return (Ri - rf_daily)", fontsize=10.5, weight="bold")
        ax0.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax0.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax0.legend(loc="upper left", frameon=True, fontsize=8.6)

        # Panel B: Residual Time Series
        ax1 = axes[0, 1]
        ax1.plot(st_diag["residuals"].index, st_diag["residuals"], color=cmap[ticker], linewidth=0.9, label="OLS Residuals ε_t")
        ax1.axhline(0.0, color="crimson", linestyle="--", linewidth=1.2)
        ax1.set_title(f"{ticker} Regression Residual Time Series", fontsize=12, weight="bold")
        ax1.set_xlabel("Trading Date", fontsize=10.5, weight="bold")
        ax1.set_ylabel("Daily Residual (Decimal)", fontsize=10.5, weight="bold")
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax1.legend(loc="upper right", frameon=True)

        # Panel C: Residual Distribution
        ax2 = axes[1, 0]
        res_vals = st_diag["residuals"].values
        sns.histplot(res_vals, bins=40, stat="density", color=cmap[ticker], alpha=0.35, edgecolor="black", ax=ax2, label="Residual Histogram")
        sns.kdeplot(res_vals, color=cmap[ticker], linewidth=2.0, ax=ax2, label="Residual KDE")
        rx = np.linspace(res_vals.min(), res_vals.max(), 200)
        ax2.plot(rx, stats.norm.pdf(rx, np.mean(res_vals), np.std(res_vals, ddof=1)), color="crimson", linestyle="--", linewidth=1.8, label="Normal Reference")
        ax2.set_title(f"{ticker} Regression Residual Distribution", fontsize=12, weight="bold")
        ax2.set_xlabel("Residual Value", fontsize=10.5, weight="bold")
        ax2.set_ylabel("Density", fontsize=10.5, weight="bold")
        ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2.legend(loc="upper right", frameon=True)

        # Panel D: Valid Rolling 126-Session & 252-Session Beta
        ax3 = axes[1, 1]
        rb126 = st_diag["rolling_beta_126"]
        rb252 = st_diag["rolling_beta_252"]
        ax3.plot(rb126.index, rb126.values, color="#ff7f0e", linewidth=1.6, label="Rolling 126-Session Beta")
        ax3.plot(rb252.index, rb252.values, color=cmap[ticker], linewidth=2.2, label="Rolling 252-Session Beta")
        ax3.axhline(row_c["Beta"], color="crimson", linestyle="--", linewidth=1.6, label=f"Full In-Sample Beta ({row_c['Beta']:.4f})")
        ax3.set_title(f"{ticker} Rolling Beta Stability (Valid Windows Only)", fontsize=12, weight="bold")
        ax3.set_xlabel("Trading Date", fontsize=10.5, weight="bold")
        ax3.set_ylabel("Rolling Beta (β)", fontsize=10.5, weight="bold")
        ax3.legend(loc="best", frameon=True, fontsize=8.8)

        plt.tight_layout()
        _save_fig(f"Beta_Regression_{ticker}.png")

    # ==========================================================================
    # Out-of-Sample NAV and Cumulative Return
    # ==========================================================================
    theo_p = bt_series_dict["theo_p"]
    theo_eq = bt_series_dict["theo_eq"]
    prac_p = bt_series_dict["prac_p"]
    dep_nav = bt_series_dict["deposit_nav"]
    init_cap = float(config["INITIAL_CAPITAL_VND"])

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True)
    axes[0].plot(theo_p.index, theo_p["Portfolio_NAV_VND"], color=cmap["PORTFOLIO_P"], linewidth=2.3,
                 label=f"Portfolio P – Theoretical Buy-and-Hold ({theo_p['Portfolio_NAV_VND'].iloc[-1]:,.0f} VND)")
    axes[0].plot(prac_p.index, prac_p["Portfolio_NAV_VND"], color="darkorange", linestyle="-.", linewidth=2.0,
                 label=f"Portfolio P – Practical Lot Execution MTM ({prac_p['Portfolio_NAV_VND'].iloc[-1]:,.0f} VND)")
    axes[0].plot(theo_eq.index, theo_eq["Portfolio_NAV_VND"], color=cmap["EQUAL_WEIGHT"], linewidth=2.1,
                 label=f"Equal Weight 50/50 – Theoretical ({theo_eq['Portfolio_NAV_VND'].iloc[-1]:,.0f} VND)")
    axes[0].plot(dep_nav.index, dep_nav.values, color=cmap["DEPOSIT"], linestyle="--", linewidth=2.0,
                 label=f"Bank Deposit Benchmark 7.0% p.a. ({dep_nav.iloc[-1]:,.0f} VND)")
    axes[0].axhline(init_cap, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_title("Out-of-Sample Portfolio NAV Evolution (03 Aug 2026 – 30 Oct 2026)", fontsize=13, weight="bold")
    axes[0].set_ylabel("Portfolio Value (VND)", fontsize=10.5, weight="bold")
    axes[0].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    axes[0].legend(loc="upper left", frameon=True, fontsize=8.8)

    axes[1].plot(theo_p.index, theo_p["Cumulative_Return"], color=cmap["PORTFOLIO_P"], linewidth=2.2, label="Portfolio P (Theoretical)")
    axes[1].plot(prac_p.index, prac_p["Cumulative_Return"], color="darkorange", linestyle="-.", linewidth=1.9, label="Portfolio P (Practical MTM)")
    axes[1].plot(theo_eq.index, theo_eq["Cumulative_Return"], color=cmap["EQUAL_WEIGHT"], linewidth=2.0, label="Equal Weight 50/50")
    axes[1].plot(dep_nav.index, (dep_nav / init_cap) - 1.0, color=cmap["DEPOSIT"], linestyle="--", linewidth=1.9, label="Bank Deposit 7.0% p.a.")
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_title("Out-of-Sample Cumulative Holding Period Return (%)", fontsize=12.5, weight="bold")
    axes[1].set_xlabel("Trading Date", fontsize=11, weight="bold")
    axes[1].set_ylabel("Cumulative Return (%)", fontsize=10.5, weight="bold")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].legend(loc="upper left", frameon=True, fontsize=8.8)
    plt.tight_layout()
    _save_fig("Out_of_Sample_NAV.png")

    # ==========================================================================
    # Out-of-Sample Drawdown
    # ==========================================================================
    plt.figure(figsize=(11, 5.5))
    plt.plot(theo_p.index, theo_p["Drawdown"], color=cmap["PORTFOLIO_P"], linewidth=2.2,
             label=f"Portfolio P (Theoretical Max DD: {theo_p['Drawdown'].min()*100:.2f}%)")
    plt.fill_between(theo_p.index, theo_p["Drawdown"], 0.0, color=cmap["PORTFOLIO_P"], alpha=0.16)

    plt.plot(theo_eq.index, theo_eq["Drawdown"], color=cmap["EQUAL_WEIGHT"], linewidth=2.0,
             label=f"Equal Weight 50/50 (Max DD: {theo_eq['Drawdown'].min()*100:.2f}%)")
    plt.fill_between(theo_eq.index, theo_eq["Drawdown"], 0.0, color=cmap["EQUAL_WEIGHT"], alpha=0.16)

    plt.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    plt.title("Out-of-Sample Running Percentage Drawdown (03 Aug 2026 – 30 Oct 2026)", fontsize=13, weight="bold")
    plt.xlabel("Trading Date", fontsize=11, weight="bold")
    plt.ylabel("Drawdown from Running Peak (%)", fontsize=11, weight="bold")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(loc="lower left", frameon=True, fontsize=9.5)
    plt.tight_layout()
    _save_fig("Out_of_Sample_Drawdown.png")

    # ==========================================================================
    # Buy-and-Hold Weight Drift
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    axes[0].plot(theo_p.index, theo_p["DCM_Weight"], color=cmap["DCM"], linewidth=2.2, label="DCM Drifted Weight")
    axes[0].plot(theo_p.index, theo_p["VCB_Weight"], color=cmap["VCB"], linewidth=2.2, label="VCB Drifted Weight")
    axes[0].axhline(theo_p["DCM_Weight"].iloc[0], color=cmap["DCM"], linestyle=":", label=f"Initial DCM Weight ({theo_p['DCM_Weight'].iloc[0]*100:.2f}%)")
    axes[0].axhline(theo_p["VCB_Weight"].iloc[0], color=cmap["VCB"], linestyle=":", label=f"Initial VCB Weight ({theo_p['VCB_Weight'].iloc[0]*100:.2f}%)")
    axes[0].set_title("Buy-and-Hold Weight Drift – Portfolio P", fontsize=12, weight="bold")
    axes[0].set_xlabel("Trading Date", fontsize=10.5, weight="bold")
    axes[0].set_ylabel("Portfolio Weight (%)", fontsize=10.5, weight="bold")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].legend(loc="center right", frameon=True, fontsize=8.5)

    axes[1].plot(theo_eq.index, theo_eq["DCM_Weight"], color=cmap["DCM"], linewidth=2.2, label="DCM Drifted Weight")
    axes[1].plot(theo_eq.index, theo_eq["VCB_Weight"], color=cmap["VCB"], linewidth=2.2, label="VCB Drifted Weight")
    axes[1].axhline(0.50, color="black", linestyle=":", label="Initial Equal Weight (50.00%)")
    axes[1].set_title("Buy-and-Hold Weight Drift – Equal Weight Portfolio", fontsize=12, weight="bold")
    axes[1].set_xlabel("Trading Date", fontsize=10.5, weight="bold")
    axes[1].set_ylabel("Portfolio Weight (%)", fontsize=10.5, weight="bold")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].legend(loc="center right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    _save_fig("Weight_Drift.png")

    # ==========================================================================
    # Terminal Wealth Comparison & PnL vs. Bank Deposit
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    strat_labels = [
        "Portfolio P\n(Theoretical)",
        "Portfolio P\n(Practical MTM)",
        "Portfolio P\n(Net Liquidation)",
        "Equal Weight\n(Theoretical)",
        "Equal Weight\n(Practical MTM)",
        "Bank Deposit\n(7.0% ACT/365)",
    ]
    p_prac_row = bt_metrics_df[bt_metrics_df["Strategy"] == "Portfolio P (Practical Lot-Size Execution)"].iloc[0]
    eq_prac_row = bt_metrics_df[bt_metrics_df["Strategy"] == "Equal Weight (Practical Lot-Size Execution)"].iloc[0]
    dep_row = bt_metrics_df[bt_metrics_df["Strategy"] == "Bank Deposit Benchmark (7.0% p.a. ACT/365)"].iloc[0]

    term_vals = [
        float(theo_p["Portfolio_NAV_VND"].iloc[-1]),
        float(p_prac_row["Ending Mark-to-Market NAV"]),
        float(p_prac_row["Net Liquidation Value"]),
        float(theo_eq["Portfolio_NAV_VND"].iloc[-1]),
        float(eq_prac_row["Ending Mark-to-Market NAV"]),
        float(dep_row["Ending Mark-to-Market NAV"]),
    ]
    bar_cols = [cmap["PORTFOLIO_P"], "darkorange", "coral", cmap["EQUAL_WEIGHT"], "#b592d6", cmap["DEPOSIT"]]

    b0 = axes[0].bar(strat_labels, term_vals, color=bar_cols, edgecolor="black", width=0.55)
    axes[0].axhline(init_cap, color="black", linestyle=":", linewidth=1.2, label="Initial Capital (50,000,000 VND)")
    axes[0].set_ylim(47_000_000, max(term_vals) * 1.04)
    axes[0].set_title("Out-of-Sample Terminal Wealth Comparison (VND)", fontsize=12, weight="bold")
    axes[0].set_ylabel("Ending Portfolio Value (VND)", fontsize=10.5, weight="bold")
    axes[0].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    for bar, v in zip(b0, term_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, v + 180_000, f"{v:,.0f}", ha="center", va="bottom", fontsize=8.2, weight="bold")
    axes[0].legend(loc="lower right", frameon=True)

    # Panel B: Excess Profit/Loss vs. 7.0% Bank Deposit
    dep_end = float(dep_row["Ending Mark-to-Market NAV"])
    excess_pnl = [v - dep_end for v in term_vals[:-1]]
    b1 = axes[1].bar(strat_labels[:-1], excess_pnl, color=bar_cols[:-1], edgecolor="black", width=0.55)
    axes[1].axhline(0.0, color="black", linestyle="-", linewidth=1.2)
    axes[1].set_title("Excess Profit / Loss versus 7.0% p.a. Bank Deposit (VND)", fontsize=12, weight="bold")
    axes[1].set_ylabel("Excess Profit / Loss vs. Deposit (VND)", fontsize=10.5, weight="bold")
    axes[1].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    for bar, v in zip(b1, excess_pnl):
        offset = 80_000 if v >= 0 else -180_000
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, v + offset, f"{v:+,.0f}", ha="center", va="bottom", fontsize=8.5, weight="bold")

    plt.tight_layout()
    _save_fig("Terminal_Wealth_Comparison.png")

    # ==========================================================================
    # Before and After Rebalancing Allocation (As of 30 Oct 2026)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10.5, 6))
    reb_labels = [
        "Hold Current Drifted\n(30 Oct 2026 Close)",
        "Rebalance to Original\nPortfolio P (31 Jul 2026)",
        "Re-optimize Updated\nPortfolio P (12.79% Target)",
        "Re-optimize Updated\nRisky Utility (A = 2.0)",
        "Re-optimize Updated\nComplete Portfolio (A = 2.0)",
    ]
    rv_vcb = rebalance_comp_df["Proposed_VCB_Weight"].values * 100.0
    rv_dcm = rebalance_comp_df["Proposed_DCM_Weight"].values * 100.0
    rv_rf = rebalance_comp_df["Proposed_Cash_Rf_Weight"].values * 100.0

    xr = np.arange(len(reb_labels))
    ax.bar(xr, rv_vcb, width=0.52, color=cmap["VCB"], edgecolor="black", label="VCB Weight (%)")
    ax.bar(xr, rv_dcm, width=0.52, bottom=rv_vcb, color=cmap["DCM"], edgecolor="black", label="DCM Weight (%)")
    ax.bar(xr, rv_rf, width=0.52, bottom=rv_vcb + rv_dcm, color="#ffbb78", edgecolor="black", label="Cash / Risk-Free Weight (%)")

    for i in range(len(xr)):
        if rv_vcb[i] > 4.0:
            ax.text(xr[i], rv_vcb[i] / 2.0, f"{rv_vcb[i]:.2f}%", ha="center", va="center", color="white", weight="bold", fontsize=9)
        if rv_dcm[i] > 4.0:
            ax.text(xr[i], rv_vcb[i] + rv_dcm[i] / 2.0, f"{rv_dcm[i]:.2f}%", ha="center", va="center", color="white", weight="bold", fontsize=9)
        if rv_rf[i] > 4.0:
            ax.text(xr[i], rv_vcb[i] + rv_dcm[i] + rv_rf[i] / 2.0, f"{rv_rf[i]:.2f}%", ha="center", va="center", color="black", weight="bold", fontsize=9)

    ax.set_xticks(xr)
    ax.set_xticklabels(reb_labels, fontsize=9.2, weight="bold")
    ax.set_ylabel("Allocation Percentage (%)", fontsize=11, weight="bold")
    ax.set_ylim(0, 112)
    ax.set_title("Before and After Rebalancing Allocation Comparison (Post-Close 30 Oct 2026)", fontsize=13, weight="bold")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    _save_fig("Before_and_After_Rebalancing_Allocation.png")

    return saved_files


def _render_styled_table_png(
    df_display: pd.DataFrame,
    title: str,
    subtitle: str,
    out_path: Path,
    legacy_path: Path,
    col_widths: List[float] = None,
    fig_size: tuple = (15, 5.5),
    highlight_rows: List[int] = None,
) -> str:
    """Render a clean, publication-grade summary table as a high-resolution PNG image."""
    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis("off")

    fig.suptitle(title, fontsize=13.5, weight="bold", y=0.97, color="#1a252f")
    if subtitle:
        ax.set_title(subtitle, fontsize=9.8, style="italic", color="#4a5568", pad=10)

    tbl = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)
    tbl.scale(1.0, 1.85)

    n_rows, n_cols = df_display.shape
    highlight_set = set(highlight_rows or [])

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#1e3a5f")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(8.8)
        else:
            data_r = row_idx - 1
            if col_idx == 0:
                cell.get_text().set_weight("bold")
            if data_r in highlight_set:
                cell.set_facecolor("#fef3c7")
                cell.get_text().set_weight("bold")
            elif data_r % 2 == 1:
                cell.set_facecolor("#f8fafc")
            else:
                cell.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    shutil.copyfile(out_path, legacy_path)
    return str(out_path)


def generate_summary_table_pngs(
    cap_shares_df: pd.DataFrame,
    rebalance_comp_df: pd.DataFrame,
    proposed_trades_df: pd.DataFrame,
    figures_dir: Path,
    legacy_dir: Path,
) -> List[str]:
    """
    Generate clean, simple PNG table images for:
      1. Capital_and_Share_Allocation_Table.png
      2. Post_OOS_Rebalancing_Summary_Table.png
      3. Post_OOS_Trade_Execution_Table.png
    """
    saved_pngs: List[str] = []

    # 1. Capital & Share Allocation across Portfolio Points
    df_cap_disp = pd.DataFrame(
        {
            "Portfolio Point": [
                "100% VCB Reference",
                "Global Minimum Variance (GMV)",
                "Equal Weight (50/50)",
                "Max Utility Risky (A = 2.0)",
                "Target Return Portfolio P (12.79%)",
                "Constrained Max Sharpe",
                "100% DCM Reference",
                "Optimal Complete Portfolio (A = 2.0)",
            ],
            "Target Weights\n(VCB / DCM / Rf)": [
                f"{r['Target Weight VCB']*100:.2f}% / {r['Target Weight DCM']*100:.2f}% / {r['Target Weight Risk-Free']*100:.2f}%"
                for _, r in cap_shares_df.iterrows()
            ],
            "Budget VCB\n(VND)": [f"{r['Target Budget VCB (VND)']:,.0f}" for _, r in cap_shares_df.iterrows()],
            "Budget DCM\n(VND)": [f"{r['Target Budget DCM (VND)']:,.0f}" for _, r in cap_shares_df.iterrows()],
            "Theoretical Shares\n(VCB / DCM)": [
                f"{r['Theoretical Exact Shares VCB']:.2f} / {r['Theoretical Exact Shares DCM']:.2f}"
                for _, r in cap_shares_df.iterrows()
            ],
            "Practical Lots\n(VCB / DCM Shares)": [
                f"{int(r['Practical Shares VCB']):,d} / {int(r['Practical Shares DCM']):,d}"
                for _, r in cap_shares_df.iterrows()
            ],
            "Stock Outlay\n+ 0.15% Fee (VND)": [
                f"{r['Total Cash Outlay for Stocks (VND)']:,.0f}" for _, r in cap_shares_df.iterrows()
            ],
            "Rf Deposit +\nResidual Cash (VND)": [
                f"{(r['Risk-Free Deposit Allocated (VND)'] + r['Uninvested Residual Cash (VND)']):,.0f}"
                for _, r in cap_shares_df.iterrows()
            ],
            "OOS Ending MTM\n(30 Oct 2026, VND)": [
                f"{r['OOS Ending MTM NAV (30 Oct 2026)']:,.0f}" for _, r in cap_shares_df.iterrows()
            ],
            "OOS MTM Profit\n(VND & %)": [
                f"{r['OOS MTM Profit/Loss (VND)']:+,.0f} ({r['OOS MTM Holding Return']*100:+.2f}%)"
                for _, r in cap_shares_df.iterrows()
            ],
        }
    )

    p1 = _render_styled_table_png(
        df_display=df_cap_disp,
        title="Capital Outlay & Share Allocation Across Portfolio Points (Initial Capital = 50,000,000 VND)",
        subtitle="Entry Date: 03 Aug 2026 (Entry Raw Prices: VCB = 60,800 VND/sh, DCM = 31,050 VND/sh | HOSE 100-Share Board Lots | Buy Fee = 0.15%)",
        out_path=figures_dir / "Capital_and_Share_Allocation_Table.png",
        legacy_path=legacy_dir / "Capital_and_Share_Allocation_Table.png",
        col_widths=[0.17, 0.13, 0.08, 0.08, 0.10, 0.10, 0.09, 0.10, 0.10, 0.11],
        fig_size=(18.5, 5.8),
        highlight_rows=[4, 7],
    )
    saved_pngs.append(p1)

    # 2. Post-OOS Rebalancing Decision Comparison Table
    prac_trades_map = {
        0: "Hold (0 VCB, 1,400 DCM)",
        1: "Buy +100 VCB (from Cash)",
        2: "Buy +300 VCB, Sell -400 DCM",
        3: "Buy +100 VCB, Sell -100 DCM",
        4: "Sell -500 DCM -> Risk-Free",
    }
    prac_cost_map = {0: 0, 1: 8445, 2: 58635, 3: 16770, 4: 41625}

    df_reb_disp = pd.DataFrame(
        {
            "Rebalancing Option (30 Oct 2026 Close)": [
                "Option 1: Hold Current Drifted Portfolio",
                "Option 2: Rebalance to Original Target P",
                "Option 3A: Re-optimize Target Return (12.79%)",
                "Option 3B: Re-optimize Risky Utility (A = 2.0)",
                "Option 3C: Re-optimize Complete Portfolio (A = 2.0)",
            ],
            "Proposed Weights\n(VCB / DCM / Rf)": [
                f"{r['Proposed_VCB_Weight']*100:.2f}% / {r['Proposed_DCM_Weight']*100:.2f}% / {r['Proposed_Cash_Rf_Weight']*100:.2f}%"
                for _, r in rebalance_comp_df.iterrows()
            ],
            "Updated E[R]\n(% p.a.)": [
                f"{r['Updated_Expected_Annual_Return']*100:.2f}%" for _, r in rebalance_comp_df.iterrows()
            ],
            "Updated Vol\n(% p.a.)": [
                f"{r['Updated_Expected_Volatility']*100:.2f}%" for _, r in rebalance_comp_df.iterrows()
            ],
            "Updated\nSharpe": [
                f"{r['Updated_Optimization_Sharpe']:.4f}" for _, r in rebalance_comp_df.iterrows()
            ],
            "One-Way\nTurnover": [
                f"{r['One_Way_Turnover_Pct']*100:.2f}%" for _, r in rebalance_comp_df.iterrows()
            ],
            "Practical HOSE\nLot Trade Action": [prac_trades_map[i] for i in range(len(rebalance_comp_df))],
            "Rebalance Cost\n(Theo / Practical VND)": [
                f"{r['Estimated_Transaction_Costs_VND']:,.0f} / {prac_cost_map[i]:,.0f}"
                for i, (_, r) in enumerate(rebalance_comp_df.iterrows())
            ],
            "Net Utility (A=2)\nAfter Cost": [
                f"{r['Net_First_Year_Utility_After_Costs']*100:.4f}%" for _, r in rebalance_comp_df.iterrows()
            ],
        }
    )

    p2 = _render_styled_table_png(
        df_display=df_reb_disp,
        title="Post-Out-of-Sample Rebalancing Decision Summary (As of 30 Oct 2026 Close)",
        subtitle="Updated 3-Year Rolling Window (30 Oct 2023 – 30 Oct 2026): E[R_VCB] = -0.36%, E[R_DCM] = +18.81%, Vol_VCB = 22.90%, Vol_DCM = 34.49%",
        out_path=figures_dir / "Post_OOS_Rebalancing_Summary_Table.png",
        legacy_path=legacy_dir / "Post_OOS_Rebalancing_Summary_Table.png",
        col_widths=[0.23, 0.13, 0.08, 0.08, 0.07, 0.07, 0.14, 0.11, 0.09],
        fig_size=(18.0, 4.6),
        highlight_rows=[0, 2, 4],
    )
    saved_pngs.append(p2)

    # 3. Detailed Practical Trade Execution Table (Shares & VND)
    prac_sub = proposed_trades_df[proposed_trades_df["Plan_Type"].str.startswith("Practical Board-Lot Plan")].copy()
    short_plan_map = {
        "Practical Board-Lot Plan (Option 1: Hold Current Drifted Positions)": "Option 1: Hold (No Trade)",
        "Practical Board-Lot Plan (Option 2: Rebalance to Original Portfolio P (10.63% / 89.37%))": "Option 2: Original Target P",
        "Practical Board-Lot Plan (Option 3A: Re-optimize Target Return Portfolio P (12.79% Target))": "Option 3A: Target 12.79%",
        "Practical Board-Lot Plan (Option 3B: Re-optimize Risky-Only Maximum Utility (A=2.0))": "Option 3B: Risky Utility A=2",
        "Practical Board-Lot Plan (Option 3C: Re-optimize Complete Portfolio Utility (A=2.0 + Rf))": "Option 3C: Complete Utility A=2",
    }
    df_trade_disp = pd.DataFrame(
        {
            "Rebalancing Option": [short_plan_map.get(r["Plan_Type"], r["Plan_Type"]) for _, r in prac_sub.iterrows()],
            "Asset": prac_sub["Asset"].values,
            "Current Shares\n/ Balance": [
                f"{int(r['Current Quantity']):,d} sh" if r["Asset"] in ("VCB", "DCM") else f"{r['Current Value']:,.0f} VND"
                for _, r in prac_sub.iterrows()
            ],
            "Current Value\n(VND)": [f"{r['Current Value']:,.0f}" for _, r in prac_sub.iterrows()],
            "Current\nWeight": [f"{r['Current Weight']*100:.2f}%" for _, r in prac_sub.iterrows()],
            "Model Target\nWeight": [f"{r['Model Target Weight']*100:.2f}%" for _, r in prac_sub.iterrows()],
            "Proposed Shares\n/ Balance": [
                f"{int(r['Proposed Quantity']):,d} sh" if r["Asset"] in ("VCB", "DCM") else f"{r['Post-Trade Cash']:,.0f} VND"
                for _, r in prac_sub.iterrows()
            ],
            "Trade Action\n(Buy + / Sell -)": [
                f"{int(r['Quantity to Buy or Sell']):+d} sh" if r["Asset"] in ("VCB", "DCM") else f"{r['Estimated Trade Value']:+,.0f} VND"
                for _, r in prac_sub.iterrows()
            ],
            "Trade Value\n(VND)": [f"{r['Estimated Trade Value']:+,.0f}" for _, r in prac_sub.iterrows()],
            "Fee & Tax\n(VND)": [f"{r['Estimated Costs']:,.0f}" for _, r in prac_sub.iterrows()],
            "Proposed\nWeight": [f"{r['Proposed Weight']*100:.2f}%" for _, r in prac_sub.iterrows()],
        }
    )

    p3 = _render_styled_table_png(
        df_display=df_trade_disp,
        title="Practical HOSE 100-Share Lot Rebalancing Trade Execution Plan (30 Oct 2026 Close)",
        subtitle="Current Practical Portfolio P NAV = 53,084,795 VND (0 VCB, 1,400 DCM @ 33,300 VND/sh = 46,620,000 VND, Cash = 6,464,795 VND | VCB Ref Price = 56,300 VND/sh)",
        out_path=figures_dir / "Post_OOS_Trade_Execution_Table.png",
        legacy_path=legacy_dir / "Post_OOS_Trade_Execution_Table.png",
        col_widths=[0.16, 0.14, 0.09, 0.09, 0.07, 0.08, 0.09, 0.10, 0.09, 0.07, 0.08],
        fig_size=(18.5, 8.2),
        highlight_rows=[6, 7, 8, 12, 13, 14],
    )
    saved_pngs.append(p3)

    return saved_pngs



def export_results(
    tables_dict: Dict[str, pd.DataFrame],
    metadata_dict: Dict[str, Any],
    tables_dir: Path,
    reports_dir: Path,
    legacy_dir: Path,
) -> Path:
    """
    Export all mandatory tables to individual CSV files and compile a multi-sheet
    Excel workbook alongside metadata JSON.
    """
    excel_path = reports_dir / "Portfolio_Analysis_VCB_DCM_Report.xlsx"
    legacy_excel_path = legacy_dir / "Portfolio_Analysis_VCB_DCM_Report.xlsx"
    meta_json_path = reports_dir / "execution_metadata_and_config.json"

    with open(meta_json_path, "w", encoding="utf-8") as f_json:
        json.dump(metadata_dict, f_json, indent=2, default=str)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for idx, (sheet_name, df_table) in enumerate(tables_dict.items(), start=1):
            csv_filename = f"{idx:02d}_{sheet_name.replace(' ', '_')}.csv"
            df_table.to_csv(tables_dir / csv_filename, index=True if df_table.index.name else False)
            safe_sheet = sheet_name[:31]
            df_table.to_excel(writer, sheet_name=safe_sheet, index=True if df_table.index.name else False)

    shutil.copyfile(excel_path, legacy_excel_path)
    return excel_path
