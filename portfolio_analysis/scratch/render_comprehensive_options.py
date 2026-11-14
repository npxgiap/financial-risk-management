import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import textwrap

data_opt = [
    {
        "Option": "Option 1: Hold (Status Quo)\nKeep Current Drifted",
        "Proposed Allocation": "10.70% VCB\n88.59% DCM\n0.71% Cash",
        "Trade Action": "No Trade\n(0 VCB, 0 DCM)",
        "Tx Cost": "0 VND",
        "Exp. Return": "17.04%",
        "Volatility": "31.75%",
        "Net Utility": "6.96%",
        "Strategic Rationale & Recommendation": textwrap.fill("BEST FOR GROWTH: Zero transaction cost; fully invested; already holds both stocks closely matching original target.", width=42)
    },
    {
        "Option": "Option 2: Target Rebalancing\nOriginal Target (10.63% / 89.37%)",
        "Proposed Allocation": "10.70% VCB\n88.59% DCM\n0.71% Cash",
        "Trade Action": "No Trade Needed\n(Drift 0.07% < 2% threshold)",
        "Tx Cost": "0 VND",
        "Exp. Return": "17.04%",
        "Volatility": "31.75%",
        "Net Utility": "6.96%",
        "Strategic Rationale & Recommendation": textwrap.fill("IDENTICAL TO OPTION 1: Because initial portfolio already holds 100 VCB & 1,400 DCM, zero trades are required to maintain target.", width=42)
    },
    {
        "Option": "Option 3A: Target 12.79% Return\n(35.75% VCB / 64.25% DCM)",
        "Proposed Allocation": "31.82% VCB\n62.73% DCM\n5.45% Cash",
        "Trade Action": "Buy +200 VCB (+11.26M)\nSell -400 DCM (-13.32M)",
        "Tx Cost": "50,190 VND",
        "Exp. Return": "12.79%",
        "Volatility": "25.53%",
        "Net Utility": "6.17%",
        "Strategic Rationale & Recommendation": textwrap.fill("BEST FOR RISK REDUCTION: Locks in exact 12.79% return target while cutting annual volatility by 6.22% (from 31.75% to 25.53%).", width=42)
    },
    {
        "Option": "Option 3B: Max Risky Utility\n(12.35% VCB / 87.65% DCM)",
        "Proposed Allocation": "10.70% VCB\n82.26% DCM\n7.04% Cash",
        "Trade Action": "Hold 100 VCB (No Trade)\nSell -100 DCM (-3.33M)",
        "Tx Cost": "8,325 VND",
        "Exp. Return": "16.55%",
        "Volatility": "30.95%",
        "Net Utility": "6.96%",
        "Strategic Rationale & Recommendation": textwrap.fill("Trims 100 DCM to reduce equity risk slightly; net utility after cost (6.9592%) is marginally lower than Option 1.", width=42)
    },
    {
        "Option": "Option 3C: Complete Portfolio\n(Optimal Risky + 40.3% Rf)",
        "Proposed Allocation": "0.00% VCB\n56.46% DCM\n43.54% Rf/Cash",
        "Trade Action": "Sell -100 VCB (-5.63M)\nSell -500 DCM (-16.65M)",
        "Tx Cost": "55,700 VND",
        "Exp. Return": "12.88%",
        "Volatility": "20.47%",
        "Net Utility": "8.59%",
        "Strategic Rationale & Recommendation": textwrap.fill("GLOBAL UTILITY OPTIMUM: Highest net utility (8.59%); shifts 23M into risk-free asset, dropping volatility down to 20.47%.", width=42)
    },
    {
        "Option": "Option 3D (Supplementary):\nBand Tolerance Rebalancing",
        "Proposed Allocation": "10.70% VCB\n88.59% DCM\n0.71% Cash",
        "Trade Action": "NO REBALANCE TRIGGERED\n(Drift = 0.07% < 2% Band)",
        "Tx Cost": "0 VND",
        "Exp. Return": "17.04%",
        "Volatility": "31.75%",
        "Net Utility": "6.96%",
        "Strategic Rationale & Recommendation": textwrap.fill("INSTITUTIONAL BEST PRACTICE: Prevents premature churn. Since portfolio drift is negligible (0.07%), do not rebalance until drift exceeds 2%.", width=42)
    },
]

df_o = pd.DataFrame(data_opt)

fig, ax = plt.subplots(figsize=(18.0, 7.5), dpi=300)
ax.axis("off")

fig.suptitle("Comprehensive Post-OOS Rebalancing Decision Matrix (Including Supplementary Tolerance Option)", 
             fontsize=13.5, weight="bold", color="#1a365d", y=0.98)

headers = [
    "Rebalancing Option",
    "Proposed Allocation\n(VCB / DCM / Cash)",
    "Practical Trade Action\n(HOSE 100-sh Lots)",
    "Estimated\nTx Cost",
    "Forward\nE[R]",
    "Annual\nVolatility",
    "Net Utility\n(A = 2.0)",
    "Strategic Rationale & Recommendation"
]

tbl = ax.table(
    cellText=df_o.values,
    colLabels=headers,
    colWidths=[0.19, 0.12, 0.16, 0.07, 0.06, 0.07, 0.07, 0.26],
    loc="center",
    cellLoc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.0, 2.5)

for (r, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cbd5e1")
    cell.set_linewidth(0.8)
    if r == 0:
        cell.set_facecolor("#1e3a5f")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
        cell.get_text().set_fontsize(8.8)
    else:
        if col == 0:
            cell.get_text().set_weight("bold")
            cell._loc = "left"
        elif col == 7:
            cell._loc = "left"
        else:
            cell._loc = "center"
            
        if r in [1, 2, 6]:
            cell.set_facecolor("#fef3c7")
        elif r % 2 == 1:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("white")

plt.tight_layout()

out1 = Path(r"C:\Users\Acer\OneDrive\Documents\FRM 2.0\portfolio_analysis\outputs\figures\Rebalancing_Options_Comprehensive_Table.png")
out2 = Path(r"C:\Users\Acer\OneDrive\Documents\FRM 2.0\output\Rebalancing_Options_Comprehensive_Table.png")
out3 = Path(r"C:\Users\Acer\.gemini\antigravity\brain\d8ff4393-4f64-4df2-81f4-9ea06a69d57e\Rebalancing_Options_Comprehensive_Table.png")

plt.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out3, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Regenerated perfectly formatted Rebalancing_Options_Comprehensive_Table.png")
