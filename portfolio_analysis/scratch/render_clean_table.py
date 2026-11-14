import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load updated table
df_cap = pd.read_csv(r"C:\Users\Acer\OneDrive\Documents\FRM 2.0\portfolio_analysis\outputs\tables\11_Capital_and_Share_Allocation.csv")

data = []
for _, r in df_cap.iterrows():
    if "Reference" in r["Portfolio Point"]:
        continue  # skip extreme 100% boundary references for a clean report table
    
    wv = r["Target Weight VCB"] * 100
    wd = r["Target Weight DCM"] * 100
    wrf = r["Target Weight Risk-Free"] * 100
    weights_str = f"{wv:.2f}% / {wd:.2f}%" if wrf == 0 else f"{wv:.2f}% / {wd:.2f}% / {wrf:.2f}%"
    
    sh_vcb = int(r["Practical Shares VCB"])
    lots_vcb = int(r["Practical Lots VCB (100 sh/lot)"])
    sh_dcm = int(r["Practical Shares DCM"])
    lots_dcm = int(r["Practical Lots DCM (100 sh/lot)"])
    
    name = r["Portfolio Point"].split(" (w_")[0]
    
    data.append({
        "Point": name,
        "Weights": weights_str,
        "Budget VCB": f"{r['Target Budget VCB (VND)']:,.0f}",
        "Budget DCM": f"{r['Target Budget DCM (VND)']:,.0f}",
        "Shares VCB": f"{sh_vcb:,d} sh ({lots_vcb} lots)",
        "Shares DCM": f"{sh_dcm:,d} sh ({lots_dcm} lots)",
        "Stock Spend": f"{r['Total Cash Outlay for Stocks (VND)']:,.0f}",
        "Remaining Cash": f"{r['Uninvested Residual Cash (VND)'] + r['Risk-Free Deposit Allocated (VND)']:,.0f}",
    })

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(15.2, 4.2), dpi=300)
ax.axis("off")

# Title only, clean and simple
fig.suptitle("Portfolio Points Capital Outlay & Share Allocation (Initial Capital: 50,000,000 VND)", 
             fontsize=13.5, weight="bold", color="#1a365d", y=0.98)

headers = [
    "Portfolio Point\n(Efficient Frontier)", 
    "Target Weights\n(VCB / DCM / Rf)", 
    "Target Budget\nVCB (VND)", 
    "Target Budget\nDCM (VND)", 
    "Practical Shares\nVCB (100 sh/lot)", 
    "Practical Shares\nDCM (100 sh/lot)", 
    "Total Stock Outlay\n+ 0.15% Fee (VND)", 
    "Cash / Deposit\nBalance (VND)"
]

tbl = ax.table(
    cellText=df.values,
    colLabels=headers,
    colWidths=[0.24, 0.15, 0.10, 0.10, 0.13, 0.13, 0.13, 0.12],
    loc="center",
    cellLoc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9.0)
tbl.scale(1.0, 2.0)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cbd5e1")
    cell.set_linewidth(0.8)
    if r == 0:
        cell.set_facecolor("#1e3a5f")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
        cell.get_text().set_fontsize(9.0)
    else:
        if c == 0:
            cell._loc = "left"
            cell.get_text().set_weight("bold")
        # Highlight Target Portfolio P (row 4 in the 6-row table)
        if "Target Return" in df.iloc[r-1]["Point"]:
            cell.set_facecolor("#fef3c7")
            cell.get_text().set_weight("bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("white")

plt.tight_layout()

out1 = Path(r"C:\Users\Acer\OneDrive\Documents\FRM 2.0\portfolio_analysis\outputs\figures\Portfolio_Points_Investment_Summary.png")
out2 = Path(r"C:\Users\Acer\OneDrive\Documents\FRM 2.0\output\Portfolio_Points_Investment_Summary.png")
out3 = Path(r"C:\Users\Acer\.gemini\antigravity\brain\d8ff4393-4f64-4df2-81f4-9ea06a69d57e\Portfolio_Points_Investment_Summary.png")

plt.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(out3, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Successfully generated clean updated Portfolio_Points_Investment_Summary.png")

