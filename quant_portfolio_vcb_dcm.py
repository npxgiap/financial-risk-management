"""
=========================================================================================
QUANTITATIVE PORTFOLIO MANAGEMENT – VCB & DCM
Model Name: Quantitative Portfolio Management & Optimization System (QPMOS)
Role: Senior Portfolio Manager, Quantitative Finance Analyst & Python Modeling Expert
=========================================================================================
MÔ TẢ CHƯƠNG TRÌNH (TIẾNG VIỆT):
Chương trình phân tích định lượng và quản lý danh mục đầu tư toàn diện cho hai cổ phiếu:
- VCB (Ngân hàng TMCP Ngoại thương Việt Nam - Vietcombank)
- DCM (CTCP Phân bón Dầu khí Cà Mau - PetroVietnam Ca Mau Fertilizer)
- Chỉ số tham chiếu thị trường (Benchmark): VN-Index

Cấu trúc gồm 19 phần tách biệt rõ ràng, thực hiện đầy đủ 16 Models trực quan hóa (bằng tiếng Anh)
và xuất báo cáo Excel 17 trang tính (sheets) chuẩn mực.
=========================================================================================
"""

# ==============================================================================
# SECTION 1: CONFIGURATION AND PARAMETERS (CẤU HÌNH THAM SỐ HỆ THỐNG)
# ==============================================================================
CONFIG = {
    "STOCK_1": "VCB",
    "STOCK_2": "DCM",
    "STOCK_1_YF": "VCB.VN",
    "STOCK_2_YF": "DCM.VN",
    "STOCK_1_NAME": "Vietcombank",
    "STOCK_2_NAME": "PetroVietnam Ca Mau Fertilizer",
    "BENCHMARK": "VNINDEX",
    "TRAINING_START": "2023-07-31",
    "TRAINING_END": "2026-07-31",
    "BACKTEST_ENTRY": "2026-08-03",
    "BACKTEST_EXIT": "2026-10-30",
    "INITIAL_CAPITAL": 50_000_000.0,     # Vốn đầu tư ban đầu: 50,000,000 VND
    "RISK_FREE_RATE": 0.045,             # Lãi suất phi rủi ro: 4.5% / năm
    "TARGET_ANNUAL_RETURN": 0.1453,      # Lợi suất kỳ vọng mục tiêu: 14.53% / năm
    "RISK_AVERSION_A": 2.0,              # Hệ số ngại rủi ro của khách hàng: A = 2
    "TRADING_DAYS_PER_YEAR": 252,        # Số ngày giao dịch quy chuẩn trong 1 năm
    "SAVINGS_DAY_COUNT_BASE": 365,       # Quy ước tính lãi tiền gửi tiết kiệm (Actual/365)
    "LOT_SIZE": 100,                     # Lô giao dịch tiêu chuẩn sàn HOSE: 100 cổ phiếu
    "TRANSACTION_COST_RATE": 0.0,        # Chi phí giao dịch (mặc định 0%, có thể tùy chỉnh)
    "TAX_RATE": 0.0,                     # Thuế giao dịch (mặc định 0%, có thể tùy chỉnh)
    "SLIPPAGE_RATE": 0.0,                # Độ trượt giá (mặc định 0%, có thể tùy chỉnh)
    "DATA_SOURCE_STOCKS": "Yahoo Finance (yfinance)",
    "DATA_SOURCE_INDEX": "DNSE Entrade Market Data API / Local CSV Cache",
    "OUTPUT_FOLDER": "output"
}

# ==============================================================================
# SECTION 2: LIBRARY INSTALLATION AND IMPORTS (KHAI BÁO THƯ VIỆN)
# ==============================================================================
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import requests

# Cấu hình hiển thị UTF-8 trên thiết bị đầu cuối Windows để in tiếng Việt không lỗi font
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Tắt các cảnh báo không cần thiết và thiết lập giao diện biểu đồ tài chính chuyên nghiệp
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

# Khởi tạo thư mục lưu trữ kết quả đầu ra
OUTPUT_DIR = Path(os.getcwd()) / CONFIG["OUTPUT_FOLDER"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_CSV_PATH = OUTPUT_DIR / "raw_market_data_vcb_dcm_vnindex.csv"
EXCEL_REPORT_PATH = OUTPUT_DIR / "Quantitative_Portfolio_Management_VCB_DCM.xlsx"


# ==============================================================================
# SECTION 3: DATA COLLECTION (THU THẬP DỮ LIỆU THỊ TRƯỜNG)
# ==============================================================================
def fetch_vnindex_data(start_date: str, end_date: str) -> pd.Series:
    """
    Hàm tải dữ liệu lịch sử chỉ số VN-Index từ API dữ liệu thị trường chứng khoán Việt Nam (DNSE Entrade)
    hoặc sử dụng dữ liệu dự phòng nếu đã lưu trong file CSV cục bộ.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    # Cộng thêm vài ngày ở biên cuối để đảm bảo bao phủ trọn ngày kết thúc
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_ts = int(start_dt.timestamp()) - 86400 * 3
    end_ts = int(end_dt.timestamp()) + 86400 * 3

    url = (
        f"https://services.entrade.com.vn/chart-api/v2/ohlcs/index"
        f"?from={start_ts}&to={end_ts}&symbol=VNINDEX&resolution=1D"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    payload = response.json()

    if not payload or "t" not in payload or len(payload["t"]) == 0:
        raise ValueError("Không nhận được dữ liệu hợp lệ cho VN-Index từ API.")

    # Chuyển đổi timestamp sang múi giờ Việt Nam (Asia/Ho_Chi_Minh) và chuẩn hóa ngày
    dates = (
        pd.to_datetime(payload["t"], unit="s")
        .tz_localize("UTC")
        .tz_convert("Asia/Ho_Chi_Minh")
        .strftime("%Y-%m-%d")
    )
    vnindex_series = pd.Series(payload["c"], index=pd.to_datetime(dates), name="VNINDEX")
    vnindex_series = vnindex_series[~vnindex_series.index.duplicated(keep="first")]
    return vnindex_series


def collect_market_data(config: dict) -> pd.DataFrame:
    """
    Thu thập dữ liệu giá đóng cửa điều chỉnh (Adjusted Close) và giá khớp lệnh thực tế (Unadjusted Close)
    của VCB, DCM từ Yahoo Finance cùng chỉ số VN-Index. Lưu trữ bản gốc vào file CSV để tái lập.
    """
    print("\n[BƯỚC 1] Đang thu thập dữ liệu thị trường cho VCB, DCM và VN-Index...")
    retrieval_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Tải dữ liệu từ đầu giai đoạn Training đến hết giai đoạn Backtest/Rebalancing
        yf_tickers = [config["STOCK_1_YF"], config["STOCK_2_YF"]]
        raw_yf = yf.download(
            yf_tickers,
            start=config["TRAINING_START"],
            end="2026-11-02",
            auto_adjust=False,
            progress=False
        )

        if raw_yf.empty:
            raise ValueError("Dữ liệu tải về từ Yahoo Finance bị trống.")

        # Trích xuất cả Adjusted Close (dùng tính tổng lợi suất) và Close (dùng kiểm tra khớp lệnh thực tế)
        adj_close = raw_yf["Adj Close"].rename(
            columns={config["STOCK_1_YF"]: "VCB", config["STOCK_2_YF"]: "DCM"}
        )
        unadj_close = raw_yf["Close"].rename(
            columns={config["STOCK_1_YF"]: "VCB_Unadj", config["STOCK_2_YF"]: "DCM_Unadj"}
        )
        volume = raw_yf["Volume"].rename(
            columns={config["STOCK_1_YF"]: "VCB_Volume", config["STOCK_2_YF"]: "DCM_Volume"}
        )

        # Tải dữ liệu VN-Index
        vnindex_series = fetch_vnindex_data(config["TRAINING_START"], config["BACKTEST_EXIT"])

        # Hợp nhất bảng dữ liệu gốc
        combined_raw = pd.concat([adj_close, vnindex_series, unadj_close, volume], axis=1)
        combined_raw.index.name = "Date"
        combined_raw["Retrieval_Date"] = retrieval_timestamp
        combined_raw["Data_Source"] = f"{config['DATA_SOURCE_STOCKS']} & {config['DATA_SOURCE_INDEX']}"

        # Lưu trữ file CSV cục bộ phục vụ tái lập không cần tải lại
        combined_raw.to_csv(RAW_DATA_CSV_PATH)
        print(f" -> Đã tải và lưu dữ liệu gốc thành công tại: {RAW_DATA_CSV_PATH}")
        return combined_raw

    except Exception as exc:
        print(f" -> Cảnh báo lỗi tải trực tuyến ({exc}). Đang kiểm tra file CSV dự phòng...")
        if RAW_DATA_CSV_PATH.exists():
            df_cached = pd.read_csv(RAW_DATA_CSV_PATH, index_col="Date", parse_dates=True)
            print(f" -> Đã khôi phục dữ liệu thành công từ file cục bộ: {RAW_DATA_CSV_PATH}")
            return df_cached
        else:
            raise RuntimeError(
                "Không thể tải dữ liệu trực tuyến và chưa có file CSV cục bộ. "
                "Vui lòng cung cấp file CSV với các cột: Date, VCB, DCM, VNINDEX."
            ) from exc


# ==============================================================================
# SECTION 4: DATA CLEANING AND VALIDATION (LÀM SẠCH VÀ KIỂM ĐỊNH DỮ LIỆU)
# ==============================================================================
def clean_and_validate_data(raw_df: pd.DataFrame, config: dict):
    """
    Kiểm tra giá trị thiếu, ngày trùng lặp, phiên nghỉ lễ không giao dịch và đồng bộ hóa
    ngày giao dịch chung giữa VCB, DCM và VN-Index. Tuyệt đối không dùng forward-fill hay nội suy.
    """
    print("[BƯỚC 2] Đang kiểm tra chất lượng, làm sạch và đồng bộ hóa ngày giao dịch...")
    df = raw_df.copy()

    # Kiểm tra và loại bỏ ngày trùng lặp nếu có
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        df = df[~df.index.duplicated(keep="first")]

    # Chỉ giữ các ngày có đầy đủ dữ liệu giao dịch hợp lệ của cả VCB, DCM và VNINDEX
    # Đồng thời loại bỏ các ngày lễ Việt Nam mà Yahoo Finance tự động điền Volume = 0
    core_cols = ["VCB", "DCM", "VNINDEX", "VCB_Unadj", "DCM_Unadj"]
    valid_mask = df[["VCB", "DCM", "VNINDEX"]].notnull().all(axis=1)
    if "VCB_Volume" in df.columns and "DCM_Volume" in df.columns:
        valid_mask = valid_mask & ((df["VCB_Volume"] > 0) | (df["DCM_Volume"] > 0))

    aligned_df = df.loc[valid_mask, core_cols].sort_index().copy()

    # Tách tập dữ liệu Training (31/07/2023 - 31/07/2026) và Out-of-Sample Backtest (03/08/2026 - 30/10/2026)
    train_df = aligned_df.loc[config["TRAINING_START"]:config["TRAINING_END"]].copy()
    backtest_df = aligned_df.loc[config["BACKTEST_ENTRY"]:config["BACKTEST_EXIT"]].copy()
    full_rebalance_df = aligned_df.loc[config["TRAINING_START"]:config["BACKTEST_EXIT"]].copy()

    # Kiểm tra bảo đảm không có hiện tượng rò rỉ dữ liệu tương lai (No Look-Ahead Bias)
    assert train_df.index.max() <= pd.Timestamp(config["TRAINING_END"]), "Lỗi: Tập Training chứa dữ liệu tương lai!"
    assert backtest_df.index.min() >= pd.Timestamp(config["BACKTEST_ENTRY"]), "Lỗi: Tập Backtest chồng lấn tập Training!"

    print(f" -> Giai đoạn Training ({train_df.index.min().date()} đến {train_df.index.max().date()}): {len(train_df)} phiên giao dịch.")
    print(f" -> Giai đoạn Backtest ({backtest_df.index.min().date()} đến {backtest_df.index.max().date()}): {len(backtest_df)} phiên giao dịch.")
    print(f" -> Giai đoạn Rebalancing ({full_rebalance_df.index.min().date()} đến {full_rebalance_df.index.max().date()}): {len(full_rebalance_df)} phiên giao dịch.")

    return aligned_df, train_df, backtest_df, full_rebalance_df


# ==============================================================================
# SECTION 5: RETURN CALCULATION (TÍNH TOÁN LỢI SUẤT ĐƠN VÀ LOGARIT)
# ==============================================================================
def calculate_returns(price_df: pd.DataFrame):
    """
    Tính toán lợi suất đơn hàng ngày (Daily Simple Return: R_t = P_t / P_{t-1} - 1)
    và lợi suất logarit hàng ngày (Daily Log Return: r_t = ln(P_t / P_{t-1})).
    """
    assets = ["VCB", "DCM", "VNINDEX"]
    prices = price_df[assets]
    simple_returns = prices.pct_change().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return simple_returns, log_returns


# ==============================================================================
# SECTION 6: HISTORICAL PRICE ANALYSIS (PHÂN TÍCH LỊCH SỬ GIÁ - MODELS 1 & 2)
# ==============================================================================
def plot_historical_prices(train_df: pd.DataFrame):
    """
    Tạo Model 1 (Adjusted Closing Price Chart) và Model 2 (Normalized Stock Price Performance = 100).
    """
    print("[BƯỚC 3] Đang khởi tạo Model 1 (Adjusted Closing Price) và Model 2 (Normalized Performance)...")

    # --- MODEL 1: Adjusted Closing Price Chart (2 Subplots riêng biệt do khác biệt thị giá) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(train_df.index, train_df["VCB"], color="#1f77b4", linewidth=1.8, label="VCB (Vietcombank)")
    axes[0].set_title("Model 1: Historical Adjusted Closing Price Movements (31 Jul 2023 – 31 Jul 2026)", fontsize=13, weight="bold")
    axes[0].set_ylabel("VCB Price (VND)", fontsize=11, weight="bold")
    axes[0].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(train_df.index, train_df["DCM"], color="#2ca02c", linewidth=1.8, label="DCM (PetroVietnam Ca Mau Fertilizer)")
    axes[1].set_xlabel("Trading Date", fontsize=11, weight="bold")
    axes[1].set_ylabel("DCM Price (VND)", fontsize=11, weight="bold")
    axes[1].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    axes[1].legend(loc="upper left", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_01_Adjusted_Closing_Price.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- MODEL 2: Normalized Stock Price Performance (Chuẩn hóa mốc khởi đầu = 100) ---
    norm_df = (train_df[["VCB", "DCM", "VNINDEX"]] / train_df[["VCB", "DCM", "VNINDEX"]].iloc[0]) * 100.0
    plt.figure(figsize=(12, 6))
    plt.plot(norm_df.index, norm_df["VCB"], label="VCB (Normalized = 100)", color="#1f77b4", linewidth=2.0)
    plt.plot(norm_df.index, norm_df["DCM"], label="DCM (Normalized = 100)", color="#2ca02c", linewidth=2.0)
    plt.plot(norm_df.index, norm_df["VNINDEX"], label="VN-Index Benchmark (Normalized = 100)", color="#7f7f7f", linestyle="--", linewidth=1.6)
    plt.axhline(100, color="black", linestyle=":", linewidth=1.0)
    plt.title("Model 2: Normalized Stock Price Performance (Base = 100 on 31 Jul 2023)", fontsize=13, weight="bold")
    plt.xlabel("Trading Date", fontsize=11, weight="bold")
    plt.ylabel("Normalized Index Value (Base = 100)", fontsize=11, weight="bold")
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_02_Normalized_Price_Performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    return norm_df


# ==============================================================================
# SECTION 7: RISK AND RETURN ANALYSIS (PHÂN TÍCH RỦI RO & LỢI SUẤT - MODELS 3 & 4)
# ==============================================================================
def compute_max_drawdown(price_series: pd.Series) -> float:
    """
    Tính mức sụt giảm tối đa từ đỉnh xuống đáy (Maximum Drawdown).
    """
    rolling_peak = price_series.cummax()
    drawdown = (price_series - rolling_peak) / rolling_peak
    return float(drawdown.min())


def analyze_risk_and_return(train_df: pd.DataFrame, simple_returns: pd.DataFrame, config: dict):
    """
    Tính toán đầy đủ 12 chỉ tiêu rủi ro và lợi suất cho VCB, DCM và VN-Index,
    đồng thời vẽ Model 3 (Return Distribution) và Model 4 (Risk and Return Comparison).
    """
    print("[BƯỚC 4] Đang tính toán các chỉ tiêu Risk & Return và vẽ Model 3, Model 4...")
    ann_factor = config["TRADING_DAYS_PER_YEAR"]
    assets = ["VCB", "DCM", "VNINDEX"]
    metrics_rows = []

    for ticker in assets:
        r = simple_returns[ticker]
        p = train_df[ticker]
        n_obs = len(r)

        mean_daily = float(r.mean())
        ann_arith_ret = mean_daily * ann_factor
        cagr = float((p.iloc[-1] / p.iloc[0]) ** (ann_factor / n_obs) - 1.0)
        daily_vol = float(r.std(ddof=1))
        ann_vol = daily_vol * np.sqrt(ann_factor)
        daily_var = float(r.var(ddof=1))
        ann_var = daily_var * ann_factor
        min_daily = float(r.min())
        max_daily = float(r.max())
        skew = float(stats.skew(r, bias=False))
        kurt = float(stats.kurtosis(r, fisher=True, bias=False))  # Excess Kurtosis
        mdd = compute_max_drawdown(p)

        metrics_rows.append({
            "Metric_Ticker": ticker,
            "Mean Daily Return": mean_daily,
            "Annualized Arithmetic Return": ann_arith_ret,
            "Annualized Geometric Return (CAGR)": cagr,
            "Daily Volatility": daily_vol,
            "Annualized Volatility": ann_vol,
            "Daily Variance": daily_var,
            "Annualized Variance": ann_var,
            "Minimum Daily Return": min_daily,
            "Maximum Daily Return": max_daily,
            "Skewness": skew,
            "Excess Kurtosis": kurt,
            "Maximum Drawdown": mdd
        })

    risk_return_df = pd.DataFrame(metrics_rows).set_index("Metric_Ticker").T

    # --- MODEL 3: Return Distribution (Histogram + Fitted Normal + KDE) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = {"VCB": "#1f77b4", "DCM": "#2ca02c"}

    for idx, ticker in enumerate(["VCB", "DCM"]):
        ax = axes[idx]
        r = simple_returns[ticker]
        mu, sigma = r.mean(), r.std(ddof=1)
        skew = stats.skew(r, bias=False)
        kurt = stats.kurtosis(r, fisher=True, bias=False)

        # Vẽ Histogram và KDE
        sns.histplot(r, bins=45, stat="density", color=colors[ticker], alpha=0.35, edgecolor="black", ax=ax, label="Observed Histogram")
        sns.kdeplot(r, color=colors[ticker], linewidth=2.0, ax=ax, label="Empirical KDE")

        # Vẽ đường cong phân phối chuẩn khớp (Fitted Normal Curve)
        x_vals = np.linspace(r.min(), r.max(), 300)
        normal_pdf = stats.norm.pdf(x_vals, mu, sigma)
        ax.plot(x_vals, normal_pdf, color="crimson", linestyle="--", linewidth=2.0, label="Fitted Normal PDF")

        ax.axvline(mu, color="navy", linestyle="-", linewidth=1.5, label=f"Mean ({mu*100:.3f}%)")
        ax.axvline(mu - sigma, color="gray", linestyle=":", linewidth=1.2, label=f"±1 Std Dev ({sigma*100:.2f}%)")
        ax.axvline(mu + sigma, color="gray", linestyle=":", linewidth=1.2)

        stats_box = (
            f"Ticker: {ticker}\n"
            f"Mean Daily: {mu*100:.3f}%\n"
            f"Daily Std: {sigma*100:.2f}%\n"
            f"Skewness: {skew:.2f}\n"
            f"Excess Kurtosis: {kurt:.2f}"
        )
        ax.text(0.04, 0.95, stats_box, transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9), fontsize=9.5, weight="bold")

        ax.set_title(f"Model 3: Daily Return Distribution – {ticker}", fontsize=12.5, weight="bold")
        ax.set_xlabel("Daily Simple Return", fontsize=11, weight="bold")
        ax.set_ylabel("Density", fontsize=11, weight="bold")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_03_Return_Distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- MODEL 4: Risk and Return Comparison Scatter Plot ---
    plt.figure(figsize=(9, 6))
    pts = {
        "VCB": (risk_return_df.loc["Annualized Volatility", "VCB"], risk_return_df.loc["Annualized Arithmetic Return", "VCB"], "#1f77b4"),
        "DCM": (risk_return_df.loc["Annualized Volatility", "DCM"], risk_return_df.loc["Annualized Arithmetic Return", "DCM"], "#2ca02c"),
        "VN-Index": (risk_return_df.loc["Annualized Volatility", "VNINDEX"], risk_return_df.loc["Annualized Arithmetic Return", "VNINDEX"], "#ff7f0e")
    }
    for label, (vol, ret_val, col) in pts.items():
        plt.scatter(vol, ret_val, s=140, color=col, edgecolors="black", zorder=5, label=label)
        plt.annotate(
            f"{label}\n(Risk: {vol*100:.2f}%, Return: {ret_val*100:.2f}%)",
            xy=(vol, ret_val),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=9.5,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=col, alpha=0.85)
        )

    plt.axhline(0, color="black", linestyle=":", linewidth=1.0)
    plt.title("Model 4: Annualized Risk and Return Comparison (31 Jul 2023 – 31 Jul 2026)", fontsize=13, weight="bold")
    plt.xlabel("Annualized Volatility (Standard Deviation σ)", fontsize=11, weight="bold")
    plt.ylabel("Annualized Expected Return E(R)", fontsize=11, weight="bold")
    plt.xaxis_fmt = plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.yaxis_fmt = plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlim(0.12, 0.45)
    plt.ylim(-0.15, 0.22)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_04_Risk_Return_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    return risk_return_df


# ==============================================================================
# SECTION 8: VALUE AT RISK (VaR) & CONDITIONAL VaR (CVaR)
# ==============================================================================
def calculate_var_cvar(simple_returns: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Tính toán Historical VaR, Parametric (Normal) VaR và Conditional VaR (CVaR / Expected Shortfall)
    tại mức độ tin cậy 95% và 99%.
    Quy ước dấu (Sign Convention): Trình bày VaR và CVaR là số dương thể hiện mức lỗ ước tính
    (Positive Loss Convention) cả theo phần trăm (%) và giá trị tiền tệ (VND) cho khoản đầu tư 50,000,000 VND.
    """
    print("[BƯỚC 5] Đang tính toán Historical VaR, Parametric VaR và CVaR (95% & 99%)...")
    capital = config["INITIAL_CAPITAL"]
    var_rows = []

    for ticker in ["VCB", "DCM"]:
        r = simple_returns[ticker].dropna()
        mu = float(r.mean())
        sigma = float(r.std(ddof=1))

        # 1. Historical VaR & CVaR (Positive Loss Convention)
        q_05 = float(np.percentile(r, 5))
        q_01 = float(np.percentile(r, 1))
        hist_var_95 = -q_05
        hist_var_99 = -q_01
        hist_cvar_95 = -float(r[r <= q_05].mean())
        hist_cvar_99 = -float(r[r <= q_01].mean())

        # 2. Parametric Normal VaR & CVaR (Positive Loss Convention)
        z_95 = stats.norm.ppf(0.95)  # ~1.64485
        z_99 = stats.norm.ppf(0.99)  # ~2.32635
        param_var_95 = -(mu - z_95 * sigma)
        param_var_99 = -(mu - z_99 * sigma)
        param_cvar_95 = -(mu - sigma * (stats.norm.pdf(z_95) / 0.05))
        param_cvar_99 = -(mu - sigma * (stats.norm.pdf(z_99) / 0.01))

        var_rows.append({
            "Ticker": ticker,
            "Historical VaR 95% (%)": hist_var_95,
            "Historical VaR 95% (VND)": hist_var_95 * capital,
            "Historical VaR 99% (%)": hist_var_99,
            "Historical VaR 99% (VND)": hist_var_99 * capital,
            "Historical CVaR 95% (%)": hist_cvar_95,
            "Historical CVaR 95% (VND)": hist_cvar_95 * capital,
            "Historical CVaR 99% (%)": hist_cvar_99,
            "Historical CVaR 99% (VND)": hist_cvar_99 * capital,
            "Parametric VaR 95% (%)": param_var_95,
            "Parametric VaR 95% (VND)": param_var_95 * capital,
            "Parametric VaR 99% (%)": param_var_99,
            "Parametric VaR 99% (VND)": param_var_99 * capital,
            "Parametric CVaR 95% (%)": param_cvar_95,
            "Parametric CVaR 95% (VND)": param_cvar_95 * capital,
            "Parametric CVaR 99% (%)": param_cvar_99,
            "Parametric CVaR 99% (VND)": param_cvar_99 * capital,
        })

    var_df = pd.DataFrame(var_rows).set_index("Ticker").T
    return var_df


# ==============================================================================
# SECTION 9: COVARIANCE AND CORRELATION ANALYSIS (MODELS 5 & 6)
# ==============================================================================
def analyze_covariance_and_correlation(simple_returns: pd.DataFrame, config: dict):
    """
    Tính toán ma trận hiệp phương sai (Daily & Annualized Covariance Matrix)
    và ma trận tương quan (2x2 Correlation Matrix). Vẽ Model 5 và Model 6.
    """
    print("[BƯỚC 6] Đang tính toán Covariance & Correlation Matrices và vẽ Model 5, Model 6...")
    pair_ret = simple_returns[["VCB", "DCM"]]
    daily_cov = pair_ret.cov()
    ann_cov = daily_cov * config["TRADING_DAYS_PER_YEAR"]
    corr_mat = pair_ret.corr(method="pearson")

    # --- MODEL 5: Covariance Matrix (Daily & Annualized Side-by-Side Heatmaps) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(daily_cov, annot=True, fmt=".6f", cmap="Blues", cbar=True, square=True, linewidths=1, ax=axes[0],
                cbar_kws={"label": "Daily Covariance"})
    axes[0].set_title("Model 5A: Daily Return Covariance Matrix", fontsize=12, weight="bold")

    sns.heatmap(ann_cov, annot=True, fmt=".4f", cmap="Greens", cbar=True, square=True, linewidths=1, ax=axes[1],
                cbar_kws={"label": "Annualized Covariance (×252)"})
    axes[1].set_title("Model 5B: Annualized Return Covariance Matrix", fontsize=12, weight="bold")

    plt.suptitle("Model 5: Covariance Matrix Analysis – VCB & DCM", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_05_Covariance_Matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- MODEL 6: Correlation Matrix Heatmap ---
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(corr_mat, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1.0, vmax=1.0, center=0.0,
                square=True, linewidths=1.2, cbar_kws={"label": "Pearson Correlation Coefficient (ρ)"})
    plt.title("Model 6: 2×2 Pearson Correlation Matrix (VCB & DCM)", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_06_Correlation_Matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    return daily_cov, ann_cov, corr_mat


# ==============================================================================
# SECTION 10: MODERN PORTFOLIO THEORY – MARKOWITZ OPTIMIZATION (MODELS 7 & 8)
# ==============================================================================
def portfolio_performance(weights: np.ndarray, mu: np.ndarray, cov_mat: np.ndarray, rf: float):
    """
    Hàm tính lợi suất kỳ vọng thường niên, độ lệch chuẩn (rủi ro), phương sai và Sharpe Ratio của danh mục.
    """
    w = np.asarray(weights, dtype=float)
    exp_ret = float(np.dot(w, mu))
    variance = float(np.dot(w.T, np.dot(cov_mat, w)))
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = (exp_ret - rf) / volatility if volatility > 1e-12 else 0.0
    return exp_ret, volatility, variance, sharpe


def run_markowitz_optimization(mu_series: pd.Series, ann_cov_df: pd.DataFrame, config: dict):
    """
    Áp dụng Lý thuyết Danh mục Hiện đại (Markowitz) với scipy.optimize:
    - Ràng buộc: 0 <= w_VCB, w_DCM <= 1, w_VCB + w_DCM = 1 (Không bán khống, không dùng đòn bẩy).
    - Tìm Global Minimum Variance Portfolio (GMVP).
    - Tìm Tangency Portfolio P (Maximum Sharpe Ratio).
    - Kiểm tra điều kiện khả thi (Feasibility Check) cho Target Return = 14.53%.
    - Vẽ Model 7 (Efficient Frontier) và Model 8 (Capital Allocation Line).
    """
    print("[BƯỚC 7] Đang giải bài toán tối ưu hóa Markowitz, kiểm định Target 14.53% và vẽ Model 7, Model 8...")
    tickers = ["VCB", "DCM"]
    mu = mu_series[tickers].values
    cov = ann_cov_df.loc[tickers, tickers].values
    rf = config["RISK_FREE_RATE"]
    target_return = config["TARGET_ANNUAL_RETURN"]

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # 1. Tìm Global Minimum Variance Portfolio (GMVP)
    res_gmvp = minimize(
        lambda w: float(np.dot(w.T, np.dot(cov, w))),
        x0=np.array([0.5, 0.5]),
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint],
        options={"ftol": 1e-12, "maxiter": 500}
    )
    if not res_gmvp.success:
        raise RuntimeError(f"Tối ưu hóa GMVP thất bại: {res_gmvp.message}")
    w_gmvp = res_gmvp.x
    ret_gmvp, vol_gmvp, var_gmvp, sr_gmvp = portfolio_performance(w_gmvp, mu, cov, rf)

    # 2. Tìm Tangency Portfolio P (Maximum Sharpe Ratio)
    res_tan = minimize(
        lambda w: -portfolio_performance(w, mu, cov, rf)[3],
        x0=np.array([0.5, 0.5]),
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint],
        options={"ftol": 1e-12, "maxiter": 500}
    )
    if not res_tan.success:
        raise RuntimeError(f"Tối ưu hóa Tangency Portfolio thất bại: {res_tan.message}")
    w_tan = res_tan.x
    ret_tan, vol_tan, var_tan, sr_tan = portfolio_performance(w_tan, mu, cov, rf)

    # 3. Equal Weight Portfolio (50% VCB - 50% DCM)
    w_eq = np.array([0.5, 0.5])
    ret_eq, vol_eq, var_eq, sr_eq = portfolio_performance(w_eq, mu, cov, rf)

    # 4. Kiểm tra tính khả thi của Target Return (14.53%)
    min_feasible_ret = float(np.min(mu))
    max_feasible_ret = float(np.max(mu))
    is_target_feasible = (min_feasible_ret - 1e-9 <= target_return <= max_feasible_ret + 1e-9)

    if is_target_feasible:
        target_constraints = [
            sum_constraint,
            {"type": "eq", "fun": lambda w: float(np.dot(w, mu)) - target_return}
        ]
        res_target = minimize(
            lambda w: float(np.dot(w.T, np.dot(cov, w))),
            x0=np.array([0.5, 0.5]),
            method="SLSQP",
            bounds=bounds,
            constraints=target_constraints,
            options={"ftol": 1e-12, "maxiter": 500}
        )
        w_target = res_target.x
        ret_target, vol_target, var_target, sr_target = portfolio_performance(w_target, mu, cov, rf)
        target_status = "Feasible under Long-Only constraints"
        return_shortfall = 0.0
    else:
        # Mục tiêu 14.53% vượt ngoài miền khả thi Long-Only: Tìm danh mục khả thi gần mục tiêu nhất
        if target_return > max_feasible_ret:
            idx_max = int(np.argmax(mu))
            w_target = np.zeros(2)
            w_target[idx_max] = 1.0
        else:
            idx_min = int(np.argmin(mu))
            w_target = np.zeros(2)
            w_target[idx_min] = 1.0

        ret_target, vol_target, var_target, sr_target = portfolio_performance(w_target, mu, cov, rf)
        return_shortfall = target_return - ret_target
        target_status = (
            f"Infeasible under Long-Only constraints (Target {target_return*100:.2f}% > "
            f"Max Feasible {max_feasible_ret*100:.2f}%; Shortfall = {return_shortfall*100:.2f}%)"
        )

    # 5. Khởi tạo tập hợp các danh mục khả thi (Feasible Portfolios Grid) để vẽ đường Efficient Frontier
    w_vcb_grid = np.linspace(0.0, 1.0, 501)
    ef_records = []
    for w1 in w_vcb_grid:
        w_vec = np.array([w1, 1.0 - w1])
        er, vol, var_val, sr = portfolio_performance(w_vec, mu, cov, rf)
        ef_records.append({
            "VCB_Weight": w1,
            "DCM_Weight": 1.0 - w1,
            "Expected_Return": er,
            "Annualized_Volatility": vol,
            "Annualized_Variance": var_val,
            "Sharpe_Ratio": sr,
            "Branch": "Upper Efficient Branch" if er >= ret_gmvp - 1e-9 else "Inefficient Lower Branch"
        })
    ef_df = pd.DataFrame(ef_records)

    # --- MODEL 7: Markowitz Efficient Frontier ---
    plt.figure(figsize=(10.5, 6.5))
    upper_df = ef_df[ef_df["Branch"] == "Upper Efficient Branch"]
    lower_df = ef_df[ef_df["Branch"] == "Inefficient Lower Branch"]

    plt.plot(upper_df["Annualized_Volatility"], upper_df["Expected_Return"],
             color="#1f77b4", linewidth=2.8, label="Upper Efficient Frontier")
    plt.plot(lower_df["Annualized_Volatility"], lower_df["Expected_Return"],
             color="#aec7e8", linestyle="--", linewidth=2.0, label="Inefficient Frontier Branch")

    # Vẽ các điểm danh mục quan trọng
    vol_vcb, ret_vcb = np.sqrt(cov[0, 0]), mu[0]
    vol_dcm, ret_dcm = np.sqrt(cov[1, 1]), mu[1]

    plt.scatter(vol_vcb, ret_vcb, color="#1f77b4", s=110, edgecolors="black", zorder=5, label="VCB (100% VCB)")
    plt.scatter(vol_dcm, ret_dcm, color="#2ca02c", s=110, edgecolors="black", zorder=5, label="DCM (100% DCM)")
    plt.scatter(vol_gmvp, ret_gmvp, color="gold", marker="D", s=140, edgecolors="black", zorder=6,
                label=f"GMVP ({w_gmvp[0]*100:.1f}% VCB, {w_gmvp[1]*100:.1f}% DCM)")
    plt.scatter(vol_tan, ret_tan, color="crimson", marker="*", s=260, edgecolors="black", zorder=7,
                label=f"Tangency Portfolio P & Closest Target ({w_tan[0]*100:.0f}% VCB, {w_tan[1]*100:.0f}% DCM)")

    # Vẽ đường ngang thể hiện Target Return = 14.53%
    plt.axhline(target_return, color="purple", linestyle="-.", linewidth=1.5,
                label=f"Client Target Return ({target_return*100:.2f}% - Infeasible Long-Only)")

    plt.title("Model 7: Markowitz Efficient Frontier – VCB & DCM", fontsize=13.5, weight="bold")
    plt.xlabel("Annualized Risk (Volatility σ)", fontsize=11, weight="bold")
    plt.ylabel("Annualized Expected Return E(R)", fontsize=11, weight="bold")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend(loc="lower left", frameon=True, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_07_Markowitz_Efficient_Frontier.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- MODEL 8: Capital Allocation Line (CAL) ---
    plt.figure(figsize=(10.5, 6.5))
    plt.plot(upper_df["Annualized_Volatility"], upper_df["Expected_Return"],
             color="#1f77b4", linewidth=2.5, label="Markowitz Efficient Frontier")
    plt.plot(lower_df["Annualized_Volatility"], lower_df["Expected_Return"],
             color="#aec7e8", linestyle="--", linewidth=1.8, label="Inefficient Frontier")

    sigma_cal = np.linspace(0.0, vol_tan * 1.25, 200)
    ret_cal = rf + sr_tan * sigma_cal
    plt.plot(sigma_cal, ret_cal, color="crimson", linewidth=2.4,
             label=f"Capital Allocation Line (CAL: Slope = Sharpe = {sr_tan:.4f})")

    plt.scatter(0.0, rf, color="forestgreen", marker="s", s=120, edgecolors="black", zorder=6,
                label=f"Risk-Free Asset (Rf = {rf*100:.2f}%)")
    plt.scatter(vol_gmvp, ret_gmvp, color="gold", marker="D", s=130, edgecolors="black", zorder=6, label="GMVP")
    plt.scatter(vol_tan, ret_tan, color="crimson", marker="*", s=260, edgecolors="black", zorder=7,
                label=f"Tangency Portfolio P (E(R)={ret_tan*100:.2f}%, σ={vol_tan*100:.2f}%)")

    plt.title("Model 8: Capital Allocation Line (CAL) & Tangency Portfolio P", fontsize=13.5, weight="bold")
    plt.xlabel("Annualized Volatility (σ)", fontsize=11, weight="bold")
    plt.ylabel("Annualized Expected Return E(R)", fontsize=11, weight="bold")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlim(-0.01, vol_tan * 1.28)
    plt.legend(loc="lower right", frameon=True, fontsize=9.5)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_08_Capital_Allocation_Line.png", dpi=300, bbox_inches="tight")
    plt.close()

    opt_results = {
        "GMVP": {"w_VCB": w_gmvp[0], "w_DCM": w_gmvp[1], "w_Rf": 0.0, "Return": ret_gmvp, "Vol": vol_gmvp, "Var": var_gmvp, "Sharpe": sr_gmvp},
        "Tangency_P": {"w_VCB": w_tan[0], "w_DCM": w_tan[1], "w_Rf": 0.0, "Return": ret_tan, "Vol": vol_tan, "Var": var_tan, "Sharpe": sr_tan},
        "Equal_Weight": {"w_VCB": w_eq[0], "w_DCM": w_eq[1], "w_Rf": 0.0, "Return": ret_eq, "Vol": vol_eq, "Var": var_eq, "Sharpe": sr_eq},
        "Target_14_53": {
            "w_VCB": w_target[0],
            "w_DCM": w_target[1],
            "w_Rf": 0.0,
            "Return": ret_target,
            "Vol": vol_target,
            "Var": var_target,
            "Sharpe": sr_target,
            "Target_Requested": target_return,
            "Is_Feasible": is_target_feasible,
            "Max_Feasible_Return": max_feasible_ret,
            "Min_Feasible_Return": min_feasible_ret,
            "Return_Shortfall": return_shortfall,
            "Status": target_status
        }
    }
    return opt_results, ef_df


# ==============================================================================
# SECTION 11: OPTIMAL INVESTOR PORTFOLIO – RISK AVERSION A = 2 (MODEL 9)
# ==============================================================================
def calculate_optimal_investor_portfolio(opt_results: dict, ef_df: pd.DataFrame, config: dict):
    """
    Xác định danh mục tối ưu của nhà đầu tư (Optimal Investor Portfolio C) với hệ số ngại rủi ro A = 2
    theo hàm hữu dụng kỳ vọng - phương sai: U = E(R_c) - 0.5 * A * sigma_c^2.
    Tỷ trọng phân bổ vào danh mục rủi ro P: y* = (E(R_P) - Rf) / (A * sigma_P^2), giới hạn trong [0, 1].
    Vẽ Model 9 (Indifference Curves and Efficient Frontier).
    """
    print("[BƯỚC 8] Đang tính toán Optimal Investor Portfolio (A = 2) và vẽ Model 9...")
    A = config["RISK_AVERSION_A"]
    rf = config["RISK_FREE_RATE"]
    tan_p = opt_results["Tangency_P"]
    ret_p, vol_p, var_p, sr_p = tan_p["Return"], tan_p["Vol"], tan_p["Var"], tan_p["Sharpe"]

    # Tính tỷ trọng tối ưu y* vào danh mục rủi ro P (áp dụng ràng buộc không vay nợ 0 <= y* <= 1)
    y_raw = (ret_p - rf) / (A * var_p) if var_p > 0 else 0.0
    y_star = float(np.clip(y_raw, 0.0, 1.0))
    w_rf = 1.0 - y_star

    # Trọng số hiệu dụng trong tổng danh mục của nhà đầu tư
    eff_w_vcb = y_star * tan_p["w_VCB"]
    eff_w_dcm = y_star * tan_p["w_DCM"]

    ret_c = rf + y_star * (ret_p - rf)
    vol_c = y_star * vol_p
    var_c = vol_c ** 2
    utility_star = ret_c - 0.5 * A * var_c
    sharpe_c = (ret_c - rf) / vol_c if vol_c > 1e-12 else 0.0

    investor_opt = {
        "Risk_Aversion_A": A,
        "Raw_Risky_Share_y": y_raw,
        "Optimal_Risky_Allocation_y": y_star,
        "Risk_Free_Allocation": w_rf,
        "Effective_VCB_Weight": eff_w_vcb,
        "Effective_DCM_Weight": eff_w_dcm,
        "Expected_Annual_Return": ret_c,
        "Annualized_Volatility": vol_c,
        "Annualized_Variance": var_c,
        "Utility_Score": utility_star,
        "Sharpe_Ratio": sharpe_c
    }
    opt_results["Optimal_Investor_A2"] = {
        "w_VCB": eff_w_vcb,
        "w_DCM": eff_w_dcm,
        "w_Rf": w_rf,
        "Return": ret_c,
        "Vol": vol_c,
        "Var": var_c,
        "Sharpe": sharpe_c,
        "Utility": utility_star
    }

    # --- MODEL 9: Indifference Curves and Efficient Frontier ---
    plt.figure(figsize=(10.5, 6.5))
    upper_df = ef_df[ef_df["Branch"] == "Upper Efficient Branch"]
    lower_df = ef_df[ef_df["Branch"] == "Inefficient Lower Branch"]

    plt.plot(upper_df["Annualized_Volatility"], upper_df["Expected_Return"],
             color="#1f77b4", linewidth=2.4, label="Markowitz Efficient Frontier")
    plt.plot(lower_df["Annualized_Volatility"], lower_df["Expected_Return"],
             color="#aec7e8", linestyle="--", linewidth=1.6, label="Inefficient Frontier")

    sigma_grid = np.linspace(0.0, vol_p * 1.25, 250)
    plt.plot(sigma_grid, rf + sr_p * sigma_grid, color="crimson", linewidth=2.2, label="Capital Allocation Line (CAL)")

    # Vẽ 3 đường bàng quan (Indifference Curves): E(R) = U + 0.5 * A * sigma^2
    u_levels = [
        (utility_star - 0.025, "Lower Utility Curve (U₁)", "gray", ":"),
        (utility_star, f"Tangent Optimal Utility Curve (U* = {utility_star*100:.2f}%)", "darkorange", "-"),
        (utility_star + 0.025, "Higher Unattainable Utility Curve (U₂)", "purple", "--")
    ]
    for u_val, u_label, u_col, u_style in u_levels:
        ic_ret = u_val + 0.5 * A * (sigma_grid ** 2)
        plt.plot(sigma_grid, ic_ret, color=u_col, linestyle=u_style, linewidth=1.8, label=u_label)

    plt.scatter(0.0, rf, color="forestgreen", marker="s", s=110, edgecolors="black", zorder=6, label="Risk-Free Asset (Rf = 4.50%)")
    plt.scatter(vol_p, ret_p, color="crimson", marker="*", s=240, edgecolors="black", zorder=7, label="Tangency Portfolio P")
    plt.scatter(vol_c, ret_c, color="darkorange", marker="o", s=150, edgecolors="black", zorder=8,
                label=f"Optimal Investor Portfolio C (y*={y_star*100:.1f}%, Rf={w_rf*100:.1f}%)")

    plt.title("Model 9: Indifference Curves (A = 2) & Optimal Investor Portfolio", fontsize=13.5, weight="bold")
    plt.xlabel("Annualized Volatility (σ)", fontsize=11, weight="bold")
    plt.ylabel("Annualized Expected Return E(R)", fontsize=11, weight="bold")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlim(-0.01, vol_p * 1.25)
    plt.ylim(-0.12, 0.25)
    plt.legend(loc="lower right", frameon=True, fontsize=8.5)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_09_Indifference_Curves_Efficient_Frontier.png", dpi=300, bbox_inches="tight")
    plt.close()

    return investor_opt


# ==============================================================================
# SECTION 12: PORTFOLIO ASSET ALLOCATION MODEL (MODEL 10)
# ==============================================================================
def build_asset_allocation_model(opt_results: dict) -> pd.DataFrame:
    """
    Tổng hợp bảng phân bổ tài sản cho 5 trường hợp danh mục và vẽ Model 10 (Stacked Bar Chart).
    """
    print("[BƯỚC 9] Đang lập bảng Asset Allocation cho 5 danh mục và vẽ Model 10...")
    cases = [
        ("Portfolio P (Max Sharpe)", opt_results["Tangency_P"]),
        ("Target Return (14.53% Closest)", opt_results["Target_14_53"]),
        ("Global Min Variance (GMVP)", opt_results["GMVP"]),
        ("Equal Weight (50/50)", opt_results["Equal_Weight"]),
        ("Optimal Investor (A = 2)", opt_results["Optimal_Investor_A2"])
    ]

    rows = []
    for name, d in cases:
        rows.append({
            "Portfolio": name,
            "VCB Weight": d["w_VCB"],
            "DCM Weight": d["w_DCM"],
            "Risk-free Weight": d["w_Rf"],
            "Expected Return": d["Return"],
            "Annualized Risk": d["Vol"],
            "Sharpe Ratio": d["Sharpe"]
        })
    alloc_df = pd.DataFrame(rows)

    # --- MODEL 10: Asset Allocation Comparison Stacked Bar Chart ---
    fig, ax = plt.subplots(figsize=(10.5, 6))
    x = np.arange(len(alloc_df))
    w_vcb = alloc_df["VCB Weight"].values * 100
    w_dcm = alloc_df["DCM Weight"].values * 100
    w_rf = alloc_df["Risk-free Weight"].values * 100

    p1 = ax.bar(x, w_vcb, width=0.55, label="VCB Weight (%)", color="#1f77b4", edgecolor="black")
    p2 = ax.bar(x, w_dcm, width=0.55, bottom=w_vcb, label="DCM Weight (%)", color="#2ca02c", edgecolor="black")
    p3 = ax.bar(x, w_rf, width=0.55, bottom=w_vcb + w_dcm, label="Risk-Free Asset (%)", color="#ffbb78", edgecolor="black")

    # Hiển thị nhãn phần trăm ngay trên từng phân đoạn của cột
    for i in range(len(x)):
        if w_vcb[i] > 4:
            ax.text(x[i], w_vcb[i] / 2, f"{w_vcb[i]:.1f}%", ha="center", va="center", color="white", weight="bold", fontsize=9.5)
        if w_dcm[i] > 4:
            ax.text(x[i], w_vcb[i] + w_dcm[i] / 2, f"{w_dcm[i]:.1f}%", ha="center", va="center", color="white", weight="bold", fontsize=9.5)
        if w_rf[i] > 4:
            ax.text(x[i], w_vcb[i] + w_dcm[i] + w_rf[i] / 2, f"{w_rf[i]:.1f}%", ha="center", va="center", color="black", weight="bold", fontsize=9.5)

    ax.set_xticks(x)
    ax.set_xticklabels(alloc_df["Portfolio"], rotation=12, ha="right", fontsize=10, weight="bold")
    ax.set_ylabel("Allocation Percentage (%)", fontsize=11, weight="bold")
    ax.set_ylim(0, 112)
    ax.set_title("Model 10: Portfolio Asset Allocation Comparison Across 5 Strategies", fontsize=13.5, weight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_10_Asset_Allocation_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    return alloc_df


# ==============================================================================
# SECTION 13: CAPM, SML AND BETA LINEAR REGRESSION (MODELS 11, 12 & 13)
# ==============================================================================
def run_capm_and_beta_regression(simple_returns: pd.DataFrame, config: dict):
    """
    Thực hiện hồi quy tuyến tính OLS (statsmodels) cho VCB và DCM với VN-Index:
    (R_i - Rf_daily) = alpha_i + beta_i * (R_m - Rf_daily) + epsilon_i.
    Xây dựng mô hình CAPM và vẽ Model 11 (SML), Model 12 (Beta Regression Scatter), Model 13 (Beta Comparison).
    """
    print("[BƯỚC 10] Đang ước lượng mô hình CAPM, hồi quy OLS Beta và vẽ Model 11, 12, 13...")
    rf_ann = config["RISK_FREE_RATE"]
    ann_factor = config["TRADING_DAYS_PER_YEAR"]
    rf_daily = rf_ann / ann_factor

    mkt_ret = simple_returns["VNINDEX"]
    mkt_excess = mkt_ret - rf_daily
    exp_mkt_ann = float(mkt_ret.mean() * ann_factor)
    mkt_risk_premium = exp_mkt_ann - rf_ann

    capm_rows = []
    reg_rows = []
    models_dict = {}

    for ticker in ["VCB", "DCM"]:
        stock_ret = simple_returns[ticker]
        stock_excess = stock_ret - rf_daily
        hist_ann_ret = float(stock_ret.mean() * ann_factor)

        X = sm.add_constant(mkt_excess)
        ols_model = sm.OLS(stock_excess, X).fit()
        models_dict[ticker] = ols_model

        alpha_daily = float(ols_model.params["const"])
        beta_val = float(ols_model.params["VNINDEX"])
        r2 = float(ols_model.rsquared)
        adj_r2 = float(ols_model.rsquared_adj)
        p_alpha = float(ols_model.pvalues["const"])
        p_beta = float(ols_model.pvalues["VNINDEX"])
        se_alpha = float(ols_model.bse["const"])
        se_beta = float(ols_model.bse["VNINDEX"])
        ci_low, ci_high = ols_model.conf_int().loc["VNINDEX"]

        capm_req_ret = rf_ann + beta_val * mkt_risk_premium
        alpha_diff_ann = hist_ann_ret - capm_req_ret

        capm_rows.append({
            "Ticker": ticker,
            "Company": config["STOCK_1_NAME"] if ticker == "VCB" else config["STOCK_2_NAME"],
            "Beta (β)": beta_val,
            "Risk-Free Rate (Rf)": rf_ann,
            "Expected Market Return E(Rm)": exp_mkt_ann,
            "Market Risk Premium (E(Rm)-Rf)": mkt_risk_premium,
            "CAPM Required Return E(Ri)": capm_req_ret,
            "Historical Annualized Return": hist_ann_ret,
            "Return Difference (Historical - CAPM)": alpha_diff_ann,
            "Position vs SML": "Above SML (Outperformed Required Return)" if alpha_diff_ann > 0 else "Below SML (Underperformed Required Return)"
        })

        reg_rows.append({
            "Ticker": ticker,
            "Beta": beta_val,
            "Beta Standard Error": se_beta,
            "Beta P-value": p_beta,
            "Beta 95% CI Lower": float(ci_low),
            "Beta 95% CI Upper": float(ci_high),
            "Daily Alpha (α)": alpha_daily,
            "Annualized Alpha (α × 252)": alpha_daily * ann_factor,
            "Alpha Standard Error": se_alpha,
            "Alpha P-value": p_alpha,
            "R-squared": r2,
            "Adjusted R-squared": adj_r2,
            "Observations": int(ols_model.nobs)
        })

    capm_df = pd.DataFrame(capm_rows)
    reg_df = pd.DataFrame(reg_rows)

    # --- MODEL 11: Security Market Line (SML) ---
    plt.figure(figsize=(10, 6))
    beta_grid = np.linspace(0.0, 1.4, 200)
    sml_line = rf_ann + beta_grid * mkt_risk_premium
    plt.plot(beta_grid, sml_line, color="crimson", linewidth=2.4, label="Security Market Line (SML)")

    plt.scatter(0.0, rf_ann, color="forestgreen", marker="s", s=110, edgecolors="black", zorder=5, label=f"Risk-Free Asset (β=0, Rf={rf_ann*100:.2f}%)")
    plt.scatter(1.0, exp_mkt_ann, color="#ff7f0e", marker="D", s=120, edgecolors="black", zorder=5, label=f"VN-Index Market (β=1.0, E(Rm)={exp_mkt_ann*100:.2f}%)")

    colors = {"VCB": "#1f77b4", "DCM": "#2ca02c"}
    for _, row in capm_df.iterrows():
        tk = row["Ticker"]
        b = row["Beta (β)"]
        req_r = row["CAPM Required Return E(Ri)"]
        hist_r = row["Historical Annualized Return"]

        # Điểm nằm trên SML (CAPM Required Return) và điểm thực tế lịch sử (Historical Return)
        plt.scatter(b, req_r, color=colors[tk], marker="o", s=100, facecolors="none", edgecolors=colors[tk], linewidth=2.0, zorder=6,
                    label=f"{tk} CAPM Required ({req_r*100:.2f}%)")
        plt.scatter(b, hist_r, color=colors[tk], marker="o", s=130, edgecolors="black", zorder=7,
                    label=f"{tk} Historical Return ({hist_r*100:.2f}%)")
        plt.plot([b, b], [req_r, hist_r], color=colors[tk], linestyle=":", linewidth=1.6)

        plt.annotate(f"{tk} (β={b:.2f})\nHist: {hist_r*100:.2f}% | CAPM: {req_r*100:.2f}%",
                     xy=(b, hist_r), xytext=(12, -10 if hist_r < req_r else 10),
                     textcoords="offset points", fontsize=9, weight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=colors[tk], alpha=0.9))

    plt.title("Model 11: Security Market Line (SML) – CAPM Assessment", fontsize=13.5, weight="bold")
    plt.xlabel("Systematic Risk (Beta β)", fontsize=11, weight="bold")
    plt.ylabel("Annualized Return E(R)", fontsize=11, weight="bold")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlim(-0.05, 1.35)
    plt.ylim(-0.15, 0.22)
    plt.legend(loc="lower right", frameon=True, fontsize=8.5)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_11_Security_Market_Line.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- MODEL 12: Beta Regression Scatter Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for idx, ticker in enumerate(["VCB", "DCM"]):
        ax = axes[idx]
        stock_excess = simple_returns[ticker] - rf_daily
        row_reg = reg_df[reg_df["Ticker"] == ticker].iloc[0]
        b = row_reg["Beta"]
        a = row_reg["Daily Alpha (α)"]
        r2 = row_reg["R-squared"]
        pval = row_reg["Beta P-value"]

        ax.scatter(mkt_excess, stock_excess, alpha=0.45, color=colors[ticker], s=25, label="Daily Excess Returns")
        x_line = np.linspace(mkt_excess.min(), mkt_excess.max(), 100)
        y_line = a + b * x_line
        ax.plot(x_line, y_line, color="crimson", linewidth=2.2, label=f"OLS Line: y = {b:.4f}x + {a:.5f}")

        info_txt = (
            f"Stock: {ticker} vs VN-Index\n"
            f"Beta (β) = {b:.4f} (p = {pval:.2e})\n"
            f"Daily Alpha (α) = {a:.5f}\n"
            f"R² = {r2:.4f} ({r2*100:.2f}%)"
        )
        ax.text(0.04, 0.95, info_txt, transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9), fontsize=9.5, weight="bold")

        ax.axhline(0, color="black", linestyle=":", linewidth=1.0)
        ax.axvline(0, color="black", linestyle=":", linewidth=1.0)
        ax.set_title(f"Model 12: Beta OLS Regression – {ticker} vs VN-Index", fontsize=12.5, weight="bold")
        ax.set_xlabel("Market Excess Return (Rm - Rf)", fontsize=11, weight="bold")
        ax.set_ylabel(f"{ticker} Excess Return (Ri - Rf)", fontsize=11, weight="bold")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_12_Beta_Regression_Scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- MODEL 13: Beta Comparison Chart ---
    plt.figure(figsize=(8, 5.5))
    tickers_b = reg_df["Ticker"].tolist()
    betas = reg_df["Beta"].tolist()
    ci_errors = [
        [row["Beta"] - row["Beta 95% CI Lower"] for _, row in reg_df.iterrows()],
        [row["Beta 95% CI Upper"] - row["Beta"] for _, row in reg_df.iterrows()]
    ]

    bars = plt.bar(tickers_b, betas, width=0.45, color=["#1f77b4", "#2ca02c"], edgecolor="black", alpha=0.88,
                   yerr=ci_errors, capsize=8, error_kw={"elinewidth": 2, "ecolor": "black"}, label="Estimated Stock Beta (with 95% CI)")
    plt.axhline(1.0, color="crimson", linestyle="--", linewidth=2.0, label="Market Benchmark Beta (VN-Index β = 1.00)")

    for bar, b_val in zip(bars, betas):
        plt.text(bar.get_x() + bar.get_width() / 2, b_val + 0.14, f"β = {b_val:.4f}", ha="center", va="bottom", fontsize=10.5, weight="bold")

    plt.ylim(0, 1.35)
    plt.title("Model 13: Systematic Risk (Beta) Comparison – VCB vs DCM", fontsize=13, weight="bold")
    plt.xlabel("Stock Ticker", fontsize=11, weight="bold")
    plt.ylabel("Estimated Beta Coefficient (β)", fontsize=11, weight="bold")
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_13_Beta_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    return capm_df, reg_df


# ==============================================================================
# SECTION 14: OUT-OF-SAMPLE BACKTESTING (03 AUG 2026 – 30 OCT 2026, MODELS 14 & 15)
# ==============================================================================
def run_out_of_sample_backtest(backtest_df: pd.DataFrame, opt_results: dict, config: dict):
    """
    Thực hiện kiểm thử Out-of-Sample nghiêm ngặt từ 03/08/2026 đến 30/10/2026 với vốn 50,000,000 VND.
    So sánh 3 chiến lược:
    1. Strategy 1 – Portfolio P (Dùng trọng số tối ưu hóa từ giai đoạn Training: 0% VCB, 100% DCM).
    2. Strategy 2 – Equal Weight Portfolio (50% VCB, 50% DCM - Buy & Hold, để trọng số tự trôi theo giá).
    3. Strategy 3 – Savings Account (Lãi suất 4.5%/năm tính theo số ngày thực tế nắm giữ Actual/365).
    Đồng thời cung cấp bảng đối chiếu số lượng cổ phiếu thực tế khớp lệnh theo lô chẵn 100 cổ phiếu sàn HOSE.
    """
    print("[BƯỚC 11] Đang chạy Out-of-Sample Backtesting (03/08/2026 - 30/10/2026) và vẽ Model 14, 15...")
    capital = config["INITIAL_CAPITAL"]
    rf_ann = config["RISK_FREE_RATE"]
    day_base = config["SAVINGS_DAY_COUNT_BASE"]
    ann_factor = config["TRADING_DAYS_PER_YEAR"]
    lot_size = config["LOT_SIZE"]

    bt_prices = backtest_df[["VCB", "DCM"]].copy()
    bt_unadj = backtest_df[["VCB_Unadj", "DCM_Unadj"]].copy()
    entry_date = bt_prices.index[0]
    exit_date = bt_prices.index[-1]

    # Trọng số ban đầu từ tập Training (Không dùng dữ liệu tương lai)
    w_p = np.array([opt_results["Tangency_P"]["w_VCB"], opt_results["Tangency_P"]["w_DCM"]])
    w_eq = np.array([0.5, 0.5])

    # 1. Định giá Buy-and-Hold liên tục (Synthetic Total-Return Buy-and-Hold Valuation)
    # Giá trị danh mục ngày t: V_t = Capital * sum(w_i * (P_{i,t} / P_{i,0}))
    rel_prices = bt_prices / bt_prices.iloc[0]
    val_port_p = capital * (w_p[0] * rel_prices["VCB"] + w_p[1] * rel_prices["DCM"])
    val_equal = capital * (w_eq[0] * rel_prices["VCB"] + w_eq[1] * rel_prices["DCM"])

    # 2. Định giá Tài khoản tiết kiệm (Savings Account: Actual/365 simple daily accrual over calendar days)
    calendar_days = (bt_prices.index - entry_date).days
    val_savings = pd.Series(
        capital * (1.0 + rf_ann * (calendar_days / day_base)),
        index=bt_prices.index,
        name="Savings_Account"
    )

    bt_daily_values = pd.DataFrame({
        "Portfolio_P_Value": val_port_p,
        "Equal_Weight_Value": val_equal,
        "Savings_Account_Value": val_savings
    })
    bt_daily_returns = bt_daily_values.pct_change().dropna()

    # Tính Drawdown hàng ngày cho các chiến lược
    dd_port_p = (val_port_p - val_port_p.cummax()) / val_port_p.cummax()
    dd_equal = (val_equal - val_equal.cummax()) / val_equal.cummax()
    bt_daily_values["Portfolio_P_Drawdown"] = dd_port_p
    bt_daily_values["Equal_Weight_Drawdown"] = dd_equal

    # 3. Kiểm tra thực thi khớp lệnh thực tế theo lô 100 cổ phiếu sàn HOSE (Executable Lot-Size Audit)
    p0_vcb = float(bt_unadj["VCB_Unadj"].iloc[0])
    p0_dcm = float(bt_unadj["DCM_Unadj"].iloc[0])
    p1_vcb = float(bt_unadj["VCB_Unadj"].iloc[-1])
    p1_dcm = float(bt_unadj["DCM_Unadj"].iloc[-1])

    # Số cổ phiếu lô chẵn 100 cho Portfolio P
    sh_p_vcb = int((capital * w_p[0]) // (p0_vcb * lot_size)) * lot_size
    sh_p_dcm = int((capital * w_p[1]) // (p0_dcm * lot_size)) * lot_size
    cash_p = capital - (sh_p_vcb * p0_vcb + sh_p_dcm * p0_dcm)
    exec_end_val_p = sh_p_vcb * p1_vcb + sh_p_dcm * p1_dcm + cash_p

    # Số cổ phiếu lô chẵn 100 cho Equal Weight
    sh_eq_vcb = int((capital * 0.5) // (p0_vcb * lot_size)) * lot_size
    sh_eq_dcm = int((capital * 0.5) // (p0_dcm * lot_size)) * lot_size
    cash_eq = capital - (sh_eq_vcb * p0_vcb + sh_eq_dcm * p0_dcm)
    exec_end_val_eq = sh_eq_vcb * p1_vcb + sh_eq_dcm * p1_dcm + cash_eq

    n_bt_returns = len(bt_daily_returns)
    total_cal_days = (exit_date - entry_date).days

    def summarize_strategy(name, alloc_str, val_series, ret_series, is_savings=False):
        final_val = float(val_series.iloc[-1])
        tot_ret = (final_val / capital) - 1.0
        pnl = final_val - capital
        if is_savings:
            ann_ret = rf_ann
            ann_vol = 0.0
            mdd = 0.0
            sharpe = 0.0
        else:
            ann_ret = float((1.0 + tot_ret) ** (ann_factor / n_bt_returns) - 1.0)
            ann_vol = float(ret_series.std(ddof=1) * np.sqrt(ann_factor))
            mdd = compute_max_drawdown(val_series)
            sharpe = (ann_ret - rf_ann) / ann_vol if ann_vol > 1e-12 else 0.0
        return {
            "Strategy": name,
            "Initial Capital (VND)": capital,
            "Initial Allocation": alloc_str,
            "Total Return (%)": tot_ret,
            "Final Portfolio Value (VND)": final_val,
            "Profit / Loss (VND)": pnl,
            "Annualized Return (%)": ann_ret,
            "Annualized Volatility (%)": ann_vol,
            "Maximum Drawdown (%)": mdd,
            "Out-of-Sample Sharpe Ratio": sharpe
        }

    bt_summary_rows = [
        summarize_strategy(
            "Portfolio P (Tangency)",
            f"{w_p[0]*100:.1f}% VCB / {w_p[1]*100:.1f}% DCM",
            val_port_p,
            bt_daily_returns["Portfolio_P_Value"]
        ),
        summarize_strategy(
            "Equal Weight (50/50)",
            "50.0% VCB / 50.0% DCM",
            val_equal,
            bt_daily_returns["Equal_Weight_Value"]
        ),
        summarize_strategy(
            "Savings Account (4.5% p.a.)",
            "100.0% Savings Deposit",
            val_savings,
            bt_daily_returns["Savings_Account_Value"],
            is_savings=True
        )
    ]
    bt_summary_df = pd.DataFrame(bt_summary_rows).set_index("Strategy").T

    lot_execution_df = pd.DataFrame([
        {
            "Strategy": "Portfolio P (100-Share Lot Implementation)",
            "Entry VCB Price (VND)": p0_vcb,
            "Entry DCM Price (VND)": p0_dcm,
            "Executable VCB Shares": sh_p_vcb,
            "Executable DCM Shares": sh_p_dcm,
            "Residual Cash (VND)": cash_p,
            "Exit VCB Price (VND)": p1_vcb,
            "Exit DCM Price (VND)": p1_dcm,
            "Ending Portfolio Value (VND)": exec_end_val_p,
            "Total Return (%)": (exec_end_val_p / capital) - 1.0
        },
        {
            "Strategy": "Equal Weight (100-Share Lot Implementation)",
            "Entry VCB Price (VND)": p0_vcb,
            "Entry DCM Price (VND)": p0_dcm,
            "Executable VCB Shares": sh_eq_vcb,
            "Executable DCM Shares": sh_eq_dcm,
            "Residual Cash (VND)": cash_eq,
            "Exit VCB Price (VND)": p1_vcb,
            "Exit DCM Price (VND)": p1_dcm,
            "Ending Portfolio Value (VND)": exec_end_val_eq,
            "Total Return (%)": (exec_end_val_eq / capital) - 1.0
        }
    ])

    # --- MODEL 14: Out-of-Sample Backtest Performance ---
    plt.figure(figsize=(11, 6))
    plt.plot(bt_daily_values.index, bt_daily_values["Portfolio_P_Value"], color="#2ca02c", linewidth=2.4,
             label=f"Portfolio P (Final: {val_port_p.iloc[-1]:,.0f} VND | {((val_port_p.iloc[-1]/capital)-1)*100:+.2f}%)")
    plt.plot(bt_daily_values.index, bt_daily_values["Equal_Weight_Value"], color="#1f77b4", linewidth=2.2,
             label=f"Equal Weight 50/50 (Final: {val_equal.iloc[-1]:,.0f} VND | {((val_equal.iloc[-1]/capital)-1)*100:+.2f}%)")
    plt.plot(bt_daily_values.index, bt_daily_values["Savings_Account_Value"], color="crimson", linestyle="--", linewidth=2.0,
             label=f"Savings Benchmark 4.5% (Final: {val_savings.iloc[-1]:,.0f} VND | {((val_savings.iloc[-1]/capital)-1)*100:+.2f}%)")

    plt.axhline(capital, color="black", linestyle=":", linewidth=1.0, label="Initial Capital (50,000,000 VND)")
    plt.title(f"Model 14: Out-of-Sample Backtest Performance ({entry_date.date()} to {exit_date.date()})", fontsize=13.5, weight="bold")
    plt.xlabel("Trading Date", fontsize=11, weight="bold")
    plt.ylabel("Portfolio Value (VND)", fontsize=11, weight="bold")
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    plt.legend(loc="upper left", frameon=True, fontsize=9.5)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_14_Out_Of_Sample_Backtest.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- MODEL 15: Backtest Drawdown Chart ---
    plt.figure(figsize=(11, 5.5))
    plt.plot(bt_daily_values.index, dd_port_p * 100, color="#2ca02c", linewidth=2.0,
             label=f"Portfolio P Drawdown (Max DD: {dd_port_p.min()*100:.2f}%)")
    plt.fill_between(bt_daily_values.index, dd_port_p * 100, 0, color="#2ca02c", alpha=0.18)

    plt.plot(bt_daily_values.index, dd_equal * 100, color="#1f77b4", linewidth=2.0,
             label=f"Equal Weight 50/50 Drawdown (Max DD: {dd_equal.min()*100:.2f}%)")
    plt.fill_between(bt_daily_values.index, dd_equal * 100, 0, color="#1f77b4", alpha=0.18)

    plt.axhline(0, color="black", linestyle="-", linewidth=1.0)
    plt.title(f"Model 15: Out-of-Sample Portfolio Drawdown Comparison ({entry_date.date()} to {exit_date.date()})", fontsize=13.5, weight="bold")
    plt.xlabel("Trading Date", fontsize=11, weight="bold")
    plt.ylabel("Drawdown (%)", fontsize=11, weight="bold")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_15_Backtest_Drawdown.png", dpi=300, bbox_inches="tight")
    plt.close()

    bt_meta = {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "calendar_days": total_cal_days,
        "trading_sessions": len(bt_prices),
        "ending_value_port_p": float(val_port_p.iloc[-1]),
        "ending_value_equal": float(val_equal.iloc[-1]),
        "ending_value_savings": float(val_savings.iloc[-1]),
        "w_p_drifted_vcb": float(w_p[0] * rel_prices["VCB"].iloc[-1] / (val_port_p.iloc[-1] / capital)),
        "w_p_drifted_dcm": float(w_p[1] * rel_prices["DCM"].iloc[-1] / (val_port_p.iloc[-1] / capital)),
        "sh_p_vcb": sh_p_vcb,
        "sh_p_dcm": sh_p_dcm,
        "cash_p": cash_p,
        "p1_vcb": p1_vcb,
        "p1_dcm": p1_dcm,
        "exec_end_val_p": exec_end_val_p
    }
    return bt_summary_df, bt_daily_values, lot_execution_df, bt_meta


# ==============================================================================
# SECTION 15: PORTFOLIO REBALANCING AS OF 30 OCTOBER 2026 (MODEL 16)
# ==============================================================================
def run_portfolio_rebalancing(full_rebalance_df: pd.DataFrame, opt_results: dict, bt_meta: dict, config: dict):
    """
    Thực hiện phân tích Tái cân bằng danh mục (Portfolio Rebalancing) tại ngày 30/10/2026
    sử dụng toàn bộ dữ liệu cập nhật từ 31/07/2023 đến 30/10/2026.
    Ước lượng lại kỳ vọng lợi suất, ma trận hiệp phương sai, Beta, Tangency Portfolio mới
    và Optimal Investor Portfolio mới (A = 2).
    Vẽ Model 16 (Before and After Rebalancing).
    """
    print("[BƯỚC 12] Đang thực hiện phân tích Rebalancing tại ngày 30/10/2026 và vẽ Model 16...")
    ann_factor = config["TRADING_DAYS_PER_YEAR"]
    rf = config["RISK_FREE_RATE"]
    A = config["RISK_AVERSION_A"]
    lot_size = config["LOT_SIZE"]

    reb_ret, _ = calculate_returns(full_rebalance_df)
    tickers = ["VCB", "DCM"]
    mu_new = reb_ret[tickers].mean().values * ann_factor
    cov_new = reb_ret[tickers].cov().values * ann_factor
    vol_new = np.sqrt(np.diag(cov_new))

    # Ước lượng lại Beta trên tập dữ liệu mở rộng (31/07/2023 - 30/10/2026)
    rf_daily = rf / ann_factor
    mkt_ex = reb_ret["VNINDEX"] - rf_daily
    betas_new = {}
    for tk in tickers:
        ols_m = sm.OLS(reb_ret[tk] - rf_daily, sm.add_constant(mkt_ex)).fit()
        betas_new[tk] = float(ols_m.params["VNINDEX"])

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # 1. Tối ưu lại Tangency Portfolio P_new (Max Sharpe Ratio)
    res_tan_new = minimize(
        lambda w: -portfolio_performance(w, mu_new, cov_new, rf)[3],
        x0=np.array([0.5, 0.5]),
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint],
        options={"ftol": 1e-12, "maxiter": 500}
    )
    w_tan_new = res_tan_new.x
    ret_tan_new, vol_tan_new, var_tan_new, sr_tan_new = portfolio_performance(w_tan_new, mu_new, cov_new, rf)

    # 2. Tối ưu lại Risky Portfolio bằng cực đại hóa hàm hữu dụng (Mean-Variance Utility Maximization, A=2)
    res_util_risky = minimize(
        lambda w: -(portfolio_performance(w, mu_new, cov_new, rf)[0] - 0.5 * A * portfolio_performance(w, mu_new, cov_new, rf)[2]),
        x0=np.array([0.5, 0.5]),
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_constraint],
        options={"ftol": 1e-12, "maxiter": 500}
    )
    w_risky_util = res_util_risky.x
    ret_ru, vol_ru, var_ru, sr_ru = portfolio_performance(w_risky_util, mu_new, cov_new, rf)

    # 3. Phân bổ tổng danh mục cập nhật của nhà đầu tư (kết hợp tài sản phi rủi ro Rf với A = 2)
    y_new_raw = (ret_tan_new - rf) / (A * var_tan_new) if var_tan_new > 0 else 0.0
    y_new_star = float(np.clip(y_new_raw, 0.0, 1.0))
    w_rf_new = 1.0 - y_new_star
    eff_vcb_new = y_new_star * w_tan_new[0]
    eff_dcm_new = y_new_star * w_tan_new[1]
    ret_c_new = rf + y_new_star * (ret_tan_new - rf)
    vol_c_new = y_new_star * vol_tan_new
    util_c_new = ret_c_new - 0.5 * A * (vol_c_new ** 2)

    # Bảng so sánh thông số trước và sau cập nhật (31/07/2026 vs 30/10/2026)
    param_comp_df = pd.DataFrame([
        {
            "Parameter": "VCB Expected Annual Return",
            "Initial Training (Up to 31 Jul 2026)": float(opt_results["GMVP"]["Return"]) if False else float(reb_ret.loc[:"2026-07-31", "VCB"].mean() * ann_factor),
            "Updated Rebalance (Up to 30 Oct 2026)": float(mu_new[0])
        },
        {
            "Parameter": "DCM Expected Annual Return",
            "Initial Training (Up to 31 Jul 2026)": float(reb_ret.loc[:"2026-07-31", "DCM"].mean() * ann_factor),
            "Updated Rebalance (Up to 30 Oct 2026)": float(mu_new[1])
        },
        {
            "Parameter": "VCB Annualized Volatility",
            "Initial Training (Up to 31 Jul 2026)": float(reb_ret.loc[:"2026-07-31", "VCB"].std(ddof=1) * np.sqrt(ann_factor)),
            "Updated Rebalance (Up to 30 Oct 2026)": float(vol_new[0])
        },
        {
            "Parameter": "DCM Annualized Volatility",
            "Initial Training (Up to 31 Jul 2026)": float(reb_ret.loc[:"2026-07-31", "DCM"].std(ddof=1) * np.sqrt(ann_factor)),
            "Updated Rebalance (Up to 30 Oct 2026)": float(vol_new[1])
        },
        {
            "Parameter": "VCB Beta (β)",
            "Initial Training (Up to 31 Jul 2026)": 0.640203,
            "Updated Rebalance (Up to 30 Oct 2026)": betas_new["VCB"]
        },
        {
            "Parameter": "DCM Beta (β)",
            "Initial Training (Up to 31 Jul 2026)": 0.930720,
            "Updated Rebalance (Up to 30 Oct 2026)": betas_new["DCM"]
        },
        {
            "Parameter": "Tangency Portfolio P Sharpe Ratio",
            "Initial Training (Up to 31 Jul 2026)": opt_results["Tangency_P"]["Sharpe"],
            "Updated Rebalance (Up to 30 Oct 2026)": sr_tan_new
        },
        {
            "Parameter": "Optimal Risky Allocation y* (A = 2)",
            "Initial Training (Up to 31 Jul 2026)": opt_results["Optimal_Investor_A2"]["w_DCM"],
            "Updated Rebalance (Up to 30 Oct 2026)": y_new_star
        }
    ])

    # Bảng giao dịch Rebalancing dựa trên giá trị kết thúc Backtest của Portfolio P tại ngày 30/10/2026
    current_total_val = bt_meta["ending_value_port_p"]
    cur_w_vcb = bt_meta["w_p_drifted_vcb"]
    cur_w_dcm = bt_meta["w_p_drifted_dcm"]
    cur_w_rf = 0.0

    # So sánh chuyển đổi từ Portfolio P hiện hữu sang Updated Optimal Investor Portfolio (A = 2)
    cur_val_vcb = current_total_val * cur_w_vcb
    cur_val_dcm = current_total_val * cur_w_dcm
    cur_val_rf = current_total_val * cur_w_rf

    tgt_val_vcb = current_total_val * eff_vcb_new
    tgt_val_dcm = current_total_val * eff_dcm_new
    tgt_val_rf = current_total_val * w_rf_new

    rebalance_trade_df = pd.DataFrame([
        {
            "Asset": "VCB",
            "Current Weight": cur_w_vcb,
            "Target Weight (Updated A=2)": eff_vcb_new,
            "Target Weight (Pure Risky P_new)": float(w_tan_new[0]),
            "Current Value (VND)": cur_val_vcb,
            "Target Value (VND)": tgt_val_vcb,
            "Buy / Sell Amount (VND)": tgt_val_vcb - cur_val_vcb
        },
        {
            "Asset": "DCM",
            "Current Weight": cur_w_dcm,
            "Target Weight (Updated A=2)": eff_dcm_new,
            "Target Weight (Pure Risky P_new)": float(w_tan_new[1]),
            "Current Value (VND)": cur_val_dcm,
            "Target Value (VND)": tgt_val_dcm,
            "Buy / Sell Amount (VND)": tgt_val_dcm - cur_val_dcm
        },
        {
            "Asset": "Risk-free Asset",
            "Current Weight": cur_w_rf,
            "Target Weight (Updated A=2)": w_rf_new,
            "Target Weight (Pure Risky P_new)": 0.0,
            "Current Value (VND)": cur_val_rf,
            "Target Value (VND)": tgt_val_rf,
            "Buy / Sell Amount (VND)": tgt_val_rf - cur_val_rf
        },
        {
            "Asset": "Total",
            "Current Weight": 1.0,
            "Target Weight (Updated A=2)": 1.0,
            "Target Weight (Pure Risky P_new)": 1.0,
            "Current Value (VND)": current_total_val,
            "Target Value (VND)": current_total_val,
            "Buy / Sell Amount (VND)": 0.0
        }
    ])

    # Tính toán giao dịch cổ phiếu thực tế theo lô 100 cổ phiếu tại ngày 30/10/2026
    exec_total_val = bt_meta["exec_end_val_p"]
    cur_sh_vcb = bt_meta["sh_p_vcb"]
    cur_sh_dcm = bt_meta["sh_p_dcm"]
    p1_vcb = bt_meta["p1_vcb"]
    p1_dcm = bt_meta["p1_dcm"]

    tgt_sh_vcb = int((exec_total_val * eff_vcb_new) // (p1_vcb * lot_size)) * lot_size
    tgt_sh_dcm = int((exec_total_val * eff_dcm_new) // (p1_dcm * lot_size)) * lot_size
    new_rf_cash = exec_total_val - (tgt_sh_vcb * p1_vcb + tgt_sh_dcm * p1_dcm)

    share_rebalance_df = pd.DataFrame([
        {
            "Stock": "VCB",
            "Price on 30 Oct 2026 (VND)": p1_vcb,
            "Current Shares": cur_sh_vcb,
            "Target Shares (Updated A=2)": tgt_sh_vcb,
            "Shares to Buy (+) / Sell (-)": tgt_sh_vcb - cur_sh_vcb,
            "Trade Value (VND)": (tgt_sh_vcb - cur_sh_vcb) * p1_vcb
        },
        {
            "Stock": "DCM",
            "Price on 30 Oct 2026 (VND)": p1_dcm,
            "Current Shares": cur_sh_dcm,
            "Target Shares (Updated A=2)": tgt_sh_dcm,
            "Shares to Buy (+) / Sell (-)": tgt_sh_dcm - cur_sh_dcm,
            "Trade Value (VND)": (tgt_sh_dcm - cur_sh_dcm) * p1_dcm
        },
        {
            "Stock": "Risk-Free Asset & Residual Cash",
            "Price on 30 Oct 2026 (VND)": 1.0,
            "Current Shares": int(bt_meta["cash_p"]),
            "Target Shares (Updated A=2)": int(new_rf_cash),
            "Shares to Buy (+) / Sell (-)": int(new_rf_cash - bt_meta["cash_p"]),
            "Trade Value (VND)": new_rf_cash - bt_meta["cash_p"]
        }
    ])

    # --- MODEL 16: Before and After Rebalancing Stacked Bar Chart ---
    labels_reb = [
        "Original Portfolio P\n(31 Jul 2026)",
        "Updated Risky Portfolio P_new\n(30 Oct 2026)",
        "Initial Optimal Investor (A=2)\n(31 Jul 2026)",
        "Rebalanced Optimal Investor (A=2)\n(30 Oct 2026)"
    ]
    vcb_vals = np.array([
        opt_results["Tangency_P"]["w_VCB"],
        w_tan_new[0],
        opt_results["Optimal_Investor_A2"]["w_VCB"],
        eff_vcb_new
    ]) * 100
    dcm_vals = np.array([
        opt_results["Tangency_P"]["w_DCM"],
        w_tan_new[1],
        opt_results["Optimal_Investor_A2"]["w_DCM"],
        eff_dcm_new
    ]) * 100
    rf_vals = np.array([
        0.0,
        0.0,
        opt_results["Optimal_Investor_A2"]["w_Rf"],
        w_rf_new
    ]) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels_reb))
    ax.bar(x, vcb_vals, width=0.5, label="VCB Weight (%)", color="#1f77b4", edgecolor="black")
    ax.bar(x, dcm_vals, width=0.5, bottom=vcb_vals, label="DCM Weight (%)", color="#2ca02c", edgecolor="black")
    ax.bar(x, rf_vals, width=0.5, bottom=vcb_vals + dcm_vals, label="Risk-Free Asset (%)", color="#ffbb78", edgecolor="black")

    for i in range(len(x)):
        if vcb_vals[i] > 4:
            ax.text(x[i], vcb_vals[i] / 2, f"{vcb_vals[i]:.1f}%", ha="center", va="center", color="white", weight="bold")
        if dcm_vals[i] > 4:
            ax.text(x[i], vcb_vals[i] + dcm_vals[i] / 2, f"{dcm_vals[i]:.1f}%", ha="center", va="center", color="white", weight="bold")
        if rf_vals[i] > 4:
            ax.text(x[i], vcb_vals[i] + dcm_vals[i] + rf_vals[i] / 2, f"{rf_vals[i]:.1f}%", ha="center", va="center", color="black", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_reb, fontsize=10, weight="bold")
    ax.set_ylabel("Portfolio Weight (%)", fontsize=11, weight="bold")
    ax.set_ylim(0, 112)
    ax.set_title("Model 16: Portfolio Allocation Before and After Rebalancing (As of 30 Oct 2026)", fontsize=13, weight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Model_16_Before_After_Rebalancing.png", dpi=300, bbox_inches="tight")
    plt.close()

    reb_summary = {
        "w_tan_new": w_tan_new,
        "ret_tan_new": ret_tan_new,
        "vol_tan_new": vol_tan_new,
        "sr_tan_new": sr_tan_new,
        "w_risky_util": w_risky_util,
        "ret_ru": ret_ru,
        "vol_ru": vol_ru,
        "y_new_star": y_new_star,
        "w_rf_new": w_rf_new,
        "eff_vcb_new": eff_vcb_new,
        "eff_dcm_new": eff_dcm_new,
        "ret_c_new": ret_c_new,
        "vol_c_new": vol_c_new,
        "util_c_new": util_c_new,
        "current_total_val": current_total_val
    }
    return param_comp_df, rebalance_trade_df, share_rebalance_df, reb_summary


# ==============================================================================
# SECTION 16: FINAL CLIENT INVESTMENT RECOMMENDATION (KHUYẾN NGHỊ ĐẦU TƯ)
# ==============================================================================
def build_final_recommendation(reb_summary: dict, opt_results: dict, config: dict) -> pd.DataFrame:
    """
    Tổng hợp bảng khuyến nghị phân bổ vốn cuối cùng cho khách hàng (A = 2) trên cả hai cơ sở vốn:
    - Cơ sở vốn khởi điểm: 50,000,000 VND.
    - Cơ sở vốn khả dụng cập nhật sau Backtest (30/10/2026): ~53,623,188 VND.
    """
    init_cap = config["INITIAL_CAPITAL"]
    upd_cap = reb_summary["current_total_val"]
    w_vcb = reb_summary["eff_vcb_new"]
    w_dcm = reb_summary["eff_dcm_new"]
    w_rf = reb_summary["w_rf_new"]

    rec_df = pd.DataFrame([
        {
            "Asset": "VCB (Vietcombank)",
            "Recommended Weight (Updated A=2)": w_vcb,
            "Allocation on Initial Capital 50M (VND)": w_vcb * init_cap,
            "Allocation on Updated Capital 30 Oct 2026 (VND)": w_vcb * upd_cap
        },
        {
            "Asset": "DCM (PetroVietnam Ca Mau Fertilizer)",
            "Recommended Weight (Updated A=2)": w_dcm,
            "Allocation on Initial Capital 50M (VND)": w_dcm * init_cap,
            "Allocation on Updated Capital 30 Oct 2026 (VND)": w_dcm * upd_cap
        },
        {
            "Asset": "Risk-Free Asset (Savings / Money Market 4.5%)",
            "Recommended Weight (Updated A=2)": w_rf,
            "Allocation on Initial Capital 50M (VND)": w_rf * init_cap,
            "Allocation on Updated Capital 30 Oct 2026 (VND)": w_rf * upd_cap
        },
        {
            "Asset": "Total Client Portfolio",
            "Recommended Weight (Updated A=2)": 1.0,
            "Allocation on Initial Capital 50M (VND)": init_cap,
            "Allocation on Updated Capital 30 Oct 2026 (VND)": upd_cap
        }
    ])
    return rec_df


# ==============================================================================
# SECTION 17: EXPORT RESULTS TO 17-SHEET EXCEL WORKBOOK (XUẤT BÁO CÁO EXCEL)
# ==============================================================================
def export_to_excel(
    raw_df: pd.DataFrame,
    simple_ret: pd.DataFrame,
    log_ret: pd.DataFrame,
    risk_return_df: pd.DataFrame,
    var_df: pd.DataFrame,
    daily_cov: pd.DataFrame,
    ann_cov: pd.DataFrame,
    corr_mat: pd.DataFrame,
    ef_df: pd.DataFrame,
    opt_results: dict,
    investor_opt: dict,
    alloc_df: pd.DataFrame,
    capm_df: pd.DataFrame,
    reg_df: pd.DataFrame,
    bt_summary_df: pd.DataFrame,
    bt_daily_values: pd.DataFrame,
    lot_exec_df: pd.DataFrame,
    param_comp_df: pd.DataFrame,
    rebalance_trade_df: pd.DataFrame,
    share_rebalance_df: pd.DataFrame,
    rec_df: pd.DataFrame
):
    """
    Xuất toàn bộ dữ liệu và mô hình tính toán ra file Excel chuyên nghiệp gồm đúng 17 Sheets theo yêu cầu.
    """
    print(f"[BƯỚC 13] Đang xuất báo cáo Excel 17 Sheets tại: {EXCEL_REPORT_PATH}...")

    # Chuẩn bị bảng Return Distribution Statistics riêng cho Sheet 4
    dist_stats_df = risk_return_df.loc[
        ["Mean Daily Return", "Daily Volatility", "Minimum Daily Return", "Maximum Daily Return", "Skewness", "Excess Kurtosis"]
    ].copy()

    # Chuẩn bị bảng Portfolio Optimization cho Sheet 9
    port_opt_df = pd.DataFrame([
        {
            "Portfolio": "Global Minimum Variance Portfolio (GMVP)",
            "VCB Weight": opt_results["GMVP"]["w_VCB"],
            "DCM Weight": opt_results["GMVP"]["w_DCM"],
            "Expected Annual Return": opt_results["GMVP"]["Return"],
            "Annualized Volatility": opt_results["GMVP"]["Vol"],
            "Annualized Variance": opt_results["GMVP"]["Var"],
            "Sharpe Ratio": opt_results["GMVP"]["Sharpe"],
            "Feasibility / Notes": "Feasible Global Minimum Risk"
        },
        {
            "Portfolio": "Tangency Portfolio P (Max Sharpe Ratio)",
            "VCB Weight": opt_results["Tangency_P"]["w_VCB"],
            "DCM Weight": opt_results["Tangency_P"]["w_DCM"],
            "Expected Annual Return": opt_results["Tangency_P"]["Return"],
            "Annualized Volatility": opt_results["Tangency_P"]["Vol"],
            "Annualized Variance": opt_results["Tangency_P"]["Var"],
            "Sharpe Ratio": opt_results["Tangency_P"]["Sharpe"],
            "Feasibility / Notes": "Optimal Risky Portfolio"
        },
        {
            "Portfolio": "Target Return Portfolio (14.53% Target)",
            "VCB Weight": opt_results["Target_14_53"]["w_VCB"],
            "DCM Weight": opt_results["Target_14_53"]["w_DCM"],
            "Expected Annual Return": opt_results["Target_14_53"]["Return"],
            "Annualized Volatility": opt_results["Target_14_53"]["Vol"],
            "Annualized Variance": opt_results["Target_14_53"]["Var"],
            "Sharpe Ratio": opt_results["Target_14_53"]["Sharpe"],
            "Feasibility / Notes": opt_results["Target_14_53"]["Status"]
        }
    ])

    # Chuẩn bị bảng CAL and Sharpe Ratio cho Sheet 10
    cal_grid = np.linspace(0.0, 1.0, 11)
    cal_df = pd.DataFrame({
        "Weight in Risky Portfolio P (y)": cal_grid,
        "Weight in Risk-Free Asset (1 - y)": 1.0 - cal_grid,
        "Expected Portfolio Return E(Rc)": CONFIG["RISK_FREE_RATE"] + cal_grid * (opt_results["Tangency_P"]["Return"] - CONFIG["RISK_FREE_RATE"]),
        "Portfolio Volatility (σc)": cal_grid * opt_results["Tangency_P"]["Vol"],
        "Sharpe Ratio": opt_results["Tangency_P"]["Sharpe"]
    })

    # Chuẩn bị bảng Covariance gộp cho Sheet 6
    cov_combined = pd.concat(
        [daily_cov.assign(Scale="Daily Covariance"), ann_cov.assign(Scale="Annualized Covariance (×252)")]
    )

    # Chuẩn bị bảng Daily Returns gộp cho Sheet 2
    returns_combined = simple_ret.rename(columns=lambda c: f"{c}_Simple_Return").join(
        log_ret.rename(columns=lambda c: f"{c}_Log_Return")
    )

    with pd.ExcelWriter(EXCEL_REPORT_PATH, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw Data")
        returns_combined.to_excel(writer, sheet_name="Daily Returns")
        risk_return_df.to_excel(writer, sheet_name="Risk and Return")
        dist_stats_df.to_excel(writer, sheet_name="Return Distribution Statistics")
        var_df.to_excel(writer, sheet_name="VaR and CVaR")
        cov_combined.to_excel(writer, sheet_name="Covariance Matrix")
        corr_mat.to_excel(writer, sheet_name="Correlation Matrix")
        ef_df.to_excel(writer, sheet_name="Efficient Frontier", index=False)
        port_opt_df.to_excel(writer, sheet_name="Portfolio Optimization", index=False)
        cal_df.to_excel(writer, sheet_name="CAL and Sharpe Ratio", index=False)
        pd.DataFrame([investor_opt]).T.rename(columns={0: "Value"}).to_excel(writer, sheet_name="Optimal Investor Portfolio")
        alloc_df.to_excel(writer, sheet_name="Asset Allocation", index=False)
        capm_df.to_excel(writer, sheet_name="CAPM and SML", index=False)
        reg_df.to_excel(writer, sheet_name="Beta Regression", index=False)

        # Sheet 15: Backtest Results (Summary + Executable Lot Audit + Daily Series)
        bt_summary_df.to_excel(writer, sheet_name="Backtest Results", startrow=0)
        lot_exec_df.to_excel(writer, sheet_name="Backtest Results", startrow=len(bt_summary_df) + 3, index=False)
        bt_daily_values.to_excel(writer, sheet_name="Backtest Results", startrow=len(bt_summary_df) + len(lot_exec_df) + 7)

        # Sheet 16: Rebalancing (Parameter Comparison + Monetary Trade Table + Share Trade Table)
        param_comp_df.to_excel(writer, sheet_name="Rebalancing", startrow=0, index=False)
        rebalance_trade_df.to_excel(writer, sheet_name="Rebalancing", startrow=len(param_comp_df) + 3, index=False)
        share_rebalance_df.to_excel(writer, sheet_name="Rebalancing", startrow=len(param_comp_df) + len(rebalance_trade_df) + 6, index=False)

        # Sheet 17: Final Recommendation
        rec_df.to_excel(writer, sheet_name="Final Recommendation", index=False)

    print(" -> Đã lưu thành công toàn bộ 17 Sheets vào Workbook Excel.")


# ==============================================================================
# SECTION 18: MASTER EXECUTION PIPELINE & FINAL SUMMARY DISPLAY
# ==============================================================================
def main():
    print("=" * 95)
    print("QUANTITATIVE PORTFOLIO MANAGEMENT & OPTIMIZATION SYSTEM (QPMOS) – VCB & DCM")
    print("=" * 95)

    # 1. Thu thập và làm sạch dữ liệu
    raw_df = collect_market_data(CONFIG)
    aligned_df, train_df, backtest_df, full_rebalance_df = clean_and_validate_data(raw_df, CONFIG)

    # 2. Tính lợi suất trên tập Training (31/07/2023 - 31/07/2026)
    simple_ret_train, log_ret_train = calculate_returns(train_df)

    # 3. Phân tích giá lịch sử (Models 1 & 2)
    plot_historical_prices(train_df)

    # 4. Phân tích rủi ro & lợi suất (Models 3 & 4)
    risk_return_df = analyze_risk_and_return(train_df, simple_ret_train, CONFIG)

    # 5. Tính toán VaR & CVaR
    var_df = calculate_var_cvar(simple_ret_train, CONFIG)

    # 6. Ma trận Hiệp phương sai & Tương quan (Models 5 & 6)
    daily_cov, ann_cov, corr_mat = analyze_covariance_and_correlation(simple_ret_train, CONFIG)

    # 7. Tối ưu hóa danh mục Markowitz, Đường biên hiệu quả & CAL (Models 7 & 8)
    mu_series = risk_return_df.loc["Annualized Arithmetic Return"]
    opt_results, ef_df = run_markowitz_optimization(mu_series, ann_cov, CONFIG)

    # 8. Danh mục tối ưu của nhà đầu tư với A = 2 (Model 9)
    investor_opt = calculate_optimal_investor_portfolio(opt_results, ef_df, CONFIG)

    # 9. Bảng phân bổ tài sản 5 chiến lược (Model 10)
    alloc_df = build_asset_allocation_model(opt_results)

    # 10. Mô hình CAPM, SML & Hồi quy Beta OLS (Models 11, 12 & 13)
    capm_df, reg_df = run_capm_and_beta_regression(simple_ret_train, CONFIG)

    # 11. Kiểm thử Out-of-Sample Backtesting (03/08/2026 - 30/10/2026) (Models 14 & 15)
    bt_summary_df, bt_daily_values, lot_exec_df, bt_meta = run_out_of_sample_backtest(backtest_df, opt_results, CONFIG)

    # 12. Tái cân bằng danh mục tại ngày 30/10/2026 (Model 16)
    param_comp_df, rebalance_trade_df, share_rebalance_df, reb_summary = run_portfolio_rebalancing(
        full_rebalance_df, opt_results, bt_meta, CONFIG
    )

    # 13. Khuyến nghị đầu tư cuối cùng
    rec_df = build_final_recommendation(reb_summary, opt_results, CONFIG)

    # 14. Xuất toàn bộ kết quả ra Excel 17 Sheets
    export_to_excel(
        aligned_df, simple_ret_train, log_ret_train, risk_return_df, var_df,
        daily_cov, ann_cov, corr_mat, ef_df, opt_results, investor_opt,
        alloc_df, capm_df, reg_df, bt_summary_df, bt_daily_values,
        lot_exec_df, param_comp_df, rebalance_trade_df, share_rebalance_df, rec_df
    )

    # 15. Hiển thị Báo cáo Tổng kết Định lượng trên màn hình
    print("\n" + "=" * 95)
    print("EXECUTIVE SUMMARY – QUANTITATIVE PORTFOLIO MANAGEMENT (VCB & DCM)")
    print("=" * 95)
    print("\n1. RISK & RETURN COMPARISON (TRAINING: 31 JUL 2023 – 31 JUL 2026):")
    print(risk_return_df.round(4).to_string())

    print(f"\n2. CORRELATION COEFFICIENT (VCB vs DCM): {corr_mat.loc['VCB', 'DCM']:.4f}")

    print("\n3. VALUE AT RISK (VaR) & CVaR SUMMARY (INITIAL CAPITAL = 50,000,000 VND):")
    print(var_df.round(4).to_string())

    print("\n4. PORTFOLIO ASSET ALLOCATION SUMMARY (5 STRATEGIES):")
    print(alloc_df.round(4).to_string(index=False))

    print("\n5. TARGET RETURN 14.53% FEASIBILITY AUDIT:")
    print(f"   - Status: {opt_results['Target_14_53']['Status']}")
    print(f"   - Maximum Feasible Return (Long-Only): {opt_results['Target_14_53']['Max_Feasible_Return']*100:.2f}%")
    print(f"   - Closest Feasible Allocation: {opt_results['Target_14_53']['w_VCB']*100:.1f}% VCB / {opt_results['Target_14_53']['w_DCM']*100:.1f}% DCM")
    print(f"   - Return Shortfall vs Target: {opt_results['Target_14_53']['Return_Shortfall']*100:.2f}%")

    print("\n6. CAPM & BETA REGRESSION RESULTS:")
    print(capm_df[["Ticker", "Beta (β)", "CAPM Required Return E(Ri)", "Historical Annualized Return", "Return Difference (Historical - CAPM)", "Position vs SML"]].round(4).to_string(index=False))

    print("\n7. OUT-OF-SAMPLE BACKTEST RESULTS (03 AUG 2026 – 30 OCT 2026):")
    print(bt_summary_df.to_string())

    print("\n8. PORTFOLIO REBALANCING TABLE AS OF 30 OCTOBER 2026:")
    print(rebalance_trade_df.round(4).to_string(index=False))

    print("\n9. FINAL CLIENT INVESTMENT RECOMMENDATION (RISK AVERSION A = 2):")
    print(rec_df.round(4).to_string(index=False))
    print("=" * 95)
    print(f"HOÀN TẤT! Toàn bộ 16 biểu đồ PNG và file Excel 17 Sheets đã được lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
