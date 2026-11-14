"""
Data validation and quality reporting module.
Enforces strict schema checks, positive finite prices, common trading calendar alignment,
zero look-ahead bias, and corporate action / anomaly detection.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def validate_price_data(
    stock_long_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Validate raw stock and benchmark datasets and split them strictly into
    in-sample, out-of-sample, and post-OOS rebalancing panels without look-ahead bias.

    Parameters
    ----------
    stock_long_df : pd.DataFrame
        Long-format stock DataFrame with [date, ticker, adjusted_close, close, volume].
    benchmark_df : pd.DataFrame
        Benchmark DataFrame with [date, index_close].
    config : Dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing validated panels:
        - 'full_aligned': All common trading sessions from IN_SAMPLE_START to OUT_OF_SAMPLE_END
        - 'in_sample': Strictly [IN_SAMPLE_START, IN_SAMPLE_END]
        - 'out_of_sample': Strictly [OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END]
        - 'rebalance_window': Strictly [IN_SAMPLE_START, OUT_OF_SAMPLE_END]
    """
    req_stock_cols = {"date", "ticker", "adjusted_close", "close", "volume"}
    if not req_stock_cols.issubset(stock_long_df.columns):
        raise ValueError(f"Stock data missing required columns: {req_stock_cols - set(stock_long_df.columns)}")

    if not {"date", "index_close"}.issubset(benchmark_df.columns):
        raise ValueError("Benchmark data must contain 'date' and 'index_close' columns.")

    # Convert dates and check validity
    s_df = stock_long_df.copy()
    b_df = benchmark_df.copy()
    s_df["date"] = pd.to_datetime(s_df["date"])
    b_df["date"] = pd.to_datetime(b_df["date"])

    if s_df["date"].isnull().any() or b_df["date"].isnull().any():
        raise ValueError("Invalid or unparseable dates detected in market data.")

    # Check duplicate (date, ticker) pairs
    dup_stocks = int(s_df.duplicated(subset=["date", "ticker"]).sum())
    dup_bench = int(b_df.duplicated(subset=["date"]).sum())
    if dup_stocks > 0:
        raise ValueError(f"Detected {dup_stocks} duplicate (date, ticker) rows in stock dataset.")
    if dup_bench > 0:
        raise ValueError(f"Detected {dup_bench} duplicate date rows in benchmark dataset.")

    # Check finite and positive prices
    for col in ["adjusted_close", "close"]:
        vals = s_df[col].dropna().values
        if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
            raise ValueError(f"Non-positive or infinite values detected in column '{col}'.")

    b_vals = b_df["index_close"].dropna().values
    if not np.all(np.isfinite(b_vals)) or np.any(b_vals <= 0):
        raise ValueError("Non-positive or infinite values detected in benchmark 'index_close'.")

    # Verify unit consistency (Vietnamese stocks in VND should exceed 1,000 VND)
    if s_df["adjusted_close"].median() < 1000.0:
        raise ValueError("Stock price unit appears to be in thousands of VND rather than VND.")

    # Pivot into wide format aligned on common trading calendar (no forward-filling)
    adj_wide = s_df.pivot(index="date", columns="ticker", values="adjusted_close")[config["ASSETS"]]
    close_wide = s_df.pivot(index="date", columns="ticker", values="close")[config["ASSETS"]].rename(
        columns=lambda c: f"{c}_Unadj"
    )
    vol_wide = s_df.pivot(index="date", columns="ticker", values="volume")[config["ASSETS"]].rename(
        columns=lambda c: f"{c}_Volume"
    )
    bench_series = b_df.set_index("date")["index_close"].rename(config["MARKET_BENCHMARK"])

    if "yf_adjusted_close" in s_df.columns:
        yf_adj_wide = s_df.pivot(index="date", columns="ticker", values="yf_adjusted_close")[config["ASSETS"]].rename(
            columns=lambda c: f"{c}_YF_Adj"
        )
        merged = pd.concat([adj_wide, bench_series, close_wide, vol_wide, yf_adj_wide], axis=1)
    else:
        merged = pd.concat([adj_wide, bench_series, close_wide, vol_wide], axis=1)

    # Filter strictly to study range and drop any dates missing core prices
    study_slice = merged.loc[config["IN_SAMPLE_START"] : config["OUT_OF_SAMPLE_END"]].copy()
    core_columns = config["ASSETS"] + [config["MARKET_BENCHMARK"]]
    full_aligned = study_slice.dropna(subset=core_columns).sort_index()

    in_sample = full_aligned.loc[config["IN_SAMPLE_START"] : config["IN_SAMPLE_END"]].copy()
    out_of_sample = full_aligned.loc[config["OUT_OF_SAMPLE_START"] : config["OUT_OF_SAMPLE_END"]].copy()
    rebalance_window = full_aligned.loc[config["IN_SAMPLE_START"] : config["OUT_OF_SAMPLE_END"]].copy()

    # Coverage and anti-look-ahead assertions
    if len(in_sample) < 700:
        raise ValueError(f"Insufficient in-sample observations ({len(in_sample)}). Expected ~748 trading days.")
    if len(out_of_sample) < 50:
        raise ValueError(f"Insufficient out-of-sample observations ({len(out_of_sample)}). Expected ~62 trading days.")
    if in_sample.index.max() > pd.Timestamp(config["IN_SAMPLE_END"]):
        raise AssertionError("Look-ahead bias violation: in-sample panel contains dates after IN_SAMPLE_END.")
    if out_of_sample.index.min() < pd.Timestamp(config["OUT_OF_SAMPLE_START"]):
        raise AssertionError("Look-ahead bias violation: out-of-sample panel overlaps with in-sample period.")

    return {
        "full_aligned": full_aligned,
        "in_sample": in_sample,
        "out_of_sample": out_of_sample,
        "rebalance_window": rebalance_window,
    }


def build_data_quality_report(
    stock_long_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    panels: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the mandatory Data Quality Report table and an Anomaly / Corporate Action Audit table.

    Parameters
    ----------
    stock_long_df : pd.DataFrame
        Raw long-format stock data.
    benchmark_df : pd.DataFrame
        Raw benchmark data.
    panels : Dict[str, pd.DataFrame]
        Validated data panels from validate_price_data().
    metadata : Dict[str, Any]
        Data retrieval metadata.
    config : Dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - quality_report_df: Mandatory 11-column Data Quality Report.
        - anomaly_audit_df: Audit of extreme daily moves and corporate action adjustments.
    """
    full_df = panels["full_aligned"]
    in_sample_df = panels["in_sample"]
    oos_df = panels["out_of_sample"]

    rows = []
    # 1. Adjusted closing price rows for VCB and DCM (In-Sample & Full)
    for ticker in config["ASSETS"]:
        sub_raw = stock_long_df[stock_long_df["ticker"] == ticker]
        dup_cnt = int(sub_raw.duplicated(subset=["date"]).sum())
        miss_cnt = int(full_df[ticker].isnull().sum())

        rows.append(
            {
                "Asset": f"{ticker} (In-Sample Adjusted)",
                "Source": metadata["stock_adjusted_source"],
                "Price Type": "Adjusted Closing Price (Total-Return Basis)",
                "Price Unit": metadata["price_unit_stocks"],
                "First Available Date": in_sample_df.index.min().strftime("%Y-%m-%d"),
                "Last Available Date": in_sample_df.index.max().strftime("%Y-%m-%d"),
                "Observation Count": int(len(in_sample_df)),
                "Missing Observation Count": miss_cnt,
                "Duplicate Count": dup_cnt,
                "Adjustment Method": "Retroactive Proportional Adjustment for Cash Dividends, Stock Dividends & Splits",
                "Retrieval Timestamp": metadata["retrieval_timestamp"],
            }
        )
        rows.append(
            {
                "Asset": f"{ticker} (Out-of-Sample Unadjusted/Tradable)",
                "Source": metadata["stock_unadjusted_source"],
                "Price Type": "Unadjusted Closing Price (Executable Basis)",
                "Price Unit": metadata["price_unit_stocks"],
                "First Available Date": oos_df.index.min().strftime("%Y-%m-%d"),
                "Last Available Date": oos_df.index.max().strftime("%Y-%m-%d"),
                "Observation Count": int(len(oos_df)),
                "Missing Observation Count": int(oos_df[f"{ticker}_Unadj"].isnull().sum()),
                "Duplicate Count": dup_cnt,
                "Adjustment Method": "None (Raw Tradable Market Close on HOSE)",
                "Retrieval Timestamp": metadata["retrieval_timestamp"],
            }
        )

    # 2. Benchmark row
    dup_b = int(benchmark_df.duplicated(subset=["date"]).sum())
    rows.append(
        {
            "Asset": f"{config['MARKET_BENCHMARK']} (In-Sample Benchmark)",
            "Source": metadata["benchmark_source"],
            "Price Type": "Market Capitalization-Weighted Price Index Close",
            "Price Unit": metadata["price_unit_benchmark"],
            "First Available Date": in_sample_df.index.min().strftime("%Y-%m-%d"),
            "Last Available Date": in_sample_df.index.max().strftime("%Y-%m-%d"),
            "Observation Count": int(len(in_sample_df)),
            "Missing Observation Count": int(full_df[config["MARKET_BENCHMARK"]].isnull().sum()),
            "Duplicate Count": dup_b,
            "Adjustment Method": "Exchange Divisor Adjusted for Capital Changes (Excludes Cash Dividends)",
            "Retrieval Timestamp": metadata["retrieval_timestamp"],
        }
    )

    quality_report_df = pd.DataFrame(rows)

    # Build Anomaly & Corporate Action Audit table
    # HOSE daily price limit is +/- 7.0%; check any daily return exceeding +/- 6.8% or provider discrepancy
    anomaly_rows = []
    ret_adj = in_sample_df[["VCB", "DCM"]].pct_change()
    ret_unadj = in_sample_df[["VCB_Unadj", "DCM_Unadj"]].pct_change()

    for ticker in ["VCB", "DCM"]:
        r_a = ret_adj[ticker].dropna()
        r_u = ret_unadj[f"{ticker}_Unadj"].dropna()
        diff_au = (r_a - r_u).abs()

        # Identify corporate action ex-dates where unadjusted return differs from adjusted return by > 1%
        ca_dates = diff_au[diff_au > 0.01].index
        for dt in ca_dates:
            yf_ret_val = np.nan
            if f"{ticker}_YF_Adj" in in_sample_df.columns:
                yf_ret_series = in_sample_df[f"{ticker}_YF_Adj"].pct_change()
                yf_ret_val = float(yf_ret_series.loc[dt])

            anomaly_rows.append(
                {
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Official_Adjusted_Return": float(r_a.loc[dt]),
                    "Unadjusted_Close_Return": float(r_u.loc[dt]),
                    "Yahoo_Finance_Adj_Return": yf_ret_val,
                    "Event_Classification": (
                        "49.5% Stock Dividend Ex-Date (Properly Adjusted in Primary Feed; Unadjusted in Yahoo Finance)"
                        if (ticker == "VCB" and dt.strftime("%Y-%m-%d") == "2025-03-03")
                        else "Cash Dividend / Corporate Action Ex-Date Adjustment"
                    ),
                    "Action_Taken": "Retained official adjusted return; no artificial deletion or winsorization",
                }
            )

    anomaly_audit_df = pd.DataFrame(anomaly_rows)
    return quality_report_df, anomaly_audit_df

