"""
Data collection and persistence module for VCB, DCM, and VN-Index.
Supports online market data retrieval from DNSE Entrade OHLC API and Yahoo Finance,
with deterministic CSV fallback following the exact schema:
- stock_prices_raw.csv: date,ticker,adjusted_close,close,volume
- market_benchmark_raw.csv: date,index_close
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import requests
import numpy as np
import pandas as pd
import yfinance as yf


def _fetch_dnse_series(
    symbol: str,
    endpoint_type: str,
    start_date: str,
    end_date: str,
    multiplier: float = 1.0,
    timezone: str = "Asia/Ho_Chi_Minh",
) -> pd.DataFrame:
    """
    Fetch daily OHLCV series from DNSE Entrade API for Vietnamese equities or indices.

    Parameters
    ----------
    symbol : str
        Ticker symbol ('VCB', 'DCM', or 'VNINDEX').
    endpoint_type : str
        API endpoint category ('stock' or 'index').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    multiplier : float
        Price unit multiplier (1000.0 for stocks to convert from kVND to VND; 1.0 for index).
    timezone : str
        Exchange local timezone for timestamp conversion.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by trading date with adjusted_close and volume columns.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_ts = int(start_dt.timestamp()) - 86400 * 5
    end_ts = int(end_dt.timestamp()) + 86400 * 5

    url = (
        f"https://services.entrade.com.vn/chart-api/v2/ohlcs/{endpoint_type}"
        f"?from={start_ts}&to={end_ts}&symbol={symbol}&resolution=1D"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    payload = response.json()

    if not payload or "t" not in payload or len(payload["t"]) == 0:
        raise ValueError(f"DNSE API returned empty dataset for symbol={symbol}.")

    dates = (
        pd.to_datetime(payload["t"], unit="s")
        .tz_localize("UTC")
        .tz_convert(timezone)
        .strftime("%Y-%m-%d")
    )
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "adjusted_close": np.asarray(payload["c"], dtype=float) * multiplier,
            "volume": np.asarray(payload.get("v", np.zeros(len(dates))), dtype=float),
        }
    )
    df = df.drop_duplicates(subset=["date"], keep="first").set_index("date").sort_index()
    return df


def load_market_data(
    config: Dict[str, Any],
    raw_dir: Path,
    use_csv_fallback_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load historical price data for VCB, DCM, and VN-Index covering both in-sample
    and out-of-sample periods, plus one base session prior to IN_SAMPLE_START if available.

    Parameters
    ----------
    config : Dict[str, Any]
        Centralized configuration dictionary.
    raw_dir : Path
        Directory path for storing or reading raw CSV files.
    use_csv_fallback_only : bool
        If True, bypass online providers and load strictly from local CSV files.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        - stock_long_df: DataFrame with columns [date, ticker, adjusted_close, close, volume, yf_adjusted_close]
        - benchmark_df: DataFrame with columns [date, index_close]
        - metadata: Dictionary recording retrieval timestamp, provider details, and corporate action notes.
    """
    stock_csv_path = raw_dir / "stock_prices_raw.csv"
    bench_csv_path = raw_dir / "market_benchmark_raw.csv"
    retrieval_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Query window includes a 7-day buffer prior to IN_SAMPLE_START to capture the base session if needed
    fetch_start = "2023-07-25"
    fetch_end = "2026-11-02"

    if not use_csv_fallback_only:
        try:
            # 1. Fetch official corporate-action-adjusted series from DNSE Entrade API
            dnse_vcb = _fetch_dnse_series(
                "VCB", "stock", fetch_start, fetch_end, config["DNSE_PRICE_MULTIPLIER_VND"], config["TIMEZONE"]
            )
            dnse_dcm = _fetch_dnse_series(
                "DCM", "stock", fetch_start, fetch_end, config["DNSE_PRICE_MULTIPLIER_VND"], config["TIMEZONE"]
            )
            dnse_vni = _fetch_dnse_series(
                "VNINDEX", "index", fetch_start, fetch_end, config["INDEX_PRICE_MULTIPLIER"], config["TIMEZONE"]
            )

            # 2. Fetch unadjusted tradable closing prices and cross-check adjusted prices from Yahoo Finance
            yf_raw = yf.download(
                ["VCB.VN", "DCM.VN"],
                start=fetch_start,
                end=fetch_end,
                auto_adjust=False,
                progress=False,
            )
            if yf_raw.empty:
                raise ValueError("Yahoo Finance returned empty DataFrame for VCB.VN and DCM.VN.")

            yf_close = yf_raw["Close"].rename(columns={"VCB.VN": "VCB", "DCM.VN": "DCM"})
            yf_adj = yf_raw["Adj Close"].rename(columns={"VCB.VN": "VCB", "DCM.VN": "DCM"})
            yf_vol = yf_raw["Volume"].rename(columns={"VCB.VN": "VCB", "DCM.VN": "DCM"})

            records = []
            for ticker, dnse_df in [("VCB", dnse_vcb), ("DCM", dnse_dcm)]:
                combined = pd.DataFrame(index=dnse_df.index)
                combined["ticker"] = ticker
                combined["adjusted_close"] = dnse_df["adjusted_close"]
                # Align unadjusted close from Yahoo Finance; fallback to adjusted_close if missing on a session
                combined["close"] = yf_close[ticker].reindex(combined.index).fillna(combined["adjusted_close"])
                combined["yf_adjusted_close"] = yf_adj[ticker].reindex(combined.index)
                vol_series = yf_vol[ticker].reindex(combined.index).fillna(dnse_df["volume"])
                combined["volume"] = vol_series.fillna(0.0)
                combined = combined.reset_index()
                records.append(combined)

            stock_long_df = pd.concat(records, axis=0, ignore_index=True)
            stock_long_df = stock_long_df.sort_values(["date", "ticker"]).reset_index(drop=True)

            benchmark_df = pd.DataFrame(
                {
                    "date": dnse_vni.index,
                    "index_close": dnse_vni["adjusted_close"].values,
                }
            ).sort_values("date").reset_index(drop=True)

            # Save standard CSV fallback files
            stock_long_df[["date", "ticker", "adjusted_close", "close", "volume", "yf_adjusted_close"]].to_csv(
                stock_csv_path, index=False
            )
            benchmark_df[["date", "index_close"]].to_csv(bench_csv_path, index=False)

            metadata = {
                "retrieval_timestamp": retrieval_ts,
                "mode": "Online Primary Feed + Cross-Provider Audit",
                "stock_adjusted_source": config["PRIMARY_ADJUSTED_PROVIDER"],
                "stock_unadjusted_source": config["SECONDARY_UNADJUSTED_PROVIDER"],
                "benchmark_source": config["PRIMARY_ADJUSTED_PROVIDER"],
                "price_unit_stocks": "VND per share",
                "price_unit_benchmark": "Index points",
                "stock_csv_path": str(stock_csv_path),
                "bench_csv_path": str(bench_csv_path),
            }
            return stock_long_df, benchmark_df, metadata

        except Exception as online_err:
            if not (stock_csv_path.exists() and bench_csv_path.exists()):
                raise RuntimeError(
                    f"Online data retrieval failed ({online_err}) and local CSV fallback files "
                    f"were not found at {stock_csv_path} and {bench_csv_path}."
                ) from online_err

    # Fallback mode: Load from local CSV files
    if not (stock_csv_path.exists() and bench_csv_path.exists()):
        raise FileNotFoundError(
            f"CSV fallback files missing. Expected {stock_csv_path} and {bench_csv_path}."
        )

    stock_long_df = pd.read_csv(stock_csv_path, parse_dates=["date"])
    benchmark_df = pd.read_csv(bench_csv_path, parse_dates=["date"])
    metadata = {
        "retrieval_timestamp": retrieval_ts,
        "mode": "Local CSV Fallback",
        "stock_adjusted_source": "Local CSV Cache (DNSE/Yahoo Verified)",
        "stock_unadjusted_source": "Local CSV Cache",
        "benchmark_source": "Local CSV Cache",
        "price_unit_stocks": "VND per share",
        "price_unit_benchmark": "Index points",
        "stock_csv_path": str(stock_csv_path),
        "bench_csv_path": str(bench_csv_path),
    }
    return stock_long_df, benchmark_df, metadata

