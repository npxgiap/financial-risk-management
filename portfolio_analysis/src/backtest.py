"""
Out-of-sample backtesting module (03 August 2026 – 30 October 2026).
Implements:
1. Theoretical Adjusted-Price Buy-and-Hold Backtest (fractional units, natural weight drift).
2. Practical Execution Backtest (unadjusted tradable prices, 100-share board lots, buy commission,
   cash ledger, mark-to-market ending NAV vs. net liquidation value after sell commission & tax).
3. Deterministic Bank Deposit Benchmark (7.0% p.a. simple interest ACT/365).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def run_theoretical_backtest(
    oos_panel: pd.DataFrame,
    initial_weights: Dict[str, float],
    config: Dict[str, Any],
    strategy_name: str,
) -> pd.DataFrame:
    """
    Run theoretical buy-and-hold backtest using fractional units and adjusted-price relatives.

    Formulas
    --------
    Asset_Value_{i,t} = Initial_Capital * Initial_Weight_i * (Adj_Price_{i,t} / Adj_Price_{i,entry})
    Portfolio_Value_t = sum_i Asset_Value_{i,t}
    Weight_{i,t} = Asset_Value_{i,t} / Portfolio_Value_t
    """
    cap = float(config["INITIAL_CAPITAL_VND"])
    w_vcb = float(initial_weights["VCB"])
    w_dcm = float(initial_weights["DCM"])
    if not np.isclose(w_vcb + w_dcm, 1.0, atol=1e-6):
        raise ValueError("Initial risky weights for theoretical backtest must sum to 1.0.")

    p_vcb = oos_panel["VCB"].astype(float)
    p_dcm = oos_panel["DCM"].astype(float)

    val_vcb = cap * w_vcb * (p_vcb / p_vcb.iloc[0])
    val_dcm = cap * w_dcm * (p_dcm / p_dcm.iloc[0])
    nav = val_vcb + val_dcm

    w_vcb_drift = val_vcb / nav
    w_dcm_drift = val_dcm / nav

    daily_ret = nav.pct_change().fillna(0.0)
    cum_ret = (nav / cap) - 1.0
    running_peak = nav.cummax()
    drawdown = (nav / running_peak) - 1.0

    df_res = pd.DataFrame(
        {
            "Strategy": strategy_name,
            "VCB_Value_VND": val_vcb,
            "DCM_Value_VND": val_dcm,
            "Cash_VND": 0.0,
            "Portfolio_NAV_VND": nav,
            "VCB_Weight": w_vcb_drift,
            "DCM_Weight": w_dcm_drift,
            "Cash_Weight": 0.0,
            "Daily_Return": daily_ret,
            "Cumulative_Return": cum_ret,
            "Running_Peak_VND": running_peak,
            "Drawdown": drawdown,
        },
        index=oos_panel.index,
    )
    return df_res


def run_practical_backtest(
    oos_panel: pd.DataFrame,
    initial_weights: Dict[str, float],
    config: Dict[str, Any],
    strategy_name: str,
    buy_comm_rate: float = None,
    sell_comm_rate: float = None,
    sell_tax_rate: float = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run practical execution backtest on unadjusted tradable prices enforcing HOSE 100-share
    board lots, buy brokerage commission, residual cash ledger, and distinguishing between
    Mark-to-Market Ending NAV (unliquidated) and Net Liquidation Value (after sell costs & tax).
    """
    cap = float(config["INITIAL_CAPITAL_VND"])
    lot_size = 1 if bool(config.get("ALLOW_ODD_LOT_TRADING", False)) else int(config["BOARD_LOT_SIZE"])
    buy_fee_rate = float(config["BUY_COMMISSION_RATE"]) if buy_comm_rate is None else float(buy_comm_rate)
    sell_fee_rate = float(config["SELL_COMMISSION_RATE"]) if sell_comm_rate is None else float(sell_comm_rate)
    tax_rate = float(config["SELL_TAX_RATE"]) if sell_tax_rate is None else float(sell_tax_rate)
    cash_rate = float(config.get("RESIDUAL_CASH_INTEREST_RATE", 0.0))

    w_vcb = float(initial_weights["VCB"])
    w_dcm = float(initial_weights["DCM"])

    p_vcb_unadj = oos_panel["VCB_Unadj"].astype(float)
    p_dcm_unadj = oos_panel["DCM_Unadj"].astype(float)
    entry_p_vcb = float(p_vcb_unadj.iloc[0])
    entry_p_dcm = float(p_dcm_unadj.iloc[0])

    # Calculate maximum whole board lots affordable within each asset's target budget including buy commission
    budget_vcb = cap * w_vcb
    budget_dcm = cap * w_dcm

    qty_vcb = int(budget_vcb // (entry_p_vcb * lot_size * (1.0 + buy_fee_rate))) * lot_size
    qty_dcm = int(budget_dcm // (entry_p_dcm * lot_size * (1.0 + buy_fee_rate))) * lot_size

    gross_buy_vcb = qty_vcb * entry_p_vcb
    gross_buy_dcm = qty_dcm * entry_p_dcm
    buy_cost_vcb = gross_buy_vcb * buy_fee_rate
    buy_cost_dcm = gross_buy_dcm * buy_fee_rate
    total_buy_costs = buy_cost_vcb + buy_cost_dcm

    initial_residual_cash = cap - (gross_buy_vcb + gross_buy_dcm + total_buy_costs)

    # If any positive target weight asset truncated to 0 lots due to granularity and remaining cash allows, allocate 1 lot
    if qty_vcb == 0 and w_vcb > 0:
        cost_1lot_vcb = entry_p_vcb * lot_size * (1.0 + buy_fee_rate)
        if initial_residual_cash >= cost_1lot_vcb:
            qty_vcb += lot_size
            gross_buy_vcb = qty_vcb * entry_p_vcb
            buy_cost_vcb = gross_buy_vcb * buy_fee_rate
            total_buy_costs = buy_cost_vcb + buy_cost_dcm
            initial_residual_cash = cap - (gross_buy_vcb + gross_buy_dcm + total_buy_costs)

    if qty_dcm == 0 and w_dcm > 0:
        cost_1lot_dcm = entry_p_dcm * lot_size * (1.0 + buy_fee_rate)
        if initial_residual_cash >= cost_1lot_dcm:
            qty_dcm += lot_size
            gross_buy_dcm = qty_dcm * entry_p_dcm
            buy_cost_dcm = gross_buy_dcm * buy_fee_rate
            total_buy_costs = buy_cost_vcb + buy_cost_dcm
            initial_residual_cash = cap - (gross_buy_vcb + gross_buy_dcm + total_buy_costs)

    if initial_residual_cash < -1e-6:
        raise AssertionError("Practical execution error: total purchase outlay plus commission exceeded initial capital.")

    # Track daily mark-to-market positions and cash ledger
    elapsed_days = (oos_panel.index - oos_panel.index[0]).days.values
    cash_series = pd.Series(
        initial_residual_cash * (1.0 + cash_rate * (elapsed_days / 365.0)),
        index=oos_panel.index,
    )

    val_vcb = qty_vcb * p_vcb_unadj
    val_dcm = qty_dcm * p_dcm_unadj
    mtm_nav = val_vcb + val_dcm + cash_series

    w_vcb_drift = val_vcb / mtm_nav
    w_dcm_drift = val_dcm / mtm_nav
    w_cash_drift = cash_series / mtm_nav

    daily_ret = mtm_nav.pct_change().fillna((mtm_nav.iloc[0] / cap) - 1.0)
    # Note: At entry close (index 0), MTM NAV is slightly below Initial Capital by total_buy_costs
    running_peak = pd.concat([pd.Series([cap], index=[oos_panel.index[0]]), mtm_nav]).cummax().iloc[1:]
    drawdown = (mtm_nav / running_peak) - 1.0

    # Calculate ending Net Liquidation Value if positions are sold at exit close
    exit_val_vcb = float(val_vcb.iloc[-1])
    exit_val_dcm = float(val_dcm.iloc[-1])
    estimated_sell_costs = (exit_val_vcb + exit_val_dcm) * (sell_fee_rate + tax_rate)
    ending_mtm_nav = float(mtm_nav.iloc[-1])
    net_liquidation_value = ending_mtm_nav - estimated_sell_costs

    df_res = pd.DataFrame(
        {
            "Strategy": strategy_name,
            "VCB_Quantity": qty_vcb,
            "DCM_Quantity": qty_dcm,
            "VCB_Value_VND": val_vcb,
            "DCM_Value_VND": val_dcm,
            "Cash_VND": cash_series,
            "Portfolio_NAV_VND": mtm_nav,
            "VCB_Weight": w_vcb_drift,
            "DCM_Weight": w_dcm_drift,
            "Cash_Weight": w_cash_drift,
            "Daily_Return": mtm_nav.pct_change().fillna(0.0),
            "Cumulative_Return": (mtm_nav / cap) - 1.0,
            "Running_Peak_VND": running_peak,
            "Drawdown": drawdown,
        },
        index=oos_panel.index,
    )

    execution_meta = {
        "strategy_name": strategy_name,
        "qty_vcb": qty_vcb,
        "qty_dcm": qty_dcm,
        "entry_price_vcb": entry_p_vcb,
        "entry_price_dcm": entry_p_dcm,
        "exit_price_vcb": float(p_vcb_unadj.iloc[-1]),
        "exit_price_dcm": float(p_dcm_unadj.iloc[-1]),
        "gross_invested_vnd": gross_buy_vcb + gross_buy_dcm,
        "entry_buy_costs_vnd": total_buy_costs,
        "initial_residual_cash_vnd": initial_residual_cash,
        "ending_cash_vnd": float(cash_series.iloc[-1]),
        "ending_mtm_nav_vnd": ending_mtm_nav,
        "estimated_exit_sell_costs_vnd": estimated_sell_costs,
        "net_liquidation_value_vnd": net_liquidation_value,
    }
    return df_res, execution_meta


def calculate_backtest_metrics(
    oos_panel: pd.DataFrame,
    portfolio_p_weights: Dict[str, float],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Execute theoretical and practical out-of-sample backtests for Portfolio P and Equal Weight,
    compute the deterministic 7.0% p.a. ACT/365 Bank Deposit benchmark, and compile the
    mandatory Out-of-Sample Performance summary table.
    """
    cap = float(config["INITIAL_CAPITAL_VND"])
    dep_rate = float(config["DEPOSIT_ANNUAL_RATE"])
    rf_ann = float(config["RISK_FREE_ANNUAL_RATE"])
    ann_days = float(config["TRADING_DAYS_PER_YEAR"])
    rf_daily = (1.0 + rf_ann) ** (1.0 / ann_days) - 1.0

    entry_dt = oos_panel.index[0]
    exit_dt = oos_panel.index[-1]
    elapsed_cal_days = (oos_panel.index - entry_dt).days.values
    total_cal_days = int(elapsed_cal_days[-1])

    # 1. Deposit benchmark: Deposit Value_t = 50,000,000 * (1 + 0.07 * Elapsed_Calendar_Days_t / 365)
    dep_nav = pd.Series(cap * (1.0 + dep_rate * (elapsed_cal_days / 365.0)), index=oos_panel.index, name="Deposit_NAV")
    dep_ending_val = float(dep_nav.iloc[-1])
    dep_profit = dep_ending_val - cap
    dep_hpr = (dep_ending_val / cap) - 1.0

    eq_weights = {"VCB": 0.5, "DCM": 0.5}

    # 2. Run Theoretical Buy-and-Hold Backtests
    theo_p_df = run_theoretical_backtest(oos_panel, portfolio_p_weights, config, "Portfolio P (Theoretical Buy-and-Hold)")
    theo_eq_df = run_theoretical_backtest(oos_panel, eq_weights, config, "Equal Weight (Theoretical Buy-and-Hold)")

    # 3. Run Practical Lot-Size Execution Backtests
    prac_p_df, prac_p_meta = run_practical_backtest(
        oos_panel, portfolio_p_weights, config, "Portfolio P (Practical Lot-Size Execution)"
    )
    prac_eq_df, prac_eq_meta = run_practical_backtest(
        oos_panel, eq_weights, config, "Equal Weight (Practical Lot-Size Execution)"
    )

    def _summarize_equity_strategy(
        label: str,
        sim_df: pd.DataFrame,
        net_liq_val: float,
        tot_tx_costs: float,
        end_cash: float,
    ) -> Dict[str, Any]:
        nav_s = sim_df["Portfolio_NAV_VND"]
        # Post-entry daily returns (excluding day 0 baseline entry session)
        r_s = nav_s.pct_change().dropna()
        end_mtm = float(nav_s.iloc[-1])
        pnl_mtm = end_mtm - cap
        hpr = (end_mtm / cap) - 1.0

        ann_vol = float(r_s.std(ddof=1) * np.sqrt(ann_days))
        ex_r = r_s - rf_daily
        realized_sr = float(np.sqrt(ann_days) * ex_r.mean() / ex_r.std(ddof=1)) if ex_r.std(ddof=1) > 1e-12 else np.nan
        max_dd = float(sim_df["Drawdown"].min())  # Negative percentage convention

        return {
            "Strategy": label,
            "Actual Entry Date": entry_dt.strftime("%Y-%m-%d"),
            "Actual End Date": exit_dt.strftime("%Y-%m-%d"),
            "Calendar Holding Days": total_cal_days,
            "Trading Sessions": int(len(oos_panel)),
            "Initial Capital": cap,
            "Ending Mark-to-Market NAV": end_mtm,
            "Net Liquidation Value": net_liq_val,
            "Profit or Loss VND": pnl_mtm,
            "Holding Period Return": hpr,
            "Annualized Volatility": ann_vol,
            "Realized Sharpe Ratio": realized_sr,
            "Maximum Drawdown": max_dd,
            "Worst Daily Return": float(r_s.min()),
            "Best Daily Return": float(r_s.max()),
            "Total Transaction Costs": tot_tx_costs,
            "Ending Cash": end_cash,
            "Excess Profit versus Deposit": pnl_mtm - dep_profit,
        }

    summary_rows = [
        _summarize_equity_strategy(
            "Portfolio P (Theoretical Buy-and-Hold)",
            theo_p_df,
            float(theo_p_df["Portfolio_NAV_VND"].iloc[-1]),
            0.0,
            0.0,
        ),
        _summarize_equity_strategy(
            "Equal Weight (Theoretical Buy-and-Hold)",
            theo_eq_df,
            float(theo_eq_df["Portfolio_NAV_VND"].iloc[-1]),
            0.0,
            0.0,
        ),
        _summarize_equity_strategy(
            "Portfolio P (Practical Lot-Size Execution)",
            prac_p_df,
            prac_p_meta["net_liquidation_value_vnd"],
            prac_p_meta["entry_buy_costs_vnd"],
            prac_p_meta["ending_cash_vnd"],
        ),
        _summarize_equity_strategy(
            "Equal Weight (Practical Lot-Size Execution)",
            prac_eq_df,
            prac_eq_meta["net_liquidation_value_vnd"],
            prac_eq_meta["entry_buy_costs_vnd"],
            prac_eq_meta["ending_cash_vnd"],
        ),
        {
            "Strategy": "Bank Deposit Benchmark (7.0% p.a. ACT/365)",
            "Actual Entry Date": entry_dt.strftime("%Y-%m-%d"),
            "Actual End Date": exit_dt.strftime("%Y-%m-%d"),
            "Calendar Holding Days": total_cal_days,
            "Trading Sessions": int(len(oos_panel)),
            "Initial Capital": cap,
            "Ending Mark-to-Market NAV": dep_ending_val,
            "Net Liquidation Value": dep_ending_val,
            "Profit or Loss VND": dep_profit,
            "Holding Period Return": dep_hpr,
            "Annualized Volatility": "N/A (Deterministic Accrual)",
            "Realized Sharpe Ratio": "N/A (Deterministic Accrual)",
            "Maximum Drawdown": 0.0,
            "Worst Daily Return": float(dep_nav.pct_change().dropna().min()),
            "Best Daily Return": float(dep_nav.pct_change().dropna().max()),
            "Total Transaction Costs": 0.0,
            "Ending Cash": dep_ending_val,
            "Excess Profit versus Deposit": 0.0,
        },
    ]

    metrics_df = pd.DataFrame(summary_rows)
    series_dict = {
        "theo_p": theo_p_df,
        "theo_eq": theo_eq_df,
        "prac_p": prac_p_df,
        "prac_eq": prac_eq_df,
        "deposit_nav": dep_nav,
    }
    meta_dict = {
        "prac_p_meta": prac_p_meta,
        "prac_eq_meta": prac_eq_meta,
        "deposit_ending_value": dep_ending_val,
        "deposit_profit": dep_profit,
        "calendar_days": total_cal_days,
    }
    return metrics_df, series_dict, meta_dict

