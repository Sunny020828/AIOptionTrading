import pandas as pd
import numpy as np
import math
from execution.strategy_pools import generate_trade_signals, mark_signals_to_market, signals_to_position_legs, apply_risk_management
from typing import Optional, Tuple, Dict, Any, List


def compute_backtest_metrics(
    scenario: Dict[str, Any] | None,
    entry_date,
    exit_date,
    option_df,
    account_size=1_000_000,
    risk_pct=0.01,
    *,
    apply_risk_controls: bool = False,
    underlying_close: Optional[pd.Series] = None,
    delta_limit: float = 300.0,
    stop_loss_ratio: float = 1.0,
    capital_alloc_pct: float = 0.1,
    logger=None,
) -> Dict[str, Any]:
    """
    Run strategy generation + MTM + optional risk management + summary metrics.
    """

    signals = []
    mtm_df = pd.DataFrame()
    strategy_type = None
    risk_exit_date = None
    risk_exit_reason = None

    if scenario is not None:
        signals = generate_trade_signals(
            scenario=scenario,
            date=entry_date,
            df=option_df,
            account_size=account_size,
            risk_pct=risk_pct,
            capital_alloc_pct= capital_alloc_pct
        )

    if signals:
        meta_list = [x["meta"] for x in signals if "meta" in x]
        if meta_list:
            strategy_type = meta_list[-1].get("strategy_type")

        mtm_df = mark_signals_to_market(
            option_df=option_df,
            signals=signals,
            start_date=entry_date,
            end_date=exit_date,
            price_col="SettlementPrice",
            contract_col="Series",
            date_col="Date",
            contract_multiplier=50.0,
        )

    if (
        apply_risk_controls
        and not mtm_df.empty
        and underlying_close is not None
    ):
        mtm_df, risk_exit_date, risk_exit_reason = apply_risk_management(
            mtm_df=mtm_df,
            option_df=option_df,
            signals=signals,
            underlying_series=underlying_close,
            stop_loss_ratio=stop_loss_ratio,
            delta_limit=delta_limit,
            logger=logger,
        )

    if mtm_df.empty:
        metrics = {
            "n_days": 0,
            "cum_pnl": np.nan,
            "return_pct": np.nan,
            "ann_return_pct": np.nan,
            "ann_vol_pct": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
            "win_rate_daily": np.nan,
        }
        return {
            "metrics": metrics,
            "mtm_df": mtm_df,
            "signals": signals,
            "strategy_type": strategy_type,
            "risk_exit_date": risk_exit_date,
            "risk_exit_reason": risk_exit_reason,
        }

    df = mtm_df.copy()
    df["daily_ret"] = df["Daily_PnL"] / account_size

    n_days = len(df)
    cum_pnl = float(df["Total_PnL"].iloc[-1])
    ret_pct = cum_pnl / account_size

    ann_return = (1 + ret_pct) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else np.nan
    ann_vol = df["daily_ret"].std(ddof=1) * np.sqrt(252) if n_days > 1 else np.nan
    sharpe = ann_return / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan

    equity = account_size + df["Total_PnL"]
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1
    mdd = float(drawdown.min()) if len(drawdown) else np.nan
    win_rate_daily = float((df["Daily_PnL"] > 0).mean()) if n_days > 0 else np.nan

    metrics = {
        "n_days": int(n_days),
        "cum_pnl": cum_pnl,
        "return_pct": ret_pct,
        "ann_return_pct": ann_return,
        "ann_vol_pct": ann_vol,
        "sharpe": sharpe,
        "max_drawdown_pct": mdd,
        "win_rate_daily": win_rate_daily,
    }

    return {
        "metrics": metrics,
        "mtm_df": mtm_df,
        "signals": signals,
        "strategy_type": strategy_type,
        "risk_exit_date": risk_exit_date,
        "risk_exit_reason": risk_exit_reason,
    }
    
    
    
def summarize_mtm_path(
    mtm_df: pd.DataFrame,
    account_size: float = 1_000_000.0,
    win_rate_mode: str = "active",
    date_col: str = "Date",
    qty_col: str = "qty",
    daily_pnl_col: str = "portfolio_daily_pnl",
    total_pnl_col: str = "portfolio_total_pnl",
) -> Dict[str, Any]:
    if mtm_df.empty:
        return {
            "n_days": 0,
            "cum_pnl": None,
            "return_pct": None,
            "ann_return_pct": None,
            "ann_vol_pct": None,
            "sharpe": None,
            "max_drawdown_pct": None,
            "win_rate_daily": None,
        }

    df = mtm_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    agg_dict = {daily_pnl_col: "sum"}
    if qty_col in df.columns:
        agg_dict[qty_col] = "sum"
    if total_pnl_col in df.columns:
        agg_dict[total_pnl_col] = "last"

    daily = (
        df.groupby(date_col, as_index=False)
        .agg(agg_dict)
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    # 更稳：总PnL自己重算
    daily[total_pnl_col] = daily[daily_pnl_col].cumsum()
    daily["daily_ret"] = daily[daily_pnl_col] / account_size

    n_days = len(daily)
    cum_pnl = float(daily[total_pnl_col].iloc[-1])
    ret_pct = cum_pnl / account_size

    ann_return = (1 + ret_pct) ** (252 / n_days) - 1 if n_days > 0 else np.nan
    ann_vol = daily["daily_ret"].std(ddof=1) * np.sqrt(252) if n_days > 1 else np.nan
    sharpe = ann_return / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan

    equity = account_size + daily[total_pnl_col]
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1
    mdd = float(drawdown.min()) if len(drawdown) else np.nan

    if win_rate_mode == "all":
        mask = pd.Series(True, index=daily.index)
    elif win_rate_mode == "nonzero":
        mask = daily[daily_pnl_col] != 0
    elif win_rate_mode == "active":
        if qty_col in daily.columns:
            mask = daily[qty_col].fillna(0).abs() > 0
        else:
            mask = daily[daily_pnl_col] != 0
    else:
        raise ValueError("win_rate_mode must be one of: 'all', 'nonzero', 'active'")

    win_rate = float((daily.loc[mask, daily_pnl_col] > 0).mean()) if mask.any() else np.nan

    def _safe(x):
        if x is None:
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x) if isinstance(x, (np.floating, float)) else x

    return {
        "n_days": int(n_days),
        "cum_pnl": _safe(cum_pnl),
        "return_pct": _safe(ret_pct),
        "ann_return_pct": _safe(ann_return),
        "ann_vol_pct": _safe(ann_vol),
        "sharpe": _safe(sharpe),
        "max_drawdown_pct": _safe(mdd),
        "win_rate_daily": _safe(win_rate),
    }


def run_monthly_backtest(
    scenario: str,
    entry_date,
    exit_date,
    option_df: pd.DataFrame,
    account_size: float = 1_000_000.0,
    risk_pct: float = 0.01,
    contract_multiplier: float = 50.0,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    One monthly strategy backtest.
    Returns:
        mtm_df, metrics, signals, meta
    """
    signals = generate_trade_signals(
        scenario=scenario,
        date=pd.to_datetime(entry_date),
        df=option_df,
        account_size=account_size,
        risk_pct=risk_pct,
    )

    if not signals:
        metrics = {
            "scenario": scenario,
            "start_date": str(pd.to_datetime(entry_date).date()),
            "end_date": str(pd.to_datetime(exit_date).date()),
            "strategy_type": None,
            "signals": [],
            **summarize_mtm_path(pd.DataFrame(), account_size=account_size),
        }
        return pd.DataFrame(), metrics, [], None

    mtm_df = mark_signals_to_market(
        option_df=option_df,
        signals=signals,
        start_date=entry_date,
        end_date=exit_date,
        contract_multiplier=contract_multiplier,
    )

    _, meta = signals_to_position_legs(signals, contract_multiplier=contract_multiplier)
    metrics = summarize_mtm_path(mtm_df, account_size=account_size)

    metrics.update({
        "scenario": scenario,
        "start_date": str(pd.to_datetime(entry_date).date()),
        "end_date": str(pd.to_datetime(exit_date).date()),
        "strategy_type": meta.get("strategy_type") if meta else None,
        "underlying_price": meta.get("underlying_price") if meta else None,
        "expiry": str(pd.to_datetime(meta.get("expiry")).date()) if meta and meta.get("expiry") is not None else None,
        "signals": signals,
    })

    return mtm_df, metrics, signals, meta


def add_transaction_costs(
    full_mtm_path_timeline: pd.DataFrame,
    exchange_fee_per_contract: float = 10.00,
    commission_levy_per_contract: float = 0.54,
    broker_commission_per_contract: float = 0.0,
    slippage_per_contract: float = 0.0,
    strategy_leg_map: dict | None = None,
    qty_col: str = "qty",
    date_col: str = "Date",
    entry_col: str = "entry_date",
    exit_col: str = "exit_date",
    strategy_col: str = "strategy_type",
    daily_pnl_col: str = "portfolio_daily_pnl",
    total_pnl_col: str = "portfolio_total_pnl",
) -> pd.DataFrame:
    df = full_mtm_path_timeline.copy()

    required_cols = [date_col, entry_col, exit_col, strategy_col, qty_col, daily_pnl_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[date_col] = pd.to_datetime(df[date_col])
    df[entry_col] = pd.to_datetime(df[entry_col])
    df[exit_col] = pd.to_datetime(df[exit_col])

    if strategy_leg_map is None:
        strategy_leg_map = {
            "call_debit_spread": 2,
            "call_credit_spread": 2,
            "put_debit_spread": 2,
            "put_credit_spread": 2,
            "bull_call_spread": 2,
            "bear_call_spread": 2,
            "bull_put_spread": 2,
            "bear_put_spread": 2,
            "vertical_spread": 2,
            "iron_condor": 4,
            "straddle": 2,
            "strangle": 2,
            "butterfly": 3,
        }

    def infer_legs(strategy_name: str) -> int:
        if pd.isna(strategy_name):
            return 2
        s = str(strategy_name).strip().lower()
        if s in strategy_leg_map:
            return strategy_leg_map[s]
        if "iron_condor" in s or "condor" in s:
            return 4
        if "butterfly" in s:
            return 3
        if "straddle" in s or "strangle" in s or "spread" in s:
            return 2
        return 2

    df["n_legs"] = df[strategy_col].apply(infer_legs)

    per_contract_side_cost = (
        exchange_fee_per_contract
        + commission_levy_per_contract
        + broker_commission_per_contract
        + slippage_per_contract
    )
    df["per_contract_side_cost"] = per_contract_side_cost

    is_entry = df[date_col].dt.normalize() == df[entry_col].dt.normalize()
    is_exit = df[date_col].dt.normalize() == df[exit_col].dt.normalize()

    df["transaction_cost"] = 0.0

    df.loc[is_entry, "transaction_cost"] += (
        df.loc[is_entry, qty_col].astype(float).abs()
        * df.loc[is_entry, "n_legs"].astype(float)
        * per_contract_side_cost
    )

    df.loc[is_exit, "transaction_cost"] += (
        df.loc[is_exit, qty_col].astype(float).abs()
        * df.loc[is_exit, "n_legs"].astype(float)
        * per_contract_side_cost
    )

    df["portfolio_daily_pnl_net"] = df[daily_pnl_col].astype(float) - df["transaction_cost"]

    sort_cols = [date_col]
    if "decision_date" in df.columns:
        sort_cols.append("decision_date")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # 注意：这里不要直接对原表 cumsum，当同一天有多行时会失真
    daily_net = (
        df.groupby(date_col, as_index=False)
        .agg(
            portfolio_daily_pnl=(daily_pnl_col, "sum"),
            transaction_cost=("transaction_cost", "sum"),
            portfolio_daily_pnl_net=("portfolio_daily_pnl_net", "sum"),
        )
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    daily_net["portfolio_total_pnl_net"] = daily_net["portfolio_daily_pnl_net"].cumsum()

    # 把日度累计净值 merge 回原表，便于后续画图/检查
    df = df.merge(
        daily_net[[date_col, "portfolio_total_pnl_net"]],
        on=date_col,
        how="left",
    )

    if total_pnl_col in df.columns:
        df["portfolio_total_pnl_gross_check"] = df[total_pnl_col]

    return df



def summarize_net_performance(
    mtm_df: pd.DataFrame,
    account_size: float = 1_000_000.0,
    date_col: str = "Date",
    gross_daily_pnl_col: str = "portfolio_daily_pnl",
    net_daily_pnl_col: str = "portfolio_daily_pnl_net",
) -> dict:
    df = mtm_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    daily = (
        df.groupby(date_col, as_index=False)
        .agg(
            gross_daily_pnl=(gross_daily_pnl_col, "sum"),
            net_daily_pnl=(net_daily_pnl_col, "sum"),
        )
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    daily["gross_daily_ret"] = daily["gross_daily_pnl"] / account_size
    daily["net_daily_ret"] = daily["net_daily_pnl"] / account_size

    gross_sharpe = np.nan
    if daily["gross_daily_ret"].std(ddof=1) > 0:
        gross_sharpe = (
            daily["gross_daily_ret"].mean()
            / daily["gross_daily_ret"].std(ddof=1)
        ) * np.sqrt(252)

    net_sharpe = np.nan
    if daily["net_daily_ret"].std(ddof=1) > 0:
        net_sharpe = (
            daily["net_daily_ret"].mean()
            / daily["net_daily_ret"].std(ddof=1)
        ) * np.sqrt(252)

    daily["cum_pnl_net"] = daily["net_daily_pnl"].cumsum()
    running_max = daily["cum_pnl_net"].cummax()
    drawdown = daily["cum_pnl_net"] - running_max
    max_drawdown_pct_net = drawdown.min() / account_size

    return {
        "gross_sharpe": float(gross_sharpe) if pd.notna(gross_sharpe) else None,
        "net_sharpe": float(net_sharpe) if pd.notna(net_sharpe) else None,
        "cum_pnl_net": float(daily["cum_pnl_net"].iloc[-1]) if len(daily) > 0 else None,
        "max_drawdown_pct_net": float(max_drawdown_pct_net) if len(daily) > 0 else None,
        "daily_net_timeline": daily,
    }