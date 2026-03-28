import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Dict, Any, Optional, Tuple, Union

from dataclasses import dataclass, field


@dataclass
class PredictionCycleResult:
    """
    One prediction-period outcome: trade list, MTM path, and performance metrics.
    Used by pipeline / backtest as the single strategy boundary.
    """

    metrics: Dict[str, Any]
    mtm_df: pd.DataFrame
    signals: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    strategy_type: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    risk_exit_date: Optional[pd.Timestamp] = None
    risk_exit_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "mtm_df": self.mtm_df,
            "signals": self.signals,
            "actions": self.actions,
            "strategy_type": self.strategy_type,
            "meta": self.meta,
            "risk_exit_date": self.risk_exit_date,
            "risk_exit_reason": self.risk_exit_reason,
        }


def _empty_period_metrics() -> Dict[str, Any]:
    return {
        "n_days": 0,
        "cum_pnl": np.nan,
        "return_pct": np.nan,
        "ann_return_pct": np.nan,
        "ann_vol_pct": np.nan,
        "sharpe": np.nan,
        "max_drawdown_pct": np.nan,
        "win_rate_daily": np.nan,
    }


def _summarize_mtm_to_metrics(mtm_df: pd.DataFrame, account_value: float) -> Dict[str, Any]:
    """Period PnL metrics from MTM path (Total_PnL / Daily_PnL columns)."""
    if mtm_df.empty:
        return _empty_period_metrics()

    df = mtm_df.copy()
    df["daily_ret"] = df["Daily_PnL"] / account_value

    n_days = len(df)
    cum_pnl = float(df["Total_PnL"].iloc[-1])
    ret_pct = cum_pnl / account_value

    ann_return = (1 + ret_pct) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else np.nan
    ann_vol = df["daily_ret"].std(ddof=1) * np.sqrt(252) if n_days > 1 else np.nan
    sharpe = ann_return / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan

    equity = account_value + df["Total_PnL"]
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1
    mdd = float(drawdown.min()) if len(drawdown) else np.nan
    win_rate_daily = float((df["Daily_PnL"] > 0).mean()) if n_days > 0 else np.nan

    return {
        "n_days": int(n_days),
        "cum_pnl": cum_pnl,
        "return_pct": ret_pct,
        "ann_return_pct": ann_return,
        "ann_vol_pct": ann_vol,
        "sharpe": sharpe,
        "max_drawdown_pct": mdd,
        "win_rate_daily": win_rate_daily,
    }


def _extract_strategy_meta(signals: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    meta_list = [x["meta"] for x in signals if isinstance(x, dict) and "meta" in x]
    if not meta_list:
        return None, None
    m = meta_list[-1]
    return m.get("strategy_type"), m


def _signals_to_actions(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in signals if isinstance(s, dict) and "action" in s]


def run_prediction_cycle_strategy(
    scenario: Optional[str],
    entry_date,
    exit_date,
    option_df: pd.DataFrame,
    account_value: float = 1_000_000,
    risk_pct: float = 0.01,
    *,
    price_col: str = "SettlementPrice",
    contract_col: str = "Series",
    date_col: str = "Date",
    contract_multiplier: float = 50.0,
    apply_risk_controls: bool = False,
    underlying_close: Optional[pd.Series] = None,
    stop_loss_pct: float = 0.5,
    delta_limit: float = 300,
    return_dataclass: bool = False,
) -> Union[PredictionCycleResult, Dict[str, Any]]:
    """
    Full strategy handling for one prediction holding period: signals → positions → MTM path → metrics.

    Pipeline should call this instead of stitching ``generate_trade_signals`` and ``mark_signals_to_market``
    separately.

    Parameters
    ----------
    scenario :
        LLM / rulebook scenario key (see ``generate_trade_signals``). None yields no trades.
    entry_date, exit_date :
        Holding window on ``option_df[date_col]`` (inclusive).
    apply_risk_controls :
        If True, run ``apply_risk_management`` (stop / delta limit). Requires ``underlying_close``
        indexed by calendar dates aligned with MTM rows.
    return_dataclass :
        If True, return ``PredictionCycleResult``; otherwise a dict (default), backward compatible
        with ``compute_backtest_metrics``.

    Returns
    -------
    Dict with keys: metrics, mtm_df, signals, actions, strategy_type, meta,
    risk_exit_date, risk_exit_reason — or ``PredictionCycleResult`` if ``return_dataclass=True``.
    """
    signals: List[Dict[str, Any]] = []
    mtm_df = pd.DataFrame()
    strategy_type: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    risk_exit_date: Optional[pd.Timestamp] = None
    risk_exit_reason: Optional[str] = None

    if scenario is not None:
        signals = generate_trade_signals(
            scenario=scenario,
            date=entry_date,
            df=option_df,
            account_value=account_value,
            risk_pct=risk_pct,
        )

    if signals:
        strategy_type, meta = _extract_strategy_meta(signals)
        mtm_df = mark_signals_to_market(
            option_df=option_df,
            signals=signals,
            start_date=entry_date,
            end_date=exit_date,
            price_col=price_col,
            contract_col=contract_col,
            date_col=date_col,
            contract_multiplier=contract_multiplier,
        )

    if apply_risk_controls and not mtm_df.empty and underlying_close is not None:
        u = underlying_close.copy()
        u.index = pd.to_datetime(u.index, errors="coerce").normalize()
        mtm_df, risk_exit_date, risk_exit_reason = apply_risk_management(
            mtm_df=mtm_df,
            option_df=option_df,
            signals=signals,
            start_date=entry_date,
            underlying_series=u,
            stop_loss_pct=stop_loss_pct,
            delta_limit=delta_limit,
        )
        if not mtm_df.empty and "Total_PnL" in mtm_df.columns:
            mtm_df = mtm_df.copy()
            mtm_df["Daily_PnL"] = mtm_df["Total_PnL"].diff().fillna(mtm_df["Total_PnL"])

    metrics = _summarize_mtm_to_metrics(mtm_df, account_value)
    actions = _signals_to_actions(signals)

    result = PredictionCycleResult(
        metrics=metrics,
        mtm_df=mtm_df,
        signals=signals,
        actions=actions,
        strategy_type=strategy_type,
        meta=meta,
        risk_exit_date=risk_exit_date,
        risk_exit_reason=risk_exit_reason,
    )
    return result if return_dataclass else result.as_dict()


def get_options_for_date(
    df: pd.DataFrame,
    date,
    exact_only: bool = False,
    forward_only: bool = True,
    max_shift_days: int | None = 10,
    return_actual_date: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Timestamp | None]:
    """
    Get option data for a target date.

    Rules
    -----
    1. Try exact date first.
    2. If exact_only=False and exact date has no rows:
       - forward_only=True: use the nearest later available trading date
       - forward_only=False: use the nearest available trading date in either direction
    3. If max_shift_days is not None, only allow fallback dates within that distance.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Date'.
    date : str / datetime-like
        Target date.
    exact_only : bool
        If True, only return exact-match rows.
    forward_only : bool
        If True, only search forward when exact date is missing.
    max_shift_days : int | None
        Maximum allowed calendar-day shift.
    return_actual_date : bool
        If True, return (subset_df, actual_date_used).

    Returns
    -------
    pd.DataFrame
        Rows for the selected date.
    or
    (pd.DataFrame, pd.Timestamp | None)
        If return_actual_date=True.
    """
    if "Date" not in df.columns:
        raise KeyError(f"'Date' not found in columns: {df.columns.tolist()}")

    target_date = pd.to_datetime(date).normalize()

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["Date"])

    # exact match first
    exact = out[out["Date"] == target_date].copy()
    if not exact.empty:
        return (exact, target_date) if return_actual_date else exact

    if exact_only:
        return (pd.DataFrame(), None) if return_actual_date else pd.DataFrame()

    available_dates = sorted(out["Date"].drop_duplicates())
    if not available_dates:
        return (pd.DataFrame(), None) if return_actual_date else pd.DataFrame()

    chosen_date = None

    if forward_only:
        future_dates = [d for d in available_dates if d >= target_date]
        if future_dates:
            chosen_date = future_dates[0]
    else:
        chosen_date = min(available_dates, key=lambda d: abs((d - target_date).days))

    if chosen_date is None:
        return (pd.DataFrame(), None) if return_actual_date else pd.DataFrame()

    if max_shift_days is not None:
        if abs((chosen_date - target_date).days) > max_shift_days:
            return (pd.DataFrame(), None) if return_actual_date else pd.DataFrame()

    selected = out[out["Date"] == chosen_date].copy()
    return (selected, chosen_date) if return_actual_date else selected




def get_target_expiry(day_data: pd.DataFrame, min_dte: int = 30, max_dte: int = 45):
    """
    Select contract month closest to 30-45 days to expiration.
    If none are in range, take the smallest dte >= min_dte.
    """
    if day_data.empty:
        return None

    expiries = day_data[["ExpDate", "dte"]].drop_duplicates()
    candidates = expiries[(expiries["dte"] >= min_dte) & (expiries["dte"] <= max_dte)]

    if not candidates.empty:
        target = candidates.iloc[
            (candidates["dte"] - (min_dte + max_dte) / 2).abs().argmin()
        ]["ExpDate"]
    else:
        candidates = expiries[expiries["dte"] >= min_dte]
        if not candidates.empty:
            target = candidates.nsmallest(1, "dte")["ExpDate"].iloc[0]
        else:
            target = None

    return target


def find_closest_strike(strikes, target):
    """Return the strike price closest to the target value."""
    return min(strikes, key=lambda x: abs(x - target))


def black_delta(S, K, T, r, sigma, option_type):
    """
    Black-76 style delta approximation.
    option_type: 'call' or 'put'
    """
    if T <= 0 or sigma <= 0 or pd.isna(sigma):
        return np.nan

    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return norm.cdf(d1)
    else:
        return -norm.cdf(-d1)


def estimate_delta(row, S, r=0.02):
    """Estimate Delta from one option row."""
    T = row["dte"] / 365.0
    sigma = row.get("ImpliedVolatility", np.nan)

    if pd.isna(sigma) or sigma <= 0:
        return np.nan

    opt_type = str(row["Type"]).lower()
    if opt_type == "c":
        opt_type = "call"
    elif opt_type == "p":
        opt_type = "put"

    return black_delta(S, row["StrikePrice"], T, r, sigma, opt_type)

def calculate_portfolio_delta(
    option_df: pd.DataFrame,
    signals: List[Dict[str, Any]],
    current_date,
    S,
    contract_col="Series",
    date_col="Date",
    multiplier=50
):
    """
    Calculate portfolio NET DELTA exposure
    """
    df = option_df.copy()

    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    current_date = pd.to_datetime(current_date).normalize()

    legs, _ = signals_to_position_legs(signals, multiplier)

    if not legs:
        return 0.0

    total_delta = 0.0

    for leg in legs:
        contract = leg["contract"]

        sub = df[
            (df[contract_col] == contract) &
            (df[date_col] == current_date)
        ]

        if sub.empty:
            continue

        row = sub.iloc[0]

        delta = estimate_delta(row, S)

        if pd.isna(delta):
            continue

        total_delta += delta * leg["qty"] * leg["sign"] * multiplier

    return total_delta

def filter_liquid_options(df: pd.DataFrame, min_volume: int = 10, min_oi: int = 10) -> pd.DataFrame:
    """
    Liquidity filtering.
    Here Open is treated as open interest, consistent with your current code.
    """
    out = df.copy()

    if "Volume" in out.columns:
        vol = pd.to_numeric(out["Volume"], errors="coerce")
        if vol.notna().any():
            out = out[vol >= min_volume]

    # if "Open" in out.columns:
    #     oi = pd.to_numeric(out["Open"], errors="coerce")
    #     if oi.notna().any():
    #         out = out[oi >= min_oi]

    return out


def calculate_position_size(
    max_loss_per_trade,
    account_value=1_000_000,
    risk_pct=0.01,
    capital_alloc_pct=0.05,
    min_qty=1
):
    """
    Hybrid sizing:
    - Risk constraint
    - Capital deployment constraint

    Position size must satisfy BOTH:
        • Max loss risk budget
        • Capital allocation budget
    """

    if max_loss_per_trade <= 0:
        return 0

    # --------------------------
    # ① Risk budget sizing
    # --------------------------
    risk_budget = account_value * risk_pct
    qty_risk = risk_budget // max_loss_per_trade

    # --------------------------
    # ② Capital deployment sizing
    # --------------------------
    capital_budget = account_value * capital_alloc_pct
    qty_capital = capital_budget // max_loss_per_trade

    # --------------------------
    # ③ Final qty must respect BOTH
    # --------------------------
    qty = int(min(qty_risk, qty_capital))

    return max(qty, min_qty)


def generate_single_spread(
    scenario_type,
    opt_type,
    delta_buy_target,
    delta_sell_target,
    S,
    subset,
    use_pct,
    contract_multiplier,
    account_value,
    risk_pct,
):
    """
    Generate one vertical spread.
    Returns dict or None.
    """
    subset = subset.copy()

    if subset.empty:
        return None

    # 1) Strike selection
    if use_pct:
        if opt_type == "call":
            buy_pct = 0.02 if abs(delta_buy_target) > 0.3 else 0.03
            sell_pct = 0.05 if abs(delta_sell_target) > 0.15 else 0.06
            buy_target = S * (1 + buy_pct)
            sell_target = S * (1 + sell_pct)
        else:
            buy_pct = 0.02 if abs(delta_buy_target) > 0.3 else 0.03
            sell_pct = 0.05 if abs(delta_sell_target) > 0.15 else 0.06
            buy_target = S * (1 - buy_pct)
            sell_target = S * (1 - sell_pct)

        strikes = sorted(subset["StrikePrice"].dropna().unique())
        if len(strikes) < 2:
            return None

        k_buy = find_closest_strike(strikes, buy_target)
        k_sell = find_closest_strike(strikes, sell_target)

    else:
        subset["Delta_abs"] = subset["Delta"].abs()

        buy_candidates = subset[subset["Delta_abs"] <= abs(delta_buy_target) * 1.2]
        if buy_candidates.empty:
            buy_candidates = subset.nsmallest(1, "Delta_abs")
        k_buy = buy_candidates.loc[
            (buy_candidates["Delta_abs"] - abs(delta_buy_target)).abs().idxmin(),
            "StrikePrice",
        ]

        sell_candidates = subset[subset["Delta_abs"] >= abs(delta_sell_target) * 0.8]
        if sell_candidates.empty:
            sell_candidates = subset.nlargest(1, "Delta_abs")
        k_sell = sell_candidates.loc[
            (sell_candidates["Delta_abs"] - abs(delta_sell_target)).abs().idxmin(),
            "StrikePrice",
        ]

    buy_contract = subset[subset["StrikePrice"] == k_buy].iloc[0]
    sell_contract = subset[subset["StrikePrice"] == k_sell].iloc[0]

    # 2) Force distinct strikes
    if buy_contract["StrikePrice"] == sell_contract["StrikePrice"]:
        strikes = sorted(subset["StrikePrice"].dropna().unique())
        if len(strikes) < 2:
            return None

        idx = strikes.index(k_buy)
        if opt_type == "call":
            if k_buy <= k_sell:
                if idx > 0:
                    k_buy = strikes[idx - 1]
                elif idx < len(strikes) - 1:
                    k_buy = strikes[idx + 1]
            else:
                if idx < len(strikes) - 1:
                    k_buy = strikes[idx + 1]
                elif idx > 0:
                    k_buy = strikes[idx - 1]
        else:
            if k_buy >= k_sell:
                if idx < len(strikes) - 1:
                    k_buy = strikes[idx + 1]
                elif idx > 0:
                    k_buy = strikes[idx - 1]
            else:
                if idx > 0:
                    k_buy = strikes[idx - 1]
                elif idx < len(strikes) - 1:
                    k_buy = strikes[idx + 1]

        buy_contract = subset[subset["StrikePrice"] == k_buy].iloc[0]

    # 3) Liquidity filter
    buy_df = filter_liquid_options(pd.DataFrame([buy_contract]))
    sell_df = filter_liquid_options(pd.DataFrame([sell_contract]))
    if buy_df.empty or sell_df.empty:
        return None

    buy_contract = buy_df.iloc[0]
    sell_contract = sell_df.iloc[0]

    # 4) Prices
    price_col = "SettlementPrice"
    if price_col not in buy_contract.index:
        return None

    buy_price = float(buy_contract[price_col])
    sell_price = float(sell_contract[price_col])

    # 5) Max loss
    spread_width = abs(float(k_sell) - float(k_buy))

    if "credit" in scenario_type:
        net_credit = sell_price - buy_price
        max_loss_per_spread = spread_width - net_credit
    else:
        net_debit = buy_price - sell_price
        max_loss_per_spread = net_debit

    max_loss_per_contract = max_loss_per_spread * contract_multiplier
    quantity = calculate_position_size(
        max_loss_per_contract,
        account_value=account_value,
        risk_pct=risk_pct,
    )

    if quantity == 0:
        return None

    return {
        "buy_contract": buy_contract,
        "sell_contract": sell_contract,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "k_buy": float(k_buy),
        "k_sell": float(k_sell),
        "quantity": int(quantity),
        "max_loss_per_spread": float(max_loss_per_spread),
        "scenario_type": scenario_type,
    }


def generate_trade_signals(scenario, date, df, account_value=1_000_000, risk_pct=0.01):
    """
    Generate trading signals using original column names.
    """
    df = df.copy()

    required_cols = ["Date", "ExpDate", "StrikePrice", "Type", "Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # normalize types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["ExpDate"] = pd.to_datetime(df["ExpDate"], errors="coerce").dt.normalize()
    date = pd.to_datetime(date).normalize()

    numeric_cols = [
        "StrikePrice", "SettlementPrice", "ImpliedVolatility", "dte", "Close",
        "Volume", "Open", "High", "Low", "Returns", "F",
        "Open_vhsi", "High_vhsi", "Low_vhsi", "Close_vhsi"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "ExpDate", "StrikePrice", "Type", "dte", "Close"])

    df["Type"] = df["Type"].astype(str).str.lower()
    df = df[df["Type"].isin(["call", "put", "c", "p"])]
    df["Type"] = df["Type"].replace({"c": "call", "p": "put"})

    day_data, actual_trade_date = get_options_for_date(df,date, return_actual_date=True)
    if day_data.empty:
        print(f"[{date}] No data")
        return []
    
    S = float(day_data["Close"].iloc[0])

    target_exp = get_target_expiry(day_data)
    if target_exp is None:
        print(f"[{date}] No suitable expiration")
        return []

    scenario_config = {
        "bull_highvol": {"type": "put_credit_spread", "opt": "put", "buy_delta": 0.12, "sell_delta": 0.28},
        "bull_lowvol": {"type": "call_debit_spread", "opt": "call", "buy_delta": 0.32, "sell_delta": 0.17},
        "bull_medianvol": {"type": "call_debit_spread", "opt": "call", "buy_delta": 0.42, "sell_delta": 0.22},
        "bear_highvol": {"type": "call_credit_spread", "opt": "call", "buy_delta": 0.12, "sell_delta": 0.28},
        "bear_lowvol": {"type": "put_debit_spread", "opt": "put", "buy_delta": -0.32, "sell_delta": -0.17},
        "bear_medianvol": {"type": "put_debit_spread", "opt": "put", "buy_delta": -0.42, "sell_delta": -0.22},
        "range_highvol": {
            "type": "iron_condor",
            "opt": None,
            "put_buy_delta": 0.10,
            "put_sell_delta": 0.22,
            "call_buy_delta": 0.10,
            "call_sell_delta": 0.22,
        },
        "range_lowvol": {
            "type": "iron_condor",
            "put_buy_delta": 0.05,
            "put_sell_delta": 0.15,
            "call_buy_delta": 0.05,
            "call_sell_delta": 0.15,
        },
        "range_medianvol": {
            "type": "iron_condor",
            "put_buy_delta": 0.10,
            "put_sell_delta": 0.25,
            "call_buy_delta": 0.10,
            "call_sell_delta": 0.25,
        },
    }

    if scenario not in scenario_config:
        print(f"[{date}] Unknown scenario: {scenario}")
        return []

    config = scenario_config[scenario]
    contract_multiplier = 50

    if config["type"] == "iron_condor":
        put_subset = day_data[(day_data["ExpDate"] == target_exp) & (day_data["Type"] == "put")].copy()
        call_subset = day_data[(day_data["ExpDate"] == target_exp) & (day_data["Type"] == "call")].copy()

        if put_subset.empty or call_subset.empty:
            print(f"[{date}] Iron condor missing put or call options")
            return []

        put_subset["Delta"] = put_subset.apply(lambda row: estimate_delta(row, S), axis=1)
        call_subset["Delta"] = call_subset.apply(lambda row: estimate_delta(row, S), axis=1)

        use_pct_put = put_subset["Delta"].isna().all()
        use_pct_call = call_subset["Delta"].isna().all()

        put_result = generate_single_spread(
            scenario_type="put_credit_spread",
            opt_type="put",
            delta_buy_target=config["put_buy_delta"],
            delta_sell_target=config["put_sell_delta"],
            S=S,
            subset=put_subset,
            use_pct=use_pct_put,
            contract_multiplier=contract_multiplier,
            account_value=account_value,
            risk_pct=risk_pct,
        )

        call_result = generate_single_spread(
            scenario_type="call_credit_spread",
            opt_type="call",
            delta_buy_target=config["call_buy_delta"],
            delta_sell_target=config["call_sell_delta"],
            S=S,
            subset=call_subset,
            use_pct=use_pct_call,
            contract_multiplier=contract_multiplier,
            account_value=account_value,
            risk_pct=risk_pct,
        )

        if put_result is None or call_result is None:
            print(f"[{date}] Iron condor contract selection failed")
            return []

        qty = min(put_result["quantity"], call_result["quantity"])
        if qty == 0:
            return []

        signals = [
            {
                "action": "SELL",
                "contract": put_result["sell_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "put_sell",
                "price_est": put_result["sell_price"],
            },
            {
                "action": "BUY",
                "contract": put_result["buy_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "put_buy",
                "price_est": put_result["buy_price"],
            },
            {
                "action": "SELL",
                "contract": call_result["sell_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "call_sell",
                "price_est": call_result["sell_price"],
            },
            {
                "action": "BUY",
                "contract": call_result["buy_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "call_buy",
                "price_est": call_result["buy_price"],
            },
            {
                "meta": {
                    "date": date,
                    "scenario": scenario,
                    "strategy_type": "iron_condor",
                    "underlying_price": S,
                    "expiry": target_exp,
                    "put_sell_strike": put_result["k_sell"],
                    "put_buy_strike": put_result["k_buy"],
                    "call_sell_strike": call_result["k_sell"],
                    "call_buy_strike": call_result["k_buy"],
                    "quantity": qty,
                    "put_max_loss": put_result["max_loss_per_spread"],
                    "call_max_loss": call_result["max_loss_per_spread"],
                }
            },
        ]
        return signals

    else:
        opt_type = config["opt"]
        subset = day_data[(day_data["ExpDate"] == target_exp) & (day_data["Type"] == opt_type)].copy()

        if subset.empty:
            print(f"[{date}] No {opt_type} options")
            return []

        subset["Delta"] = subset.apply(lambda row: estimate_delta(row, S), axis=1)
        use_pct = subset["Delta"].isna().all()

        result = generate_single_spread(
            scenario_type=config["type"],
            opt_type=opt_type,
            delta_buy_target=config["buy_delta"],
            delta_sell_target=config["sell_delta"],
            S=S,
            subset=subset,
            use_pct=use_pct,
            contract_multiplier=contract_multiplier,
            account_value=account_value,
            risk_pct=risk_pct,
        )

        if result is None:
            print(f"[{date}] Spread selection failed")
            return []

        qty = result["quantity"]
        signals = []

        if "credit" in config["type"]:
            signals.append({
                "action": "SELL",
                "contract": result["sell_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "sell_leg",
                "price_est": result["sell_price"],
            })
            signals.append({
                "action": "BUY",
                "contract": result["buy_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "buy_leg",
                "price_est": result["buy_price"],
            })
        else:
            signals.append({
                "action": "BUY",
                "contract": result["buy_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "buy_leg",
                "price_est": result["buy_price"],
            })
            signals.append({
                "action": "SELL",
                "contract": result["sell_contract"]["Series"],
                "quantity": qty,
                "strategy": scenario,
                "leg": "sell_leg",
                "price_est": result["sell_price"],
            })

        signals.append({
            "meta": {
                "date": date,
                "scenario": scenario,
                "strategy_type": config["type"],
                "underlying_price": S,
                "expiry": target_exp,
                "buy_strike": result["k_buy"],
                "sell_strike": result["k_sell"],
                "quantity": qty,
                "max_loss_per_spread": result["max_loss_per_spread"],
            }
        })

        return signals


def signals_to_position_legs(signals: List[Dict[str, Any]], contract_multiplier: float = 50.0):
    """
    Convert signals into position legs.
    """
    legs = []
    meta = None

    for sig in signals:
        if "meta" in sig:
            meta = sig["meta"]
            continue

        action = sig["action"].upper()
        sign = 1 if action == "BUY" else -1

        legs.append({
            "contract": sig["contract"],
            "sign": sign,
            "qty": float(sig["quantity"]),
            "entry_price": float(sig["price_est"]),
            "leg": sig.get("leg"),
            "strategy": sig.get("strategy"),
            "multiplier": contract_multiplier,
        })

    return legs, meta


def mark_signals_to_market(
    option_df: pd.DataFrame,
    signals: List[Dict[str, Any]],
    start_date,
    end_date,
    price_col: str = "SettlementPrice",
    contract_col: str = "Series",
    date_col: str = "Date",
    contract_multiplier: float = 50.0,
) -> pd.DataFrame:
    """
    Generic MTM engine using original column names.
    pnl_leg_t = sign * (price_t - entry_price) * qty * multiplier
    """
    df = option_df.copy()

    if date_col not in df.columns:
        raise KeyError(f"{date_col} not found in option_df. Available columns: {df.columns.tolist()}")
    if contract_col not in df.columns:
        raise KeyError(f"{contract_col} not found in option_df. Available columns: {df.columns.tolist()}")
    if price_col not in df.columns:
        raise KeyError(f"{price_col} not found in option_df. Available columns: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()

    legs, meta = signals_to_position_legs(signals, contract_multiplier=contract_multiplier)
    if not legs:
        return pd.DataFrame()

    all_dates = sorted(
        df.loc[(df[date_col] >= start_date) & (df[date_col] <= end_date), date_col].unique()
    )
    if not all_dates:
        return pd.DataFrame()

    contract_prices = {}
    for leg in legs:
        c = leg["contract"]
        sub = df.loc[df[contract_col] == c, [date_col, price_col]].copy()
        sub = sub.sort_values(date_col).drop_duplicates(subset=[date_col])
        sub = sub.set_index(date_col)[price_col]
        contract_prices[c] = sub

    rows = []
    last_price = {leg["contract"]: leg["entry_price"] for leg in legs}

    for d in all_dates:
        total_pnl = 0.0
        row = {"Date": d}

        for i, leg in enumerate(legs, start=1):
            c = leg["contract"]
            px_series = contract_prices.get(c)

            if px_series is not None and d in px_series.index and pd.notna(px_series.loc[d]):
                px = float(px_series.loc[d])
                last_price[c] = px
            else:
                px = last_price[c]

            pnl = leg["sign"] * (px - leg["entry_price"]) * leg["qty"] * leg["multiplier"]
            row[f"Price_leg_{i}"] = px
            row[f"PnL_leg_{i}"] = pnl
            total_pnl += pnl

        row["Total_PnL"] = total_pnl
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    out["Daily_PnL"] = out["Total_PnL"].diff().fillna(out["Total_PnL"])
    return out

def apply_risk_management(
    mtm_df: pd.DataFrame,
    option_df: pd.DataFrame,
    signals: List[Dict[str, Any]],
    start_date,
    underlying_series: pd.Series,
    stop_loss_pct=0.5,
    delta_limit=300
):
    """
    Apply:

    1. FLOATING STOP LOSS (% of max risk)
    2. PORTFOLIO NET DELTA LIMIT

    Returns truncated MTM until exit date
    """

    if mtm_df.empty:
        return mtm_df, None, None

    # get meta info
    meta = [x["meta"] for x in signals if "meta" in x][0]

    if "put_max_loss" in meta:
        max_risk = (meta["put_max_loss"] + meta["call_max_loss"]) * meta["quantity"] * 50
    else:
        max_risk = meta["max_loss_per_spread"] * meta["quantity"] * 50

    stop_threshold = -stop_loss_pct * max_risk

    mtm_df = mtm_df.copy()

    for i in range(len(mtm_df)):

        d = mtm_df.iloc[i]["Date"]

        pnl = mtm_df.iloc[i]["Total_PnL"]

        S = underlying_series.loc[d]

        # ---- STOP LOSS CHECK ----
        if pnl <= stop_threshold:
            return mtm_df.iloc[:i+1], d, "STOP_LOSS"

        # ---- DELTA CHECK ----
        net_delta = calculate_portfolio_delta(
            option_df,
            signals,
            d,
            S
        )

        if abs(net_delta) >= delta_limit:
            return mtm_df.iloc[:i+1], d, f"DELTA_BREACH({int(net_delta)})"

    return mtm_df, None, None