import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Dict, Any, Tuple, Optional



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
    """
    if "Date" not in df.columns:
        raise KeyError(f"'Date' not found in columns: {df.columns.tolist()}")

    target_date = pd.to_datetime(date).normalize()

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["Date"])

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
    Black-Scholes style delta approximation.
    option_type: 'call' or 'put'
    """
    if pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(sigma):
        return np.nan
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan

    option_type = str(option_type).lower()
    if option_type not in {"call", "put"}:
        return np.nan

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def estimate_delta(row, S, r=0.02):
    """Estimate option delta from one option row."""
    T = row.get("dte", np.nan) / 365.0
    if pd.isna(T) or T <= 0:
        return np.nan

    sigma = row.get("ImpliedVolatility", np.nan)
    if pd.isna(sigma) or sigma <= 0:
        return np.nan

    if sigma > 3:
        sigma = sigma / 100.0

    opt_type = str(row.get("Type", "")).lower()
    if opt_type == "c":
        opt_type = "call"
    elif opt_type == "p":
        opt_type = "put"

    if opt_type not in {"call", "put"}:
        return np.nan

    K = row.get("StrikePrice", np.nan)
    if pd.isna(K) or K <= 0:
        return np.nan

    underlying = row.get("F", np.nan)
    if pd.isna(underlying) or underlying <= 0:
        underlying = S

    return black_delta(underlying, K, T, r, sigma, opt_type)


def filter_liquid_options(df: pd.DataFrame, min_volume: int = 10, min_oi: int = 10) -> pd.DataFrame:
    """
    Liquidity filtering.
    Here Open is treated as open interest if needed.
    """
    out = df.copy()

    if "Volume" in out.columns:
        vol = pd.to_numeric(out["Volume"], errors="coerce")
        if vol.notna().any():
            out = out[vol >= min_volume]

    # Optional if "Open" is really OI:
    # if "Open" in out.columns:
    #     oi = pd.to_numeric(out["Open"], errors="coerce")
    #     if oi.notna().any():
    #         out = out[oi >= min_oi]

    return out


def calculate_position_size(
    max_loss_per_trade,
    account_value=1_000_000,
    risk_pct=0.01,
    capital_alloc_pct=0.1,
    min_qty=1
):
    if pd.isna(max_loss_per_trade) or max_loss_per_trade <= 0:
        return 0

    risk_budget = account_value * risk_pct
    qty_risk = risk_budget // max_loss_per_trade

    capital_budget = account_value * capital_alloc_pct
    qty_capital = capital_budget // max_loss_per_trade

    qty = int(min(qty_risk, qty_capital))

    if qty < min_qty:
        return 0
    return qty


def select_contract_by_delta_or_pct(subset, target_delta, S, opt_type, use_pct):
    """
    Select one contract by delta target; fall back to pct-based strike targeting.
    """
    subset = subset.copy()
    subset = subset.dropna(subset=["StrikePrice"])
    if subset.empty:
        return None

    strikes = sorted(subset["StrikePrice"].dropna().unique())
    if len(strikes) == 0:
        return None

    if use_pct or "Delta" not in subset.columns or subset["Delta"].isna().all():
        td = abs(target_delta)

        if opt_type == "call":
            pct = 0.02 if td > 0.30 else 0.04 if td > 0.15 else 0.06
            target_strike = S * (1 + pct)
        else:
            pct = 0.02 if td > 0.30 else 0.04 if td > 0.15 else 0.06
            target_strike = S * (1 - pct)

        k = find_closest_strike(strikes, target_strike)
        pick = subset[subset["StrikePrice"] == k]
        if pick.empty:
            return None
        return pick.iloc[0]

    valid = subset.dropna(subset=["Delta"]).copy()
    if valid.empty:
        return None

    valid["delta_diff"] = (valid["Delta"] - target_delta).abs()
    valid = valid.sort_values(["delta_diff", "StrikePrice"])
    return valid.iloc[0]



def select_contract_by_strike(subset: pd.DataFrame, strike: float):
    subset = subset.copy()
    subset = subset.dropna(subset=["StrikePrice"])
    if subset.empty:
        return None

    strikes = sorted(subset["StrikePrice"].dropna().unique())
    if not strikes:
        return None

    k = find_closest_strike(strikes, strike)
    pick = subset[subset["StrikePrice"] == k]
    if pick.empty:
        return None
    return pick.iloc[0]



def get_strategy_config(direction: str, volatility: str, strength: str) -> Dict[str, Any]:
    """
    Map (direction, volatility, strength) -> strategy config.

    Philosophy
    ----------
    - lowvol + directional: favor debit spreads
    - highvol + directional: favor credit spreads
    - medianvol: split by conviction
        * weak/fragile directional view -> conservative credit spread
        * strong clean directional view -> debit spread
    - range:
        * medium/high vol -> iron condor
        * low vol -> no trade
    """
    direction = str(direction).lower()
    volatility = str(volatility).lower()
    strength = str(strength).lower()

    if direction not in {"bull", "bear", "range"}:
        raise ValueError(f"Invalid direction={direction}")
    if volatility not in {"lowvol", "medianvol", "highvol"}:
        raise ValueError(f"Invalid volatility={volatility}")
    if strength not in {"weak", "medium", "strong"}:
        raise ValueError(f"Invalid strength={strength}")

    # -------------------------
    # RANGE
    # -------------------------
    if direction == "range":
        if volatility == "lowvol":
            return {"type": "no_trade"}

        if volatility == "medianvol":
            if strength == "weak":
                return {
                    "type": "iron_condor",
                    "put_sell_delta": -0.12,
                    "call_sell_delta": 0.12,
                    "wing_width": 200,
                }
            elif strength == "medium":
                return {
                    "type": "iron_condor",
                    "put_sell_delta": -0.15,
                    "call_sell_delta": 0.15,
                    "wing_width": 400,
                }
            else:
                return {
                    "type": "iron_condor",
                    "put_sell_delta": -0.18,
                    "call_sell_delta": 0.18,
                    "wing_width": 400,
                }

        if volatility == "highvol":
            if strength == "weak":
                return {"type": "no_trade"}
            elif strength == "medium":
                return {
                    "type": "iron_condor",
                    "put_sell_delta": -0.14,
                    "call_sell_delta": 0.14,
                    "wing_width": 400,
                }
            else:
                return {
                    "type": "iron_condor",
                    "put_sell_delta": -0.18,
                    "call_sell_delta": 0.18,
                    "wing_width": 400,
                }

    # -------------------------
    # BULL
    # -------------------------
    if direction == "bull":
        if volatility == "lowvol":
            if strength == "weak":
                return {
                    "type": "call_debit_spread",
                    "opt": "call",
                    "buy_delta": 0.25,   # anchor long call
                }
            elif strength == "medium":
                return {
                    "type": "call_debit_spread",
                    "opt": "call",
                    "buy_delta": 0.32,
                }
            else:
                return {
                    "type": "call_debit_spread",
                    "opt": "call",
                    "buy_delta": 0.40,
                }

        if volatility == "medianvol":
            if strength == "weak":
                return {
                    "type": "put_credit_spread",
                    "opt": "put",
                    "sell_delta": -0.15,   # anchor short put
                }
            elif strength == "medium":
                return {
                    "type": "put_credit_spread",
                    "opt": "put",
                    "sell_delta": -0.20,

                }
            else:
                return {
                    "type": "call_debit_spread",
                    "opt": "call",
                    "buy_delta": 0.38,
                }

        if volatility == "highvol":
            if strength == "weak":
                return {
                    "type": "put_credit_spread",
                    "opt": "put",
                    "sell_delta": -0.16,
                }
            elif strength == "medium":
                return {
                    "type": "put_credit_spread",
                    "opt": "put",
                    "sell_delta": -0.24,
                }
            else:
                return {
                    "type": "put_credit_spread",
                    "opt": "put",
                    "sell_delta": -0.30,
                }

    # -------------------------
    # BEAR
    # -------------------------
    if direction == "bear":
        if volatility == "lowvol":
            if strength == "weak":
                return {
                    "type": "put_debit_spread",
                    "opt": "put",
                    "buy_delta": -0.25,   # anchor long put
                }
            elif strength == "medium":
                return {
                    "type": "put_debit_spread",
                    "opt": "put",
                    "buy_delta": -0.32,
                }
            else:
                return {
                    "type": "put_debit_spread",
                    "opt": "put",
                    "buy_delta": -0.40,
                }

        if volatility == "medianvol":
            if strength == "weak":
                return {
                    "type": "call_credit_spread",
                    "opt": "call",
                    "sell_delta": 0.15,   # anchor short call
                }
            elif strength == "medium":
                return {
                    "type": "call_credit_spread",
                    "opt": "call",
                    "sell_delta": 0.20,
                }
            else:
                return {
                    "type": "put_debit_spread",
                    "opt": "put",
                    "buy_delta": -0.38,
                }

        if volatility == "highvol":
            if strength == "weak":
                return {
                    "type": "call_credit_spread",
                    "opt": "call",
                    "sell_delta": 0.16,
                }
            elif strength == "medium":
                return {
                    "type": "call_credit_spread",
                    "opt": "call",
                    "sell_delta": 0.24,
                }
            else:
                return {
                    "type": "call_credit_spread",
                    "opt": "call",
                    "sell_delta": 0.30,
                }

    raise ValueError(
        f"Unsupported regime combination: direction={direction}, volatility={volatility}, strength={strength}"
    )

def generate_iron_condor(
    S,
    put_subset: pd.DataFrame,
    call_subset: pd.DataFrame,
    put_sell_delta: float,
    call_sell_delta: float,
    wing_width: float,
    contract_multiplier: float,
    account_size: float,
    risk_pct: float,
    capital_alloc_pct: float,
    min_credit_rr: float = 0.15,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Build iron condor as:
    - short put / short call selected by delta
    - long wings selected by fixed width from short strikes

    Returns
    -------
    (result_dict | None, fail_reason | None)
    """
    put_subset = put_subset.copy().dropna(subset=["StrikePrice", "SettlementPrice"])
    call_subset = call_subset.copy().dropna(subset=["StrikePrice", "SettlementPrice"])

    if put_subset.empty:
        return None, "put_subset_empty"
    if call_subset.empty:
        return None, "call_subset_empty"

    # 1) short legs by delta
    short_put = select_contract_by_delta_or_pct(
        subset=put_subset,
        target_delta=put_sell_delta,
        S=S,
        opt_type="put",
        use_pct=put_subset["Delta"].isna().all() if "Delta" in put_subset.columns else True,
    )
    short_call = select_contract_by_delta_or_pct(
        subset=call_subset,
        target_delta=call_sell_delta,
        S=S,
        opt_type="call",
        use_pct=call_subset["Delta"].isna().all() if "Delta" in call_subset.columns else True,
    )

    if short_put is None:
        return None, "short_put_none"
    if short_call is None:
        return None, "short_call_none"

    short_put_k = float(short_put["StrikePrice"])
    short_call_k = float(short_call["StrikePrice"])

    # Sanity: short put should be below short call
    if short_put_k >= short_call_k:
        return None, f"invalid_short_structure_put={short_put_k}_call={short_call_k}"

    # 2) long wings by fixed width
    target_long_put_k = short_put_k - wing_width
    target_long_call_k = short_call_k + wing_width

    long_put = select_contract_by_strike(put_subset, target_long_put_k)
    long_call = select_contract_by_strike(call_subset, target_long_call_k)

    if long_put is None:
        return None, "long_put_none"
    if long_call is None:
        return None, "long_call_none"

    long_put_k = float(long_put["StrikePrice"])
    long_call_k = float(long_call["StrikePrice"])

    # Ensure protective structure is correct
    if not (long_put_k < short_put_k):
        return None, f"invalid_put_wing_long={long_put_k}_short={short_put_k}"
    if not (long_call_k > short_call_k):
        return None, f"invalid_call_wing_long={long_call_k}_short={short_call_k}"

    put_width = short_put_k - long_put_k
    call_width = long_call_k - short_call_k

    if put_width <= 0:
        return None, f"invalid_put_width_{put_width}"
    if call_width <= 0:
        return None, f"invalid_call_width_{call_width}"

    # 3) prices
    short_put_price = float(pd.to_numeric(short_put["SettlementPrice"], errors="coerce"))
    long_put_price = float(pd.to_numeric(long_put["SettlementPrice"], errors="coerce"))
    short_call_price = float(pd.to_numeric(short_call["SettlementPrice"], errors="coerce"))
    long_call_price = float(pd.to_numeric(long_call["SettlementPrice"], errors="coerce"))

    if any(pd.isna(x) for x in [short_put_price, long_put_price, short_call_price, long_call_price]):
        return None, "invalid_leg_prices"

    put_credit = short_put_price - long_put_price
    call_credit = short_call_price - long_call_price
    total_credit = put_credit + call_credit

    if put_credit <= 0:
        return None, f"invalid_put_credit_{put_credit}"
    if call_credit <= 0:
        return None, f"invalid_call_credit_{call_credit}"
    if total_credit <= 0:
        return None, f"invalid_total_credit_{total_credit}"

    # 4) per-spread risk
    put_max_loss = put_width - put_credit
    call_max_loss = call_width - call_credit

    if put_max_loss <= 0:
        return None, f"invalid_put_max_loss_{put_max_loss}"
    if call_max_loss <= 0:
        return None, f"invalid_call_max_loss_{call_max_loss}"

    # Iron condor loss is max(one side loss), not sum of both sides
    condor_max_loss = max(put_max_loss, call_max_loss)

    # reward/risk filter
    reward_risk_ratio = total_credit / condor_max_loss
    if reward_risk_ratio < min_credit_rr:
        return None, f"condor_rr_too_low_{reward_risk_ratio:.4f}"

    max_loss_per_unit = condor_max_loss * contract_multiplier

    quantity = calculate_position_size(
        max_loss_per_unit,
        account_value=account_size,
        risk_pct=risk_pct,
        capital_alloc_pct=capital_alloc_pct,
    )

    if quantity <= 0:
        return None, f"quantity_zero|max_loss_per_unit={max_loss_per_unit}"

    return {
        "short_put": short_put,
        "long_put": long_put,
        "short_call": short_call,
        "long_call": long_call,
        "short_put_k": short_put_k,
        "long_put_k": long_put_k,
        "short_call_k": short_call_k,
        "long_call_k": long_call_k,
        "short_put_price": short_put_price,
        "long_put_price": long_put_price,
        "short_call_price": short_call_price,
        "long_call_price": long_call_price,
        "put_width": float(put_width),
        "call_width": float(call_width),
        "put_credit": float(put_credit),
        "call_credit": float(call_credit),
        "total_credit": float(total_credit),
        "put_max_loss": float(put_max_loss),
        "call_max_loss": float(call_max_loss),
        "condor_max_loss": float(condor_max_loss),
        "reward_risk_ratio": float(reward_risk_ratio),
        "quantity": int(quantity),
        "strategy_type": "iron_condor",
    }, None
    
    
def generate_single_spread(
    strategy_type,
    opt_type,
    delta_buy_target,
    delta_sell_target,
    S,
    subset,
    use_pct,
    contract_multiplier,
    account_size,
    risk_pct,
    capital_alloc_pct,
    unit_risk_frac: float = 0.6,
    min_credit_rr: float = 0.2,
    min_debit_rr: float = 0.7,
    min_width: float = 200,
    max_width: float = 800,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate one vertical spread using:
    - anchor leg by delta
    - other leg by risk-budget-constrained search

    Logic
    -----
    credit spread:
        short leg selected by delta_sell_target
        long protective leg searched from outer strikes under per-unit risk budget

    debit spread:
        long leg selected by delta_buy_target
        short leg searched from favorable-direction strikes under per-unit risk budget
    """
    subset = subset.copy()

    if subset.empty:
        return None, "subset_empty_initial"

    subset = subset.dropna(subset=["StrikePrice", "SettlementPrice"])
    if subset.empty:
        return None, "subset_empty_after_dropna"

    subset["StrikePrice"] = pd.to_numeric(subset["StrikePrice"], errors="coerce")
    subset["SettlementPrice"] = pd.to_numeric(subset["SettlementPrice"], errors="coerce")
    subset = subset.dropna(subset=["StrikePrice", "SettlementPrice"])

    if subset.empty:
        return None, "subset_empty_after_numeric_clean"

    strikes = sorted(subset["StrikePrice"].dropna().unique())
    if len(strikes) < 2:
        return None, "not_enough_strikes"

    total_risk_budget = account_size * risk_pct
    per_unit_risk_budget = total_risk_budget * unit_risk_frac

    if per_unit_risk_budget <= 0:
        return None, f"invalid_per_unit_risk_budget_{per_unit_risk_budget}"

    # -------------------------
    # CREDIT SPREAD
    # -------------------------
    if "credit" in strategy_type:
        anchor_contract = select_contract_by_delta_or_pct(
            subset=subset,
            target_delta=delta_sell_target,
            S=S,
            opt_type=opt_type,
            use_pct=use_pct,
        )
        if anchor_contract is None:
            return None, "short_contract_none"

        short_contract = anchor_contract
        short_k = float(short_contract["StrikePrice"])
        short_price = float(pd.to_numeric(short_contract["SettlementPrice"], errors="coerce"))

        if pd.isna(short_price):
            return None, "invalid_short_price"

        if strategy_type == "put_credit_spread":
            candidate_df = subset[
                subset["StrikePrice"] < short_k
            ].copy().sort_values("StrikePrice", ascending=False)
        elif strategy_type == "call_credit_spread":
            candidate_df = subset[
                subset["StrikePrice"] > short_k
            ].copy().sort_values("StrikePrice", ascending=True)
        else:
            return None, f"unsupported_credit_type_{strategy_type}"

        if candidate_df.empty:
            return None, "no_long_leg_candidates"

        feasible = []

        for _, row in candidate_df.iterrows():
            long_k = float(row["StrikePrice"])
            long_price = float(pd.to_numeric(row["SettlementPrice"], errors="coerce"))
            if pd.isna(long_price):
                continue

            spread_width = abs(short_k - long_k)
            if spread_width <= 0:
                continue

            # optional: keep same width filter for credit too
            if min_width is not None and spread_width < min_width:
                continue
            if max_width is not None and spread_width > max_width:
                continue

            net_credit = short_price - long_price
            if pd.isna(net_credit) or net_credit <= 0:
                continue

            max_loss_per_spread = spread_width - net_credit
            if pd.isna(max_loss_per_spread) or max_loss_per_spread <= 0:
                continue

            max_loss_per_unit = max_loss_per_spread * contract_multiplier
            if max_loss_per_unit > per_unit_risk_budget:
                continue

            reward_risk_ratio = net_credit / max_loss_per_spread
            if reward_risk_ratio < min_credit_rr:
                continue

            risk_utilization = max_loss_per_unit / per_unit_risk_budget
            target_util = 0.7
            score = (
                abs(risk_utilization - target_util) * 100
                - reward_risk_ratio * 50
                + spread_width * 0.05
            )

            feasible.append({
                "buy_contract": row,
                "sell_contract": short_contract,
                "k_buy": long_k,
                "k_sell": short_k,
                "buy_price": long_price,
                "sell_price": short_price,
                "spread_width": spread_width,
                "max_loss_per_spread": max_loss_per_spread,
                "max_profit_per_spread": net_credit,
                "reward_risk_ratio": reward_risk_ratio,
                "max_loss_per_unit": max_loss_per_unit,
                "score": score,
            })

        if not feasible:
            return None, "no_feasible_credit_spread_under_risk_budget"

        best = min(feasible, key=lambda x: (x["score"], x["spread_width"], -x["reward_risk_ratio"]))

    # -------------------------
    # DEBIT SPREAD
    # -------------------------
    else:
        anchor_contract = select_contract_by_delta_or_pct(
            subset=subset,
            target_delta=delta_buy_target,
            S=S,
            opt_type=opt_type,
            use_pct=use_pct,
        )
        if anchor_contract is None:
            return None, "long_contract_none"

        long_contract = anchor_contract
        long_k = float(long_contract["StrikePrice"])
        long_price = float(pd.to_numeric(long_contract["SettlementPrice"], errors="coerce"))

        if pd.isna(long_price):
            return None, "invalid_long_price"

        if strategy_type == "call_debit_spread":
            candidate_df = subset[
                subset["StrikePrice"] > long_k
            ].copy().sort_values("StrikePrice", ascending=True)
        elif strategy_type == "put_debit_spread":
            candidate_df = subset[
                subset["StrikePrice"] < long_k
            ].copy().sort_values("StrikePrice", ascending=False)
        else:
            return None, f"unsupported_debit_type_{strategy_type}"

        if candidate_df.empty:
            return None, "no_short_leg_candidates"

        feasible = []

        for _, row in candidate_df.iterrows():
            short_k = float(row["StrikePrice"])
            short_price = float(pd.to_numeric(row["SettlementPrice"], errors="coerce"))
            if pd.isna(short_price):
                continue

            spread_width = abs(short_k - long_k)
            if spread_width <= 0:
                continue

            # key change: enforce minimum width
            if min_width is not None and spread_width < min_width:
                continue
            if max_width is not None and spread_width > max_width:
                continue

            net_debit = long_price - short_price
            if pd.isna(net_debit) or net_debit <= 0:
                continue

            max_loss_per_spread = net_debit
            max_profit_per_spread = spread_width - net_debit

            if pd.isna(max_profit_per_spread) or max_profit_per_spread <= 0:
                continue

            max_loss_per_unit = max_loss_per_spread * contract_multiplier
            if max_loss_per_unit > per_unit_risk_budget:
                continue

            reward_risk_ratio = max_profit_per_spread / max_loss_per_spread
            if reward_risk_ratio < min_debit_rr:
                continue

            score = (
                spread_width * 0.2
                - reward_risk_ratio * 100.0
                + net_debit * 0.05
            )

            feasible.append({
                "buy_contract": long_contract,
                "sell_contract": row,
                "k_buy": long_k,
                "k_sell": short_k,
                "buy_price": long_price,
                "sell_price": short_price,
                "spread_width": spread_width,
                "max_loss_per_spread": max_loss_per_spread,
                "max_profit_per_spread": max_profit_per_spread,
                "reward_risk_ratio": reward_risk_ratio,
                "max_loss_per_unit": max_loss_per_unit,
                "score": score,
            })

        if not feasible:
            return None, f"no_feasible_debit_spread|min_width={min_width}|max_width={max_width}"

        best = min(feasible, key=lambda x: (x["score"], x["spread_width"], -x["reward_risk_ratio"]))

    quantity = calculate_position_size(
        best["max_loss_per_unit"],
        account_value=account_size,
        risk_pct=risk_pct,
        capital_alloc_pct=capital_alloc_pct,
    )

    if quantity <= 0:
        return None, f"quantity_zero|max_loss_per_unit={best['max_loss_per_unit']}"

    print(
        f"[generate_single_spread] selected | strategy={strategy_type} | "
        f"k_buy={best['k_buy']} | k_sell={best['k_sell']} | "
        f"buy_price={best['buy_price']} | sell_price={best['sell_price']} | "
        f"width={best['spread_width']} | max_loss_per_spread={best['max_loss_per_spread']} | "
        f"rr={best['reward_risk_ratio']:.4f} | per_unit_risk_budget={per_unit_risk_budget:.2f} | "
        f"max_loss_per_unit={best['max_loss_per_unit']:.2f} | qty={quantity}"
    )

    return {
        "buy_contract": best["buy_contract"],
        "sell_contract": best["sell_contract"],
        "buy_price": float(best["buy_price"]),
        "sell_price": float(best["sell_price"]),
        "k_buy": float(best["k_buy"]),
        "k_sell": float(best["k_sell"]),
        "spread_width": float(best["spread_width"]),
        "quantity": int(quantity),
        "max_loss_per_spread": float(best["max_loss_per_spread"]),
        "max_profit_per_spread": float(best["max_profit_per_spread"]),
        "reward_risk_ratio": float(best["reward_risk_ratio"]),
        "max_loss_per_unit": float(best["max_loss_per_unit"]),
        "per_unit_risk_budget": float(per_unit_risk_budget),
        "strategy_type": strategy_type,
    }, None
    
    
def generate_trade_signals(scenario, date, df, account_size=1_000_000, risk_pct=0.02, capital_alloc_pct=0.1) -> List[Dict[str, Any]]:
    """
    Generate trading signals from scenario dict:
    scenario = {
        "direction": "bull" | "bear" | "range",
        "volatility": "lowvol" | "medianvol" | "highvol",
        "strength": "weak" | "medium" | "strong",
        ...
    }
    """
    df = df.copy()

    required_cols = [
        "Date", "ExpDate", "StrikePrice", "Type", "Close",
        "SettlementPrice", "dte", "Series"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if not isinstance(scenario, dict):
        raise TypeError("scenario must be a dict with keys: direction, volatility, strength")

    direction = str(scenario.get("direction", "")).lower()
    volatility = str(scenario.get("volatility", "")).lower()
    strength = str(scenario.get("strength", "")).lower()

    if direction not in {"bull", "bear", "range"}:
        raise ValueError(f"Invalid direction: {direction}")
    if volatility not in {"lowvol", "medianvol", "highvol"}:
        raise ValueError(f"Invalid volatility: {volatility}")
    if strength not in {"weak", "medium", "strong"}:
        raise ValueError(f"Invalid strength: {strength}")

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

    config = get_strategy_config(direction, volatility, strength)

    day_data, actual_trade_date = get_options_for_date(df, date, return_actual_date=True)
    if day_data.empty:
        print(f"[{date}] No option data found")
        return []

    S = float(day_data["Close"].iloc[0])

    print(
        f"[{date}] Start signal generation | actual_trade_date={actual_trade_date} | "
        f"direction={direction} | volatility={volatility} | strength={strength} | "
        f"strategy_type={config['type']} | underlying={S}"
    )

    if config["type"] == "no_trade":
        print(f"[{date}] No trade under current rule")
        return [{
            "meta": {
                "date": actual_trade_date,
                "scenario": scenario,
                "strategy_type": "no_trade",
                "underlying_price": S,
                "expiry": None,
                "quantity": 0,
            }
        }]

    target_exp = get_target_expiry(day_data)
    if target_exp is None:
        print(f"[{date}] No suitable expiration found")
        return []

    target_dte_rows = day_data.loc[day_data["ExpDate"] == target_exp, "dte"]
    target_dte = target_dte_rows.iloc[0] if not target_dte_rows.empty else None

    print(f"[{date}] Selected expiry={target_exp} | dte={target_dte}")

    contract_multiplier = 50

    if config["type"] == "iron_condor":
        put_subset = day_data[
            (day_data["ExpDate"] == target_exp) & (day_data["Type"] == "put")
        ].copy()
        call_subset = day_data[
            (day_data["ExpDate"] == target_exp) & (day_data["Type"] == "call")
        ].copy()

        if put_subset.empty or call_subset.empty:
            print(f"[{date}] Iron condor missing put or call options")
            return []

        put_subset["Delta"] = put_subset.apply(lambda row: estimate_delta(row, S), axis=1)
        call_subset["Delta"] = call_subset.apply(lambda row: estimate_delta(row, S), axis=1)

        print(
            f"[{date}] Iron condor prep | "
            f"put_n={len(put_subset)} | call_n={len(call_subset)} | "
            f"put_sell_delta={config['put_sell_delta']} | call_sell_delta={config['call_sell_delta']} | "
            f"wing_width={config['wing_width']}"
        )

        condor_result, condor_fail_reason = generate_iron_condor(
            S=S,
            put_subset=put_subset,
            call_subset=call_subset,
            put_sell_delta=config["put_sell_delta"],
            call_sell_delta=config["call_sell_delta"],
            wing_width=config["wing_width"],
            contract_multiplier=contract_multiplier,
            account_size=account_size,
            risk_pct=risk_pct,
            capital_alloc_pct=capital_alloc_pct,
            min_credit_rr=0.15,
        )

        if condor_result is None:
            print(f"[{date}] Iron condor contract selection failed | reason={condor_fail_reason}")
            return []

        qty = condor_result["quantity"]

        print(
            f"[{date}] Iron condor selected | "
            f"put_short={condor_result['short_put_k']} | put_long={condor_result['long_put_k']} | "
            f"call_short={condor_result['short_call_k']} | call_long={condor_result['long_call_k']} | "
            f"put_width={condor_result['put_width']} | call_width={condor_result['call_width']} | "
            f"credit={condor_result['total_credit']} | qty={qty} | rr={condor_result['reward_risk_ratio']:.4f}"
        )

        return [
            {
                "action": "SELL",
                "contract": condor_result["short_put"]["Series"],
                "quantity": qty,
                "strategy": "iron_condor",
                "leg": "put_sell",
                "price_est": condor_result["short_put_price"],
            },
            {
                "action": "BUY",
                "contract": condor_result["long_put"]["Series"],
                "quantity": qty,
                "strategy": "iron_condor",
                "leg": "put_buy",
                "price_est": condor_result["long_put_price"],
            },
            {
                "action": "SELL",
                "contract": condor_result["short_call"]["Series"],
                "quantity": qty,
                "strategy": "iron_condor",
                "leg": "call_sell",
                "price_est": condor_result["short_call_price"],
            },
            {
                "action": "BUY",
                "contract": condor_result["long_call"]["Series"],
                "quantity": qty,
                "strategy": "iron_condor",
                "leg": "call_buy",
                "price_est": condor_result["long_call_price"],
            },
            {
                "meta": {
                    "date": actual_trade_date,
                    "scenario": scenario,
                    "strategy_type": "iron_condor",
                    "underlying_price": S,
                    "expiry": target_exp,
                    "put_sell_strike": condor_result["short_put_k"],
                    "put_buy_strike": condor_result["long_put_k"],
                    "call_sell_strike": condor_result["short_call_k"],
                    "call_buy_strike": condor_result["long_call_k"],
                    "quantity": qty,
                    "put_max_loss": condor_result["put_max_loss"],
                    "call_max_loss": condor_result["call_max_loss"],
                    "condor_max_loss": condor_result["condor_max_loss"],
                    "total_credit": condor_result["total_credit"],
                    "reward_risk_ratio": condor_result["reward_risk_ratio"],
                }
            },
        ]

    else:
        opt_type = config["opt"]
        subset = day_data[
            (day_data["ExpDate"] == target_exp) & (day_data["Type"] == opt_type)
        ].copy()

        if subset.empty:
            print(f"[{date}] No {opt_type} options for selected expiry")
            return []

        subset["Delta"] = subset.apply(lambda row: estimate_delta(row, S), axis=1)
        use_pct = subset["Delta"].isna().all()

        print(
            f"[{date}] Single spread prep | strategy={config['type']} | opt_type={opt_type} | "
            f"n_options={len(subset)} | strike_min={subset['StrikePrice'].min()} | "
            f"strike_max={subset['StrikePrice'].max()} | use_pct={use_pct} | "
            f"buy_delta={config.get('buy_delta')} | sell_delta={config.get('sell_delta')} | "
            f"wing_width={config.get('wing_width')}"
        )

        result, fail_reason = generate_single_spread(
            strategy_type=config["type"],
            opt_type=opt_type,
            delta_buy_target=config.get("buy_delta"),
            delta_sell_target=config.get("sell_delta"),
            S=S,
            subset=subset,
            use_pct=use_pct,
            contract_multiplier=contract_multiplier,
            account_size=account_size,
            risk_pct=risk_pct,
            capital_alloc_pct=capital_alloc_pct,
        )

        if result is None:
            print(f"[{date}] Spread selection failed | reason={fail_reason}")
            return []

        qty = result["quantity"]

        print(
            f"[{date}] Spread selected | strategy={config['type']} | "
            f"k_buy={result['k_buy']} | k_sell={result['k_sell']} | "
            f"buy_price={result['buy_price']} | sell_price={result['sell_price']} | "
            f"qty={qty} | max_loss_per_spread={result['max_loss_per_spread']}"
        )

        signals = []

        if "credit" in config["type"]:
            signals.append({
                "action": "SELL",
                "contract": result["sell_contract"]["Series"],
                "quantity": qty,
                "strategy": config["type"],
                "leg": "sell_leg",
                "price_est": result["sell_price"],
            })
            signals.append({
                "action": "BUY",
                "contract": result["buy_contract"]["Series"],
                "quantity": qty,
                "strategy": config["type"],
                "leg": "buy_leg",
                "price_est": result["buy_price"],
            })
        else:
            signals.append({
                "action": "BUY",
                "contract": result["buy_contract"]["Series"],
                "quantity": qty,
                "strategy": config["type"],
                "leg": "buy_leg",
                "price_est": result["buy_price"],
            })
            signals.append({
                "action": "SELL",
                "contract": result["sell_contract"]["Series"],
                "quantity": qty,
                "strategy": config["type"],
                "leg": "sell_leg",
                "price_est": result["sell_price"],
            })

        signals.append({
            "meta": {
                "date": actual_trade_date,
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

    all_dates = sorted(
        df.loc[(df[date_col] >= start_date) & (df[date_col] <= end_date), date_col].unique()
    )
    if not all_dates:
        return pd.DataFrame()

    if not legs:
        out = pd.DataFrame({"Date": all_dates})
        out["Total_PnL"] = 0.0
        out["Daily_PnL"] = 0.0
        return out

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



def calculate_portfolio_delta(
    option_df: pd.DataFrame,
    signals: List[Dict[str, Any]],
    current_date,
    S,
    contract_col="Series",
    date_col="Date",
    multiplier=50.0,
):
    """
    Calculate portfolio net delta exposure on a given date.
    """
    if option_df.empty or not signals:
        return 0.0

    df = option_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    current_date = pd.to_datetime(current_date).normalize()

    legs, _ = signals_to_position_legs(signals, contract_multiplier=multiplier)
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

    return float(total_delta)


def apply_risk_management(
    mtm_df: pd.DataFrame,
    option_df: pd.DataFrame,
    signals: List[Dict[str, Any]],
    underlying_series: pd.Series,
    stop_loss_ratio: float = 0.4,
    delta_limit: float = 300.0,
    logger=None,
):
    """
    Apply execution-level risk controls to an existing MTM path.

    Rules
    -----
    1. Stop loss:
       Exit when Total_PnL <= - stop_loss_ratio * max_risk

    2. Delta control:
       Exit when abs(portfolio_net_delta) >= delta_limit

    Returns
    -------
    truncated_mtm_df, risk_exit_date, risk_exit_reason
    """
    if mtm_df.empty:
        return mtm_df, None, None

    meta_list = [x["meta"] for x in signals if isinstance(x, dict) and "meta" in x]
    if not meta_list:
        if logger:
            logger.warning("RISK MGMT SKIPPED | no meta found in signals")
        return mtm_df, None, None

    meta = meta_list[-1]

    if meta.get("strategy_type") == "no_trade":
        if logger:
            logger.info("RISK MGMT SKIPPED | strategy_type=no_trade")
        return mtm_df, None, None

    qty = float(meta.get("quantity", 0) or 0)

    if "condor_max_loss" in meta:
        max_risk = float(meta["condor_max_loss"]) * qty * 50.0
    elif "put_max_loss" in meta and "call_max_loss" in meta:
        max_risk = max(float(meta["put_max_loss"]), float(meta["call_max_loss"])) * qty * 50.0
    else:
        max_loss_per_spread = float(meta.get("max_loss_per_spread", np.nan))
        max_risk = max_loss_per_spread * qty * 50.0

    if pd.isna(max_risk) or max_risk <= 0:
        if logger:
            logger.warning("RISK MGMT SKIPPED | invalid max_risk=%s", max_risk)
        return mtm_df, None, None

    stop_threshold = -stop_loss_ratio * max_risk
    out = mtm_df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()

    u = underlying_series.copy()
    u.index = pd.to_datetime(u.index, errors="coerce").normalize()

    if logger:
        logger.info(
            "RISK MGMT START | strategy_type=%s | qty=%s | max_risk=%.2f | stop_threshold=%.2f | delta_limit=%.2f",
            meta.get("strategy_type"),
            qty,
            max_risk,
            stop_threshold,
            delta_limit,
        )

    for i in range(len(out)):
        d = out.iloc[i]["Date"]
        pnl = float(out.iloc[i]["Total_PnL"])

        # 1) Stop loss
        if pnl <= stop_threshold:
            if logger:
                logger.warning(
                    "RISK EXIT | reason=STOP_LOSS | date=%s | pnl=%.2f | threshold=%.2f",
                    d, pnl, stop_threshold
                )
            truncated = out.iloc[: i + 1].copy()
            truncated["Daily_PnL"] = truncated["Total_PnL"].diff().fillna(truncated["Total_PnL"])
            return truncated, d, "STOP_LOSS"

        # 2) Delta control
        S = u.get(d, np.nan)
        if pd.notna(S):
            net_delta = calculate_portfolio_delta(
                option_df=option_df,
                signals=signals,
                current_date=d,
                S=S,
                contract_col="Series",
                date_col="Date",
                multiplier=50.0,
            )

            if abs(net_delta) >= delta_limit:
                if logger:
                    logger.warning(
                        "RISK EXIT | reason=DELTA_BREACH | date=%s | net_delta=%.2f | limit=%.2f | pnl=%.2f",
                        d, net_delta, delta_limit, pnl
                    )
                truncated = out.iloc[: i + 1].copy()
                truncated["Daily_PnL"] = truncated["Total_PnL"].diff().fillna(truncated["Total_PnL"])
                return truncated, d, f"DELTA_BREACH({int(net_delta)})"

    if logger:
        logger.info("RISK MGMT END | no risk exit triggered")

    return out, None, None