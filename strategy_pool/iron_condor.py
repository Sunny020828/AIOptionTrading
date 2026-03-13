import pandas as pd
import numpy as np


def iron_condor(
    option_df: pd.DataFrame,
    start_date,
    end_date,
    price_col: str = "SettlementPrice",
    underlying_col: str = "Close",
    expiry_col: str = "ExpDate",
    date_col: str = "Date",
    strike_col: str = "StrikePrice",
    type_col: str = "Type",
    qty: int = 1,
    multiplier: float = 1.0,
    short_offset: int = 1,
    wing_width: int = 2,
):
    """
    Build and mark-to-market an Iron Condor from start_date to end_date.

    Strategy on entry date:
    - Choose the earliest trading date >= start_date
    - Choose nearest expiry >= end_date
    - Find ATM strike using underlying_col
    - Sell 1 OTM put  (ATM lower by `short_offset` strike steps)
    - Buy 1 further OTM put  (additional `wing_width` lower strike steps)
    - Sell 1 OTM call (ATM higher by `short_offset` strike steps)
    - Buy 1 further OTM call (additional `wing_width` higher strike steps)
    - Hold through end_date
    - Return daily MTM floating PnL path

    Parameters
    ----------
    option_df : pd.DataFrame
        Must contain columns:
        [date_col, expiry_col, strike_col, type_col, price_col, underlying_col]
    start_date : str or datetime
    end_date : str or datetime
    price_col : str
        Option price column
    underlying_col : str
        Underlying futures/spot column, used to locate ATM
    expiry_col, date_col, strike_col, type_col : str
        Column names
    qty : int
        Number of condors
    multiplier : float
        Contract multiplier
    short_offset : int
        How many strike steps away from ATM to place the short legs
    wing_width : int
        How many additional strike steps for the long protection wings

    Returns
    -------
    result : dict
        {
            "summary": dict,
            "legs": pd.DataFrame,
            "mtm": pd.DataFrame
        }
    """

    df = option_df.copy()

    # ---------- 0) basic cleaning ----------
    required_cols = [date_col, expiry_col, strike_col, type_col, price_col, underlying_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce")
    df[strike_col] = pd.to_numeric(df[strike_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[underlying_col] = pd.to_numeric(df[underlying_col], errors="coerce")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df = df.dropna(subset=[date_col, expiry_col, strike_col, price_col, type_col])

    # normalize option type
    df[type_col] = df[type_col].astype(str).str.lower().str.strip()
    df[type_col] = df[type_col].replace({
        "p": "put",
        "c": "call",
        "put": "put",
        "call": "call",
    })

    df = df[df[type_col].isin(["put", "call"])].copy()
    if df.empty:
        raise ValueError("No valid option rows after type normalization.")

    # ---------- 1) choose actual entry date ----------
    candidate_days = sorted(df.loc[df[date_col] >= start_date, date_col].drop_duplicates())
    if len(candidate_days) == 0:
        raise ValueError(
            f"No option data on or after start_date={start_date.date()}."
        )

    actual_start_date = pd.to_datetime(candidate_days[0])

    # ---------- 2) choose expiry from entry date ----------
    open_day = df[df[date_col] == actual_start_date].copy()
    if open_day.empty:
        raise ValueError("No option rows found on actual_start_date.")

    valid_expiries = sorted(open_day.loc[open_day[expiry_col] >= end_date, expiry_col].dropna().unique())
    if len(valid_expiries) == 0:
        raise ValueError("No expiry on/after end_date available from entry date chain.")

    chosen_expiry = pd.to_datetime(valid_expiries[0])
    open_day = open_day[open_day[expiry_col] == chosen_expiry].copy()

    if open_day.empty:
        raise ValueError("No option rows left after selecting chosen expiry.")

    # ---------- 3) locate ATM strike ----------
    if open_day[underlying_col].dropna().empty:
        raise ValueError(f"No valid '{underlying_col}' values on entry date.")

    underlying_px = float(open_day[underlying_col].dropna().iloc[0])

    strikes = np.sort(open_day[strike_col].dropna().unique())
    if len(strikes) == 0:
        raise ValueError("No strikes found on selected expiry.")

    atm_idx = int(np.argmin(np.abs(strikes - underlying_px)))

    put_short_idx = atm_idx - short_offset
    put_long_idx = put_short_idx - wing_width
    call_short_idx = atm_idx + short_offset
    call_long_idx = call_short_idx + wing_width

    if put_long_idx < 0 or put_short_idx < 0 or call_short_idx >= len(strikes) or call_long_idx >= len(strikes):
        raise ValueError("Not enough strikes to construct iron condor with current offsets.")

    k_put_long = float(strikes[put_long_idx])
    k_put_short = float(strikes[put_short_idx])
    k_call_short = float(strikes[call_short_idx])
    k_call_long = float(strikes[call_long_idx])

    # print(
    #     f"Selected strikes - put long: {k_put_long}, put short: {k_put_short}, "
    #     f"call short: {k_call_short}, call long: {k_call_long}"
    # )

    # ---------- 4) helper to fetch opening leg ----------
    def _get_open_leg(day_df: pd.DataFrame, strike: float, opt_type: str) -> pd.Series:
        leg = day_df[
            (day_df[strike_col] == strike) &
            (day_df[type_col] == opt_type)
        ].copy()

        if leg.empty:
            raise ValueError(f"Missing opening leg: {opt_type} @ strike {strike}")
        return leg.iloc[0]

    open_put_long = _get_open_leg(open_day, k_put_long, "put")
    open_put_short = _get_open_leg(open_day, k_put_short, "put")
    open_call_short = _get_open_leg(open_day, k_call_short, "call")
    open_call_long = _get_open_leg(open_day, k_call_long, "call")

    # ---------- 5) define legs ----------
    legs = pd.DataFrame([
        {
            "leg_name": "long_put",
            "side": +1,
            "type": "put",
            "strike": k_put_long,
            "open_price": float(open_put_long[price_col]),
        },
        {
            "leg_name": "short_put",
            "side": -1,
            "type": "put",
            "strike": k_put_short,
            "open_price": float(open_put_short[price_col]),
        },
        {
            "leg_name": "short_call",
            "side": -1,
            "type": "call",
            "strike": k_call_short,
            "open_price": float(open_call_short[price_col]),
        },
        {
            "leg_name": "long_call",
            "side": +1,
            "type": "call",
            "strike": k_call_long,
            "open_price": float(open_call_long[price_col]),
        },
    ])

    # side convention:
    #   +1 = long option
    #   -1 = short option
    # position value = side * current option price * qty * multiplier
    opening_position_value = float((legs["side"] * legs["open_price"] * qty * multiplier).sum())

    # net opening premium:
    # negative => net credit received
    # positive => net debit paid
    opening_net_cost = opening_position_value

    # ---------- 6) build daily MTM ----------
    hold_df = df[
        (df[date_col] >= actual_start_date) &
        (df[date_col] <= end_date) &
        (df[expiry_col] == chosen_expiry)
    ].copy()

    daily_dates = sorted(hold_df[date_col].dropna().unique())
    mtm_rows = []

    for d in daily_dates:
        day = hold_df[hold_df[date_col] == d].copy()
        # print(
        #     f"Processing date {pd.to_datetime(d).date()}, "
        #     f"available strikes: {sorted(day[strike_col].dropna().unique())}"
        # )

        leg_values = {}
        leg_prices = {}
        missing = False
        missing_info = None

        for _, leg in legs.iterrows():
            sub = day[
                (day[strike_col] == leg["strike"]) &
                (day[type_col] == leg["type"])
            ].copy()

            if sub.empty:
                missing = True
                missing_info = (leg["strike"], leg["type"])
                break

            px = float(sub.iloc[0][price_col])

            leg_prices[f"{leg['leg_name']}_price"] = px
            leg_values[leg["leg_name"]] = float(leg["side"] * px * qty * multiplier)

        if missing:
            print(
                f"Missing leg price for date {pd.to_datetime(d).date()}, "
                f"strike {missing_info[0]}, type {missing_info[1]}, skipping MTM."
            )
            continue

        current_position_value = float(sum(leg_values.values()))
        pnl = current_position_value - opening_position_value

        row = {
            "Date": pd.to_datetime(d),
            "position_value": current_position_value,
            "pnl": pnl,
            "underlying": float(day[underlying_col].dropna().iloc[0])
            if underlying_col in day.columns and day[underlying_col].notna().any()
            else np.nan,
        }
        row.update(leg_prices)
        row.update(leg_values)
        mtm_rows.append(row)

    mtm = pd.DataFrame(mtm_rows).sort_values("Date").reset_index(drop=True)
    if mtm.empty:
        raise ValueError("No MTM rows could be constructed. Check data completeness for all four legs.")

    # ---------- 7) summary ----------
    put_spread_width = float(k_put_short - k_put_long)
    call_spread_width = float(k_call_long - k_call_short)
    spread_width = float(max(put_spread_width, call_spread_width))

    net_credit = max(0.0, -opening_net_cost)
    net_debit = max(0.0, opening_net_cost)

    max_profit = None
    max_loss = None
    breakeven_low = None
    breakeven_high = None

    # standard iron condor: usually entered for credit
    if net_credit > 0:
        max_profit = net_credit
        max_loss = spread_width * qty * multiplier - net_credit
        breakeven_low = k_put_short - (net_credit / (qty * multiplier))
        breakeven_high = k_call_short + (net_credit / (qty * multiplier))
    else:
        # unusual debit condor case
        max_profit = None
        max_loss = net_debit

    summary = {
        "strategy": "iron_condor",
        "start_date": start_date,
        "entry_date": actual_start_date,
        "end_date": end_date,
        "chosen_expiry": chosen_expiry,
        "underlying_at_open": underlying_px,
        "atm_strike": float(strikes[atm_idx]),
        "put_long_strike": k_put_long,
        "put_short_strike": k_put_short,
        "call_short_strike": k_call_short,
        "call_long_strike": k_call_long,
        "put_spread_width": put_spread_width,
        "call_spread_width": call_spread_width,
        "spread_width_used": spread_width,
        "opening_net_cost": opening_net_cost,   # < 0 means net credit received
        "net_credit": net_credit,
        "net_debit": net_debit,
        "max_profit_est": max_profit,
        "max_loss_est": max_loss,
        "breakeven_low_est": breakeven_low,
        "breakeven_high_est": breakeven_high,
        "final_position_value": float(mtm["position_value"].iloc[-1]),
        "final_pnl": float(mtm["pnl"].iloc[-1]),
        "n_obs": len(mtm),
    }

    return {
        "summary": summary,
        "legs": legs,
        "mtm": mtm,
    }