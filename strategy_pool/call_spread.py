import pandas as pd
import numpy as np


def call_debit_spread(
    option_df: pd.DataFrame,
    start_date,
    end_date,
    price_col: str = "SettlementPrice",
    underlying_col: str = "F",
    expiry_col: str = "ExpDate",
    date_col: str = "Date",
    strike_col: str = "StrikePrice",
    type_col: str = "Type",
    qty: int = 1,
    multiplier: float = 1.0,
    short_offset: int = 2,
):
    """
    Build and mark-to-market a call debit spread from start_date to end_date.

    Strategy:
    - On start_date, choose the nearest expiry >= end_date
    - Buy 1 ATM call
    - Sell 1 higher-strike call (ATM higher by `short_offset` strike steps)
    - Hold through end_date
    - Return daily MTM floating PnL path

    Parameters
    ----------
    option_df : pd.DataFrame
        Must contain columns:
        [Date, Expiry, Strike, Type, price_col, underlying_col]
    start_date : str or datetime
    end_date : str or datetime
    price_col : str
        Option price column used for valuation
    underlying_col : str
        Column used to determine ATM strike on entry date, usually F or S
    qty : int
        Number of spreads
    multiplier : float
        Contract multiplier
    short_offset : int
        Number of strike steps above ATM for the short call leg

    Returns
    -------
    result : dict
        {
            "summary": ...,
            "mtm_path": DataFrame
        }
    """

    df = option_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[expiry_col] = pd.to_datetime(df[expiry_col])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # normalize type
    df[type_col] = df[type_col].astype(str).str.lower().str.strip()
    df[type_col] = df[type_col].replace({
        "Call": "call",
        "Put": "put",
        "c": "call",
        "p": "put",
    })

    # keep calls only
    calls = df[df[type_col] == "call"].copy()

    # ---------- 1) choose expiry ----------
    entry_universe = calls[
        (calls[date_col] >= start_date) &
        (calls[expiry_col] >= end_date)
    ].copy()
    # print(entry_universe)
    if entry_universe.empty:
        raise ValueError("No eligible call options on start_date with Expiry >= end_date.")

    chosen_expiry = entry_universe[expiry_col].min()
    entry_slice = entry_universe[entry_universe[expiry_col] == chosen_expiry].copy()

    # ---------- 2) choose ATM long strike ----------
    if underlying_col not in entry_slice.columns:
        raise ValueError(f"Missing underlying column: {underlying_col}")

    f0 = float(entry_slice[underlying_col].iloc[0])
    entry_slice["atm_dist"] = (entry_slice[strike_col] - f0).abs()
    entry_slice = entry_slice.sort_values(["atm_dist", strike_col])

    k_long = float(entry_slice.iloc[0][strike_col])

    # ---------- 3) choose higher strike short leg ----------
    strikes = sorted(entry_slice[strike_col].dropna().unique())
    if k_long not in strikes:
        raise ValueError("ATM strike selection failed.")

    atm_idx = strikes.index(k_long)
    short_idx = atm_idx + short_offset
    if short_idx >= len(strikes):
        raise ValueError("Not enough higher strikes to form the short call leg.")

    k_short = float(strikes[short_idx])

    # ---------- 4) entry prices ----------
    entry_long_row = entry_slice[entry_slice[strike_col] == k_long]
    entry_short_row = entry_slice[entry_slice[strike_col] == k_short]
    # print(entry_long_row)
    # print(entry_short_row)
    if entry_long_row.empty or entry_short_row.empty:
        raise ValueError("Missing entry leg prices.")

    entry_long_price = float(entry_long_row.iloc[0][price_col])
    entry_short_price = float(entry_short_row.iloc[0][price_col])

    entry_debit_points = entry_long_price - entry_short_price
    entry_debit_cash = entry_debit_points * qty * multiplier

    # ---------- 5) MTM path ----------
    path_dates = sorted(
        calls.loc[
            (calls[date_col] >= start_date) &
            (calls[date_col] <= end_date),
            date_col
        ].drop_duplicates()
    )

    records = []
    for d in path_dates:
        daily = calls[
            (calls[date_col] == d) &
            (calls[expiry_col] == chosen_expiry)
        ].copy()

        long_row = daily[daily[strike_col] == k_long]
        short_row = daily[daily[strike_col] == k_short]
        # print(long_row)
        # skip dates where one leg is missing
        if long_row.empty or short_row.empty:
            continue

        long_price = float(long_row.iloc[0][price_col])
        short_price = float(short_row.iloc[0][price_col])

        spread_value_points = long_price - short_price
        pnl_points = (spread_value_points - entry_debit_points) * qty
        pnl_cash = pnl_points * multiplier

        records.append({
            "Date": d,
            "Expiry": chosen_expiry,
            "K_long": k_long,
            "K_short": k_short,
            "long_price": long_price,
            "short_price": short_price,
            "spread_value_points": spread_value_points,
            "entry_debit_points": entry_debit_points,
            "floating_pnl_points": pnl_points,
            "floating_pnl_cash": pnl_cash,
        })


    mtm_path = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)

    if mtm_path.empty:
        raise ValueError("No MTM path could be constructed between start_date and end_date.")

    final_row = mtm_path.iloc[-1]

    summary = {
        "start_date": start_date,
        "end_date": end_date,
        "chosen_expiry": chosen_expiry,
        "underlying_at_entry": f0,
        "K_long": k_long,
        "K_short": k_short,
        "entry_long_price": entry_long_price,
        "entry_short_price": entry_short_price,
        "entry_debit_points": entry_debit_points,
        "entry_debit_cash": entry_debit_cash,
        "final_spread_value_points": float(final_row["spread_value_points"]),
        "final_floating_pnl_points": float(final_row["floating_pnl_points"]),
        "final_floating_pnl_cash": float(final_row["floating_pnl_cash"]),
        "n_obs": len(mtm_path),
    }

    return {
        "summary": summary,
        "mtm_path": mtm_path,
    }