#计算持仓delta 
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
#止损
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
#调用方法
mtm = mark_signals_to_market(
    option_df,
    signals,
    entry_date,
    expiry_date
)

mtm_adj, exit_date, exit_reason = apply_risk_management(
    mtm_df       = mtm,
    option_df    = option_df,
    signals      = signals,
    start_date   = entry_date,
    underlying_series = underlying_close_series,
    stop_loss_pct = 0.5,
    delta_limit  = 300
)