import os
from openai import OpenAI
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import os
import math

LOOKBACK_LEN=3
PRED_LEN=1

def get_data():
    price_chain_df=pd.read_csv('data/Monthly_HSI_Options.csv')
    hsi_df=pd.read_csv('data/HSI_data.csv')
    vhsi_df=pd.read_csv('data/VHSI_data.csv')

    def parse_number(x):
        """Parse numbers like '21,123.0' -> 21123.0"""
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace('"','').replace(',','')
        if s == "" or s.lower() in {"nan", "null"}:
            return np.nan
        try:
            return float(s)
        except:
            return np.nan
        


    future_df = pd.read_csv('data/Hang Seng Futures Historical Price Data.csv')
    future_df.columns = future_df.columns.str.replace('"', '', regex=False).str.strip()
    future_df["Date"] = pd.to_datetime(future_df["Date"], dayfirst=True).dt.normalize()

    for col in ["Price","Open","High","Low"]:
        future_df[col] = future_df[col].map(parse_number)


    # 最终你大概率只需要 Date + F(=Price)
    future_df = future_df.rename(columns={"Price":"F"})[["Date","F"]]
    future_df = future_df.sort_values("Date").reset_index(drop=True)
    future=future_df.copy().reset_index(drop=True)
    
    return price_chain_df, hsi_df, vhsi_df, future



def prepare_datasets(
    price_chain_df: pd.DataFrame,
    hsi_df: pd.DataFrame,
    vhsi_df: pd.DataFrame,
    future: pd.DataFrame,
    bad_contract_threshold: float = 0.85,
):
    """
    Clean raw datasets.

    Returns:
        pc_clean, hsi_clean, vhsi_clean
    """
    # ---------------- price_chain_df ----------------
    pc = price_chain_df.copy()

    for col in ["Date", "ExpDate"]:
        if col in pc.columns:
            pc[col] = pd.to_datetime(pc[col], errors="coerce")

    pc["dte"] = (pc["ExpDate"] - pc["Date"]).dt.days

    for col in ["StrikePrice", "SettlementPrice", "ImpliedVolatility", "dte"]:
        if col in pc.columns:
            pc[col] = pd.to_numeric(pc[col], errors="coerce")

    pc = pc.dropna(subset=["Date"]).drop_duplicates()

    grp = (
        pc.groupby(["ExpDate", "StrikePrice", "Type"])["SettlementPrice"]
        .agg(
            n_obs=lambda x: x.notna().sum(),
            pct_one=lambda x: np.mean(x.dropna() == 1.0) if x.notna().sum() > 0 else np.nan,
            n_unique=lambda x: x.dropna().nunique(),
        )
        .reset_index()
    )

    bad_contracts = grp.loc[
        grp["pct_one"] >= bad_contract_threshold,
        ["ExpDate", "StrikePrice", "Type"]
    ].copy()
    bad_contracts["drop_flag"] = 1

    pc = pc.merge(
        bad_contracts,
        on=["ExpDate", "StrikePrice", "Type"],
        how="left",
    )
    pc = pc.loc[pc["drop_flag"] != 1].drop(columns=["drop_flag"])

    sort_cols = [c for c in ["Date", "ExpDate", "StrikePrice", "Type"] if c in pc.columns]
    if sort_cols:
        pc = pc.sort_values(sort_cols).reset_index(drop=True)

    # ---------------- hsi_df ----------------
    hsi = hsi_df.copy()
    if "Date" in hsi.columns:
        hsi["Date"] = pd.to_datetime(hsi["Date"], errors="coerce")
        hsi = hsi.dropna(subset=["Date"]).sort_values("Date")

    price_cols = ["Close", "High", "Low", "Open", "Volume"]
    for col in price_cols:
        if col in hsi.columns:
            hsi[col] = pd.to_numeric(hsi[col], errors="coerce")
            hsi[col] = hsi[col].ffill()

    if "Returns" in hsi.columns:
        hsi["Returns"] = pd.to_numeric(hsi["Returns"], errors="coerce")

    hsi = hsi.drop_duplicates(subset=["Date"]).reset_index(drop=True)

    fut = future.copy()
    if "Date" in fut.columns:
        fut["Date"] = pd.to_datetime(fut["Date"], errors="coerce")
        fut = fut.dropna(subset=["Date"]).sort_values("Date")
        fut = fut.drop_duplicates(subset=["Date"]).reset_index(drop=True)

    hsi = hsi.merge(fut, on="Date", how="left", validate="one_to_one")

    # ---------------- vhsi_df ----------------
    vhsi = vhsi_df.copy()
    if "Date" in vhsi.columns:
        vhsi["Date"] = pd.to_datetime(vhsi["Date"], errors="coerce")
        vhsi = vhsi.dropna(subset=["Date"]).sort_values("Date")

    for col in ["Open", "High", "Low", "Close"]:
        if col in vhsi.columns:
            vhsi[col] = pd.to_numeric(vhsi[col], errors="coerce")

    vhsi = vhsi.drop_duplicates(subset=["Date"]).reset_index(drop=True)

    return pc, hsi, vhsi



def _days_since_event(event: pd.Series) -> pd.Series:
    """
    event: boolean Series
    return: number of rows since last True event
    """
    event = event.fillna(False).astype(bool)
    out = np.full(len(event), np.nan)
    last_idx = -1

    for i, flag in enumerate(event.to_numpy()):
        if flag:
            last_idx = i
            out[i] = 0
        elif last_idx >= 0:
            out[i] = i - last_idx

    return pd.Series(out, index=event.index)


def _rolling_log_trend_stats(price: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """
    Rolling linear regression on log(price).
    Returns:
        slope: daily slope in log-price space
        r2: trend quality
    """
    y = np.log(price.astype(float).replace(0, np.nan).to_numpy())
    x = np.arange(window, dtype=float)

    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    slope = np.full(len(price), np.nan)
    r2 = np.full(len(price), np.nan)

    for i in range(window - 1, len(price)):
        yi = y[i - window + 1 : i + 1]
        if not np.isfinite(yi).all():
            continue

        y_mean = yi.mean()
        cov_xy = ((x - x_mean) * (yi - y_mean)).sum()
        beta = cov_xy / x_var

        y_hat = y_mean + beta * (x - x_mean)
        ss_res = ((yi - y_hat) ** 2).sum()
        ss_tot = ((yi - y_mean) ** 2).sum()

        slope[i] = beta
        r2[i] = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    return pd.Series(slope, index=price.index), pd.Series(r2, index=price.index)


def _rolling_percentile_of_last(x: np.ndarray) -> float:
    """
    Percentile rank of the last observation within the rolling window.
    Output in [0, 1].
    """
    if len(x) == 0 or np.isnan(x[-1]):
        return np.nan
    valid = x[np.isfinite(x)]
    if len(valid) == 0:
        return np.nan
    return (valid <= x[-1]).mean()


def add_hsi_features(hsi_df: pd.DataFrame) -> pd.DataFrame:
    df = hsi_df.copy().sort_values("Date").reset_index(drop=True)

    close = df["Close"].astype(float)

    # --------------------------------------------------
    # 1) daily return base
    # --------------------------------------------------
    df["hsi_close_lag1"] = close.shift(1)
    df["hsi_ret_1d"] = close / df["hsi_close_lag1"] - 1.0
    df["hsi_log_ret_1d"] = np.log(close / df["hsi_close_lag1"])

    # --------------------------------------------------
    # 2) return horizons: 1d / 3d / 7d / 1m / 3m
    # 1m ~ 20 trading days, 3m ~ 60 trading days
    # --------------------------------------------------
    for win in [3, 7, 20, 60]:
        df[f"hsi_ret_{win}d"] = close / close.shift(win) - 1.0

    # --------------------------------------------------
    # 3) realized vol horizons
    # --------------------------------------------------
    for win in [3, 7, 20]:
        df[f"hsi_vol_{win}d_ann"] = (
            df["hsi_ret_1d"]
            .rolling(window=win, min_periods=win)
            .std()
            .shift(1)
            * np.sqrt(252)
        )

    # --------------------------------------------------
    # 4) moving averages + above/bias
    # keep your original structure
    # --------------------------------------------------
    for win in [20, 50, 100]:
        ma_col = f"hsi_ma_{win}"
        df[ma_col] = (
            close.rolling(window=win, min_periods=win)
            .mean()
            .shift(1)
        )

        valid = df[ma_col].notna()
        df[f"hsi_above_ma_{win}"] = np.where(
            valid,
            (close > df[ma_col]).astype(float),
            np.nan,
        )
        df[f"hsi_bias_{win}"] = (close - df[ma_col]) / df[ma_col]

    # --------------------------------------------------
    # 5) transition / acceleration features
    # --------------------------------------------------
    df["hsi_ret_accel_3d_vs_1m"] = df["hsi_ret_3d"] - df["hsi_ret_20d"] / (20 / 3)
    df["hsi_ret_accel_7d_vs_1m"] = df["hsi_ret_7d"] - df["hsi_ret_20d"] / (20 / 7)

    df["hsi_ma20_ma50_spread_pct"] = df["hsi_ma_20"] / df["hsi_ma_50"] - 1.0
    df["hsi_ma20_ma50_spread_delta_5d"] = df["hsi_ma20_ma50_spread_pct"].diff(5)

    # price cross MA20
    price_vs_ma20 = np.sign(close - df["hsi_ma_20"])
    price_cross_ma20 = (
        price_vs_ma20.ne(price_vs_ma20.shift(1))
        & price_vs_ma20.ne(0)
        & price_vs_ma20.shift(1).ne(0)
    )
    df["hsi_days_since_price_cross_ma20"] = _days_since_event(price_cross_ma20)

    # MA20 cross MA50
    ma20_vs_ma50 = np.sign(df["hsi_ma_20"] - df["hsi_ma_50"])
    ma20_cross_ma50 = (
        ma20_vs_ma50.ne(ma20_vs_ma50.shift(1))
        & ma20_vs_ma50.ne(0)
        & ma20_vs_ma50.shift(1).ne(0)
    )
    df["hsi_days_since_ma20_cross_ma50"] = _days_since_event(ma20_cross_ma50)

    # --------------------------------------------------
    # 6) trend quality: slope + R^2
    # helps LLM tell "trend diffusion" vs "one-off spike"
    # --------------------------------------------------
    df["hsi_trend_slope_7d"], df["hsi_trend_r2_7d"] = _rolling_log_trend_stats(close, 7)
    df["hsi_trend_slope_20d"], df["hsi_trend_r2_20d"] = _rolling_log_trend_stats(close, 20)

    # --------------------------------------------------
    # 7) price location / extension
    # percentile of current close within recent window
    # --------------------------------------------------
    df["hsi_close_pctile_20d"] = (
        close.rolling(20, min_periods=20)
        .apply(_rolling_percentile_of_last, raw=True)
    )
    df["hsi_close_pctile_60d"] = (
        close.rolling(60, min_periods=60)
        .apply(_rolling_percentile_of_last, raw=True)
    )

    # optional: distance to recent highs/lows
    rolling_high_20 = close.rolling(20, min_periods=20).max().shift(1)
    rolling_low_20 = close.rolling(20, min_periods=20).min().shift(1)
    rolling_high_60 = close.rolling(60, min_periods=60).max().shift(1)
    rolling_low_60 = close.rolling(60, min_periods=60).min().shift(1)

    df["hsi_dist_to_20d_high"] = close / rolling_high_20 - 1.0
    df["hsi_dist_to_20d_low"] = close / rolling_low_20 - 1.0
    df["hsi_dist_to_60d_high"] = close / rolling_high_60 - 1.0
    df["hsi_dist_to_60d_low"] = close / rolling_low_60 - 1.0

    return df


def add_vhsi_features(vhsi_df: pd.DataFrame) -> pd.DataFrame:
    df = vhsi_df.copy().sort_values("Date").reset_index(drop=True)

    # avoid collision with HSI OHLC names
    rename_map = {}
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            rename_map[col] = f"{col}_vhsi"
    df = df.rename(columns=rename_map)

    close = df["Close_vhsi"].astype(float)

    # --------------------------------------------------
    # 1) daily base
    # --------------------------------------------------
    df["vhsi_close_lag1"] = close.shift(1)
    df["vhsi_ret_1d"] = close / df["vhsi_close_lag1"] - 1.0
    df["vhsi_diff_1d"] = close - df["vhsi_close_lag1"]

    # --------------------------------------------------
    # 2) horizons: 1d / 3d / 7d / 1m
    # --------------------------------------------------
    for win in [3, 7, 20]:
        df[f"vhsi_ret_{win}d"] = close / close.shift(win) - 1.0
        df[f"vhsi_diff_{win}d"] = close - close.shift(win)

    # --------------------------------------------------
    # 3) realized vol of VHSI itself
    # --------------------------------------------------
    for win in [3, 7, 20]:
        df[f"vhsi_vol_{win}d"] = (
            df["vhsi_ret_1d"]
            .rolling(window=win, min_periods=win)
            .std()
            .shift(1)
        )

    # --------------------------------------------------
    # 4) MA / bias / high-vol-regime
    # --------------------------------------------------
    df["vhsi_ma_20"] = (
        close.rolling(window=20, min_periods=20)
        .mean()
        .shift(1)
    )

    valid = df["vhsi_ma_20"].notna()
    df["vhsi_above_ma_20"] = np.where(
        valid,
        (close > df["vhsi_ma_20"]).astype(float),
        np.nan,
    )
    df["vhsi_bias_20"] = (close - df["vhsi_ma_20"]) / df["vhsi_ma_20"]

    df["vhsi_high_vol_regime"] = np.where(
        valid,
        (close > df["vhsi_ma_20"]).astype(float),
        np.nan,
    )

    # --------------------------------------------------
    # 5) vol expansion / fading clues
    # --------------------------------------------------
    df["vhsi_ret_accel_3d_vs_1m"] = df["vhsi_ret_3d"] - df["vhsi_ret_20d"] / (20 / 3)
    df["vhsi_ret_accel_7d_vs_1m"] = df["vhsi_ret_7d"] - df["vhsi_ret_20d"] / (20 / 7)

    df["vhsi_close_pctile_20d"] = (
        close.rolling(20, min_periods=20)
        .apply(_rolling_percentile_of_last, raw=True)
    )
    df["vhsi_close_pctile_60d"] = (
        close.rolling(60, min_periods=60)
        .apply(_rolling_percentile_of_last, raw=True)
    )

    # optional: shock persistence / fading
    df["vhsi_ma20_slope_5d"] = df["vhsi_ma_20"].diff(5) / 5.0

    return df


def _safe_to_datetime(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c])
    return out


def _prepare_call_put_wide(
    df: pd.DataFrame,
    price_col: str,
) -> pd.DataFrame:
    """
    For each Date x ExpDate x Strike, pivot Call/Put prices side by side.
    """
    required = ["Date", "ExpDate", "StrikePrice", "Type", price_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tmp = df[required].copy()
    tmp["Type"] = tmp["Type"].astype(str).str.lower().str.strip()

    # normalize common labels
    type_map = {
        "call": "call",
        "c": "call",
        "put": "put",
        "p": "put",
    }
    tmp["Type"] = tmp["Type"].map(type_map)

    wide = (
        tmp.pivot_table(
            index=["Date", "ExpDate", "StrikePrice"],
            columns="Type",
            values=price_col,
            aggfunc="first",
        )
        .reset_index()
    )

    # ensure columns exist
    for col in ["call", "put"]:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["abs_cp_diff"] = (wide["call"] - wide["put"]).abs()
    return wide


def compute_implied_forward_by_parity(
    df: pd.DataFrame,
    price_col: str = "SettlementPrice",
    rate_col: Optional[str] = None,
    r_default: float = 0.0,
    dte_col: str = "dte",
) -> pd.DataFrame:
    """
    Literature-style implied forward F0 for each Date x ExpDate:

    1) find the strike where |Call - Put| is smallest
    2) define ATM strike = that strike
    3) compute F0 = K + exp(r * tau) * (Call - Put)

    Returns one row per Date x ExpDate with:
    - Date, ExpDate
    - atm_strike_parity
    - call_atm, put_atm
    - tau
    - r
    - implied_forward_parity
    - abs_cp_diff_min
    """
    df = _safe_to_datetime(df, ["Date", "ExpDate"]).copy()

    if dte_col not in df.columns:
        raise ValueError(f"Missing '{dte_col}' column.")
    if price_col not in df.columns:
        raise ValueError(f"Missing '{price_col}' column.")

    wide = _prepare_call_put_wide(df, price_col=price_col)

    # need dte / rate at Date x ExpDate x Strike level for the selected ATM strike
    aux_cols = ["Date", "ExpDate", "StrikePrice", dte_col]
    if rate_col is not None and rate_col in df.columns:
        aux_cols.append(rate_col)

    aux = (
        df[aux_cols]
        .drop_duplicates(subset=["Date", "ExpDate", "StrikePrice"])
        .copy()
    )

    wide = wide.merge(aux, on=["Date", "ExpDate", "StrikePrice"], how="left")
    # keep rows with both call and put
    wide = wide.dropna(subset=["call", "put", dte_col]).copy()
    if wide.empty:
        raise ValueError("No valid call-put pairs available to compute implied forward.")

    # pick the strike minimizing |C - P| per Date x ExpDate
    wide = wide.sort_values(
        ["Date", "ExpDate", "abs_cp_diff", "StrikePrice"],
        ascending=[True, True, True, True]
    )

    atm = (
        wide.groupby(["Date", "ExpDate"], as_index=False)
        .first()
        .rename(columns={"StrikePrice": "atm_strike_parity"})
    )

    # tau in years
    atm["tau"] = atm[dte_col] / 365.0

    # risk-free rate
    if rate_col is not None and rate_col in atm.columns:
        atm["r"] = atm[rate_col].fillna(r_default)
    else:
        atm["r"] = r_default

    # literature formula
    atm["implied_forward_parity"] = (
        atm["atm_strike_parity"] + np.exp(atm["r"] * atm["tau"]) * (atm["call"] - atm["put"])
    )

    atm = atm.rename(columns={
        "call": "call_atm",
        "put": "put_atm",
        "abs_cp_diff": "abs_cp_diff_min",
    })

    return atm[
        [
            "Date",
            "ExpDate",
            "atm_strike_parity",
            "call_atm",
            "put_atm",
            dte_col,
            "tau",
            "r",
            "implied_forward_parity",
            "abs_cp_diff_min",
        ]
    ].copy()


def build_otm_smirk_points(
    df: pd.DataFrame,
    forwards_df: pd.DataFrame,
    iv_col: str = "ImpliedVolatility",
    x_band: float = np.inf,
    dte_col: str = "dte",
) -> pd.DataFrame:
    """
    Build OTM smirk fitting points using:
      x = ln(K / F0)

    Keep only:
      - Put when K < F0
      - Call when K > F0
    and only in the near-ATM band abs(x) <= x_band.
    """
    df = _safe_to_datetime(df, ["Date", "ExpDate"]).copy()

    required = ["Date", "ExpDate", "StrikePrice", "Type", iv_col, dte_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.merge(
        forwards_df[["Date", "ExpDate", "implied_forward_parity", "atm_strike_parity", "tau", "r"]],
        on=["Date", "ExpDate"],
        how="left",
    ).copy()

    out = out.dropna(subset=["implied_forward_parity", iv_col]).copy()
    out["Type_norm"] = out["Type"].astype(str).str.lower().str.strip()
    out["Type_norm"] = out["Type_norm"].replace({
        "c": "call",
        "call": "call",
        "p": "put",
        "put": "put",
    })

    out["x"] = np.log(out["StrikePrice"] / out["implied_forward_parity"])
    # print(out["x"])
    # OTM rule from the paper
    is_otm_put = (out["StrikePrice"] < out["implied_forward_parity"]) & (out["Type_norm"] == "put")
    is_otm_call = (out["StrikePrice"] > out["implied_forward_parity"]) & (out["Type_norm"] == "call")
    out["is_otm_selected"] = is_otm_put | is_otm_call

    out = out[out["is_otm_selected"]].copy()
    out = out[out["x"].abs() <= x_band].copy()

    # remove impossible / dirty rows
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["x", iv_col])

    return out.sort_values(["Date", "ExpDate", "StrikePrice", "Type_norm"]).copy()


def _fit_quadratic_with_anchor(
    x: np.ndarray,
    iv: np.ndarray,
    iv_atm: float,
) -> Tuple[float, float, float, float]:
    """
    Fit:
        IV(x) = a + b x + c x^2
    with a forced to equal iv_atm.

    Then:
        y = IV - iv_atm = b x + c x^2

    Returns:
        a, b, c, rmse
    """
    x = np.asarray(x, dtype=float)
    iv = np.asarray(iv, dtype=float)

    y = iv - iv_atm
    X = np.column_stack([x, x**2])

    # OLS on [b, c]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b, c = beta[0], beta[1]

    fitted = iv_atm + b * x + c * x**2
    rmse = float(np.sqrt(np.mean((iv - fitted) ** 2)))

    return float(iv_atm), float(b), float(c), rmse


def fit_iv_surface_shape_daily(
    df: pd.DataFrame,
    price_col: str = "SettlementPrice",
    iv_col: str = "ImpliedVolatility",
    rate_col: Optional[str] = None,
    r_default: float = 0.0,
    dte_col: str = "dte",
    x_band: float = np.inf,
    min_points: int = 5,
) -> pd.DataFrame:
    """
    Main pipeline:

    Step 1: compute implied forward F0 by literature parity method
    Step 2: x = ln(K / F0)
    Step 3: keep near-ATM strikes |x| <= x_band
    Step 4: keep OTM puts for K < F0 and OTM calls for K > F0
    Step 5: fit IV(x) = a + b x + c x^2, forcing a = ATM IV at parity strike

    Returns one row per Date x ExpDate with:
    - Date, ExpDate, dte, tau, r
    - implied_forward_parity
    - atm_strike_parity
    - atm_iv
    - level, slope, curvature
    - n_points, rmse
    """
    df = _safe_to_datetime(df, ["Date", "ExpDate"]).copy()

    # ---------- Step 1 ----------
    forwards = compute_implied_forward_by_parity(
        df=df,
        price_col=price_col,
        rate_col=rate_col,
        r_default=r_default,
        dte_col=dte_col,
    )

    # ATM IV at the parity-selected strike:
    atm_iv_df = (
        df.merge(
            forwards[["Date", "ExpDate", "atm_strike_parity"]],
            left_on=["Date", "ExpDate", "StrikePrice"],
            right_on=["Date", "ExpDate", "atm_strike_parity"],
            how="inner",
        )
        .copy()
    )

    # if both call and put exist at ATM strike, take their average IV
    atm_iv_df["Type_norm"] = atm_iv_df["Type"].astype(str).str.lower().str.strip()
    atm_iv_df["Type_norm"] = atm_iv_df["Type_norm"].replace({
        "c": "call",
        "call": "call",
        "p": "put",
        "put": "put",
    })

    atm_iv_agg = (
        atm_iv_df.groupby(["Date", "ExpDate", "atm_strike_parity"], as_index=False)[iv_col]
        .mean()
        .rename(columns={iv_col: "atm_iv"})
    )

    forwards = forwards.merge(
        atm_iv_agg,
        on=["Date", "ExpDate", "atm_strike_parity"],
        how="left",
    )

    # ---------- Steps 2-4 ----------
    pts = build_otm_smirk_points(
        df=df,
        forwards_df=forwards,
        iv_col=iv_col,
        x_band=x_band,
        dte_col=dte_col,
    )

    # ---------- Step 5 ----------
    results = []
    for (date, ExpDate), g in pts.groupby(["Date", "ExpDate"], sort=True):
        row_f = forwards.loc[
            (forwards["Date"] == date) & (forwards["ExpDate"] == ExpDate)
        ]
        if row_f.empty:
            continue

        frow = row_f.iloc[0]
        atm_iv = frow.get("atm_iv", np.nan)

        if pd.isna(atm_iv):
            continue

        g = g.dropna(subset=["x", iv_col]).copy()
        if len(g) < min_points:
            continue

        try:
            a, b, c, rmse = _fit_quadratic_with_anchor(
                x=g["x"].to_numpy(),
                iv=g[iv_col].to_numpy(),
                iv_atm=float(atm_iv),
            )

        except np.linalg.LinAlgError:
            continue

        # simple R^2
        fitted = a + b * g["x"].to_numpy() + c * (g["x"].to_numpy() ** 2)
        y = g[iv_col].to_numpy()
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot

        results.append({
            "Date": date,
            "ExpDate": ExpDate,
            dte_col: int(frow[dte_col]),
            "tau": float(frow["tau"]),
            "r": float(frow["r"]),
            "implied_forward_parity": float(frow["implied_forward_parity"]),
            "atm_strike_parity": float(frow["atm_strike_parity"]),
            "atm_iv": float(atm_iv),
            "level": float(a)/100.0,
            "slope": float(b)/100.0,
            "curvature": float(c)/100.0,
            "n_points": int(len(g)),
            "rmse": float(rmse),
            "r2": float(r2) if pd.notna(r2) else np.nan,
            "x_min": float(g["x"].min()),
            "x_max": float(g["x"].max()),
        })

    res = pd.DataFrame(results).sort_values(["Date", "ExpDate"]).reset_index(drop=True)
    return res


def build_smirk_points_with_fit(
    df: pd.DataFrame,
    fitted_df: pd.DataFrame,
    price_col: str = "SettlementPrice",
    iv_col: str = "ImpliedVolatility",
    rate_col: Optional[str] = None,
    r_default: float = 0.0,
    dte_col: str = "dte",
    x_band: float = np.inf,
) -> pd.DataFrame:
    """
    Optional helper: return fitting points joined with fitted level/slope/curvature,
    useful for plotting diagnostics.
    """
    forwards = compute_implied_forward_by_parity(
        df=df,
        price_col=price_col,
        rate_col=rate_col,
        r_default=r_default,
        dte_col=dte_col,
    )
    pts = build_otm_smirk_points(
        df=df,
        forwards_df=forwards,
        iv_col=iv_col,
        x_band=x_band,
        dte_col=dte_col,
    )
    out = pts.merge(
        fitted_df[["Date", "ExpDate", "level", "slope", "curvature"]],
        on=["Date", "ExpDate"],
        how="left",
    )
    out["iv_fit"] = out["level"] + out["slope"] * out["x"] + out["curvature"] * (out["x"] ** 2)
    return out



def add_option_shape_daily_features(
    shape_df: pd.DataFrame,
    date_col: str = "Date",
    expiry_col: str = "ExpDate",
    dte_col: str = "dte",
    min_points: int = 10,
    min_r2: float = 0.5,
    add_normalized: bool = False,
    norm_window: int = 252,
    clip_z: float = 5.0,
) -> pd.DataFrame:
    """
    Convert Date x Expiry IV-shape panel into compact Date-level daily option shape features.

    Keep only the most useful IV features for regime / news overlay:
    - near-term level and slope
    - next-near term structure in level and slope
    - fit quality info
    - short-horizon changes (1d / 3d)
    - optional rolling z-scores for a small core set

    Expected input columns:
    - Date
    - ExpDate
    - dte
    - level
    - slope
    - curvature
    - n_points
    - r2
    Optional:
    - rmse

    Output:
    One row per Date.
    """

    df = shape_df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df[expiry_col] = pd.to_datetime(df[expiry_col]).dt.normalize()

    required = [date_col, expiry_col, dte_col, "level", "slope", "n_points", "r2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_rmse = "rmse" in df.columns

    # quality filter
    df = df[(df["r2"] >= min_r2) & (df["n_points"] >= min_points)].copy()

    base_cols = [
        date_col,
        "level_near", "slope_near", "dte_near",
        "level_next", "slope_next", "dte_next",
        "term_level_next_near", "term_slope_next_near",
        "r2_near", "n_points_near",
        "r2_next", "n_points_next",
    ]
    if has_rmse:
        base_cols += ["rmse_near", "rmse_next"]

    if df.empty:
        out = pd.DataFrame(columns=base_cols)
        return out

    df = df.sort_values([date_col, dte_col, expiry_col]).copy()

    rows = []
    for d, g in df.groupby(date_col, sort=True):
        g = g.sort_values([dte_col, expiry_col]).reset_index(drop=True)

        row = {date_col: d}

        # ---------- near ----------
        if len(g) >= 1:
            row["level_near"] = g.loc[0, "level"]
            row["slope_near"] = g.loc[0, "slope"]
            row["dte_near"] = g.loc[0, dte_col]
            row["r2_near"] = g.loc[0, "r2"]
            row["n_points_near"] = g.loc[0, "n_points"]
            if has_rmse:
                row["rmse_near"] = g.loc[0, "rmse"]
        else:
            row["level_near"] = np.nan
            row["slope_near"] = np.nan
            row["dte_near"] = np.nan
            row["r2_near"] = np.nan
            row["n_points_near"] = np.nan
            if has_rmse:
                row["rmse_near"] = np.nan

        # ---------- next ----------
        if len(g) >= 2:
            row["level_next"] = g.loc[1, "level"]
            row["slope_next"] = g.loc[1, "slope"]
            row["dte_next"] = g.loc[1, dte_col]
            row["r2_next"] = g.loc[1, "r2"]
            row["n_points_next"] = g.loc[1, "n_points"]
            if has_rmse:
                row["rmse_next"] = g.loc[1, "rmse"]
        else:
            row["level_next"] = np.nan
            row["slope_next"] = np.nan
            row["dte_next"] = np.nan
            row["r2_next"] = np.nan
            row["n_points_next"] = np.nan
            if has_rmse:
                row["rmse_next"] = np.nan

        # ---------- term structure ----------
        row["term_level_next_near"] = (
            row["level_next"] - row["level_near"]
            if pd.notna(row["level_next"]) and pd.notna(row["level_near"])
            else np.nan
        )
        row["term_slope_next_near"] = (
            row["slope_next"] - row["slope_near"]
            if pd.notna(row["slope_next"]) and pd.notna(row["slope_near"])
            else np.nan
        )

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(date_col).reset_index(drop=True)

    # ---------- short-horizon changes ----------
    chg_cols = [
        "level_near",
        "slope_near",
        "term_level_next_near",
        "term_slope_next_near",
    ]

    for col in chg_cols:
        s = out[col].astype(float)
        out[f"{col}_chg_1d"] = s.diff(1)
        out[f"{col}_chg_3d"] = s.diff(3)

    # ---------- rolling normalized features ----------
    if add_normalized:
        norm_cols = [
            "level_near",
            "slope_near",
            "term_level_next_near",
            "term_slope_next_near",
        ]

        for col in norm_cols:
            s = out[col].astype(float)
            rolling_mean = s.shift(1).rolling(
                norm_window,
                min_periods=max(20, norm_window // 5)
            ).mean()
            rolling_std = s.shift(1).rolling(
                norm_window,
                min_periods=max(20, norm_window // 5)
            ).std()

            z = (s - rolling_mean) / rolling_std
            if clip_z is not None:
                z = z.clip(-clip_z, clip_z)

            out[f"{col}_z"] = z

    return out


def summarize_index_window(idx_win: pd.DataFrame) -> dict:
    out = {}

    close = idx_win["Close"].dropna()
    vhsi_close = idx_win["Close_vhsi"].dropna()
    log_ret = idx_win["hsi_log_ret_1d"].dropna()

    out["hsi_last_close"] = close.iloc[-1] if len(close) else np.nan
    out["hsi_lookback_ret"] = close.iloc[-1] / close.iloc[0] - 1 if len(close) >= 2 else np.nan

    out["hsi_mean_1d_log_ret"] = log_ret.mean() if len(log_ret) else np.nan
    out["hsi_ann_log_ret"] = log_ret.mean() * 252 if len(log_ret) else np.nan

    out["hsi_last_3d_ret"] = idx_win["hsi_ret_3d"].iloc[-1]
    out["hsi_last_5d_ret"] = idx_win["hsi_ret_5d"].iloc[-1]
    out["hsi_last_10d_ret"] = idx_win["hsi_ret_10d"].iloc[-1]
    out["hsi_last_bias_20"] = idx_win["hsi_bias_20"].iloc[-1]
    out["hsi_last_bias_50"] = idx_win["hsi_bias_50"].iloc[-1]
    daily_std = log_ret.std(ddof=1) if len(log_ret) > 1 else np.nan
    out["hsi_realized_vol"] = daily_std * np.sqrt(252) if pd.notna(daily_std) else np.nan
    out["hsi_sharpe"] = (
        out["hsi_ann_log_ret"] / out["hsi_realized_vol"]
        if pd.notna(out["hsi_ann_log_ret"]) and pd.notna(out["hsi_realized_vol"]) and out["hsi_realized_vol"] > 0
        else np.nan
    )

    for win in [20, 50, 100]:
        flag_col = f"hsi_above_ma_{win}"
        bias_col = f"hsi_bias_{win}"

        if flag_col in idx_win.columns:
            out[f"hsi_frac_above_ma_{win}"] = idx_win[flag_col].mean()

        if bias_col in idx_win.columns:
            out[f"hsi_mean_bias_{win}"] = idx_win[bias_col].mean()

    out["vhsi_mean_raw"] = vhsi_close.mean() if len(vhsi_close) else np.nan
    out["vhsi_mean"] = vhsi_close.mean() / 100.0 if len(vhsi_close) else np.nan
    out["vhsi_std"] = vhsi_close.std(ddof=1) / 100.0 if len(vhsi_close) > 1 else np.nan
    out["vhsi_net_change"] = vhsi_close.iloc[-1] - vhsi_close.iloc[0] if len(vhsi_close) >= 2 else np.nan
    out["vhsi_high_vol_regime_frac"] = idx_win["vhsi_high_vol_regime"].mean()
    out["vhsi_last_close"] = idx_win["Close_vhsi"].iloc[-1] / 100.0
    
    return out



def summarize_option_window(opt_win: pd.DataFrame) -> dict:
    """
    Summarize daily option shape features over the lookback window.

    Expected input: one row per Date, with columns like
    - level_near, slope_near, curvature_near
    - level_next, slope_next, curvature_next
    - level_far, slope_far, curvature_far
    - term_level_next_near, term_slope_next_near, term_curvature_next_near
    - term_level_far_near, term_slope_far_near, term_curvature_far_near
    """
    out = {}
    if opt_win is None or opt_win.empty:
        return out

    feature_cols = [
        "level_near", "slope_near", "curvature_near",
        "level_next", "slope_next", "curvature_next",
        "level_far", "slope_far", "curvature_far",
        "term_level_next_near", "term_slope_next_near", "term_curvature_next_near",
        "term_level_far_near", "term_slope_far_near", "term_curvature_far_near",
    ]

    for col in feature_cols:
        if col not in opt_win.columns:
            continue

        s = opt_win[col].dropna()
        if s.empty:
            continue

        out[f"{col}_mean"] = s.mean()
        out[f"{col}_std"] = s.std()
        out[f"{col}_last"] = s.iloc[-1]

        # 20-trading-day change
        if len(s) >= 21:
            out[f"{col}_chg_20d"] = s.iloc[-1] - s.iloc[-21]
        else:
            out[f"{col}_chg_20d"] = np.nan

    # optional: quality diagnostics
    if "level_near" in opt_win.columns:
        out["option_obs_count"] = len(opt_win)

    return out


def build_periodic_features(index_features: pd.DataFrame,
                            option_features: Optional[pd.DataFrame] = None,
                            lookback: int = LOOKBACK_LEN,
                            pred_len: int = PRED_LEN) -> pd.DataFrame:
    """
    Build one row per decision date.
    - lookback: months of history used for features (e.g. 3)
    - pred_len: months ahead for prediction horizon (e.g. 1)
    """
    idx = index_features.sort_values("Date").copy()
    opt = option_features.sort_values("Date").copy()

    # use last trading day of each month as decision date
    month_ends = (
        idx.groupby(idx["Date"].dt.to_period("M"))["Date"]
           .max()
           .sort_values()
           .tolist()
    )

    rows = []
    for d in month_ends:
        d = pd.to_datetime(d)
        lookback_start = d - pd.DateOffset(months=lookback)

        idx_win = idx[(idx["Date"] > lookback_start) & (idx["Date"] <= d)]
        opt_win = opt[(opt["Date"] > lookback_start) & (opt["Date"] <= d)]

        if len(idx_win) < 20:
            continue  # not enough history

        idx_summary = summarize_index_window(idx_win)
        opt_summary = summarize_option_window(opt_win)

        row = {
            "decision_date": d,
            "lookback_months": lookback,
            "pred_len_months": pred_len,
        }
        row.update(idx_summary)
        row.update(opt_summary)
        rows.append(row)

    features = pd.DataFrame(rows).sort_values("decision_date").reset_index(drop=True)
    return features