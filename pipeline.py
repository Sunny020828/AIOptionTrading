""" 

conda activate ophr
python pipeline.py --start 2025-01-01 --end 2025-12-31 --risk_pct 0.05 --alloc_pct 0.1 --stop_loss 0.6 --delta_limit 300.0 --no-search_news --no-save_db 

"""

import os
import re
import json
import math
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple
import asyncio 
import numpy as np
import pandas as pd
import time
import traceback
import argparse
from tqdm.auto import tqdm
from features.data import *
from get_news.urls import *
from execution.backtest import compute_backtest_metrics, summarize_mtm_path
from db.init import init_database, get_connection
from db.operations import save_news_one, save_performance_record, save_news_digestion_one, load_digestion
from models.update_model import news_weekly_update
from models.trader_model_v2 import call_llm
# from models.trader_model_v3 import call_llm
from execution.choose_scenario import choose_scenario
# from execution.strategy_pools_v2 import run_prediction_cycle_strategy

from models.keyword_model import suggest_keywords_for_hsi
from get_news.news_search import harvest_news, build_source_row
from utils.context import fetch_news_for_context


NEWS_CONTEXT_DAYS = 7 
NEWS_SEARCH_LEN = 1
METRICS_LOOKBACK=3


def _to_ts(x) -> pd.Timestamp:
    return pd.to_datetime(x).normalize()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full option trading timeline backtest."
    )
    parser.add_argument("--start",type=str, default="2024-01-01", help="Backtest start date, e.g. 2024-01-01")
    parser.add_argument("--end",type=str, default="2024-12-31", help="Backtest end date, e.g. 2024-12-31")
    parser.add_argument("--account_size", type=float, default=1_000_000, help="Initial account size")
    parser.add_argument("--risk_pct", type=float, default=0.02, help="Risk percentage per trade")
    parser.add_argument("--alloc_pct", type=float, default=0.1, help="Capital allocation percentage per trade")
    parser.add_argument("--stop_loss", type=float, default=0.6, help="Stop loss rate")
    parser.add_argument("--delta_limit", type=float, default=300, help="Delta limit for option selection")
    parser.add_argument("--model_name", type=str, default="deepseek-reasoner", help="LLM model name")
    parser.add_argument("--search_news", action=argparse.BooleanOptionalAction,default=True, help="Whether to search for new news")
    parser.add_argument("--save_db", action=argparse.BooleanOptionalAction,default=True, help="Whether to save results into database")
    return parser.parse_args()



def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def _extract_json_from_text_local(text: str) -> Dict[str, Any]:
    """
    Local lightweight parser for logging only.
    """
    if text is None:
        return {}

    text = str(text).strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            parsed = json.loads(m.group(1))
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            pass

    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        try:
            parsed = json.loads(m.group(1))
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            pass

    return {"raw_text": text}

def _safe_json(obj, max_len: int = 3000):
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        text = str(obj)
    if len(text) > max_len:
        text = text[:max_len] + f"... [truncated {len(text) - max_len} chars]"
    return text


def get_month_key(ts) -> str:
    return pd.to_datetime(ts).strftime("%Y-%m")


def get_monthly_windows_from_trading_dates(
    option_df: pd.DataFrame,
    start_date,
    end_date,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Returns:
        [(decision_date, entry_date, exit_date), ...]

    Logic:
    - decision_date = each month's last trading day
    - entry_date    = next month's first trading day
    - exit_date     = next month's last trading day
    - keep only windows whose entry_date is within [start_date, end_date]
    """
    df = option_df.copy()
    date_col = "date" if "date" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()

    trading_dates = (
        df[date_col]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    cal = pd.DataFrame({"Date": trading_dates})
    cal["year_month"] = cal["Date"].dt.strftime("%Y-%m")

    month_last = (
        cal.groupby("year_month", as_index=False)["Date"]
        .max()
        .rename(columns={"Date": "decision_date"})
        .sort_values("decision_date")
        .reset_index(drop=True)
    )

    windows = []
    for i in range(len(month_last) - 1):
        decision_date = pd.to_datetime(month_last.loc[i, "decision_date"]).normalize()

        next_month_decision = pd.to_datetime(month_last.loc[i + 1, "decision_date"]).normalize()
        next_month_key = next_month_decision.strftime("%Y-%m")

        next_month_dates = cal[cal["year_month"] == next_month_key]["Date"].sort_values()
        if next_month_dates.empty:
            continue

        entry_date = pd.to_datetime(next_month_dates.iloc[0]).normalize()
        exit_date = pd.to_datetime(next_month_dates.iloc[-1]).normalize()

        if entry_date < start_date or entry_date > end_date:
            continue

        windows.append((decision_date, entry_date, exit_date))

    return windows


def get_weekly_update_dates_for_month(
    option_df: pd.DataFrame,
    decision_date,
) -> List[pd.Timestamp]:
    """
    For a given decision_date, return all weekly update dates
    within that calendar month up to decision_date.
    Rule: use each ISO week's last trading day.
    """
    df = option_df.copy()
    date_col = "date" if "date" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    decision_date = pd.to_datetime(decision_date).normalize()
    month_start = decision_date.replace(day=1)

    dates = (
        df.loc[(df[date_col] >= month_start) & (df[date_col] <= decision_date), date_col]
        .drop_duplicates()
        .sort_values()
    )

    if dates.empty:
        return []

    cal = pd.DataFrame({"Date": dates})
    iso = cal["Date"].dt.isocalendar()
    cal["iso_year"] = iso.year
    cal["iso_week"] = iso.week

    weekly_dates = (
        cal.groupby(["iso_year", "iso_week"], as_index=False)["Date"]
        .max()["Date"]
        .sort_values()
        .tolist()
    )

    return [pd.to_datetime(x).normalize() for x in weekly_dates]


def previous_calendar_month_window(decision_date) -> Tuple[str, str]:
    decision_date = _to_ts(decision_date)
    month_start = decision_date.replace(day=1)
    prev_month_end = month_start - pd.Timedelta(days=1)
    prev_month_start = prev_month_end.replace(day=1)
    return prev_month_start.strftime("%Y-%m-%d"), prev_month_end.strftime("%Y-%m-%d")


def setup_logger(run_id: str, log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{run_id}.log"), encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger

def harvest_and_save_new_news(
    *,
    kwq_bundle: Dict[str, Any],
    anchor_date,
    lookback_months: int,
    known_url_hashes: set[str],
    keywords: Optional[List[str]] = None,
    logger=None,
) -> List[Dict[str, Any]]:
    """
    Harvest recent news and save only items not already in DB.

    Returns
    -------
    saved_news : list[dict]
        Newly saved rows.
    """
    try:
        raw_rss = harvest_news(
            kwq_bundle=kwq_bundle or {},
            target_recent=24,
            max_queries=None,
            anchor_date=anchor_date,
            lookback_months=lookback_months,
            stop_after_valid=30,
            type_targets={
                "macro": 8,
                "spillover": 8,
                "index_direct": 8,
            },
            locale_soft_targets={
                "en_hk": 10,
                "zh_cn": 7,
                "zh_hk": 7,
            },
            verbose=True,
        )
    except Exception as e:
        if logger:
            logger.warning("[news][harvest-error] %s", e)
        else:
            print(f"[news][harvest-error] {e}")
        return []

    if not raw_rss:
        return []

    # ---------- date window ----------
    anchor_ts = pd.to_datetime(anchor_date).normalize()
    sd = (anchor_ts - pd.DateOffset(months=lookback_months)).date()
    ed = anchor_ts.date()

    def in_range(x, sd, ed):
        try:
            d = pd.to_datetime(x).date()
            return sd <= d <= ed
        except Exception:
            return False

    # ---------- sort + filter ----------
    items_sorted = sorted(
        raw_rss,
        key=lambda x: str(x.get("approx_date") or ""),
        reverse=True,
    )
    pool = [it for it in items_sorted if in_range(it.get("approx_date"), sd, ed)]
    pool = dedup_by_url_hash(pool)

    # ---------- only keep unseen ----------
    new_pool = [it for it in pool if it.get("_url_hash") not in known_url_hashes]

    if logger:
        logger.info(
            "[news] harvest done | total=%d | in_range=%d | new=%d",
            len(raw_rss), len(pool), len(new_pool)
        )
    else:
        print(
            f"[news] harvest done | total={len(raw_rss)} | in_range={len(pool)} | new={len(new_pool)}"
        )

    if not new_pool:
        return []

    saved_news: List[Dict[str, Any]] = []
    conn = None
    try:
        conn = get_connection(include_database=True)
        for item in new_pool:
            row = build_source_row(
                item,
                keywords=keywords,
                known_url_hashes=known_url_hashes,
            )
            if not row:
                continue

            try:
                save_news_one(conn, row)
                conn.commit()
                known_url_hashes.add(row["url_hash"])
                saved_news.append(row)
            except Exception as e:
                conn.rollback()
                if logger:
                    logger.warning(
                        "[news][db-save-failed] url=%s | err=%s",
                        row.get("url"), e
                    )
                else:
                    print(f"[news][db-save-failed] {row.get('url')} | err={e}")
    finally:
        if conn is not None:
            conn.close()

    return saved_news



def build_monthly_digest_from_memo_and_overlays(
    *,
    decision_date,
    month_key: str,
    # monthly_memo_result: Dict[str, Any],
    weekly_update_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # monthly_memo_text = monthly_memo_result.get("monthly_memo_text", "") or ""

    weekly_text_blocks = []
    for x in weekly_update_results or []:
        anchor_date_raw = x.get("decision_date") or x.get("anchor_date")
        anchor_date = pd.to_datetime(anchor_date_raw).date() if anchor_date_raw is not None else None
        txt = x.get("weekly_update_text", "") or ""
        if not txt:
            continue
        if anchor_date:
            weekly_text_blocks.append(f"[Weekly overlay | {anchor_date}]\n{txt}")
        else:
            weekly_text_blocks.append(txt)

    digest_parts = []
    # if monthly_memo_text:
    #     digest_parts.append(f"[Monthly memo]\n{monthly_memo_text}")
    if weekly_text_blocks:
        digest_parts.append(
            "[Accumulated weekly overlays this month]\n" + "\n\n".join(weekly_text_blocks)
        )

    digest_text = "\n\n".join(digest_parts).strip() or "N/A"

    return {
        "report_type": "monthly_digest",
        "month_key": month_key,
        "decision_date": str(pd.to_datetime(decision_date).date()),
        "digest_text": digest_text,
    }




def build_flat_mtm(option_df, start_date, end_date):
    df = option_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()

    dates = sorted(
        df.loc[(df["Date"] >= start_date) & (df["Date"] <= end_date), "Date"]
        .dropna()
        .unique()
    )

    if not dates:
        return pd.DataFrame(columns=["Date", "Total_PnL", "Daily_PnL"])

    out = pd.DataFrame({"Date": dates})
    out["Total_PnL"] = 0.0
    out["Daily_PnL"] = 0.0
    return out



async def run_full_timeline(
    start_date: str,
    end_date: str,
    model_name: str = "deepseek-reasoner",
    account_size: float = 1_000_000.0,
    risk_pct: float = 0.01,
    capital_alloc_pct: float = 0.1,
    stop_loss: float = 0.6,
    delta_limit: float = 300.0,
    search_news: bool = False,
    save_db: bool = True,
) -> Dict[str, Any]:
    run_id = f"timeline_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}"
    logger = setup_logger(run_id)
    t0 = time.perf_counter()

    logger.info(
        "RUN START | run_id=%s | start_date=%s | end_date=%s | model_name=%s | account_size=%.2f | risk_pct=%.4f | save_db=%s | search_news=%s",
        run_id, start_date, end_date, model_name, account_size, risk_pct, save_db, search_news
    )

    try:
        init_database()

        # ============================================================
        # 1) Load raw data
        # ============================================================
        price_chain_df, hsi_df, vhsi_df, future = get_data()

        # ============================================================
        # 2) Prepare datasets
        # ============================================================
        pc, hsi, vhsi = prepare_datasets(
            price_chain_df=price_chain_df,
            hsi_df=hsi_df,
            vhsi_df=vhsi_df,
            future=future,
            bad_contract_threshold=0.85,
        )

        merged_df = pd.merge(pc, hsi, on="Date", how="left")

        # ============================================================
        # 3) Build DAILY features
        # ============================================================
        hsi_features = add_hsi_features(hsi)
        vhsi_features = add_vhsi_features(vhsi)

        index_features = (
            hsi_features.merge(
                vhsi_features,
                on="Date",
                how="inner",
                validate="one_to_one",
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        smirk_daily = fit_iv_surface_shape_daily(
            df=pc,
            price_col="SettlementPrice",
            iv_col="ImpliedVolatility",
            rate_col=None,
            r_default=0.0,
            dte_col="dte",
            x_band=np.inf,
            min_points=5,
        )

        option_features = add_option_shape_daily_features(shape_df=smirk_daily)

        daily_features = (
            index_features.merge(
                option_features,
                on="Date",
                how="left",
                validate="one_to_one",
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )

        logger.info("DAILY FEATURES READY | rows=%d", len(daily_features))

        # ============================================================
        # 4) Build MONTHLY windows
        # ============================================================
        windows = get_monthly_windows_from_trading_dates(
            option_df=merged_df,
            start_date=start_date,
            end_date=end_date,
        )

        if not windows:
            raise ValueError("No monthly trading windows found.")

        logger.info("Timeline ready | run_id=%s | n_windows=%d", run_id, len(windows))

        monthly_rows = []
        mtm_parts = []
        cumulative_offset = 0.0

        
        

        # ============================================================
        # 5) Monthly loop
        # ============================================================
        for i, (decision_date, entry_date, exit_date) in enumerate(
            tqdm(windows, desc="Running monthly timeline", total=len(windows)),
            start=1,
        ):
            month_t0 = time.perf_counter()
            decision_date = pd.to_datetime(decision_date).normalize()
            entry_date = pd.to_datetime(entry_date).normalize()
            exit_date = pd.to_datetime(exit_date).normalize()
            month_key = get_month_key(decision_date)

            logger.info(
                "[%02d/%02d] MONTH START | decision_date=%s | entry_date=%s | exit_date=%s | month_key=%s | cumulative_offset_before=%.2f",
                i, len(windows), decision_date.date(), entry_date.date(), exit_date.date(), month_key, cumulative_offset
            )

            try:
                # ----------------------------------------------------
                # 5.1 Monthly memo (once per month)
                # ----------------------------------------------------
                # monthly_memo_cache: Dict[str, Dict[str, Any]] = {}
                monthly_keywords_cache: Dict[str, List[str]] = {}
                kwq_bundle = await suggest_keywords_for_hsi(use_static=True)
                keywords = kwq_bundle.get("keywords") if isinstance(kwq_bundle, dict) else None
                monthly_keywords_cache[month_key] = keywords or []

                if search_news:
                    known_url_hashes = set()
                    try:
                        existing_news = fetch_news_for_context(
                            start_date=(pd.to_datetime(decision_date) - pd.DateOffset(months=1)).date(),
                            end_date=pd.to_datetime(decision_date).date(),
                            keyword=None,
                        )
                        for x in existing_news or []:
                            h = x.get("url_hash")
                            if h:
                                known_url_hashes.add(h)
                    except Exception as e:
                        logger.warning("[%s] load existing news failed | err=%s", decision_date.date(), e)

                    saved_news = harvest_and_save_new_news(
                        kwq_bundle=kwq_bundle,
                        anchor_date=decision_date,
                        lookback_months=1,
                        known_url_hashes=known_url_hashes,
                        keywords=keywords,
                        logger=logger,
                    )

                    logger.info(
                        "[%s] news refresh done | newly_saved=%d",
                        decision_date.date(),
                        len(saved_news),
                        )

                #     monthly_memo_result = await news_monthly_memo(
                #         country_code="HK",
                #         decision_date=decision_date,
                #         keyword=None,
                #         keywords=keywords,
                #         lookback_days=30,
                #         limit_fetch=30,
                #         limit_model_items=12,
                #         topic="Hang Seng Index",
                #         run_id=f"{run_id}_{month_key}_memo",
                #     )

                #     monthly_memo_cache[month_key] = monthly_memo_result

                #     logger.info(
                #         "[%s] MONTHLY MEMO CREATED | n_items=%s",
                #         decision_date.date(),
                #         monthly_memo_result.get("n_items"),
                #     )

                # monthly_memo_result = monthly_memo_cache[month_key]
                # monthly_memo_text = monthly_memo_result.get("monthly_memo_text", {})
                # keywords = monthly_keywords_cache.get(month_key, [])

                # ----------------------------------------------------
                # 5.2 Load or generate weekly overlay for this decision_date
                # ----------------------------------------------------
                weekly_report = None

                try:
                    conn = get_connection(include_database=True)
                    try:
                        weekly_report = load_digestion(
                            conn=conn,
                            decision_date=decision_date,
                            country_code="HK",
                            market_code="HSI",
                        )
                    finally:
                        conn.close()
                except Exception as e:
                    logger.warning(
                        "[%s] load cached weekly digestion failed | err=%s",
                        decision_date.date(),
                        e,
                    )

                if weekly_report is not None and isinstance(weekly_report, dict):
                    weekly_report = dict(weekly_report)
                    weekly_report["decision_date"] = decision_date

                    logger.info(
                        "[%s] WEEKLY OVERLAY LOADED FROM DB",
                        decision_date.date(),
                    )

                else:
                    logger.info(
                        "[%s] WEEKLY OVERLAY NOT FOUND IN DB | generating...",
                        decision_date.date(),
                    )

                    weekly_update_result = await news_weekly_update(
                        country_code="HK",
                        decision_date=decision_date,
                        keyword=None,
                        keywords=keywords,
                        lookback_days=7,
                        limit_fetch=20,
                        limit_model_items=10,
                        topic="Hang Seng Index",
                        run_id=f"{run_id}_{decision_date:%Y%m%d}_weekly",
                    )
                    weekly_update_result = dict(weekly_update_result or {})
                    weekly_update_result["decision_date"] = decision_date

                    logger.info(
                        "[%s] WEEKLY OVERLAY GENERATED | n_items=%s",
                        decision_date.date(),
                        weekly_update_result.get("n_items"),
                    )

                    # ----------------------------------------------------
                    # 5.3 Build monthly digest report
                    # ----------------------------------------------------
                    weekly_update_results = [weekly_update_result] if weekly_update_result else []

                    weekly_report = build_monthly_digest_from_memo_and_overlays(
                        decision_date=decision_date,
                        month_key=month_key,
                        weekly_update_results=weekly_update_results,
                    )

                    used_url_hashes = set()
                    for overlay in weekly_update_results:
                        for h in overlay.get("used_news", []) or []:
                            if isinstance(h, dict):
                                uh = h.get("url_hash")
                                if uh:
                                    used_url_hashes.add(uh)
                            elif isinstance(h, str):
                                used_url_hashes.add(h)

                    latest_weekly_update = weekly_update_results[-1] if weekly_update_results else {}

                    digest_row = {
                        "run_id": f"{run_id}_{decision_date:%Y%m%d}_monthly_digest",
                        "country_code": "HK",
                        "market_code": "HSI",
                        "anchor_date": pd.to_datetime(decision_date).date(),
                        "lookback_months": 1,
                        "model_name": model_name,
                        "news_window_start": pd.to_datetime(decision_date).replace(day=1).date(),
                        "news_window_end": pd.to_datetime(decision_date).date(),
                        "used_url_hashes_json": sorted(used_url_hashes),
                        "analyst_report_json": weekly_report,
                        "signal_strength": latest_weekly_update.get("signal_strength"),
                        "confidence": latest_weekly_update.get("confidence"),
                    }
                    if save_db:
                        conn = get_connection(include_database=True)
                        try:
                            save_news_digestion_one(conn, digest_row)
                            conn.commit()
                        finally:
                            conn.close()

                # ----------------------------------------------------
                # 5.4 Trader model (monthly decision only)
                # ----------------------------------------------------
                reasoning, final_text = call_llm(
                    features_df=daily_features,
                    context=weekly_report,
                    as_of_date=decision_date,
                )
                scenario = choose_scenario(final_text)
                direction = scenario["direction"]
                volatility = scenario["volatility"]
                strength = scenario["strength"]
                direction_probs = scenario["direction_probs"]
                volatility_probs = scenario["volatility_probs"]
                
                action_guidance = scenario.get("action_guidance", {})
                warning_flag = scenario.get("warning_flag", False)
                warning_type = scenario.get("warning_type", "none")
                warning_message = scenario.get("warning_message", "")
                entry_mode = action_guidance.get("entry_mode", "normal")
                size_multiplier = float(action_guidance.get("size_multiplier", 1.0) or 1.0)
                effective_risk_pct = risk_pct * size_multiplier
                effective_risk_pct = max(0.0, min(float(effective_risk_pct), float(risk_pct)))
                logger.info(
                    "[%s] TRADER OUTPUT | direction=%s | vol=%s | strength=%s | direction_probs=%s | vol_probs=%s | entry_mode=%s | effective_risk_pct=%s | warning_msg=%s |",
                    decision_date.date(),
                    direction,
                    volatility,
                    strength,
                    _safe_json(direction_probs),
                    _safe_json(volatility_probs),
                    entry_mode,
                    effective_risk_pct,
                    warning_message if warning_flag else "None",
                )

                # ----------------------------------------------------
                # 5.5 Backtest next month
                # ----------------------------------------------------
                underlying_close = (
                    hsi.set_index("Date")["Close"]
                    if "Date" in hsi.columns and "Close" in hsi.columns
                    else None
                )
                
                bt = compute_backtest_metrics(
                    scenario=scenario,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    option_df=merged_df,
                    account_size=account_size,
                    risk_pct=effective_risk_pct,
                    apply_risk_controls=True,
                    underlying_close=underlying_close,
                    stop_loss_ratio=stop_loss,
                    delta_limit=delta_limit,
                    capital_alloc_pct=capital_alloc_pct,
                    logger=logger,
                )

                metrics = bt["metrics"]
                mtm_df = bt["mtm_df"]
                signals = bt["signals"]
                strategy_type = bt["strategy_type"]
                risk_exit_date = bt.get("risk_exit_date")
                risk_exit_reason = bt.get("risk_exit_reason")
                t_bt = time.perf_counter()
                # # Single strategy boundary: signals → MTM → period metrics (see utils.strategy_pools)
                # cycle = run_prediction_cycle_strategy(
                #     strategy=scenario,
                #     entry_date=entry_date,
                #     exit_date=exit_date,
                #     option_df=merged_df,
                #     account_value=account_size,
                #     risk_pct=risk_pct,
                #     logger=logger,
                #     apply_risk_controls=True,
                # )

                # logger.debug("[%s] prediction cycle done | elapsed=%.3fs", decision_date.date(), time.perf_counter() - t_bt)

                # metrics = cycle["metrics"]
                # mtm_df = cycle["mtm_df"]
                # signals = cycle["signals"]
                # strategy_type = cycle["strategy_type"]

                logger.debug("[%s] strategy_type=%s", decision_date.date(), strategy_type)
                logger.debug("[%s] signals=%s", decision_date.date(), _safe_json(signals))
                logger.debug("[%s] metrics=%s", decision_date.date(), _safe_json(metrics))
                # # logger.debug(_df_debug_summary(f"mtm_df[{decision_date.date()}]", mtm_df, date_col="Date"))
                # logger.info(
                #     "[%s] MONTH END | scenario=%s | strategy=%s | cum_pnl=%s | return_pct=%s | sharpe=%s | max_dd=%s | risk_exit_date=%s | risk_exit_reason=%s | elapsed=%.3fs",
                #     decision_date.date(),
                #     scenario,
                #     strategy_type,
                #     metrics.get("cum_pnl"),
                #     metrics.get("return_pct"),
                #     metrics.get("sharpe"),
                #     metrics.get("max_drawdown_pct"),
                #     risk_exit_date,
                #     risk_exit_reason,
                #     time.perf_counter() - month_t0,
                # )
                # ----------------------------------------------------
                # 5.6 Merge MTM
                # ----------------------------------------------------
                if mtm_df.empty:
                    mtm_df = build_flat_mtm(merged_df, entry_date, exit_date)

                if not mtm_df.empty:
                    mtm_df = mtm_df.copy()
                    mtm_date_col = _pick_col(mtm_df, ["Date", "date"])
                    mtm_df[mtm_date_col] = pd.to_datetime(mtm_df[mtm_date_col]).dt.normalize()

                    mtm_df["decision_date"] = decision_date
                    mtm_df["entry_date"] = entry_date
                    mtm_df["exit_date"] = exit_date
                    mtm_df["scenario"] = scenario
                    mtm_df["strategy_type"] = strategy_type

                    mtm_df["portfolio_total_pnl"] = mtm_df["Total_PnL"] + cumulative_offset
                    mtm_df["portfolio_daily_pnl"] = mtm_df["Daily_PnL"]

                    cumulative_offset = float(mtm_df["portfolio_total_pnl"].iloc[-1])

                    mtm_parts.append(mtm_df)

                # ----------------------------------------------------
                # 5.7 Save performance
                # ----------------------------------------------------
                if save_db:
                    reasoning_text = "\n\n".join([
                        f"Decision date: {decision_date.date()}",
                        f"Chosen scenario: {scenario}",
                        f"Strategy type: {strategy_type}",
                        f"Weekly digest report:\n{weekly_report.get('digest_text', 'N/A')}",
                        f"LLM reasoning:\n{reasoning}" if reasoning else "LLM reasoning: None",
                    ])

                    save_performance_record(
                        run_id=run_id,
                        decision_date=decision_date,
                        final_text=final_text,
                        performance_metrics=metrics,
                        reasoning=reasoning_text,
                        model=model_name,
                    )

                    monthly_rows.append({
                        "decision_date": decision_date,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "month_key": month_key,
                        "direction_probs": direction_probs,
                        "volatility_probs": volatility_probs,
                        "scenario": scenario,
                        "strategy_type": strategy_type,
                        "signals": signals,
                        "digest_text": weekly_report.get("digest_text"),
                        "risk_exit_date": risk_exit_date,
                        "risk_exit_reason": risk_exit_reason,
                        "cum_pnl": metrics.get("cum_pnl"),
                        "return_pct": metrics.get("return_pct"),
                        "ann_return_pct": metrics.get("ann_return_pct"),
                        "ann_vol_pct": metrics.get("ann_vol_pct"),
                        "sharpe": metrics.get("sharpe"),
                        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                        "win_rate_daily": metrics.get("win_rate_daily"),
                    })

                logger.info(
                    "[%s] MONTH END ｜ cum_pnl=%s | return_pct=%s | sharpe=%s | max_dd=%s | elapsed=%.3fs",
                    decision_date.date(),
                    # scenario,
                    # strategy_type,
                    metrics.get("cum_pnl"),
                    metrics.get("return_pct"),
                    metrics.get("sharpe"),
                    metrics.get("max_drawdown_pct"),
                    time.perf_counter() - month_t0,
                )

            except Exception as e:
                logger.exception(
                    "[%s] MONTH FAILED | entry_date=%s | exit_date=%s | error=%s",
                    decision_date, entry_date, exit_date, e
                )
                raise

        # ============================================================
        # 6) Final assemble
        # ============================================================
        monthly_df = pd.DataFrame(monthly_rows)

        if mtm_parts:
            full_mtm_df = pd.concat(mtm_parts, ignore_index=True)
            full_mtm_df["Date"] = pd.to_datetime(full_mtm_df["Date"]).dt.normalize()
            full_mtm_df = full_mtm_df.sort_values("Date").reset_index(drop=True)
            full_mtm_df = full_mtm_df.drop_duplicates(subset=["Date"], keep="last")
            mtm_date_col = _pick_col(full_mtm_df, ["Date", "date"])
            full_mtm_df = full_mtm_df.sort_values(mtm_date_col).reset_index(drop=True)
            full_mtm_df["portfolio_daily_pnl"] = (
                full_mtm_df["portfolio_total_pnl"]
                .diff()
                .fillna(full_mtm_df["portfolio_total_pnl"])
            )
        else:
            full_mtm_df = pd.DataFrame()

        overall_metrics = summarize_mtm_path(
            mtm_df=full_mtm_df,
            account_size=account_size,
        )

        logger.info("OVERALL METRICS=%s", _safe_json(overall_metrics))
        logger.info("RUN END | run_id=%s | total_elapsed=%.3fs", run_id, time.perf_counter() - t0)

        return {
            "run_id": run_id,
            "monthly_df": monthly_df,
            "full_mtm_df": full_mtm_df,
            "overall_metrics": overall_metrics,
        }

    except Exception as e:
        logger.exception("RUN FAILED | run_id=%s | error=%s", run_id, e)
        raise
    
    

async def main():
    
    args = parse_args()

    result = await run_full_timeline(
        start_date=args.start,
        end_date=args.end,
        model_name=args.model_name,
        account_size=args.account_size,
        risk_pct=args.risk_pct,
        capital_alloc_pct=args.alloc_pct,
        stop_loss=args.stop_loss,
        delta_limit=args.delta_limit,
        search_news=args.search_news,
        save_db=args.save_db,
    )

    print(result["run_id"])
    if not result["monthly_df"].empty:
        print(result["monthly_df"][[
            "decision_date", "scenario", "strategy_type", "cum_pnl", "sharpe"
        ]])
    print(result["overall_metrics"])

    out_path = "./outputs"
    os.makedirs(out_path, exist_ok=True)

    run_id = result["run_id"]
    result["monthly_df"].to_csv(
        os.path.join(out_path, f"monthly_summary_{run_id}.csv"),
        index=False
    )
    result["full_mtm_df"].to_csv(
        os.path.join(out_path, f"full_mtm_path_{run_id}.csv"),
        index=False
    )


if __name__ == "__main__":
    asyncio.run(main())