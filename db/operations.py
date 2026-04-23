from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import json 
from db.init import get_connection
from utils.utils import *
from utils.utils import _extract_json_from_text

def normalize_datetime(dt_value: Optional[Union[str, datetime]]) -> Optional[str]:
    if dt_value is None:
        return None

    if isinstance(dt_value, datetime):
        return dt_value.strftime("%Y-%m-%d %H:%M:%S")

    if not isinstance(dt_value, str):
        return None

    text = dt_value.strip()
    if not text:
        return None

    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in fmts:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    return None


def save_news_one(conn, news: Dict[str, Any]) -> None:
    sql = f"""
    INSERT INTO {TABLE_NEWS} (
        title,
        url,
        url_hash,
        source,
        published_at,
        keyword,
        summary,
        content,
        relevance_score
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        title = VALUES(title),
        url = VALUES(url),
        source = VALUES(source),
        published_at = VALUES(published_at),
        keyword = VALUES(keyword),
        summary = VALUES(summary),
        content = VALUES(content),
        relevance_score = VALUES(relevance_score)
    """

    values = (
        news.get("title"),
        news.get("url"),
        news.get("url_hash"),
        news.get("source"),
        normalize_datetime(news.get("published_at") or news.get("approx_date")),
        news.get("keyword"),
        news.get("summary"),
        news.get("content"),
        news.get("relevance_score", 1.0),
    )

    cursor = conn.cursor()
    cursor.execute(sql, values)
    cursor.close()


def save_news(items: Union[Dict[str, Any], List[Dict[str, Any]]]) -> int:
    if isinstance(items, dict):
        items = [items]

    if not items:
        return 0

    conn = get_connection()
    affected = 0

    try:
        for item in items:
            if not item:
                continue
            save_news_one(conn, item)
            affected += 1

        conn.commit()
        print(f"[save_news] affected={affected}")
        return affected

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()
        

from datetime import datetime, date, time
from typing import Optional, Union

def normalize_datetime(
    dt_value: Optional[Union[str, datetime, date]],
    *,
    end_of_day: bool = False,
) -> Optional[str]:
    if dt_value is None:
        return None

    if isinstance(dt_value, datetime):
        return dt_value.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(dt_value, date):
        dt = datetime.combine(
            dt_value,
            time.max if end_of_day else time.min
        )
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    if not isinstance(dt_value, str):
        return None

    text = dt_value.strip()
    if not text:
        return None

    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in fmts:
        try:
            dt = datetime.strptime(text, fmt)
            if fmt in ("%Y-%m-%d", "%Y/%m/%d"):
                dt = datetime.combine(
                    dt.date(),
                    time.max if end_of_day else time.min
                )
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            continue

    return None


def fetch_news(start_date, end_date, limit=None, keyword=None):
    conn = get_connection(include_database=True)
    cursor = conn.cursor(dictionary=True)

    sql = f"SELECT * FROM {TABLE_NEWS} WHERE 1=1"
    params = []

    if start_date:
        sql += " AND published_at >= %s"
        params.append(normalize_datetime(start_date, end_of_day=False))

    if end_date:
        sql += " AND published_at <= %s"
        params.append(normalize_datetime(end_date, end_of_day=True))

    if keyword:
        sql += " AND keyword = %s"
        params.append(keyword)

    sql += " ORDER BY published_at DESC"

    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))

    cursor.execute(sql, params)
    results = cursor.fetchall()

    cursor.close()
    conn.close()
    return results


def save_performance_record(
    run_id: str,
    decision_date,
    final_text: Union[str, Dict[str, Any], None],
    performance_metrics: Union[Dict[str, Any], str, None],
    reasoning: Optional[str] = None,
    model: str = "deepseek-reasoner",
) -> int:
    """
    Insert one row into performance table.

    Parameters
    ----------
    run_id : str
    decision_date : str | datetime
        Monthly decision date. Will be written into prediction_json.
    final_text : str | dict | None
        Raw LLM output or parsed dict.
        prediction_json will be built inside this function.
    performance_metrics : dict | str | None
    reasoning : str | None
    model : str

    Returns
    -------
    int
        inserted row id
    """
    conn = get_connection()
    cur = None

    try:
        if final_text is None:
            return None

        if isinstance(final_text, dict):
            parsed = to_json_safe(final_text)
        elif isinstance(final_text, str):
            parsed = _extract_json_from_text(final_text)
        else:
            parsed = None

        if parsed is None:
            return None

        direction_probs = parsed.get("direction_probs") if isinstance(parsed, dict) else None
        volatility_probs = parsed.get("volatility_probs") if isinstance(parsed, dict) else None

        chosen_scenario = None
        if isinstance(parsed, dict):
            chosen_scenario = (
                parsed.get("chosen_scenario")
                or parsed.get("scenario")
            )

        payload = {
            "decision_date": str(pd.to_datetime(decision_date).date()) if decision_date is not None else None,
            "direction_probs": direction_probs,
            "volatility_probs": volatility_probs,
            "chosen_scenario": chosen_scenario,
            "raw_llm_json": parsed,
        }
        prediction_json=to_json_safe(payload)

        if isinstance(performance_metrics, dict):
            perf_json = to_json_safe(performance_metrics)
        elif isinstance(performance_metrics, str):
            perf_json = _extract_json_from_text(performance_metrics)
        else:
            perf_json = None

        sql = f"""
            INSERT INTO {TABLE_PERFORMANCE} (
                run_id,
                created_at,
                model,
                reasoning,
                prediction_json,
                performance_metrics
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        created_at = dt.datetime.now()

        cur = conn.cursor()
        cur.execute(
            sql,
            (
                run_id,
                created_at,
                model,
                reasoning,
                json.dumps(prediction_json, ensure_ascii=False) if prediction_json is not None else None,
                json.dumps(perf_json, ensure_ascii=False) if perf_json is not None else None,
            ),
        )
        inserted_id = cur.lastrowid
        conn.commit()
        return inserted_id

    except Exception:
        conn.rollback()
        raise

    finally:
        if cur is not None:
            cur.close()
        conn.close()

def save_news_digestion_one(conn, row: Dict[str, Any]) -> None:
    sql =f"""
    INSERT INTO {TABLE_DIGEST} (
        run_id,
        country_code,
        market_code,
        anchor_date,
        lookback_months,
        model_name,
        news_window_start,
        news_window_end,
        used_url_hashes_json,
        analyst_report_json,
        signal_strength,
        confidence
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        country_code = VALUES(country_code),
        market_code = VALUES(market_code),
        anchor_date = VALUES(anchor_date),
        lookback_months = VALUES(lookback_months),
        model_name = VALUES(model_name),
        news_window_start = VALUES(news_window_start),
        news_window_end = VALUES(news_window_end),
        used_url_hashes_json = VALUES(used_url_hashes_json),
        analyst_report_json = VALUES(analyst_report_json),
        signal_strength = VALUES(signal_strength),
        confidence = VALUES(confidence)
    """
    cursor = conn.cursor()
    cursor.execute(
        sql,
        (
            row["run_id"],
            row.get("country_code", "HK"),
            row.get("market_code", "HSI"),
            row.get("anchor_date"),
            row.get("lookback_months", 1),
            row.get("model_name"),
            row.get("news_window_start"),
            row.get("news_window_end"),
            json.dumps(row.get("used_url_hashes_json", []), ensure_ascii=False),
            json.dumps(row.get("analyst_report_json", {}), ensure_ascii=False),
            row.get("signal_strength"),
            row.get("confidence"),
        ),
    )
    cursor.close()


def load_digestion(
    conn,
    *,
    decision_date,
    country_code: str = "HK",
    market_code: str = "HSI",
) -> Optional[Dict[str, Any]]:
    sql = f"""
    SELECT analyst_report_json
    FROM {TABLE_DIGEST}
    WHERE country_code = %s
      AND market_code = %s
      AND anchor_date = %s
    """
    params = [
        country_code,
        market_code,
        pd.to_datetime(decision_date).date(),
    ]


    sql += " ORDER BY updated_at DESC, id DESC LIMIT 1"

    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql, tuple(params))
    row = cursor.fetchone()
    cursor.close()

    if not row:
        return None

    raw = row.get("analyst_report_json")
    try:
        return raw if isinstance(raw, dict) else json.loads(raw)
    except Exception:
        return None