# File: src/loansys_llm_cro/news_search.py

import os
import re
import datetime
import urllib.parse
from typing import List, Dict, Any, Optional

import httpx
import feedparser
from urls import *
from email.utils import parsedate_to_datetime
from fetch_fulltext import get_google_params, get_origin_url


# ============================================================
# Config
# ============================================================

GOOGLE_HL = os.getenv("GOOGLE_HL", "en")
GOOGLE_GL = os.getenv("GOOGLE_GL", "US")
GOOGLE_CEID = os.getenv("GOOGLE_CEID", "US:en")

TARGET_RECENT = 20
DATE_RANGE_START = datetime.date(2024, 1, 1)
DATE_RANGE_END = datetime.date(2026, 3, 1)


# ============================================================
# Helpers
# ============================================================

def _to_iso_date(published: str, published_parsed=None) -> Optional[str]:
    """
    Convert RSS published fields into YYYY-MM-DD if possible.
    """
    if published_parsed:
        try:
            y, m, d = published_parsed.tm_year, published_parsed.tm_mon, published_parsed.tm_mday
            return datetime.date(y, m, d).isoformat()
        except Exception:
            pass

    if published:
        try:
            dt = parsedate_to_datetime(published)
            return dt.date().isoformat()
        except Exception:
            return None
    return None


def _clean_html_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"<[^>]+>", " ", s).strip()


def _safe_decode_google_news_url(rss_url: str) -> Optional[str]:
    """
    Decode a Google News RSS redirect URL into the original publisher URL.
    Returns None if decoding fails.
    """
    try:
        source, sign, ts = get_google_params(rss_url)
        return get_origin_url(source, sign, ts)
    except Exception as e:
        print(f"[decode-origin-url-error] {e} | rss_url={rss_url}")
        return None


def _append_if_in_range(
    it: Dict[str, Any],
    buf: List[Dict[str, Any]],
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
) -> None:
    """
    Normalize item schema and keep only items inside [start_date, end_date].
    """
    approx = it.get("approx_date")
    if not in_range(approx, start_date, end_date):
        return

    buf.append({
        "title": it.get("title", ""),
        "url": it.get("url", ""),                 # Google News RSS URL
        "origin_url": it.get("origin_url"),       # decoded original publisher URL
        "abstract": it.get("abstract", ""),
        "approx_date": approx,
        "source": it.get("source", "google_rss"),
        **({"query": it["query"]} if it.get("query") else {}),
    })



# ============================================================
# Google News RSS
# ============================================================

def google_news_rss(
    q: str,
    when: str = "30d",
    *,
    country_code: str = "",
    language: Optional[str] = None,
    decode_origin: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch Google News RSS results for a query.
    Output keeps:
      - url: Google News RSS redirect link
      - origin_url: decoded original publisher link (optional)
    """
    base = "https://news.google.com/rss/search"

    hl = (language or GOOGLE_HL or "en").strip()
    gl = (country_code or GOOGLE_GL or "US").strip().upper()
    ceid = f"{gl}:{hl}" if country_code else (GOOGLE_CEID or f"{gl}:{hl}")

    # important: put when inside q for Google News RSS search
    q_full = q.strip()
    if when:
        q_full = f"{q_full} when:{when}"

    params = {
        "q": q_full,
        "hl": hl,
        "gl": gl,
        "ceid": ceid,
    }
    url = base + "?" + urllib.parse.urlencode(params)

    try:
        resp = httpx.get(
            url,
            timeout=20.0,
            follow_redirects=True,
            trust_env=False,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
            },
        )
        resp.raise_for_status()
        fp = feedparser.parse(resp.text)
    except Exception as e:
        print(f"[rss-fetch-error] {e} | q={q} | url={url}")
        return []

    items: List[Dict[str, Any]] = []

    for ent in fp.entries:
        rss_link = ent.get("link") or ""
        title = ent.get("title", "") or ""
        summary = ent.get("summary", "") or ""

        approx = (
            _to_iso_date(ent.get("published"), ent.get("published_parsed"))
            or infer_date_from_url_or_text(rss_link, summary or title)
        )

        origin_url = _safe_decode_google_news_url(rss_link) if (decode_origin and rss_link) else None

        items.append({
            "title": title,
            "url": rss_link,   # Google News RSS link
            "origin_url": origin_url,
            "abstract": _clean_html_text(summary),
            "approx_date": approx if approx and re.match(r"^\d{4}-\d{2}-\d{2}$", str(approx)) else None,
            "source": "google_rss",
            "query": q,
        })

    return items


# ============================================================
# Public API
# ============================================================

def harvest_news(
    topic: str,
    kwq_bundle: Dict[str, Dict[str, List[str]]],
    target_recent: int = TARGET_RECENT,
    max_queries: int = 40,
    start_date: Optional[datetime.date] = DATE_RANGE_START,
    end_date: Optional[datetime.date] = DATE_RANGE_END,
    *,
    country_code: str = "",
    language: Optional[str] = None,
    decode_origin: bool = True,
) -> List[Dict[str, Any]]:
    """
    RSS-only harvesting.

    Expected kwq_bundle shape from keyword_model:
    {
      "rss": {
        "keywords": [...],
        "queries": [...]
      }
    }

    Returns normalized items:
    [
      {
        "title": ...,
        "url": ...,          # Google News RSS URL
        "origin_url": ...,   # decoded original publisher URL, for fetch_fulltext.read(...)
        "abstract": ...,
        "approx_date": ...,
        "source": "google_rss",
        "query": ...
      },
      ...
    ]
    """
    mode = (kwq_bundle or {}).get("rss", {}) or {}
    mode_queries: List[str] = [str(q).strip() for q in (mode.get("queries") or []) if str(q).strip()]

    if not mode_queries:
        print("[harvest][rss] empty query list")
        return []

    total_days = ((end_date - start_date).days + 1) if (start_date and end_date) else 60
    if total_days <= 7:
        when = "7d"
    elif total_days <= 14:
        when = "14d"
    elif total_days <= 30:
        when = "30d"
    elif total_days <= 60:
        when = "60d"
    elif total_days <= 90:
        when = "90d"
    elif total_days <= 180:
        when = "180d"
    else:
        when = "365d"

    harvested: List[Dict[str, Any]] = []
    used = 0

    for q in mode_queries:
        if used >= max_queries:
            break

        try:
            batch = google_news_rss(
                q=q,
                when=when,
                country_code=country_code,
                language=language,
                decode_origin=decode_origin,
            )
            for it in batch or []:
                _append_if_in_range(it, harvested, start_date, end_date)
        except Exception as e:
            print(f"[harvest][rss] err {e} | q={q}")

        used += 1

    harvested = dedup_items(harvested)

    recent = [
        it for it in harvested
        if in_range(it.get("approx_date"), start_date, end_date)
    ]

    print(f"    ✓ Harvested {len(harvested)} items ({len(recent)} recent) | mode=RSS-only | queries_used={used}")

    return recent if len(recent) >= max(8, target_recent // 2) else harvested


def build_fulltext_jobs(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience helper:
    prepare items for fetch_fulltext.read(...)

    Output:
    [
      {
        "title": ...,
        "origin_url": ...,
        "approx_date": ...,
        ...
      }
    ]
    """
    jobs = []
    for it in news_items or []:
        origin_url = it.get("origin_url")
        if not origin_url:
            continue
        jobs.append(it)
    return jobs