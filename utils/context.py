import asyncio
import hashlib
from typing import List, Dict, Any, Optional

import httpx
import pandas as pd

from db.operations import fetch_news
from get_news.fetch_fulltext import read
from utils.utils import *
from get_news.urls import *

# ============================================================
# Normalization / dedup
# ============================================================

def normalize_db_news_items(db_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for it in db_items or []:
        row = dict(it)

        source_url = (
            row.get("url")
            or row.get("origin_url")
            or row.get("canon_url")
            or ""
        ).strip()

        canon_url = source_url
        if source_url:
            try:
                canon_url = canonicalize_url(source_url) or source_url
            except Exception:
                canon_url = source_url

        chash = (
            str(row.get("url_hash") or "").strip()
            or (url_hash(canon_url) if canon_url else "")
        )

        dt = pd.to_datetime(
            row.get("_news_date") or row.get("published_at") or row.get("approx_date"),
            errors="coerce",
        )

        row["_canon_url"] = canon_url
        row["_url_hash"] = chash
        row["_news_date"] = dt.normalize() if pd.notna(dt) else pd.NaT
        out.append(row)

    return out


def dedup_news_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []

    for row in items or []:
        row = dict(row)
        chash = str(row.get("_url_hash") or row.get("url_hash") or "").strip()

        if not chash:
            url = (
                row.get("_canon_url")
                or row.get("canon_url")
                or row.get("origin_url")
                or row.get("url")
                or ""
            ).strip()
            if not url:
                continue
            chash = url_hash(url)

        if chash in seen:
            continue

        seen.add(chash)
        row["_url_hash"] = chash
        out.append(row)

    return out


def filter_news_by_date(
    news_list: List[Dict[str, Any]],
    *,
    start_date,
    end_date,
) -> List[Dict[str, Any]]:
    sd = pd.to_datetime(start_date).normalize()
    ed = pd.to_datetime(end_date).normalize()

    out = []
    for item in news_list or []:
        row = dict(item)
        dt = pd.to_datetime(
            row.get("_news_date") or row.get("published_at") or row.get("approx_date"),
            errors="coerce",
        )
        if pd.isna(dt):
            continue
        dt = dt.normalize()
        if sd <= dt <= ed:
            row["_news_date"] = dt
            out.append(row)

    out.sort(key=lambda x: x["_news_date"], reverse=True)
    return out


# ============================================================
# DB fetch
# ============================================================

def fetch_news_for_context(
    *,
    start_date,
    end_date,
    keyword: Optional[str] = None,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    raw = fetch_news(
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        keyword=keyword,
    )

    normalized = normalize_db_news_items(raw)
    filtered = filter_news_by_date(
        normalized,
        start_date=start_date,
        end_date=end_date,
    )
    deduped = dedup_news_items(filtered)
    deduped.sort(
        key=lambda x: x.get("_news_date", pd.NaT),
        reverse=True,
    )
    return deduped


# ============================================================
# Fulltext enrichment
# ============================================================

async def _read_fulltext_safe(url: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        try:
            return await asyncio.to_thread(read, url)
        except httpx.HTTPStatusError as e:
            status = getattr(e.response, "status_code", None)
            print(f"[context][http-error] status={status} url={url}")
            return ""
        except Exception as e:
            print(f"[context][read-failed] url={url} err={e}")
            return ""


async def enrich_news_with_fulltext(
    items: List[Dict[str, Any]],
    *,
    max_items: int = 12,
    min_existing_content_chars: int = 500,
    max_chars: int = 1200,
    max_concurrency: int = 5,
) -> List[Dict[str, Any]]:
    base_rows: List[Dict[str, Any]] = []

    for it in (items or [])[:max_items]:
        row = dict(it)

        title = (row.get("title") or "").strip()
        url = (
            row.get("url")
            or row.get("origin_url")
            or row.get("_canon_url")
            or row.get("canon_url")
            or ""
        ).strip()
        url_hash_ = (
            str(row.get("_url_hash") or "").strip()
            or str(row.get("url_hash") or "").strip()
        )
        published_at = row.get("published_at") or row.get("approx_date")
        source = row.get("source") or row.get("source_api") or "unknown"
        summary = (row.get("summary") or row.get("abstract") or "").strip()
        content = (row.get("content") or "").strip()
        news_date = pd.to_datetime(
            row.get("_news_date") or published_at,
            errors="coerce",
        )

        if not title or not url:
            continue

        base_rows.append({
            "title": title,
            "published_at": published_at,
            "url": url,
            "url_hash": url_hash_,
            "source": source,
            "summary": summary,
            "content": content,
            "_news_date": news_date.normalize() if pd.notna(news_date) else pd.NaT,
        })

    if not base_rows:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []
    for row in base_rows:
        existing = (row.get("content") or "").strip()
        if len(existing) < min_existing_content_chars:
            tasks.append(_read_fulltext_safe(row["url"], semaphore))
        else:
            tasks.append(asyncio.sleep(0, result=""))

    fetched = await asyncio.gather(*tasks)

    out: List[Dict[str, Any]] = []
    for row, fulltext in zip(base_rows, fetched):
        content = (row.get("content") or "").strip()

        if fulltext and len(fulltext.strip()) > len(content):
            content = fulltext.strip()

        if not content:
            content = row.get("summary") or row.get("title") or ""

        row["content"] = content
        row["text_for_model"] = trim_to_chars(content, max_chars)
        out.append(row)

    return out


# ============================================================
# Bundle packing
# ============================================================

def build_news_bundle_text(
    items: List[Dict[str, Any]],
    *,
    start_date,
    end_date,
    label: str = "News Bundle",
) -> str:
    sd = pd.to_datetime(start_date).normalize()
    ed = pd.to_datetime(end_date).normalize()

    lines: List[str] = []
    lines.append(f"## {label}")
    lines.append(f"Window start: {sd.date()}")
    lines.append(f"Window end: {ed.date()}")
    lines.append("")

    if not items:
        lines.append("No relevant news articles were found.")
        return "\n".join(lines)

    for i, item in enumerate(items, start=1):
        dt = pd.to_datetime(item.get("_news_date") or item.get("published_at"), errors="coerce")
        dt_str = str(dt.date()) if pd.notna(dt) else "(unknown)"
        source = item.get("source") or "unknown"
        title = item.get("title") or "(no title)"
        url = item.get("url") or "(no url)"
        text_for_model = (
            item.get("text_for_model")
            or item.get("content")
            or item.get("summary")
            or title
        )

        lines.append(f"[{i}]")
        lines.append(f"Date: {dt_str}")
        lines.append(f"Source: {source}")
        lines.append(f"URL: {url}")
        lines.append(f"Title: {title}")
        lines.append(f"Text: {text_for_model}")
        lines.append("")

    return "\n".join(lines).strip()


# ============================================================
# Main context entrypoint
# ============================================================

async def build_news_context_from_db(
    *,
    start_date,
    end_date,
    keyword: Optional[str] = None,
    limit_fetch: int = 30,
    limit_model_items: int = 12,
    fulltext_chars: int = 1200,
    min_existing_content_chars: int = 500,
    max_concurrency: int = 5,
    label: str = "News Bundle",
) -> Dict[str, Any]:
    raw_items = fetch_news_for_context(
        start_date=start_date,
        end_date=end_date,
        keyword=keyword,
        limit=limit_fetch,
    )

    if not raw_items:
        return {
            "start_date": str(pd.to_datetime(start_date).date()),
            "end_date": str(pd.to_datetime(end_date).date()),
            "used_news": [],
            "bundle_text": build_news_bundle_text(
                [],
                start_date=start_date,
                end_date=end_date,
                label=label,
            ),
            "n_items": 0,
        }

    used_news = await enrich_news_with_fulltext(
        raw_items,
        max_items=limit_model_items,
        min_existing_content_chars=min_existing_content_chars,
        max_chars=fulltext_chars,
        max_concurrency=max_concurrency,
    )

    bundle_text = build_news_bundle_text(
        used_news,
        start_date=start_date,
        end_date=end_date,
        label=label,
    )

    return {
        "start_date": str(pd.to_datetime(start_date).date()),
        "end_date": str(pd.to_datetime(end_date).date()),
        "used_news": used_news,
        "bundle_text": bundle_text,
        "n_items": len(used_news),
    }