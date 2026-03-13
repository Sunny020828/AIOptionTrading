# File: src/loansys_llm_cro/models/reader_model.py
import asyncio
import datetime
import json
import re
from typing import List, Dict, Any, Optional

import httpx
import pymysql
from tqdm.auto import tqdm
import os
from urls import *  # provides in_range, canonicalize_url, url_hash, publisher_from_url, etc.
from loansys_llm_cro.utils import *  # IMPORTANT: reader_model uses many globals from utils (HTTP_TIMEOUT, USER_AGENT, etc.)
from loansys_llm_cro.db.credential import DB_CONFIG
from loansys_llm_cro.db.storage import TABLE_EVIDENCE, evidence_exists_by_hash
from fetch_fulltext import read
from news_search import harvest_news
from keyword_model import build_topic_queries, achat_complete


BATCH_SIZE = int(os.getenv("READER_BATCH_SIZE", "50"))
MIN_VALID_TO_STOP = int(os.getenv("READER_MIN_VALID_TO_CONTINUE", "10"))

# -----------------------------
# Local helper (utils.py does NOT have date_sort_key)
# -----------------------------
def date_sort_key(item: Dict[str, Any]) -> str:
    """
    Sort key for candidate items by date.
    Assumes ISO date strings (YYYY-MM-DD) sort lexicographically.
    """
    d = item.get("approx_date") or item.get("time") or ""
    return str(d or "")


# =========================
# 2) reader_model
# =========================
READER_SYSTEM = """You are a fast, no-nonsense **news relevance judge** for e-commerce industries on Amazon.

ARTICLE GATE (run this BEFORE scoring relevance):
- Determine if the URL/fulltext represents a **single article/press release/advisory** (with coherent paragraphs and ideally a publish date/byline).
- Treat as **NOT an article page** if it looks like: homepage, topic/section/tag/category/archive pages, company/product landing pages, shopping/marketplace listings, author index, “news and articles” hubs, search results, media libraries, podcast index, or pages whose main content is a list of headlines/links.
- URL patterns that usually indicate NOT an article: paths starting with or ending in `topic/`, `section/`, `tag/`, `category/`, `press/`, `press-releases/`, `news/` (when it’s a hub), `news-and-articles/`, `archive/`, `company/`, `subjects/`, `by/`, `pages/`; NYT-like `nytimes.com/topic/...`, `.../section/...`; `.rss`, `.atom`, `.xml`, `/feed`.
- If **NOT an article page**: OUTPUT JSON with EXACT schema and these values:
  {
    "is_article": false,
    "relevant score": 1,
    "summary": "",
    "reason": "not an article page",
    "mechanisms": [],
    "impact_direction": "unclear",
    "impact_channels": [],
    "regions": [],
    "has_china_linkage": false,
    "has_amazon_linkage": false,
    "time_horizon": "unclear"
  }


Your job (ONLY if the page passes the Article Gate):
1) Read the **Fulltext** and use the **Keywords List** to decide whether the article materially affects the specified **Industry/Category** on **Amazon** in the user-provided market/country context.
2) If relevant, produce:
   - a concise English abstract (6-8 sentences, ≤300 words) stating the concrete mechanism of impact (policy/fees/recalls/compliance/logistics/seasonality/costs/prices/demand/macro shocks), and
   - a **one-sentence reason** explaining **why** you assigned this score (what signals/mechanisms justify it).
3) Output **JSON only** with EXACT schema (no extra keys, no prose):
{
  "is_article": true,
  "relevant score": <integer 1-10>,
  "summary": "<English abstract>",
  "reason": "<≤50 words why this score>",
  "mechanisms": [
    "tariff" | "platform_fee" | "logistics" | "recall_safety" |
    "demand" | "macro_fx" | "competition" | "other"
  ],
  "impact_direction": "negative" | "positive" | "mixed" | "unclear",
  "impact_channels": [
    "sales_level" | "volatility" | "costs" | "inventory" | "other"
  ],
  "regions": [
    "US" | "EU" | "UK" | "global" | "other"
  ],
  "has_china_linkage": true/false,
  "has_amazon_linkage": true/false,
  "time_horizon": "0-1 months" | "1-3 months" | "3-12 months" | "unclear"
}
- Use "mechanisms" and "impact_channels" to tag how this article affects Amazon sellers in the specified market:
  tariffs, platform fees, logistics, recalls/safety, demand, macro FX, etc.
- "has_china_linkage" should be true if CN exporters, CN manufacturing, CN lanes, or CN-based sellers are clearly involved.
- "has_amazon_linkage" should be true if the article is about Amazon marketplace, FBA/SFP, Buy with Prime, Amazon fees, or Amazon listing/recall actions.

(Output constraints and rubric omitted here for brevity in this file; keep your original as-is.)
"""


async def fetch_fulltext(url: str) -> str:
    if not url:
        return ""
    try:
        text = ""
        if read:
            try:
                text = await asyncio.to_thread(read, url)
            except Exception:
                text = ""

        if not text:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client_http:
                r = await client_http.get(url)
                r.raise_for_status()
                text = clean_html_to_text(r.text)

        if text and len(text) > 60000:
            text = text[:60000]
        return text.strip()
    except Exception:
        return ""


def build_reader_user(
    title: str,
    abstract: str,
    fulltext: str,
    keywords: List[str],
    *,
    url: str = "",
    industry: str = "",
    country_code: str = "",
    currency: str = "",
    query: str = "",
    query_tier: str = "",
    source_api: str = "",
    ) -> str:
    head = "Keywords List: " + ", ".join([k for k in keywords if k]) + "\n\n"
    meta = []
    if industry:
        meta.append(f"Industry/Category: {industry}")
    if country_code:
        meta.append(f"Country: {country_code}")
    if currency:
        meta.append(f"Currency: {currency}")
    if query:
        meta.append(f"Search query: {query}")
    if query_tier:
        meta.append(f"Query tier: {query_tier}")
    if source_api:
        meta.append(f"Source API: {source_api}")  # e.g. cse / rss

    body = (
        "\n".join(meta) + ("\n\n" if meta else "") +
        f"Title: {title or '(none)'}\n"
        f"URL: {url or '(none)'}\n"
        f"Snippet (from search): {abstract or '(none)'}\n\n"
        f"Fulltext (raw, may contain boilerplate):\n{fulltext or '(failed)'}\n\n"
        "Return JSON only."
    )

    return head + body


MODEL_MAX_CONN = int(os.getenv("MODEL_MAX_CONN", "3"))
MODEL_SEM = asyncio.Semaphore(MODEL_MAX_CONN)
async def process_one_article(article: Dict[str, Any], keywords: List[str]) -> Optional[Dict[str, str]]:
    """
    Assumes upstream provides canonical source URL at article["url"].
    """
    title: str = (article.get("title") or "").strip()
    abstract: str = (article.get("abstract") or "").strip()
    canon_url: str = (article.get("url") or "").strip()
    industry: str = article.get("industry", "")
    country_code: str = article.get("country_code", "")
    currency: str = article.get("currency", "")
    query: str = (article.get("query") or "").strip()
    query_tier: str = (article.get("query_tier") or "").strip()
    source_api: str = (article.get("source_api") or "").strip()

    if not canon_url:
        return None

    try:
        chash = url_hash(canon_url)
    except Exception:
        chash = ""

    try:
        fulltext = await fetch_fulltext(canon_url)
    except Exception:
        fulltext = ""
    if not fulltext:
        fulltext = (title + ". " + abstract).strip()

    user = build_reader_user(
        title=title,
        abstract=abstract,
        fulltext=fulltext,
        keywords=keywords,
        url=canon_url,
        industry=industry,
        country_code=country_code,
        currency=currency,
        query=query,
        query_tier=query_tier,
        source_api=source_api,
    )

    try:
        async with MODEL_SEM:
            raw = await achat_complete(READER_MODEL, READER_SYSTEM, user, temperature=TEMPERATURE, seed=SEED)
    except Exception as e:
        print(f"[reader][error-model] model={READER_MODEL} url={canon_url} err={repr(e)}")
        return None

    # if not raw or len(raw.strip()) < 10:
    #     print(f"[reader][empty-raw] url={canon_url} raw_len={len(raw or '')}")

    try:
        s, e = raw.find("{"), raw.rfind("}")
        js = json.loads(raw[s:e + 1]) if (s >= 0 and e > s) else {}

        score = js.get("relevant score", js.get("score"))
        try:
            score = int(score)
        except Exception:
            m = re.search(r'"relevant\s*score"\s*:\s*(\d+)', raw, flags=re.I)
            if not m:
                m = re.search(r'"score"\s*:\s*(\d+)', raw, flags=re.I)
            score = int(m.group(1)) if m else None

        reason = (js.get("reason") or js.get("why") or "").strip()
        if not reason:
            if score is None:
                # print("[reader][no-score-raw-head]", (raw or "")[:300])
                reason = "No parsable score."
            elif score >= 9:
                reason = "Explicit category-level tariff/recall/logistics change impacting Amazon CN sellers."
            elif score >= 7:
                reason = "Category and CN Amazon sellers likely affected by policy/fees/compliance/logistics."
            elif score >= 4:
                reason = "General or indirect signals; limited category linkage."
            else:
                reason = "Weak linkage to the target category/sellers."

        print(f"[reader] {score or '-'} | {canon_url} | {title[:80]} | reason: {reason[:60]}")

        if score is None or score < READER_SCORE_THRESHOLD:
            return None

        is_article = js.get("is_article")
        if isinstance(is_article, str):
            is_article = is_article.strip().lower() in ("true", "1", "yes", "y")
        if is_article is None:
            is_article = True

        mechanisms = js.get("mechanisms") or []
        impact_direction = js.get("impact_direction") or ""
        impact_channels = js.get("impact_channels") or []
        regions = js.get("regions") or []
        has_china_linkage = bool(js.get("has_china_linkage"))
        has_amazon_linkage = bool(js.get("has_amazon_linkage"))
        time_horizon = js.get("time_horizon") or ""

        summary = trim_to_chars((js.get("summary") or "").strip(), 200)
        if not summary:
            return None

        try:
            existed_prior = (
                evidence_exists_by_hash(chash, country_code=country_code, category_root=industry)
                if chash else False
            )
        except Exception:
            existed_prior = False

        date_str = article.get("approx_date") or ""
        issuer = publisher_from_url(canon_url)

        return {
            "title": title,
            "url": canon_url,
            "url_hash": chash,
            "summary": summary,
            "fulltext": fulltext,
            "summary_line": f"time: {date_str or '(unknown)'} | issuer: {issuer or '(unknown)'} | abstract: {summary}",
            "score": str(score),
            "score_reason": reason,
            "time": date_str,
            "issuer": issuer,
            "reason": reason,
            "is_new": (not existed_prior),
            "is_article": is_article,
            "mechanisms": mechanisms,
            "impact_direction": impact_direction,
            "impact_channels": impact_channels,
            "regions": regions,
            "has_china_linkage": has_china_linkage,
            "has_amazon_linkage": has_amazon_linkage,
            "time_horizon": time_horizon,
        }

    except Exception as e:
        print(f"[reader][parse-failed] url={canon_url} err={e}")
        return None


async def refine_for_market(
    country_code: str,
    currency: str,
    category_root: str,
    keywords: List[str],
    *,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    kwq_bundle: Optional[Dict[str, Dict[str, List[str]]]] = None,
    rss_only: bool = False,
) -> List[Dict[str, str]]:
    sd = start_date or DATE_RANGE_START
    ed = end_date or DATE_RANGE_END
    is_today_win = (ed is None) or (ed == datetime.date.today())

    topic, _ = build_topic_queries(category_root, keywords, [], country_code=country_code)

    # Load known URL hashes for this market category
    known_url_hashes: set[str] = set()
    try:
        conn = pymysql.connect(
            host=DB_CONFIG["host"],
            port=int(DB_CONFIG["port"]),
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            charset=DB_CONFIG.get("charset", "utf8mb4"),
            autocommit=True,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT url FROM {TABLE_EVIDENCE} WHERE country_code = %s AND category_root = %s;",
                    ((country_code or "").strip(), (category_root or "").strip()),
                )
                for (db_url,) in cur.fetchall():
                    u = canonicalize_url(db_url or "")
                    if u:
                        known_url_hashes.add(url_hash(u))
        finally:
            conn.close()
    except Exception as e:
        print(f"[reader][known-hash-load-failed] {e}")

    valid_this_run: List[Dict[str, str]] = []
    seen_in_current_run: set[str] = set()

    ARTICLE_CAP = 30
    articles_cnt = 0
    reached_cap = False

    def _is_article_by_reason(r: Dict[str, Any]) -> bool:
        v = r.get("is_article")
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "yes", "y", "1"):
                return True
            if s in ("false", "no", "n", "0"):
                return False

        reason = str(r.get("reason", "") or "").lower()
        if reason:
            return ("not an article page" not in reason)

        title = (r.get("title") or "").strip()
        url = (r.get("canonical_url") or r.get("url") or "").strip()
        return len(title) >= 6 and url.startswith("http")


    async def _refine_batch(items: List[Dict[str, Any]], *, phase: str):
        nonlocal valid_this_run, seen_in_current_run, articles_cnt, reached_cap

        # items_sorted = sorted(items, key=date_sort_key, reverse=True)
        items_sorted = sorted(items, key=date_sort_key, reverse=True)
        items_sorted.sort(key=lambda it: (it.get("_tier_rank", 9), it.get("_seq", 10**12)))

        pool = [it for it in items_sorted if in_range(it.get("approx_date"), sd, ed)]
        if not pool:
            print(f"[reader][{phase}] no items in date window {sd}→{ed}; skipping.")
            return

        # -----------------------------
        # Batch controls
        # -----------------------------


        # recommended that "valid cnt" be calculated based on the newly added valid entries in this phase (to avoid mutual interference between CSE and RSS).
        valid_before_phase = len(valid_this_run)

        sem = asyncio.Semaphore(MAX_CONN)

        async def _one(article: Dict[str, Any]) -> Optional[Dict[str, str]]:
            async with sem:
                article.setdefault("industry", category_root)
                article.setdefault("country_code", country_code)
                article.setdefault("currency", currency)
                return await process_one_article(article, keywords)

        print(
            f"[reader][{phase}] start refining candidates={len(pool)} "
            f"(batch_size={BATCH_SIZE}, stop_when_phase_added>={MIN_VALID_TO_STOP})"
        )

        # -----------------------------
        # Process in batches
        # -----------------------------
        total = len(pool)
        batch_idx = 0

        for start in range(0, total, BATCH_SIZE):
            if reached_cap:
                break

            batch_idx += 1
            end = min(total, start + BATCH_SIZE)
            batch_pool = pool[start:end]

            # If phase already has enough valid, stop before launching new work
            phase_added_so_far = len(valid_this_run) - valid_before_phase
            if phase_added_so_far >= MIN_VALID_TO_STOP:
                print(f"[reader][{phase}] stop early before batch {batch_idx}: phase_added={phase_added_so_far}")
                break

            print(f"[reader][{phase}] batch {batch_idx}: size={len(batch_pool)} (idx {start}..{end-1})")

            tasks = [asyncio.create_task(_one(it)) for it in batch_pool]

            with tqdm(total=len(tasks), desc=f"reader:{(category_root or '')[:24]}:{phase}:b{batch_idx}", unit="art") as bar:
                try:
                    for fut in asyncio.as_completed(tasks):
                        if reached_cap:
                            break

                        r = await fut
                        bar.update(1)

                        if r is None or isinstance(r, Exception):
                            continue

                        u = r.get("canonical_url") or r.get("url") or ""
                        chash = r.get("url_hash") or (url_hash(u) if u else "")
                        if not u or not chash:
                            continue

                        if chash in seen_in_current_run:
                            continue

                        approx_date = r.get("time") or ""
                        if not in_range(approx_date, sd, ed):
                            continue

                        seen_in_current_run.add(chash)
                        r["is_new"] = (chash not in known_url_hashes)
                        valid_this_run.append(r)

                        if _is_article_by_reason(r):
                            articles_cnt += 1
                            if articles_cnt >= ARTICLE_CAP:
                                reached_cap = True

                        tag = "added-new" if r["is_new"] else "added-known"
                        print(f"[reader][{phase}][{tag}] {r.get('score','-')} | {u} | {r.get('title','')[:80]}")

                finally:
                    # If we stop early, cancel remaining tasks in this batch
                    if reached_cap:
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                    bar.close()

            phase_added_so_far = len(valid_this_run) - valid_before_phase
            print(
                f"[reader][{phase}] batch {batch_idx} done → "
                f"phase_added={phase_added_so_far} | total_valid={len(valid_this_run)} | "
                f"articles_cnt={articles_cnt} | reached_cap={reached_cap}"
            )

            # only if valid cnt < 10 continue next batch
            if phase_added_so_far >= MIN_VALID_TO_STOP:
                print(f"[reader][{phase}] stop early after batch {batch_idx}: phase_added={phase_added_so_far}")
                break

        print(f"[reader][{phase}] phase done → valid={len(valid_this_run)} | articles_cnt={articles_cnt} | reached_cap={reached_cap}")


    if rss_only:
            # ✅ Force RSS only, regardless of today_win
            if not reached_cap:
                try:
                    raw_rss = harvest_news(
                        topic,
                        kwq_bundle or {},
                        target_recent=max(TARGET_RECENT, 30),
                        max_queries=200,
                        start_date=sd,
                        end_date=ed,
                        use_api="rss",
                        country_code=country_code,
                        market_name=country_name_for(country_code),
                    )
                    await _refine_batch(raw_rss, phase="RSS(FORCED)")
                except Exception as e:
                    print(f"[refine][RSS-forced-phase-error] {e}")
    else:
        # Phase 1: CSE (only for today window)
        if is_today_win and not reached_cap:
            try:
                raw_cse = harvest_news(
                    topic,
                    kwq_bundle or {},
                    target_recent=TARGET_RECENT,
                    max_queries=120,
                    start_date=sd,
                    end_date=ed,
                    use_api="cse",
                    country_code=country_code,
                    market_name=country_name_for(country_code),
                )
                await _refine_batch(raw_cse, phase="CSE")
            except Exception as e:
                print(f"[refine][CSE-phase-error] {e}")

        # Phase 2: RSS (fallback / non-today)
        MIN_VALID = max(6, TARGET_RECENT // 3)
        need_rss = (not is_today_win) or (len(valid_this_run) < MIN_VALID)
        if need_rss and not reached_cap:
            try:
                raw_rss = harvest_news(
                    topic,
                    kwq_bundle or {},
                    target_recent=max(TARGET_RECENT, 30),
                    max_queries=200,
                    start_date=sd,
                    end_date=ed,
                    use_api="rss",
                    country_code=country_code,
                    market_name=country_name_for(country_code),
                )
                await _refine_batch(raw_rss, phase="RSS")
            except Exception as e:
                print(f"[refine][RSS-phase-error] {e}")

    valid_this_run.sort(key=date_sort_key, reverse=True)
    return valid_this_run
