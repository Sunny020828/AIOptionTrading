
import json
import re
from pathlib import Path
from string import Template
from typing import Dict, List, Tuple
import asyncio
from openai import OpenAI

KEYWORDS_MODEL='gpt-oss:120b-cloud'
API_KEY = 'sk-ef018c3ada414210851f7578ae15b15b'
MODEL='deepseek-chat'
# ============================================================
# Constants
# ============================================================

TOPIC_LABEL = "Hang Seng Index"

# Core anchors that should frequently appear in RSS-relevant queries
HSI_CORE_TERMS = [
    "Hang Seng Index",
    "HSI",
    "Hong Kong stocks",
    "Hong Kong equity market",
    "Hong Kong market",
    "Hang Seng",
]

# Macro / policy / flow / derivatives related signals
HSI_SIGNAL_BUCKETS = {
    "macro_policy": [
        "China stimulus",
        "China policy",
        "PBOC",
        "Fed",
        "rate cut",
        "interest rates",
        "inflation",
        "GDP",
        "PMI",
        "property sector",
        "regulation",
    ],
    "hk_market": [
        "HKEX",
        "southbound flows",
        "Stock Connect",
        "northbound flows",
        "HIBOR",
        "HKD",
        "peg",
        "liquidity",
        "earnings",
    ],
    "derivatives_vol": [
        "VHSI",
        "volatility",
        "implied volatility",
        "options",
        "index options",
        "derivatives",
        "risk sentiment",
    ],
    "sector_drivers": [
        "tech stocks",
        "property stocks",
        "banks",
        "internet stocks",
        "Alibaba",
        "Tencent",
        "Meituan",
        "AIA",
        "ICBC",
    ],
}

# Handcrafted backup queries for deterministic fallback
FALLBACK_RSS_QUERIES = [
    "Hang Seng Index",
    "HSI Hong Kong stocks",
    "Hang Seng Index outlook",
    "Hang Seng Index market close",
    "Hang Seng Index China stimulus",
    "Hang Seng Index Fed rates",
    "Hang Seng Index PBOC policy",
    "Hang Seng Index HKEX",
    "Hang Seng Index southbound flows",
    "Hang Seng Index Stock Connect",
    "Hang Seng Index HIBOR",
    "Hang Seng Index HKD peg",
    "Hang Seng Index volatility",
    "Hang Seng Index VHSI",
    "Hang Seng Index options",
    "Hang Seng tech stocks",
    "Hang Seng property stocks",
    "Hong Kong stocks China economy",
    "Hong Kong stocks earnings",
    "Hong Kong market risk sentiment",
]

FALLBACK_RSS_KEYWORDS = [
    "Hang Seng Index",
    "HSI",
    "Hang Seng",
    "Hong Kong stocks",
    "Hong Kong equity market",
    "HKEX",
    "southbound flows",
    "Stock Connect",
    "China stimulus",
    "PBOC",
    "Fed",
    "HIBOR",
    "HKD",
    "VHSI",
    "volatility",
    "index options",
    "tech stocks",
    "property stocks",
    "earnings",
    "risk sentiment",
]


# ============================================================
# Prompt Templates (RSS only)
# ============================================================

RSS_KEYWORD_SYSTEM = """
You generate ultra-high-precision RSS keywords and queries for collecting news/articles relevant to the Hang Seng Index and its trading regime.

Return JSON only:
{
  "keywords": [...],
  "queries": [...]
}

HARD RULES:
- Output exactly 20 keywords and exactly 20 queries.
- Queries must be concise, natural headline-style search phrases for RSS matching.
- Each query must be <= 8 words.
- No parentheses, no quotes, no colon, no boolean operators.
- Prefer terms that appear directly in financial news headlines.
- Queries must stay tightly related to Hang Seng Index, Hong Kong equities, Hong Kong macro/liquidity, China macro spillover to Hong Kong, or HSI options/volatility.
- Avoid generic world macro queries with no Hong Kong / HSI anchor.
- Avoid duplicate or near-duplicate wording.
"""


RSS_KEYWORD_USER_TMPL = Template("""
Topic: $topic

Core anchors:
$core_anchors

Relevant signal buckets:
$signal_buckets

Task:
Generate high-precision RSS keywords and queries for collecting news that can help classify HSI market regime and support HSI options/index trading.

Coverage should include:
- HSI index moves and market commentary
- Hong Kong equity market liquidity and flows
- China macro / policy signals that affect HSI
- Hong Kong rates / HKD / HIBOR / Fed spillover
- HSI volatility / options / derivatives
- Major sector and heavyweight drivers in HSI

Return JSON only.
""".strip())


# ============================================================
# Helpers
# ============================================================

def chat_complete(model: str, sysmsg: str, user: str) -> str:
    """
    Synchronous call, using the exact signature you requested.
    """
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    model='deepseek-reasoner'
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": user},
        ],
        stream=False,
        temperature=0.14,
    )
    msg = response.choices[0].message
    content = (msg.content or "").strip()
    return content

async def achat_complete(model: str, sysmsg: str, user: str) -> str:
    """
    Async wrapper: run the synchronous call in a thread to allow concurrency with asyncio.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: chat_complete(model, sysmsg, user))


def _dedup_keep(xs: List[str], n: int | None = None) -> List[str]:
    out = []
    seen = set()
    for x in xs or []:
        s = str(x).strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if n is not None and len(out) >= n:
            break
    return out


def _clean_keyword(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "").replace(":", "")
    return s.strip()


def _clean_query(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', "").replace(":", "")
    s = s.replace("(", "").replace(")", "")
    # remove explicit boolean operators for RSS simplicity
    s = re.sub(r"\b(AND|OR|NOT)\b", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    # keep short
    toks = s.split()
    if len(toks) > 8:
        s = " ".join(toks[:8])
    return s


def _fmt_lines(xs: List[str]) -> str:
    xs = [str(x).strip() for x in (xs or []) if str(x).strip()]
    return "\n".join(f"- {x}" for x in xs) if xs else "- (none)"


def _fmt_signal_buckets(buckets: Dict[str, List[str]]) -> str:
    lines = []
    for k, vals in buckets.items():
        pretty = ", ".join(vals)
        lines.append(f"- {k}: {pretty}")
    return "\n".join(lines)


def _pad_to(xs: List[str], pool: List[str], n: int) -> List[str]:
    xs = _dedup_keep(xs)
    existing = {x.lower() for x in xs}
    for item in pool:
        if len(xs) >= n:
            break
        if item.lower() not in existing:
            xs.append(item)
            existing.add(item.lower())
    return xs[:n]


def _is_hsi_relevant_query(q: str) -> bool:
    """
    Enforce at least one HSI/Hong Kong anchor to avoid drifting into generic macro.
    """
    ql = q.lower()
    anchors = [
        "hang seng",
        "hsi",
        "hong kong",
        "hkex",
        "hibor",
        "hkd",
        "vhsi",
        "stock connect",
        "southbound",
    ]
    return any(a in ql for a in anchors)


# ============================================================
# Public API
# ============================================================

async def suggest_keywords_for_hsi() -> Dict[str, Dict[str, List[str]]]:
    """
    RSS-only keyword/query generator for HSI news collection.

    Returns:
    {
      "rss": {
        "keywords": [...],
        "queries": [...]
      }
    }
    """

    user_rss = RSS_KEYWORD_USER_TMPL.substitute(
        topic=TOPIC_LABEL,
        core_anchors=_fmt_lines(HSI_CORE_TERMS),
        signal_buckets=_fmt_signal_buckets(HSI_SIGNAL_BUCKETS),
    )

    raw_rss = await achat_complete(
        KEYWORDS_MODEL,
        RSS_KEYWORD_SYSTEM,
    )

    js = {}
    try:
        s, e = raw_rss.find("{"), raw_rss.rfind("}")
        if s >= 0 and e > s:
            js = json.loads(raw_rss[s:e + 1])
    except Exception:
        js = {}

    rss_keywords = [_clean_keyword(x) for x in js.get("keywords", [])]
    rss_queries = [_clean_query(x) for x in js.get("queries", [])]

    # filter low-quality / drifted queries
    rss_keywords = [x for x in rss_keywords if x]
    rss_queries = [x for x in rss_queries if x and _is_hsi_relevant_query(x)]

    rss_keywords = _dedup_keep(rss_keywords)
    rss_queries = _dedup_keep(rss_queries)

    # pad with deterministic fallback
    rss_keywords = _pad_to(rss_keywords, FALLBACK_RSS_KEYWORDS, 20)
    rss_queries = _pad_to(rss_queries, FALLBACK_RSS_QUERIES, 20)

    return {
        "rss": {
            "keywords": rss_keywords,
            "queries": rss_queries,
        }
    }


def build_topic_queries(
    keywords: List[str],
    queries: List[str],
) -> Tuple[str, List[str]]:
    """
    Backward-compatible helper for reader / thinking pipeline.

    Returns:
      (topic, query_list)
    """
    topic = TOPIC_LABEL

    if queries:
        cleaned = [_clean_query(x) for x in queries]
        cleaned = [x for x in cleaned if x and _is_hsi_relevant_query(x)]
        cleaned = _dedup_keep(cleaned)
        if cleaned:
            return topic, cleaned

    return topic, FALLBACK_RSS_QUERIES[:]