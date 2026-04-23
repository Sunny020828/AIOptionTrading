
import json
import re
from pathlib import Path
from string import Template
from typing import Dict, List, Tuple, Any
import asyncio
from openai import OpenAI
from utils.chat_completion import achat_complete, KEYWORDS_MODEL

# ============================================================
# Constants
# ============================================================

TOPIC_LABEL = "Hang Seng Index"

HSI_CORE_TERMS = [
    "Hang Seng Index",
    "HSI",
    "Hong Kong equities",
    "Hong Kong stocks",
    "Hong Kong market",
    "Hang Seng",
]

HSI_SIGNAL_BUCKETS = {
    "china_policy": [
        "China stimulus",
        "China fiscal stimulus",
        "China monetary easing",
        "PBOC",
        "RRR cut",
        "rate cut",
        "NPC",
        "Ministry of Finance",
        "property support",
        "consumption stimulus",
    ],
    "geopolitics_trade": [
        "US China trade",
        "tariffs",
        "export controls",
        "sanctions",
        "EU China trade",
        "trade tensions",
        "semiconductor restrictions",
        "Taiwan tensions",
        "global trade shock",
        "financial sanctions",
    ],
    "hk_transmission": [
        "HKEX",
        "southbound flows",
        "Stock Connect",
        "HIBOR",
        "HKD",
        "peg",
        "liquidity",
        "funding costs",
        "market turnover",
        "cross border flows",
    ],
    "sector_heavyweights": [
        "tech stocks",
        "property stocks",
        "banks",
        "internet stocks",
        "Alibaba",
        "Tencent",
        "Meituan",
        "AIA",
        "ICBC",
        "China developers",
    ],
    "derivatives_vol": [
        "VHSI",
        "implied volatility",
        "index options",
        "derivatives",
        "skew",
        "hedging demand",
        "risk sentiment",
    ],
}

# Handcrafted backup queries for deterministic fallback
FALLBACK_RSS_QUERIES = [
    "China stimulus Hong Kong stocks",
    "PBOC easing Hong Kong equities",
    "China property support Hang Seng",
    "Fed rates Hong Kong market",
    "US China trade Hong Kong stocks",
    "Tariffs China tech Hong Kong",
    "Stock Connect southbound flows",
    "HIBOR HKD Hong Kong equities",
    "Tencent Alibaba Hang Seng",
    "VHSI Hang Seng options",
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
You generate ultra-high-precision RSS keywords and queries for collecting news/articles that have predictive value for the Hang Seng Index regime and HSI options trading.

Return JSON only:
{
  "keywords": [...],
  "queries": [...]
}

HARD RULES:
- Output exactly 10 keywords and 10 queries.
- Queries must be concise, natural headline-style search phrases for RSS matching.
- Each query must be <= 8 words.
- No parentheses, no quotes, no colon, no boolean operators.
- Prefer causal and leading indicators over market wrap or market close commentary.
- Prioritize China policy, fiscal/monetary easing, trade tensions, tariffs, export controls, Hong Kong liquidity, Stock Connect flows, HKD/HIBOR, and heavyweight sector drivers.
- Keep queries tightly anchored to Hong Kong equities, Hang Seng, or clear China-to-Hong-Kong transmission.
- Avoid generic macro queries with no Hong Kong or China equities linkage.
- Avoid duplicate or near-duplicate wording.
"""

RSS_KEYWORD_USER_TMPL = Template("""
Topic: $topic

Core anchors:
$core_anchors

Relevant signal buckets:
$signal_buckets

Task:
Generate high-precision RSS keywords and queries for collecting news that helps predict HSI market regime and supports HSI options/index trading.

Prioritize:
- China fiscal and monetary policy shifts
- Trade, tariff, sanctions, export-control, and geopolitical shocks affecting China/Hong Kong equities
- Hong Kong liquidity, HKD, HIBOR, Stock Connect, and cross-border flows
- Heavyweight and sector drivers that materially move HSI
- Volatility and options signals only as secondary confirmation

De-prioritize:
- market close summaries
- generic daily market commentary
- purely backward-looking index recap articles

Return JSON only.
""".strip())


# ============================================================
# Helpers
# ============================================================


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


def build_hsi_kwq_bundle() -> Dict[str, Dict[str, Any]]:
    keywords_by_type = {
        "macro": {
            "anchor_en": ["China", "Hong Kong", "PBOC", "Fed"],
            "anchor_zh": ["中国", "香港", "央行", "美联储"],
            "context_en": ["stimulus", "easing", "trade", "tariffs", "property support", "rates"],
            "context_zh": ["刺激政策", "宽松", "贸易", "关税", "地产支持", "利率"],
        },
        "spillover": {
            "anchor_en": ["HKD", "HIBOR", "Stock Connect", "southbound flows"],
            "anchor_zh": ["港元", "HIBOR", "港股通", "南向资金"],
            "context_en": ["Hong Kong liquidity", "capital flows", "spillover"],
            "context_zh": ["香港流动性", "资金流动", "中港传导"],
        },
        "index_direct": {
            "anchor_en": ["Hang Seng", "HSI", "VHSI", "Tencent", "Alibaba"],
            "anchor_zh": ["恒生指数", "恒指", "恒指波动率", "腾讯", "阿里巴巴"],
            "context_en": ["index options", "sector rotation", "Hong Kong equities"],
            "context_zh": ["指数期权", "板块轮动", "港股"],
        },
    }

    queries_by_type = {
        "macro": {
            "en": [
                "China Hong Kong",
                "US Hong Kong",
                "China stimulus Hong Kong stocks",
                "PBOC easing Hong Kong equities",
                "Fed rates Hong Kong market",
                "US China trade Hong Kong stocks",
            ],
            "zh": [
                "中国 香港",
                "美国 香港",
                "中国刺激政策 港股",
                "央行宽松 香港股市",
                "美联储 利率 港股",
                "中美贸易 港股",
            ],
        },
        "spillover": {
            "en": [
                "HKMA",
                "Southbound flows",
                "HKD HIBOR Hong Kong equities",
                "Stock Connect southbound flows",
            ],
            "zh": [
                "金管局",
                "南向资金",
                "港元 HIBOR 港股",
                "南向资金 港股通",
            ],
        },
        "index_direct": {
            "en": [
                "HSI",
                "Hang Seng Index",
                "Hong Kong market",
                "Tencent Alibaba Hang Seng",
                "VHSI Hang Seng options",
                "Hong Kong stocks sector rotation",
            ],
            "zh": [
                "恒指",
                "恒生指数",
                "香港市场",
                "腾讯 阿里巴巴 恒指",
                "恒指 波动率 期权",
                "港股 板块轮动",
            ],
        },
    }

    query_plan: List[Dict[str, Any]] = []

    def add_query(q: str, news_type: str, query_lang: str, query_tier: str):
        if query_lang == "en":
            locales = [
                {"name": "en_hk", "language": "en", "country_code": "HK"},
            ]
        else:
            locales = [
                {"name": "zh_cn", "language": "zh-CN", "country_code": "CN"},
                {"name": "zh_hk", "language": "zh-HK", "country_code": "HK"},
            ]

        query_plan.append({
            "query": q,
            "news_type": news_type,
            "query_lang": query_lang,
            "query_tier": query_tier,
            "locales": locales,
        })

    for news_type, bucket in queries_by_type.items():
        for q in bucket.get("en", []):
            add_query(q, news_type, "en", "mixed")
        for q in bucket.get("zh", []):
            add_query(q, news_type, "zh", "mixed")

    anchor_keywords: List[str] = []
    context_keywords: List[str] = []

    for bucket in keywords_by_type.values():
        anchor_keywords.extend(bucket.get("anchor_en", []))
        anchor_keywords.extend(bucket.get("anchor_zh", []))
        context_keywords.extend(bucket.get("context_en", []))
        context_keywords.extend(bucket.get("context_zh", []))

    flat_queries = [x["query"] for x in query_plan]

    return {
        "rss": {
            "keywords_by_type": keywords_by_type,
            "queries_by_type": queries_by_type,
            "anchor_keywords": list(dict.fromkeys(anchor_keywords)),
            "context_keywords": list(dict.fromkeys(context_keywords)),
            "keywords": list(dict.fromkeys(anchor_keywords + context_keywords)),
            "queries": list(dict.fromkeys(flat_queries)),
            "query_plan": query_plan,
        }
    }
    
    
# ============================================================
# Public API
# ============================================================

async def suggest_keywords_for_hsi(use_static=False) -> Dict[str, Dict[str, List[str]]]:
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
    if use_static:
        return build_hsi_kwq_bundle()
    else:
        user_rss = RSS_KEYWORD_USER_TMPL.substitute(
            topic=TOPIC_LABEL,
            core_anchors=_fmt_lines(HSI_CORE_TERMS),
            signal_buckets=_fmt_signal_buckets(HSI_SIGNAL_BUCKETS),
        )

        raw_rss = await achat_complete(
            KEYWORDS_MODEL,
            RSS_KEYWORD_SYSTEM,
            user_rss
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