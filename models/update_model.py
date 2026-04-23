import uuid
from typing import List, Dict, Any, Optional

import pandas as pd

from utils.chat_completion import achat_complete, READER_MODEL
from utils.context import build_news_context_from_db


UPDATE_MODEL = READER_MODEL


WEEKLY_UPDATE_SYSTEM = """
You are a market news analyst writing a weekly highlight for an HSI options trader.

You will receive:
a bundled set of recent news articles from the last 7 days

Your task is to answer only these questions based on the last 7 days of news:

1. Policy:
Were there any clearly important new policy signals, policy support, tightening signals,
stimulus signals, regulatory shifts, or government actions that may affect liquidity
expectations or market sentiment?

2. Macro / Liquidity:
Were there any clearly important macro or liquidity developments, such as rates, inflation,
central bank signals, fiscal signals, growth surprises, or other macro changes that may
affect market sentiment or liquidity expectations?

3. IPO / Financing / Large Events:
Were there any clearly important IPOs, placements, financing activity, large corporate events,
or other major events that may matter for market attention or sentiment?


Rules:
- Be concise.
- Focus only on clearly important developments.
- Ignore repetitive or low-information headlines.
- If there is no clearly important drivers in a category, say "No clearly important drivers."
- Be conservative and evidence-based.
- Do not give a direct trade recommendation.
- Do not use JSON.

Return plain text only in EXACTLY this format:

Policy:
...
Macro / Liquidity:
...
IPO / Financing / Large Events:
...
"""


def empty_weekly_update_text(
    summary_text: str = "No usable recent news items were available."
) -> str:
    return (
        "Policy:\n"
        "No clearly important update.\n\n"
        "Macro / Liquidity:\n"
        "No clearly important update.\n\n"
        "IPO / Financing / Large Events:\n"
        "No clearly important update.\n\n"
        "Overall Weekly Take:\n"
        f"{summary_text}"
    )


def build_weekly_update_user_prompt(
    *,
    # monthly_memo_text: str,
    bundle_text: str,
    topic: str = "Hang Seng Index",
    country_code: str = "HK",
    anchor_date=None,
    keywords: Optional[List[str]] = None,
) -> str:
    anchor_text = str(pd.to_datetime(anchor_date).date()) if anchor_date is not None else "(none)"
    keyword_text = ", ".join([k for k in (keywords or []) if k][:30]) or "(none)"

    parts = [
        f"Topic: {topic}",
        f"Country code: {country_code}",
        "Market: Hong Kong",
        f"Anchor date: {anchor_text}",
        f"Keywords List: {keyword_text}",
        "",
        # "## Existing Monthly Memo",
        # monthly_memo_text,
        "",
        bundle_text,
        "",
        "Return plain text only in the required format.",
    ]
    return "\n".join(parts)


async def build_weekly_update_with_llm(
    *,
    # monthly_memo_text: str,
    bundle_text: str,
    topic: str = "Hang Seng Index",
    country_code: str = "HK",
    anchor_date=None,
    keywords: Optional[List[str]] = None,
) -> str:
    if not bundle_text or "No relevant news articles were found." in bundle_text:
        return empty_weekly_update_text()

    user_prompt = build_weekly_update_user_prompt(
        # monthly_memo_text=monthly_memo_text,
        bundle_text=bundle_text,
        topic=topic,
        country_code=country_code,
        anchor_date=anchor_date,
        keywords=keywords,
    )

    try:
        raw = await achat_complete(
            UPDATE_MODEL,
            WEEKLY_UPDATE_SYSTEM,
            user_prompt,
        )
        text = (raw or "").strip()
        return text if text else empty_weekly_update_text("LLM returned empty weekly update.")
    except Exception as e:
        print(f"[update-model][llm-error] {e}")
        return empty_weekly_update_text("LLM weekly update generation failed.")


async def news_weekly_update(
    country_code: str,
    *,
    decision_date,
    # monthly_memo_text: str,
    keyword: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    lookback_days: int = 7,
    limit_fetch: int = 20,
    limit_model_items: int = 10,
    fulltext_chars: int = 1000,
    min_existing_content_chars: int = 500,
    max_concurrency: int = 5,
    topic: str = "Hang Seng Index",
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    run_id = run_id or f"weeklyupd_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    decision_date = pd.to_datetime(decision_date).normalize()
    start_date = decision_date - pd.Timedelta(days=lookback_days)
    end_date = decision_date

    context_result = await build_news_context_from_db(
        start_date=start_date,
        end_date=end_date,
        keyword=keyword,
        limit_fetch=limit_fetch,
        limit_model_items=limit_model_items,
        fulltext_chars=fulltext_chars,
        min_existing_content_chars=min_existing_content_chars,
        max_concurrency=max_concurrency,
        label="Weekly News Bundle",
    )


    weekly_update_text = await build_weekly_update_with_llm(
        # monthly_memo_text=monthly_memo_text,
        bundle_text=context_result["bundle_text"],
        topic=topic,
        country_code=country_code,
        anchor_date=decision_date,
        keywords=keywords,
    )

    return {
        "run_id": run_id,
        "decision_date": str(decision_date.date()),
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "weekly_update_text": weekly_update_text,
        "bundle_text": context_result["bundle_text"],
        "used_news": context_result["used_news"],
        "n_items": context_result["n_items"],
    }