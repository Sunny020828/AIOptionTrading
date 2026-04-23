
import json
import re
import pandas as pd
import numpy as np
from openai import OpenAI
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from features.data import LOOKBACK_LEN, PRED_LEN
from utils.utils import API_KEY

def _safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _fmt_num(x, digits=4):
    return "N/A" if pd.isna(x) else f"{float(x):.{digits}f}"


def _fmt_pct(x, digits=2):
    return "N/A" if pd.isna(x) else f"{100 * float(x):.{digits}f}%"


def _fmt_close(x, digits=2):
    return "N/A" if pd.isna(x) else f"{float(x):.{digits}f}"


def _fmt_text(x):
    if x is None:
        return "N/A"
    s = str(x).strip()
    return s if s else "N/A"

def render_monthly_memo_text(monthly_memo):
    if monthly_memo is None:
        return "N/A"

    if isinstance(monthly_memo, str):
        return monthly_memo.strip() or "N/A"

    if not isinstance(monthly_memo, dict):
        return str(monthly_memo)

    themes = monthly_memo.get("main_themes") or []
    if not isinstance(themes, list):
        themes = []
    themes_text = "; ".join([str(x).strip() for x in themes if str(x).strip()]) or "None"

    lines = [
        f"Market mood: {_fmt_text(monthly_memo.get('market_mood'))}",
        f"Main themes: {themes_text}",
        f"Market bias: {_fmt_text(monthly_memo.get('market_bias'))}",
        f"Risk bias: {_fmt_text(monthly_memo.get('risk_bias'))}",
        f"Signal strength: {_fmt_text(monthly_memo.get('signal_strength'))}",
        f"Insufficient news signal: {_fmt_text(monthly_memo.get('insufficient_news_signal'))}",
        f"Summary: {_fmt_text(monthly_memo.get('summary_text'))}",
    ]
    return "\n".join(lines)


def render_weekly_update_text(weekly_update):
    if weekly_update is None:
        return "N/A"
    if isinstance(weekly_update, str):
        return weekly_update.strip() or "N/A"
    return str(weekly_update)

def summarize_hsi_state(row: pd.Series) -> Dict[str, str]:
    ret_3d = _safe_float(row.get("hsi_ret_3d"))
    ret_7d = _safe_float(row.get("hsi_ret_7d"))
    ret_20d = _safe_float(row.get("hsi_ret_20d"))
    ret_60d = _safe_float(row.get("hsi_ret_60d"))

    bias_20 = _safe_float(row.get("hsi_bias_20"))
    bias_50 = _safe_float(row.get("hsi_bias_50"))

    slope_7d = _safe_float(row.get("hsi_trend_slope_7d"))
    slope_20d = _safe_float(row.get("hsi_trend_slope_20d"))
    r2_7d = _safe_float(row.get("hsi_trend_r2_7d"))
    r2_20d = _safe_float(row.get("hsi_trend_r2_20d"))

    accel_3d = _safe_float(row.get("hsi_ret_accel_3d_vs_1m"))
    accel_7d = _safe_float(row.get("hsi_ret_accel_7d_vs_1m"))

    pct_20 = _safe_float(row.get("hsi_close_pctile_20d"))
    pct_60 = _safe_float(row.get("hsi_close_pctile_60d"))

    spread = _safe_float(row.get("hsi_ma20_ma50_spread_pct"))
    spread_delta = _safe_float(row.get("hsi_ma20_ma50_spread_delta_5d"))

    # 1) direction
    direction_score = 0
    for v in [ret_7d, ret_20d, bias_20, bias_50, spread]:
        if pd.isna(v):
            continue
        direction_score += 1 if v > 0 else -1

    if direction_score >= 3:
        direction_state = "bullish"
    elif direction_score <= -3:
        direction_state = "bearish"
    else:
        direction_state = "range_or_mixed"

    # 2) trend quality
    valid_r2 = [x for x in [r2_7d, r2_20d] if not pd.isna(x)]
    avg_r2 = np.mean(valid_r2) if valid_r2 else np.nan

    if pd.isna(avg_r2):
        trend_quality = "unclear"
    elif avg_r2 >= 0.70:
        trend_quality = "strong_persistent_trend"
    elif avg_r2 >= 0.45:
        trend_quality = "moderate_trend"
    else:
        trend_quality = "noisy_or_choppy"

    # 3) extension
    if not pd.isna(pct_20) and not pd.isna(pct_60):
        if pct_20 >= 0.85 and pct_60 >= 0.85:
            extension_state = "extended_upside"
        elif pct_20 <= 0.15 and pct_60 <= 0.15:
            extension_state = "extended_downside"
        elif pct_20 >= 0.70:
            extension_state = "near_upper_range"
        elif pct_20 <= 0.30:
            extension_state = "near_lower_range"
        else:
            extension_state = "mid_range"
    else:
        extension_state = "unclear"

    # 4) transition / exhaustion
    transition_flags = []
    if not pd.isna(accel_3d):
        transition_flags.append("short_term_accel_up" if accel_3d > 0 else "short_term_accel_down")
    if not pd.isna(accel_7d):
        transition_flags.append("weekly_accel_up" if accel_7d > 0 else "weekly_accel_down")
    if not pd.isna(spread_delta):
        transition_flags.append("trend_spread_widening" if spread_delta > 0 else "trend_spread_narrowing")

    turning_risk = ", ".join(transition_flags) if transition_flags else "no_clear_transition_signal"

    return {
        "direction_state": direction_state,
        "trend_quality": trend_quality,
        "extension_state": extension_state,
        "turning_risk": turning_risk,
    }
    
def summarize_vhsi_state(row: pd.Series) -> Dict[str, str]:
    ret_3d = _safe_float(row.get("vhsi_ret_3d"))
    ret_7d = _safe_float(row.get("vhsi_ret_7d"))
    ret_20d = _safe_float(row.get("vhsi_ret_20d"))

    bias_20 = _safe_float(row.get("vhsi_bias_20"))
    pct_20 = _safe_float(row.get("vhsi_close_pctile_20d"))
    pct_60 = _safe_float(row.get("vhsi_close_pctile_60d"))
    accel_3d = _safe_float(row.get("vhsi_ret_accel_3d_vs_1m"))
    accel_7d = _safe_float(row.get("vhsi_ret_accel_7d_vs_1m"))
    ma20_slope_5d = _safe_float(row.get("vhsi_ma20_slope_5d"))

    if not pd.isna(pct_60):
        if pct_60 >= 0.85:
            vol_level_state = "high_vol"
        elif pct_60 >= 0.60:
            vol_level_state = "elevated_vol"
        elif pct_60 <= 0.20:
            vol_level_state = "low_vol"
        else:
            vol_level_state = "medium_vol"
    else:
        vol_level_state = "unclear"

    direction_score = 0
    for v in [ret_7d, accel_3d, accel_7d, ma20_slope_5d]:
        if pd.isna(v):
            continue
        direction_score += 1 if v > 0 else -1

    if direction_score >= 2:
        vol_direction_state = "vol_expanding"
    elif direction_score <= -2:
        vol_direction_state = "vol_fading"
    else:
        vol_direction_state = "vol_stable_or_mixed"

    if vol_level_state in {"high_vol", "elevated_vol"} and vol_direction_state == "vol_expanding":
        vol_regime_interpretation = "stress_building"
    elif vol_level_state in {"high_vol", "elevated_vol"} and vol_direction_state == "vol_fading":
        vol_regime_interpretation = "elevated_but_cooling"
    elif vol_level_state == "low_vol" and vol_direction_state == "vol_expanding":
        vol_regime_interpretation = "low_vol_but_risk_repricing"
    elif vol_level_state == "low_vol":
        vol_regime_interpretation = "complacent_or_orderly"
    else:
        vol_regime_interpretation = "mixed_vol_signal"

    return {
        "vol_level_state": vol_level_state,
        "vol_direction_state": vol_direction_state,
        "vol_regime_interpretation": vol_regime_interpretation,
    }
    
def summarize_iv_state(row: pd.Series) -> Dict[str, str]:
    level_near = _safe_float(row.get("level_near"))
    slope_near = _safe_float(row.get("slope_near"))
    term_level = _safe_float(row.get("term_level_next_near"))

    level_chg_1d = _safe_float(row.get("level_near_chg_1d"))
    level_chg_3d = _safe_float(row.get("level_near_chg_3d"))
    slope_chg_3d = _safe_float(row.get("slope_near_chg_3d"))
    term_level_chg_3d = _safe_float(row.get("term_level_next_near_chg_3d"))

    if pd.isna(level_chg_3d):
        iv_level_state = "unclear"
    elif level_chg_3d > 0.02:
        iv_level_state = "near_iv_rising_fast"
    elif level_chg_3d < -0.02:
        iv_level_state = "near_iv_falling"
    else:
        iv_level_state = "near_iv_stable"

    if pd.isna(slope_near):
        skew_state = "unclear"
    elif slope_near < -0.02:
        skew_state = "downside_skew_elevated"
    elif slope_near < 0:
        skew_state = "mild_negative_skew"
    else:
        skew_state = "flat_or_positive_skew"

    if pd.isna(term_level):
        term_structure_state = "unclear"
    elif term_level < 0:
        term_structure_state = "front_end_richer_than_next"
    elif term_level > 0:
        term_structure_state = "normal_upward_term_structure"
    else:
        term_structure_state = "flat_term_structure"

    changes = []
    if not pd.isna(level_chg_1d):
        changes.append("iv_up_1d" if level_chg_1d > 0 else "iv_down_1d")
    if not pd.isna(slope_chg_3d):
        changes.append("skew_steepening" if slope_chg_3d < 0 else "skew_softening")
    if not pd.isna(term_level_chg_3d):
        changes.append("front_risk_rising" if term_level_chg_3d < 0 else "front_risk_easing")

    iv_recent_change = ", ".join(changes) if changes else "no_clear_recent_iv_shift"

    return {
        "iv_level_state": iv_level_state,
        "skew_state": skew_state,
        "term_structure_state": term_structure_state,
        "iv_recent_change": iv_recent_change,
    }
    
def build_market_state_summary(row: pd.Series) -> Dict[str, Any]:
    hsi_state = summarize_hsi_state(row)
    vhsi_state = summarize_vhsi_state(row)
    iv_state = summarize_iv_state(row)

    # cross-asset interpretation
    direction_state = hsi_state["direction_state"]
    vol_direction_state = vhsi_state["vol_direction_state"]
    iv_level_state = iv_state["iv_level_state"]

    if direction_state == "bullish" and vol_direction_state == "vol_fading":
        direction_vs_vol_alignment = "bullish_spot_with_easing_risk"
    elif direction_state == "bearish" and vol_direction_state == "vol_expanding":
        direction_vs_vol_alignment = "bearish_spot_with_rising_risk"
    elif vol_direction_state == "vol_expanding":
        direction_vs_vol_alignment = "spot_and_vol_tension_rising"
    else:
        direction_vs_vol_alignment = "mixed"

    if vol_direction_state == "vol_expanding" and iv_level_state == "near_iv_rising_fast":
        spot_vs_iv_alignment = "realized_and_implied_risk_both_rising"
    elif vol_direction_state == "vol_fading" and iv_level_state == "near_iv_falling":
        spot_vs_iv_alignment = "risk_premium_easing_broadly"
    else:
        spot_vs_iv_alignment = "mixed"

    overall_market_state = (
        f"{hsi_state['direction_state']}; "
        f"{hsi_state['trend_quality']}; "
        f"{hsi_state['extension_state']}; "
        f"{vhsi_state['vol_regime_interpretation']}; "
        f"{iv_state['term_structure_state']}"
    )

    return {
        "hsi_state": hsi_state,
        "vhsi_state": vhsi_state,
        "iv_state": iv_state,
        "cross_asset_read": {
            "direction_vs_vol_alignment": direction_vs_vol_alignment,
            "spot_vs_iv_alignment": spot_vs_iv_alignment,
            "overall_market_state": overall_market_state,
        },
    }
    
    
def build_regime_prompt(
    features_df: pd.DataFrame,
    as_of_date,
    analyst_report: Optional[Dict[str, Any]] = None,
    lookback: int = LOOKBACK_LEN,
    pred_len: int = PRED_LEN,
):
    as_of_date = pd.to_datetime(as_of_date)
    df = features_df.copy()

    if lookback is not None and "lookback_months" in df.columns:
        df = df[df["lookback_months"] == lookback]
    if pred_len is not None and "pred_len_months" in df.columns:
        df = df[df["pred_len_months"] == pred_len]

    decision_col = "decision_date" if "decision_date" in df.columns else "Date"
    df[decision_col] = pd.to_datetime(df[decision_col])
    df = df[df[decision_col] <= as_of_date].sort_values(decision_col)

    if df.empty:
        raise ValueError(f"No feature row found on or before {as_of_date}")

    row = df.iloc[-1]
    market_state = build_market_state_summary(row)

    digest_text = "N/A"
    if analyst_report:
        digest_text = _fmt_text(analyst_report.get("digest_text"))

    system_msg = """
You are a quantitative index-and-options trader.

Task:
Given:
1. a compressed quantitative market-state summary
2. supporting market metrics
3. weekly news digestion reports derived from the past one month of news

Estimate the probability distribution for:
1) next-period market direction: bull / bear / range
2) next-period volatility regime: low_vol / medium_vol / high_vol

How to read the inputs:
- Read the quantitative market-state summary first.
- Use supporting metrics as the main source of truth about current market conditions.
- Read the weekly news digestion report only as contextual information that may help explain the metrics.
- Use the digestion report to assess whether recent moves or shocks are:
  - still building,
  - still being digested,
  - already fading,
  - or largely unrelated to the observed metrics.

Decision principles:
- Quantitative metrics are primary. News is secondary.
- If the digestion report clearly supports and explains the quantitative state, allow it to strengthen the corresponding direction or volatility view.
- If the digestion report appears weak, vague, stale, speculative, or only loosely related to the quantitative state, keep the decision anchored on metrics.
- If the digestion report and metrics disagree, prefer metrics unless the digestion report provides a very clear and direct explanation for a likely transition that is already partially visible in the metrics.
- Do not over-infer from policy headlines, macro commentary, or narrative language alone.
- Do not assume every policy or event creates a lasting regime shift.
- Distinguish carefully between:
  - trend continuation,
  - temporary shock,
  - fading impulse,
  - and consolidation after a move.
- Treat volatility separately from direction.
- Missing values should be ignored, not treated as neutral evidence.

Output rules:
- Output valid JSON only.
- Do not explain your reasoning.
- Use this exact schema:

{
  "direction_probs": {
    "bull": ...,
    "bear": ...,
    "range": ...
  },
  "volatility_probs": {
    "low_vol": ...,
    "medium_vol": ...,
    "high_vol": ...
  }
}

Constraints:
- Each probability must be between 0 and 1.
- direction_probs must sum exactly to 1.0.
- volatility_probs must sum exactly to 1.0.
"""

    facts = [
        f"decision_date: {pd.to_datetime(row[decision_col]).date()}",
        f"forecast_horizon_months: {pred_len}",
        f"lookback_window_months: {lookback}",
        "",
        "[MARKET_STATE_SUMMARY]",
        f"hsi_direction_state: {market_state['hsi_state']['direction_state']}",
        f"hsi_trend_quality: {market_state['hsi_state']['trend_quality']}",
        f"hsi_extension_state: {market_state['hsi_state']['extension_state']}",
        f"hsi_turning_risk: {market_state['hsi_state']['turning_risk']}",
        f"vhsi_vol_level_state: {market_state['vhsi_state']['vol_level_state']}",
        f"vhsi_vol_direction_state: {market_state['vhsi_state']['vol_direction_state']}",
        f"vhsi_vol_regime_interpretation: {market_state['vhsi_state']['vol_regime_interpretation']}",
        f"iv_level_state: {market_state['iv_state']['iv_level_state']}",
        f"iv_skew_state: {market_state['iv_state']['skew_state']}",
        f"iv_term_structure_state: {market_state['iv_state']['term_structure_state']}",
        f"iv_recent_change: {market_state['iv_state']['iv_recent_change']}",
        f"direction_vs_vol_alignment: {market_state['cross_asset_read']['direction_vs_vol_alignment']}",
        f"spot_vs_iv_alignment: {market_state['cross_asset_read']['spot_vs_iv_alignment']}",
        f"overall_market_state: {market_state['cross_asset_read']['overall_market_state']}",
        "",
        "[HSI_SUPPORTING_METRICS]",
        f"hsi_close: {_fmt_close(row.get('Close'))}",
        f"hsi_ret_3d: {_fmt_pct(row.get('hsi_ret_3d'))}",
        f"hsi_ret_7d: {_fmt_pct(row.get('hsi_ret_7d'))}",
        f"hsi_ret_20d: {_fmt_pct(row.get('hsi_ret_20d'))}",
        f"hsi_ret_60d: {_fmt_pct(row.get('hsi_ret_60d'))}",
        f"hsi_vol_7d_ann: {_fmt_pct(row.get('hsi_vol_7d_ann'))}",
        f"hsi_vol_20d_ann: {_fmt_pct(row.get('hsi_vol_20d_ann'))}",
        f"hsi_bias_20: {_fmt_pct(row.get('hsi_bias_20'))}",
        f"hsi_bias_50: {_fmt_pct(row.get('hsi_bias_50'))}",
        f"hsi_trend_slope_7d: {_fmt_num(row.get('hsi_trend_slope_7d'), 5)}",
        f"hsi_trend_r2_7d: {_fmt_num(row.get('hsi_trend_r2_7d'), 4)}",
        f"hsi_trend_slope_20d: {_fmt_num(row.get('hsi_trend_slope_20d'), 5)}",
        f"hsi_trend_r2_20d: {_fmt_num(row.get('hsi_trend_r2_20d'), 4)}",
        f"hsi_close_pctile_20d: {_fmt_num(row.get('hsi_close_pctile_20d'), 4)}",
        f"hsi_close_pctile_60d: {_fmt_num(row.get('hsi_close_pctile_60d'), 4)}",
        f"hsi_ret_accel_3d_vs_1m: {_fmt_num(row.get('hsi_ret_accel_3d_vs_1m'), 5)}",
        f"hsi_ret_accel_7d_vs_1m: {_fmt_num(row.get('hsi_ret_accel_7d_vs_1m'), 5)}",
        "",
        "[VHSI_SUPPORTING_METRICS]",
        f"vhsi_close: {_fmt_close(row.get('Close_vhsi'))}",
        f"vhsi_ret_3d: {_fmt_pct(row.get('vhsi_ret_3d'))}",
        f"vhsi_ret_7d: {_fmt_pct(row.get('vhsi_ret_7d'))}",
        f"vhsi_ret_20d: {_fmt_pct(row.get('vhsi_ret_20d'))}",
        f"vhsi_bias_20: {_fmt_pct(row.get('vhsi_bias_20'))}",
        f"vhsi_close_pctile_20d: {_fmt_num(row.get('vhsi_close_pctile_20d'), 4)}",
        f"vhsi_close_pctile_60d: {_fmt_num(row.get('vhsi_close_pctile_60d'), 4)}",
        f"vhsi_ret_accel_3d_vs_1m: {_fmt_num(row.get('vhsi_ret_accel_3d_vs_1m'), 5)}",
        f"vhsi_ret_accel_7d_vs_1m: {_fmt_num(row.get('vhsi_ret_accel_7d_vs_1m'), 5)}",
        f"vhsi_ma20_slope_5d: {_fmt_num(row.get('vhsi_ma20_slope_5d'), 5)}",
        "",
        "[IV_SUPPORTING_METRICS]",
        f"level_near: {_fmt_num(row.get('level_near'), 4)}",
        f"slope_near: {_fmt_num(row.get('slope_near'), 4)}",
        f"dte_near: {_fmt_num(row.get('dte_near'), 2)}",
        f"level_next: {_fmt_num(row.get('level_next'), 4)}",
        f"slope_next: {_fmt_num(row.get('slope_next'), 4)}",
        f"dte_next: {_fmt_num(row.get('dte_next'), 2)}",
        f"term_level_next_near: {_fmt_num(row.get('term_level_next_near'), 4)}",
        f"term_slope_next_near: {_fmt_num(row.get('term_slope_next_near'), 4)}",
        f"level_near_chg_1d: {_fmt_num(row.get('level_near_chg_1d'), 4)}",
        f"level_near_chg_3d: {_fmt_num(row.get('level_near_chg_3d'), 4)}",
        f"slope_near_chg_3d: {_fmt_num(row.get('slope_near_chg_3d'), 4)}",
        f"term_level_next_near_chg_3d: {_fmt_num(row.get('term_level_next_near_chg_3d'), 4)}",
        "",
        "[MONTHLY_DIGEST_REPORT]",
        digest_text,
        "",
        "Return JSON only.",
    ]

    user_msg = "\n".join(facts)
    return system_msg, user_msg


def call_llm(features_df, context, as_of_date="2024-10-01"):
    system_msg, user_msg = build_regime_prompt(
        features_df,
        as_of_date=as_of_date,
        analyst_report=context,
    )
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        stream=False,
        temperature=0.14,
    )

    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    final_text = msg.content
    return reasoning, final_text



def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction from LLM text.
    """
    text = (text or "").strip()

    # Try full text first
    try:
        return json.loads(text)
    except Exception:
        pass

    # fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # first {...} block
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    raise ValueError("Failed to parse JSON from LLM output.")

