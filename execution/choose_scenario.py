import json
import re
from typing import Optional, Tuple, Dict, List, Any
# from models.trader_model_v3 import normalize_llm_probs

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


def normalize_llm_probs(obj: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Normalize keys and probabilities.
    """
    direction = obj.get("direction_probs", {}) or {}
    vol = obj.get("volatility_probs", {}) or {}

    dir_map = {
        "bull": "bull",
        "bear": "bear",
        "range": "range",
    }

    vol_map = {
        "low_vol": "low_vol",
        "low": "low_vol",
        "medium_vol": "medium_vol",
        "medium": "medium_vol",
        "median_vol": "medium_vol",
        "mid_vol": "medium_vol",
        "high_vol": "high_vol",
        "high": "high_vol",
    }

    d = {dir_map[k]: float(v) for k, v in direction.items() if k in dir_map}
    v = {vol_map[k]: float(v) for k, v in vol.items() if k in vol_map}

    for key in ["bull", "bear", "range"]:
        d.setdefault(key, 0.0)
    for key in ["low_vol", "medium_vol", "high_vol"]:
        v.setdefault(key, 0.0)

    d_sum = sum(d.values())
    v_sum = sum(v.values())

    if d_sum > 0:
        d = {k: val / d_sum for k, val in d.items()}
    if v_sum > 0:
        v = {k: val / v_sum for k, val in v.items()}

    return {
        "direction_probs": d,
        "volatility_probs": v
    }
    
    
def choose_scenario(final_text: str, method: str = "conviction") -> Dict[str, Any]:
    """
    Convert LLM probabilities into three regime labels:

    Returns
    -------
    {
        "direction": "bull" | "bear" | "range",
        "volatility": "lowvol" | "medianvol" | "highvol",
        "strength": "weak" | "medium" | "strong",
        "direction_probs": {...},
        "volatility_probs": {...},
    }
    """
    obj = extract_json_from_text(final_text)
    prob_obj = normalize_llm_probs(obj)

    d = prob_obj["direction_probs"]
    v = prob_obj["volatility_probs"]

    bear = float(d.get("bear", 0.0))
    bull = float(d.get("bull", 0.0))
    rng  = float(d.get("range", 0.0))

    lowv  = float(v.get("low_vol", 0.0))
    medv  = float(v.get("medium_vol", 0.0))
    highv = float(v.get("high_vol", 0.0))

    vol_name_map = {
        "low_vol": "lowvol",
        "medium_vol": "medianvol",
        "high_vol": "highvol",
    }

    # -----------------------------
    # 1) Direction
    # -----------------------------
    if method == "argmax_pair":
        direction = max(d.items(), key=lambda x: x[1])[0]

    elif method == "joint":
        direction = max(d.items(), key=lambda x: x[1])[0]

    else:
        trend_score = bull - bear
        trend_abs = abs(trend_score)

        dir_sorted = sorted(
            [("bull", bull), ("bear", bear), ("range", rng)],
            key=lambda x: x[1],
            reverse=True
        )
        top_dir, top_prob = dir_sorted[0]
        second_dir, second_prob = dir_sorted[1]
        dir_gap = top_prob - second_prob

        bull_vs_range_gap = bull - rng
        bear_vs_range_gap = bear - rng

        trend_conviction = trend_abs * (1.0 - rng)

        if rng >= 0.45:
            direction = "range"
        elif dir_gap < 0.08:
            direction = "range"
        elif top_dir == "bull":
            if bull < 0.50:
                direction = "range"
            elif bull_vs_range_gap < 0.10:
                direction = "range"
            elif trend_conviction < 0.20:
                direction = "range"
            else:
                direction = "bull"
        elif top_dir == "bear":
            if bear < 0.50:
                direction = "range"
            elif bear_vs_range_gap < 0.10:
                direction = "range"
            elif trend_conviction < 0.20:
                direction = "range"
            else:
                direction = "bear"
        else:
            direction = "range"

    # -----------------------------
    # 2) Volatility
    # -----------------------------
    if method == "argmax_pair":
        best_vol_name = max(v.items(), key=lambda x: x[1])[0]
        volatility = vol_name_map[best_vol_name]
    else:
        vol_sorted = sorted(
            [("low_vol", lowv), ("medium_vol", medv), ("high_vol", highv)],
            key=lambda x: x[1],
            reverse=True
        )
        best_vol_name, best_vol_prob = vol_sorted[0]
        second_vol_name, second_vol_prob = vol_sorted[1]
        vol_gap = best_vol_prob - second_vol_prob

        if vol_gap < 0.08:
            volatility = "medianvol"
        else:
            volatility = vol_name_map[best_vol_name]

        if direction in {"bull", "bear"}:
            directional_gap = max(bull - rng, bear - rng)
            if directional_gap < 0.10 and volatility == "highvol":
                volatility = "medianvol"

    # -----------------------------
    # 3) Strength
    # -----------------------------
    if direction == "bull":
        strength_gap = bull - rng
    elif direction == "bear":
        strength_gap = bear - rng
    else:
        strength_gap = rng - max(bull, bear)

    if strength_gap < 0.08:
        strength = "weak"
    elif strength_gap < 0.20:
        strength = "medium"
    else:
        strength = "strong"

    return {
        "direction": direction,
        "volatility": volatility,
        "strength": strength,
        "direction_probs": d,
        "volatility_probs": v,
    }
    
    
def choose_scenario(final_text: str, method: str = "conviction") -> Dict[str, Any]:
    """
    Convert LLM probabilities into regime labels.

    Returns
    -------
    {
        "direction": "bull" | "bear" | "range",
        "volatility": "lowvol" | "medianvol" | "highvol",
        "strength": "weak" | "medium" | "strong",
        "direction_probs": {...},
        "volatility_probs": {...},
        "action_guidance": {...},
        "warning_flag": bool,
        "warning_type": str,
        "warning_message": str,
        "base_quant": {...},
        "news_review": {...},
    }
    """
    obj = extract_json_from_text(final_text)
    prob_obj = normalize_llm_probs(obj)

    d = prob_obj["direction_probs"]
    v = prob_obj["volatility_probs"]

    base_quant = prob_obj.get("base_quant", {}) or {}
    news_review = prob_obj.get("news_review", {}) or {}
    action_guidance = prob_obj.get("action_guidance", {}) or {}

    bear = float(d.get("bear", 0.0))
    bull = float(d.get("bull", 0.0))
    rng = float(d.get("range", 0.0))

    lowv = float(v.get("low_vol", 0.0))
    medv = float(v.get("medium_vol", 0.0))
    highv = float(v.get("high_vol", 0.0))

    vol_name_map = {
        "low_vol": "lowvol",
        "medium_vol": "medianvol",
        "high_vol": "highvol",
    }

    # -----------------------------
    # 1) Direction
    # -----------------------------
    if method == "argmax_pair":
        direction = max(d.items(), key=lambda x: x[1])[0]

    elif method == "joint":
        direction = max(d.items(), key=lambda x: x[1])[0]

    else:
        dir_sorted = sorted(
            [("bull", bull), ("bear", bear), ("range", rng)],
            key=lambda x: x[1],
            reverse=True
        )
        top_dir, top_prob = dir_sorted[0]
        second_dir, second_prob = dir_sorted[1]
        dir_gap = top_prob - second_prob

        bull_vs_range_gap = bull - rng
        bear_vs_range_gap = bear - rng

        # clear range regime
        if rng >= 0.45:
            direction = "range"

        # too close to call
        elif dir_gap < 0.1:
            direction = "range"

        # bullish tilt
        elif top_dir == "bull":
            if bull_vs_range_gap >= 0.1:
                direction = "bull"
            else:
                direction = "range"

        # bearish tilt
        elif top_dir == "bear":
            if bear_vs_range_gap >= 0.1:
                direction = "bear"
            else:
                direction = "range"

        else:
            direction = "range"

    # -----------------------------
    # 2) Volatility
    # -----------------------------
    if method == "argmax_pair":
        best_vol_name = max(v.items(), key=lambda x: x[1])[0]
        volatility = vol_name_map[best_vol_name]
    else:
        vol_sorted = sorted(
            [("low_vol", lowv), ("medium_vol", medv), ("high_vol", highv)],
            key=lambda x: x[1],
            reverse=True
        )
        best_vol_name, best_vol_prob = vol_sorted[0]
        _, second_vol_prob = vol_sorted[1]
        vol_gap = best_vol_prob - second_vol_prob

        if vol_gap < 0.08:
            volatility = "medianvol"
        else:
            volatility = vol_name_map[best_vol_name]

        if direction in {"bull", "bear"}:
            directional_gap = max(bull - rng, bear - rng)
            if directional_gap < 0.10 and volatility == "highvol":
                volatility = "medianvol"

    # -----------------------------
    # 3) Strength
    # -----------------------------
    model_strength = str(prob_obj.get("strength", "medium")).strip().lower()

    if direction == "bull":
        strength_gap = bull - rng
    elif direction == "bear":
        strength_gap = bear - rng
    else:
        strength_gap = rng - max(bull, bear)

    if strength_gap < 0.08:
        inferred_strength = "weak"
    elif strength_gap < 0.20:
        inferred_strength = "medium"
    else:
        inferred_strength = "strong"

    if model_strength in {"weak", "medium", "strong"}:
        strength = model_strength
    else:
        strength = inferred_strength

    # -----------------------------
    # 4) Warning / execution info
    # -----------------------------
    warning_flag = bool(news_review.get("warning_flag", False))
    warning_type = str(news_review.get("warning_type", "none"))
    warning_message = str(news_review.get("warning_message", ""))

    return {
        "direction": direction,
        "volatility": volatility,
        "strength": strength,
        "direction_probs": d,
        "volatility_probs": v,
        "action_guidance": action_guidance,
        "warning_flag": warning_flag,
        "warning_type": warning_type,
        "warning_message": warning_message,
        "base_quant": base_quant,
        "news_review": news_review,
    }