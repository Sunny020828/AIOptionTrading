
import pandas as pd
from openai import OpenAI
from pathlib import Path

API_KEY='sk-ef018c3ada414210851f7578ae15b15b'

def build_regime_prompt(
    features_df: pd.DataFrame,
    as_of_date,
    lookback: int = 3,
    pred_len: int = 1,
):
    """Build (system_msg, user_msg) for direction/volatility regime inference using HSI, VHSI, and IV surface shape features."""
    as_of_date = pd.to_datetime(as_of_date)

    df = features_df.copy()
    if lookback is not None:
        df = df[df["lookback_months"] == lookback]
    if pred_len is not None:
        df = df[df["pred_len_months"] == pred_len]

    df = df[df["decision_date"] <= as_of_date].sort_values("decision_date")
    if df.empty:
        raise ValueError(f"No feature row found on or before {as_of_date}")

    row = df.iloc[-1]

    lookback = int(row["lookback_months"])
    pred_len = int(row["pred_len_months"])

    system_msg = f"""You are a quantitative trader specializing in equity index regime analysis.

Task:
Based ONLY on the aggregated historical features provided, estimate:

1. The probability distribution of the market direction regime for the NEXT {pred_len} month(s):
   - bull
   - bear
   - range

2. The probability distribution of the market volatility regime for the NEXT {pred_len} month(s):
   - low_vol
   - medium_vol
   - high_vol

Market context:
- Underlying market: Hang Seng Index (HSI)
- Volatility reference: VHSI
- Additional option information is provided through fitted implied-volatility surface shape features.
- The input features summarize recent HSI price behavior, trend condition, realized volatility,
  recent VHSI behavior, and recent option-implied volatility surface structure.
- Use only the provided information.
- Do not use future data.
- Do not introduce any external macro, news, policy, or market information.
- Treat the task as a pure feature-to-probability mapping problem.

Definitions:

Direction regimes:
- bull: upward directional bias dominates.
- bear: downward directional bias dominates.
- range: no strong persistent directional trend dominates.

Volatility regimes:
- low_vol: volatility is relatively compressed and stable.
- medium_vol: volatility is moderate or normal.
- high_vol: volatility is elevated, unstable, or stressed.

Interpretation of option surface shape features:
- level: near-ATM implied volatility level. Higher level usually indicates that the options market is pricing more uncertainty or risk.
- slope: first-order tilt of the implied-volatility smile around ATM. More negative slope typically suggests stronger downside skew or stronger pricing of downside risk.
- curvature: degree of bend in the implied-volatility smile around ATM. Higher curvature suggests a more strongly curved smile and stronger tail/convexity pricing.
- term_* features: differences between next-expiry and near-expiry shape parameters. They describe the term structure of IV level, skew, and curvature.
- *_chg_20d features: recent 20-trading-day changes in those option-surface shape measures.
- Treat option-surface features as supporting signals, not deterministic rules.

Decision guidance:
- Positive lookback return, positive average daily return, and high fractions above MA20/50/100
  suggest bull.
- Negative lookback return, negative average daily return, and low fractions above MA20/50/100
  suggest bear.
- If directional evidence is weak, mixed, or close to neutral, assign more probability to range.
- Higher realized volatility, higher VHSI level, higher VHSI variability, and a larger fraction
  of high-volatility days suggest higher probability of high_vol.
- Higher IV level and rising IV level can support higher-volatility regimes.
- More negative IV slope can support downside-risk or bearish interpretations, especially when consistent with weak index trend features.
- Higher IV curvature can support elevated tail-risk pricing and can be consistent with higher-volatility regimes.
- Moderate volatility evidence can support medium_vol.
- If evidence is mixed or weak, use a more diffuse probability distribution.
- Avoid extremely concentrated probabilities unless the signals are very strong.


Output requirements:
- Output JSON only.
- Use exactly this structure:
  {{
    "direction_probs": {{
      "bull": ...,
      "bear": ...,
      "range": ...
    }},
    "volatility_probs": {{
      "low_vol": ...,
      "medium_vol": ...,
      "high_vol": ...
    }}
  }}
- All probabilities must be between 0 and 1.
- The three direction probabilities must sum exactly to 1.0.
- The three volatility probabilities must sum exactly to 1.0.
- Return valid JSON only. Do not use markdown fences.
- Do not output any extra text.
"""

    user_msg = f"""Decision date: {row['decision_date'].date()}
Forecast horizon: next {pred_len} month(s)
Lookback window: last {lookback} month(s) of data.

Aggregated HSI features over the lookback window:
- Last HSI close: {row['hsi_last_close']:.2f}
- Lookback HSI total return: {row['hsi_lookback_ret']:.4f}
- Mean daily return: {row['hsi_mean_1d_ret']:.4f}
- Daily return standard deviation: {row['hsi_std_1d_ret']:.4f}
- Realized volatility (annualized): {row['hsi_realized_vol']:.4f}
- Fraction of days HSI above MA20: {row['hsi_frac_above_ma_20']:.3f}
- Fraction of days HSI above MA50: {row['hsi_frac_above_ma_50']:.3f}
- Fraction of days HSI above MA100: {row['hsi_frac_above_ma_100']:.3f}

Aggregated VHSI features over the lookback window:
- VHSI mean: {row['vhsi_mean']:.4f}
- VHSI standard deviation: {row['vhsi_std']:.4f}
- Fraction of days in high-vol regime (from VHSI): {row['vhsi_high_vol_regime_frac']:.3f}

Aggregated option IV surface shape features over the lookback window:
- Near-term IV level mean: {row.get('level_near_mean', float('nan')):.4f}
- Near-term IV slope mean: {row.get('slope_near_mean', float('nan')):.4f}
- Near-term IV curvature mean: {row.get('curvature_near_mean', float('nan')):.4f}
- Near-term IV level 20d change: {row.get('level_near_chg_20d', float('nan')):.4f}
- Near-term IV slope 20d change: {row.get('slope_near_chg_20d', float('nan')):.4f}
- Near-term IV curvature 20d change: {row.get('curvature_near_chg_20d', float('nan')):.4f}
- Next-minus-near IV level term mean: {row.get('term_level_next_near_mean', float('nan')):.4f}
- Next-minus-near IV slope term mean: {row.get('term_slope_next_near_mean', float('nan')):.4f}
- Next-minus-near IV curvature term mean: {row.get('term_curvature_next_near_mean', float('nan')):.4f}

Estimate:
1. the probability distribution over direction regimes (bull, bear, range), and
2. the probability distribution over volatility regimes (low_vol, medium_vol, high_vol),
for the next {pred_len} month(s)."""

    return system_msg, user_msg


def call_llm(context, as_of_date="2024-10-01"):
    system_msg, user_msg = build_regime_prompt(context, as_of_date=as_of_date)
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
    model='deepseek-reasoner'
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

    # print("=== reasoning ===")
    # print(reasoning)

    # print("\n=== final answer ===")
    # print(final_text)
    return reasoning, final_text
