import json
from datetime import datetime

PRICES = {
    "o3-mini": {
        "input_per_1k": 0.004,
        "output_per_1k": 0.016,
        "notes": "Pricing not published by OpenAI, estimated via o3-mini"
    },
    "o4-mini": {
        "input_per_1k": 0.004,
        "output_per_1k": 0.016,
        "notes": "NA"
    },
    "Gemini-2.5-Flash-Preview-04-17": {
        "input_per_1k": 0.00030,
        "output_per_1k": 0.00250,
        "notes": "Free tier available; paid tier $0.30/1M input, $2.50/1M output"
    },
    "claude-3-5-sonnet-20241022": {
        "input_per_1k": 0.00300,
        "output_per_1k": 0.01500,
        "notes": "Anthropic API pricing ($3/1M input, $15/1M output)"
    },
    "claude-3-5-haiku-20241022": {
        "input_per_1k": 0.00300,
        "output_per_1k": 0.01500,
        "notes": "Anthropic API pricing ($3/1M input, $15/1M output)"
    },
    "claude-3-7-sonnet-latest": {
        "input_per_1k": 0.00300,
        "output_per_1k": 0.01500,
        "notes": "Same as Claude-3.5-Sonnet"
    },
    "gpt-5-2025-08-07": {
        "input_per_1k": 0.00125,
        "output_per_1k": 0.01000,
        "notes": "OpenAI API pricing ($1.25/1M input, $10/1M output)"
    },
    "claude-sonnet-4-20250514": {
        "input_per_1k": 0.00300,
        "output_per_1k": 0.01500,
        "notes": "Anthropic API pricing ($3/1M input, $15/1M output); discounts via caching/batching"
    },
    "claude-opus-4-1-20250805": {
        "input_per_1k": 0.015,
        "output_per_1k": 0.075,
        "notes": "Same as Claude-4; extended reasoning mode"
    }
}

LOG_FILE = "api_usage_log.jsonl"
TOTAL_COST = 0.0
LAST_WARNING_THRESHOLD = 0.0  # track last $10 threshold crossed


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 characters in English)."""
    return max(1, len(text) // 4)


def log_and_track(model: str, input_text: str, output_text: str, log: bool = False):
    """
    Track usage & warn every $10 spent.
    
    Args:
        model (str): Model name (must exist in PRICES).
        input_text (str): Prompt sent to model.
        output_text (str): Model's response text.
        log (bool): If True, save JSON log entry. Defaults False.
    """
    global TOTAL_COST, LAST_WARNING_THRESHOLD

    if model not in PRICES:
        raise ValueError(f"Model '{model}' not found in PRICES dictionary")

    pricing = PRICES[model]
    if pricing["input_per_1k"] is None or pricing["output_per_1k"] is None:
        print(f"[{model}] ⚠️ No pricing data available (skipping cost tracking).")
        return TOTAL_COST

    # Token estimation
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)

    # Cost calculation
    input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
    output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
    total_cost = input_cost + output_cost

    # Update cumulative cost
    TOTAL_COST += total_cost

    # Print feedback
    print(f"[{model}] Call: ${total_cost:.4f}, Cumulative: ${TOTAL_COST:.4f}")

    # Warn at every $10 threshold
    if TOTAL_COST - LAST_WARNING_THRESHOLD >= 10.0:
        LAST_WARNING_THRESHOLD = (TOTAL_COST // 10) * 10
        print(f"⚠️  WARNING: Budget consumed reached ${LAST_WARNING_THRESHOLD:.2f}")

    # Optional logging
    if log:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "input_preview": input_text[:200],
            "output_preview": output_text[:200],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "cumulative_cost": round(TOTAL_COST, 6),
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return TOTAL_COST

