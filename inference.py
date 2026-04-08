#!/usr/bin/env python3
"""
inference.py — Baseline LLM inference script for CryptoRiskEnv.

Connects an LLM to the CryptoRiskEnv OpenEnv environment via the OpenAI
Python client (compatible with Groq, Together, OpenAI, etc.).

Environment variables:
    OPENAI_API_KEY   — API key (falls back to HF_TOKEN if not set)
    API_BASE_URL     — Base URL of the OpenAI-compatible API
    MODEL_NAME       — Model identifier (e.g. "llama-3.3-70b-versatile")
    HF_TOKEN         — Hugging Face token (used as API key fallback)
    ENV_BASE_URL     — URL where CryptoRiskEnv is running

Runs all 3 tasks (easy, medium, hard) and prints reproducible scores.
Designed to complete in under 20 minutes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — reads from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_RETRIES = 3
RETRY_DELAY = 2.0

# ---------------------------------------------------------------------------
# System prompt — teaches the LLM risk management theory
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional cryptocurrency trading agent trained in risk management.

You follow three core formulas from professional trading:

FORMULA 1 — Risk/Reward Ratio:
- Never enter a trade unless the potential reward is ≥ 2× your risk.
- The observation provides "suggested_stop_loss" and "reward_target" for a 1:2 ratio.
- Risk = Entry - StopLoss. Reward = TakeProfit - Entry. Always aim for R:R ≥ 1:2.

FORMULA 2 — Expectancy:
- Your system must be mathematically profitable over many trades.
- Expectancy = (WinRate × AvgWin) - (LossRate × AvgLoss)
- Even with a 33% win rate, if your avg win is 2× your avg loss, you're profitable.
- "You don't need to win often; you need to win big when right and lose small when wrong."

FORMULA 3 — Position Sizing:
- Position Size = (Account × 1%) / (Entry - StopLoss)
- The observation provides "suggested_position_size" calculated using this formula.
- NEVER risk more than 1% of your portfolio on any single trade.
- The "max_trade_size" field shows the maximum allowed dollar amount.

At each step you receive a JSON observation with these fields:
- current_price, price_change_pct: price data
- ema_9, ema_21, ema_50: trend indicators (fast/medium/slow)
- macd, macd_signal: momentum (buy when MACD crosses above signal)
- rsi: overbought (>70) / oversold (<30) indicator
- atr: volatility measure (used for stop-loss placement)
- bollinger_upper, bollinger_lower: volatility bands
- suggested_stop_loss: ATR-based stop-loss price (2×ATR below entry)
- risk_per_share: dollar risk per unit if stopped out
- suggested_position_size: optimal trade size from Formula 3
- reward_target: price target for 1:2 risk/reward
- portfolio_value, cash_balance, position_size, position_pct, unrealized_pnl
- risk_budget_remaining, max_trade_size: your risk limits
- step_number, total_steps

You must respond with a valid JSON object:
{
  "action": "Buy" | "Sell" | "Hold",
  "amount": <number or null>,
  "stop_loss": <number or null>,
  "take_profit": <number or null>,
  "reasoning": "<brief explanation>"
}

CRITICAL RULES:
1. amount must be ≤ max_trade_size (1% risk rule)
2. Use suggested_position_size as your guide for trade sizing
3. Set stop_loss to suggested_stop_loss from the observation
4. Set take_profit to reward_target for 1:2 R:R
5. Set amount and stop_loss and take_profit to null for Hold actions
6. Always respond with ONLY the JSON object, no other text.
"""

TASK_PROMPTS = {
    "easy": (
        "TASK: Market Data Parsing & Hold.\n"
        "You are being tested on your ability to parse market data correctly.\n"
        "The ONLY correct strategy is to HOLD on every single step.\n"
        "Do not buy or sell. Just respond with:\n"
        '{"action": "Hold", "amount": null, "stop_loss": null, "take_profit": null, '
        '"reasoning": "Holding as instructed"}'
    ),
    "medium": (
        "TASK: Position Sizing & Risk-Constrained Trading.\n"
        "You are tested on applying the position sizing formula correctly.\n"
        "You MUST actively trade (buy and sell) but size positions correctly:\n"
        "  Position Size = (Account × 1%) / (Entry - StopLoss)\n"
        "  The observation gives you 'suggested_position_size' — use it!\n\n"
        "Use technical indicators to find entries:\n"
        "- BUY when: RSI < 35 or (MACD > macd_signal and EMA_9 > EMA_21)\n"
        "  → Set amount to suggested_position_size (or less)\n"
        "  → Set stop_loss to suggested_stop_loss\n"
        "  → Set take_profit to reward_target\n"
        "- SELL when: RSI > 65 or (MACD < macd_signal and EMA_9 < EMA_21)\n"
        "  → Sell your position to lock in profit or cut losses\n"
        "- HOLD when: signals are mixed\n\n"
        "NEVER exceed max_trade_size. Conservative sizing is key."
    ),
    "hard": (
        "TASK: Build a Positive-Expectancy Trading System.\n"
        "You are tested on creating a mathematically profitable system.\n"
        "Expectancy = (WinRate × AvgWin) - (LossRate × AvgLoss)\n\n"
        "Strategy for positive expectancy under extreme volatility:\n"
        "- Only enter HIGH-CONVICTION setups (confluence of multiple signals)\n"
        "- BUY when: price near bollinger_lower AND RSI < 30 AND MACD > macd_signal\n"
        "  → Use 60-80% of suggested_position_size (conservative in volatility)\n"
        "  → Set stop at suggested_stop_loss, target at reward_target (1:2 R:R)\n"
        "- SELL when: price near bollinger_upper AND RSI > 70\n"
        "  → Or sell if unrealized_pnl is negative and price breaks stop-loss\n"
        "- HOLD when: no strong confluence signal\n\n"
        "KEY INSIGHT: You can lose 7 out of 10 trades and still profit if:\n"
        "  - Average win = 2× average loss (1:2 risk/reward)\n"
        "  - You cut losses quickly at the stop-loss\n"
        "  - You let winners run to the reward target\n\n"
        "NEVER exceed max_trade_size. Risk discipline is non-negotiable."
    ),
}

# ---------------------------------------------------------------------------
# Environment client helpers
# ---------------------------------------------------------------------------


def env_request(
    method: str, path: str, json_data: Optional[Dict] = None, retries: int = MAX_RETRIES
) -> Dict[str, Any]:
    """Make a request to the CryptoRiskEnv API with retries."""
    url = f"{ENV_BASE_URL}{path}"
    for attempt in range(retries):
        try:
            if method == "GET":
                resp = requests.get(url, timeout=10.0)
            else:
                resp = requests.post(url, json=json_data, timeout=10.0)
            if not resp.ok:
                print(f"Server returned error {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                print(f"  [Retry {attempt+1}/{retries}] {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise


def reset_env(task_id: str) -> Dict[str, Any]:
    return env_request("POST", "/reset", {"task_id": task_id})


def step_env(action: Dict[str, Any]) -> Dict[str, Any]:
    return env_request("POST", "/step", action)


def grade_env() -> Dict[str, Any]:
    return env_request("POST", "/grade")


def get_tasks() -> Dict[str, Any]:
    return env_request("GET", "/tasks")


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def get_llm_action(
    client: OpenAI,
    observation: Dict[str, Any],
    task_id: str,
    step_num: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Query the LLM for a trading action given the current observation."""
    user_content = (
        f"Step {step_num + 1}/{max_steps}\n"
        f"Observation: {json.dumps(observation, indent=2)}\n"
        f"\nRespond with your action as a JSON object."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + TASK_PROMPTS.get(task_id, "")},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=250,
            )
            content = response.choices[0].message.content.strip()

            # Extract JSON from various LLM output formats
            action = _parse_llm_json(content)

            # Validate required fields
            if "action" not in action:
                action = {"action": "Hold", "amount": None, "stop_loss": None,
                          "take_profit": None, "reasoning": "Parse fallback"}

            # Enforce risk limit from observation
            max_size = observation.get("max_trade_size", 1000)
            if action.get("amount") is not None and action["amount"] > max_size:
                action["amount"] = round(max_size * 0.8, 2)
                action["reasoning"] = (action.get("reasoning", "") +
                                       " [amount clamped to risk limit]")

            return action

        except Exception as e:
            print(f"  [LLM Error attempt {attempt+1}] {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {"action": "Hold", "amount": None, "stop_loss": None,
                        "take_profit": None, "reasoning": f"LLM error: {e}"}

    return {"action": "Hold", "amount": None, "stop_loss": None,
            "take_profit": None, "reasoning": "Max retries exceeded"}


def _parse_llm_json(content: str) -> Dict[str, Any]:
    """Robustly parse JSON from LLM output, handling markdown wraps etc."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    if "```" in content:
        lines = content.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        if json_lines:
            try:
                return json.loads("\n".join(json_lines))
            except json.JSONDecodeError:
                pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {"action": "Hold", "amount": None, "stop_loss": None,
            "take_profit": None, "reasoning": "Could not parse LLM response"}


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------


def run_task(client: OpenAI, task_id: str, task_name: str, max_steps: int) -> Dict[str, Any]:
    """Run a single task episode and return the result."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_name} ({task_id}) -- {max_steps} steps")
    print(f"{'='*60}")

    print(f"[START] {task_id}")

    reset_data = reset_env(task_id)
    observation = reset_data
    print(f"  Initial price: ${observation['current_price']:,.2f}")
    print(f"  Risk budget:   ${observation['max_trade_size']:,.2f}")
    print(f"  Stop-loss:     ${observation['suggested_stop_loss']:,.2f}")
    print(f"  Reward target: ${observation['reward_target']:,.2f}")

    done = False
    step_num = 0

    while not done and step_num < max_steps:
        action = get_llm_action(client, observation, task_id, step_num, max_steps)
        act_type = action.get('action', 'Hold')
        print(f"  Step {step_num + 1:2d}: {act_type:4s}", end="")
        if action.get("amount"):
            print(f"  ${action['amount']:>10,.2f}", end="")
        else:
            print(f"  {'--':>11s}", end="")

        print(f"\n[STEP] {json.dumps(action)}")

        step_data = step_env(action)
        observation = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        step_info = step_data.get("info", {})

        risk_pen = reward.get("risk_penalty", 0)
        if step_info.get("risk_violation"):
            print(f"  !! VIOLATION", end="")
        if risk_pen < 0:
            print(f"  (pen: {risk_pen:.3f})", end="")
        print()
        step_num += 1

    grade_result = grade_env()
    score = grade_result["score"]
    reason = grade_result["reason"]
    breakdown = grade_result.get("breakdown", {})

    print(f"[END] {score}")
    print(f"\n  Score:  {score:.4f}")
    print(f"  Reason: {reason}")

    return {
        "task_id": task_id,
        "name": task_name,
        "difficulty": "Easy" if task_id == "easy" else ("Medium" if task_id == "medium" else "Hard"),
        "score": score,
        "reason": reason,
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    start_time = time.time()

    print("=" * 60)
    print("  CryptoRiskEnv -- LLM Inference Baseline")
    print("=" * 60)
    print(f"  Model:    {MODEL_NAME}")
    print(f"  API Base: {API_BASE_URL}")
    print(f"  Env URL:  {ENV_BASE_URL}")
    print()

    if not OPENAI_API_KEY:
        print("ERROR: No API key set. Set OPENAI_API_KEY or HF_TOKEN env variable.")
        sys.exit(1)

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=API_BASE_URL,
        timeout=15.0,
    )

    try:
        health = env_request("GET", "/health")
        print(f"  Environment health: {health['status']}")
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {ENV_BASE_URL}: {e}")
        sys.exit(1)

    tasks_data = get_tasks()
    tasks = tasks_data["tasks"]
    print(f"  Available tasks: {len(tasks)}")

    results = {}
    for task in tasks:
        result = run_task(client, task["task_id"], task["name"], task["max_steps"])
        results[task["task_id"]] = result

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    total_score = 0.0
    for tid, r in results.items():
        print(f"  [{r['difficulty']:6s}] {r['name']:50s} -> {r['score']:.4f}")
        total_score += max(0.01, min(0.99, float(r["score"])))
    avg_score = total_score / len(results) if results else 0.01
    avg_score = max(0.01, min(0.99, avg_score))
    print(f"\n  Average Score: {avg_score:.4f}")
    print(f"  Total Time:   {elapsed:.1f}s")
    print(f"{'='*60}")

    print("\n--- JSON RESULTS ---")
    json_results = {
        "model": MODEL_NAME,
        "average_score": round(avg_score, 4),
        "total_time_seconds": round(elapsed, 1),
        "tasks": {tid: {"score": max(0.01, min(0.99, float(r["score"]))), "breakdown": r.get("breakdown", {})} for tid, r in results.items()},
    }
    print(json.dumps(json_results, indent=2))


if __name__ == "__main__":
    main()
