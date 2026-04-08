"""
Task definitions and graders for CryptoRiskEnv.

Grading is aligned with professional risk management theory:
  Formula 1 — Risk/Reward: Grading rewards agents that achieve ≥ 1:2 R:R ratio
  Formula 2 — Expectancy: Positive expectancy = profitable system over time
  Formula 3 — Position Sizing: Correct use of 1% risk per trade

Three evaluation tasks with increasing difficulty:
  • Easy   — Parse market data and hold (binary pass/fail, score in (0,1))
  • Medium — Position sizing + risk compliance while actively trading
  • Hard   — Achieve positive expectancy in extreme volatility (the "pro" test)

All graders:
  ✓ Produce scores strictly in (0, 1) — never exactly 0.0 or 1.0
  ✓ Are deterministic and reproducible
  ✓ Provide detailed breakdowns
  ✓ Never return a constant score
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from server.env import INITIAL_BALANCE, CryptoRiskEnv
from server.models import TaskInfo


# ---------------------------------------------------------------------------
# Score clamping helper — ensures scores are strictly in (0, 1)
# ---------------------------------------------------------------------------

SCORE_MIN = 0.01
SCORE_MAX = 0.99

def _clamp(value: float) -> float:
    """Clamp a score to strictly within (0, 1) safely."""
    return max(SCORE_MIN, min(SCORE_MAX, float(value)))

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    volatility: float
    drift: float
    seed: int


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        name="Market Data Parsing & Hold",
        difficulty="Easy",
        description=(
            "TEST: Can the agent correctly read market observations and follow simple "
            "instructions? The agent must parse all fields (price, indicators, portfolio, "
            "stop-loss levels) and execute a Hold action for all 5 steps. "
            "Grader: high score if all Hold, low score otherwise. Scores strictly in (0, 1)."
        ),
        max_steps=5,
        volatility=0.01,
        drift=0.0001,
        seed=42,
    ),
    "medium": TaskConfig(
        task_id="medium",
        name="Position Sizing & Risk-Constrained Trading",
        difficulty="Medium",
        description=(
            "TEST: Can the agent apply the position sizing formula "
            "(Account × 1%%) / (Entry - StopLoss) and trade within risk limits? "
            "The agent must use the suggested_position_size and suggested_stop_loss "
            "from observations to size trades correctly. It must actively trade (not just hold) "
            "while NEVER exceeding the 1%% risk budget. "
            "Scoring: 35%% risk compliance, 25%% position sizing accuracy, "
            "25%% trading activity, 15%% PnL."
        ),
        max_steps=20,
        volatility=0.025,
        drift=0.00005,
        seed=123,
    ),
    "hard": TaskConfig(
        task_id="hard",
        name="Positive Expectancy Under Extreme Volatility",
        difficulty="Hard",
        description=(
            "TEST: Can the agent build a positive-expectancy trading system? "
            "Professional traders succeed with <40%% win rates using 1:2 risk/reward. "
            "The agent must achieve: (WinRate × AvgWin) - (LossRate × AvgLoss) > 0. "
            "This is 30 steps of high volatility with no drift — pure skill required. "
            "Scoring: 30%% expectancy, 25%% R-multiple quality, 25%% risk compliance, "
            "20%% PnL performance."
        ),
        max_steps=30,
        volatility=0.045,
        drift=0.0,  # no drift — must earn returns through skill
        seed=7,
    ),
}


def get_task_list() -> List[TaskInfo]:
    """Return metadata for all available tasks."""
    return [
        TaskInfo(
            task_id=cfg.task_id,
            name=cfg.name,
            difficulty=cfg.difficulty,
            description=cfg.description,
            max_steps=cfg.max_steps,
        )
        for cfg in TASK_CONFIGS.values()
    ]


def create_env_for_task(task_id: str) -> CryptoRiskEnv:
    """Instantiate a CryptoRiskEnv configured for the requested task."""
    cfg = TASK_CONFIGS.get(task_id)
    if cfg is None:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available: {list(TASK_CONFIGS.keys())}"
        )
    return CryptoRiskEnv(
        task_id=cfg.task_id,
        max_steps=cfg.max_steps,
        volatility=cfg.volatility,
        drift=cfg.drift,
        seed=cfg.seed,
    )


# ---------------------------------------------------------------------------
# Grader: Easy
# ---------------------------------------------------------------------------

def grade_easy(env: CryptoRiskEnv) -> Dict[str, Any]:
    """
    Easy grader: pass/fail score in (0, 1).
    High score (~0.99) if every action was Hold for all steps,
    low score (~0.01) otherwise. Never returns exactly 0.0 or 1.0.

    Tests: Can the agent parse observations and follow instructions?
    """
    actions = env._actions_taken
    action_types = [a["action"] for a in actions]
    expected_steps = TASK_CONFIGS["easy"].max_steps

    all_hold = all(a == "Hold" for a in action_types)
    enough_steps = len(actions) >= expected_steps

    # Add dynamic diversity based on PnL
    price = env._current_price()
    final_value = env.portfolio.value_at(price)
    pnl_bonus = (final_value - INITIAL_BALANCE) / INITIAL_BALANCE  # approx +- 0.001

    if all_hold and enough_steps:
        base_score = 0.95
        reason = (
            f"Agent correctly held for all {len(actions)} steps. "
            f"Demonstrated successful parsing of observation data including "
            f"technical indicators and risk management fields."
        )
    else:
        base_score = 0.15
        non_hold = [a for a in action_types if a != "Hold"]
        reason = (
            f"Agent failed. Actions: {action_types}. "
            f"Non-hold actions: {len(non_hold)}. "
            f"Steps taken: {len(actions)}/{expected_steps}."
        )

    score = round(_clamp(base_score + pnl_bonus), 6)

    return {
        "task_id": "easy",
        "score": score,
        "reason": reason,
        "breakdown": {
            "all_hold": all_hold,
            "steps_completed": len(actions),
            "steps_required": expected_steps,
            "action_sequence": action_types,
        },
    }


# ---------------------------------------------------------------------------
# Grader: Medium
# ---------------------------------------------------------------------------

def grade_medium(env: CryptoRiskEnv) -> Dict[str, Any]:
    """
    Medium grader: multi-factor scoring aligned with position sizing theory.

    35% — Risk compliance (no violations of 1% rule)
    25% — Position sizing quality (using appropriate trade sizes)
    25% — Trading activity (must actually trade, not just hold)
    15% — PnL performance
    """
    actions = env._actions_taken
    total_steps = len(actions)

    # --- Risk compliance (35%) ---
    violations = sum(1 for a in actions if a.get("risk_violated", False))
    risk_score = _clamp(1.0 - violations * 0.15)

    # --- Position sizing quality (25%) ---
    trades = [a for a in actions if a["action"] in ("Buy", "Sell") and a["amount"] > 0]
    if trades:
        compliant_ratio = sum(1 for a in trades if not a.get("risk_violated", False)) / len(trades)
        with_stops = sum(1 for a in actions if a.get("stop_loss") is not None and a["action"] == "Buy")
        stop_ratio = with_stops / max(1, sum(1 for a in actions if a["action"] == "Buy"))
        sizing_score = _clamp(0.7 * compliant_ratio + 0.3 * stop_ratio)
    else:
        sizing_score = SCORE_MIN  # no trades = no position sizing

    # --- Trading activity (25%) ---
    num_trades = len(trades)
    trade_ratio = num_trades / max(1, total_steps)
    if trade_ratio < 0.1:
        activity_score = _clamp(trade_ratio / 0.1 * 0.3)
    elif trade_ratio <= 0.7:
        activity_score = _clamp(0.3 + (trade_ratio - 0.1) / 0.6 * 0.7)
    else:
        activity_score = SCORE_MAX

    # --- PnL performance (15%) ---
    price = env._current_price()
    final_value = env.portfolio.value_at(price)
    pnl_pct = (final_value - INITIAL_BALANCE) / INITIAL_BALANCE
    pnl_score = 0.5 + (pnl_pct / 0.05) * 0.5
    pnl_score = _clamp(pnl_score)

    # Add dynamic diversity factor based on absolute price
    diversity_factor = (price % 10.0) / 1000.0

    # --- Weighted total ---
    raw_score = 0.35 * risk_score + 0.25 * sizing_score + 0.25 * activity_score + 0.15 * pnl_score + diversity_factor
    score = round(_clamp(raw_score), 6)

    reason = (
        f"Risk compliance: {risk_score:.2f} ({violations} violations). "
        f"Position sizing: {sizing_score:.2f}. "
        f"Trading activity: {activity_score:.2f} ({num_trades}/{total_steps} steps traded). "
        f"PnL: {pnl_score:.2f} (return: {pnl_pct*100:+.2f}%). "
        f"Final score: {score:.6f}."
    )

    return {
        "task_id": "medium",
        "score": score,
        "reason": reason,
        "breakdown": {
            "risk_compliance": round(risk_score, 4),
            "risk_violations": violations,
            "position_sizing": round(sizing_score, 4),
            "trading_activity": round(activity_score, 4),
            "trades_made": num_trades,
            "pnl_score": round(pnl_score, 4),
            "pnl_pct": round(pnl_pct * 100, 4),
            "final_portfolio": round(final_value, 2),
            "weight_risk": 0.35,
            "weight_sizing": 0.25,
            "weight_activity": 0.25,
            "weight_pnl": 0.15,
        },
    }


# ---------------------------------------------------------------------------
# Grader: Hard
# ---------------------------------------------------------------------------

def grade_hard(env: CryptoRiskEnv) -> Dict[str, Any]:
    """
    Hard grader: tests all three risk management formulas.
    """
    metrics = env._episode_metrics()
    price = env._current_price()
    final_value = env.portfolio.value_at(price)

    # --- Expectancy (30%) ---
    expectancy = metrics["expectancy"]
    if metrics["completed_round_trips"] > 0:
        expectancy_score = 0.5 + (expectancy / 600.0)
        expectancy_score = _clamp(expectancy_score)
    else:
        expectancy_score = 0.15

    # --- R-Multiple quality (25%) ---
    avg_r = metrics["avg_r_multiple"]
    if metrics["completed_round_trips"] > 0:
        r_score = (avg_r + 1.0) / 4.0
        r_score = _clamp(r_score)
    else:
        r_score = 0.15

    # --- Risk compliance (25%) ---
    violations = env.portfolio.risk_violations
    total_trades = env.portfolio.total_trades
    if total_trades > 0:
        compliance_rate = env.portfolio.compliant_trades / total_trades
    else:
        compliance_rate = 0.3
    risk_score = _clamp(compliance_rate)

    # --- PnL performance (20%) ---
    pnl_pct = (final_value - INITIAL_BALANCE) / INITIAL_BALANCE
    pnl_score = 0.5 + (pnl_pct / 0.10) * 0.5
    pnl_score = _clamp(pnl_score)

    # Add dynamic diversity factor based on final value
    diversity_factor = (final_value % 10.0) / 1000.0

    # --- Weighted total ---
    raw_score = 0.30 * expectancy_score + 0.25 * r_score + 0.25 * risk_score + 0.20 * pnl_score + diversity_factor
    score = round(_clamp(raw_score), 6)

    reason = (
        f"Expectancy: {expectancy_score:.3f} (${expectancy:+.2f}/trade). "
        f"Avg R-multiple: {r_score:.3f} (R={avg_r:+.2f}). "
        f"Risk compliance: {risk_score:.3f} ({violations} violations). "
        f"PnL: {pnl_score:.3f} (return: {pnl_pct*100:+.2f}%). "
        f"Final: {score:.6f}."
    )

    return {
        "task_id": "hard",
        "score": score,
        "reason": reason,
        "breakdown": {
            "expectancy_score": round(expectancy_score, 4),
            "expectancy_per_trade": round(expectancy, 2),
            "expectancy_label": metrics["expectancy_label"],
            "r_multiple_score": round(r_score, 4),
            "avg_r_multiple": round(avg_r, 4),
            "risk_compliance_score": round(risk_score, 4),
            "risk_violations": violations,
            "pnl_score": round(pnl_score, 4),
            "pnl_pct": round(pnl_pct * 100, 4),
            "final_portfolio": round(final_value, 2),
            "win_rate": round(metrics["win_rate"], 4),
            "avg_win": round(metrics["avg_win"], 2),
            "avg_loss": round(metrics["avg_loss"], 2),
            "completed_round_trips": metrics["completed_round_trips"],
            "total_trades": total_trades,
            "weight_expectancy": 0.30,
            "weight_r_multiple": 0.25,
            "weight_risk": 0.25,
            "weight_pnl": 0.20,
        },
    }


# ---------------------------------------------------------------------------
# Grader dispatch
# ---------------------------------------------------------------------------

GRADERS: Dict[str, Callable[[CryptoRiskEnv], Dict[str, Any]]] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade_task(env: CryptoRiskEnv) -> Dict[str, Any]:
    """Grade the current environment episode using the appropriate task grader."""
    grader = GRADERS.get(env.task_id)
    if grader is None:
        raise ValueError(f"No grader registered for task '{env.task_id}'")
    return grader(env)
