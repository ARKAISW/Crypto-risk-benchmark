"""
Strict Pydantic models for CryptoRiskEnv — OpenEnv specification.

Aligned with professional risk management theory (position sizing,
risk/reward ratio, expectancy). Models define the typed spaces for
evaluating LLM agents on cryptocurrency risk management discipline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Observation Space
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Market observation returned to the agent at each step.

    Provides everything a professional trader needs:
    - Price + technical indicators to find setups
    - ATR-based stop-loss levels for position sizing
    - Portfolio context including risk budget
    """

    # Price data
    current_price: float = Field(..., description="Current asset price in USD")
    price_change_pct: float = Field(..., description="Price change from previous step (%)")

    # Trend indicators
    ema_9: float = Field(..., description="9-period EMA (fast signal)")
    ema_21: float = Field(..., description="21-period EMA (medium signal)")
    ema_50: float = Field(..., description="50-period EMA (slow trend)")

    # Momentum indicators
    macd: float = Field(..., description="MACD line (EMA-12 minus EMA-26)")
    macd_signal: float = Field(..., description="MACD signal line (9-period EMA of MACD)")
    rsi: float = Field(..., description="Relative Strength Index (0-100)")

    # Volatility indicators
    atr: float = Field(..., description="Average True Range (volatility measure)")
    bollinger_upper: float = Field(..., description="Upper Bollinger Band (2 std dev)")
    bollinger_lower: float = Field(..., description="Lower Bollinger Band (2 std dev)")

    # Stop-loss & position sizing context (from risk management theory)
    suggested_stop_loss: float = Field(
        ..., description="ATR-based stop-loss price (entry - 2×ATR). "
        "Use with position sizing formula: shares = (Account × 1%) / (Entry - StopLoss)"
    )
    risk_per_share: float = Field(
        ..., description="Dollar risk per unit if stopped out (current_price - stop_loss)"
    )
    suggested_position_size: float = Field(
        ..., description="Optimal position size in USD using the formula: "
        "(portfolio × 1%) / risk_per_share × current_price"
    )
    reward_target: float = Field(
        ..., description="Reward target price for 1:2 risk/reward (entry + 2×risk_per_share)"
    )

    # Portfolio context (the agent must see its own risk state)
    portfolio_value: float = Field(..., description="Current total portfolio value in USD")
    cash_balance: float = Field(..., description="Available cash in USD")
    position_size: float = Field(..., description="Current position value in USD")
    position_pct: float = Field(..., description="Position as % of portfolio (exposure)")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss on open position")

    # Risk context
    risk_budget_remaining: float = Field(
        ..., description="Max dollar risk for this trade (1% of portfolio)"
    )
    max_trade_size: float = Field(
        ..., description="Maximum allowed trade size using position sizing formula"
    )
    step_number: int = Field(..., description="Current step in the episode")
    total_steps: int = Field(..., description="Total steps in this episode")


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Discrete action types the agent can take."""
    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"


class Action(BaseModel):
    """Trading action submitted by the agent.

    The agent should use the position sizing formula from risk management theory:
      Position Size = (Account × Risk%) / (Entry - Stop Loss)

    Risk Rule: No single trade may risk more than 1% of portfolio value.
    The amount should be ≤ the suggested_position_size from the observation.
    """

    action: ActionType = Field(..., description="Trading action: Buy, Sell, or Hold")
    amount: Optional[float] = Field(
        None,
        ge=0,
        description="Dollar amount to trade. Should be ≤ suggested_position_size. Ignored for Hold.",
    )
    stop_loss: Optional[float] = Field(
        None,
        description="Stop-loss price for this trade. Used for risk/reward ratio calculation.",
    )
    take_profit: Optional[float] = Field(
        None,
        description="Take-profit target price. Should give at least 1:2 risk/reward ratio.",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief reasoning for this action (used for evaluation).",
    )

    @field_validator("action", mode="before")
    @classmethod
    def normalise_action(cls, v: Any) -> str:
        """Accept case-insensitive action strings."""
        if isinstance(v, str):
            mapping = {"buy": "Buy", "sell": "Sell", "hold": "Hold"}
            normalised = mapping.get(v.strip().lower())
            if normalised is None:
                raise ValueError(f"Invalid action '{v}'. Must be Buy, Sell, or Hold.")
            return normalised
        return v


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Reward signal returned after each step.

    Multi-dimensional feedback aligned with risk management theory:
    - Risk compliance: following position sizing rules
    - PnL: portfolio performance
    - R-multiple: quality of wins vs losses (risk/reward ratio)
    """

    step_reward: float = Field(..., description="Net reward for this step")
    cumulative_reward: float = Field(..., description="Cumulative episode reward")
    risk_penalty: float = Field(0.0, description="Risk violation penalty this step (≤ 0)")
    pnl_reward: float = Field(0.0, description="PnL-based reward component this step")
    compliance_bonus: float = Field(0.0, description="Bonus for following risk rules")


# ---------------------------------------------------------------------------
# OpenEnv API Request/Response Schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""
    task_id: str = Field("easy", description="ID of the task to start (easy/medium/hard)")


class ResetResponse(BaseModel):
    """Response from /reset endpoint."""
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)


class StepRequest(BaseModel):
    """Request body for /step endpoint."""
    action: Action


class StepResponse(BaseModel):
    """Response from /step endpoint."""
    observation: Observation
    reward: Reward
    done: bool = Field(..., description="Whether the episode has ended")
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    """Response from /state endpoint — full environment state snapshot."""
    observation: Observation
    portfolio: Dict[str, Any]
    step_count: int
    done: bool
    task_id: str
    episode_metrics: Dict[str, Any] = Field(default_factory=dict)
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    """Metadata for a single evaluation task."""
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int


class TaskListResponse(BaseModel):
    """Response listing all available tasks."""
    tasks: List[TaskInfo]


class GradeRequest(BaseModel):
    """Optional request body for /grade endpoint."""
    pass


class GradeResponse(BaseModel):
    """Response from the grading endpoint."""
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str
    breakdown: Dict[str, Any] = Field(default_factory=dict)
