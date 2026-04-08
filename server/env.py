"""
CryptoRiskEnv — Core environment engine.

Aligned with professional risk management theory:
  Formula 1 — Risk/Reward Ratio: Every observation includes stop-loss, take-profit
      targets, and the suggested 1:2 R:R ratio. Trades are evaluated on R-multiples.
  Formula 2 — Expectancy: (WinRate × AvgWin) - (LossRate × AvgLoss). Computed at
      episode end and used in grading. A positive expectancy = profitable system.
  Formula 3 — Position Sizing: (Account × Risk%) / (Entry - StopLoss). The observation
      provides ATR-based stop-loss and the correctly calculated position size.

Environment features:
  • $100,000 starting balance
  • Synthetic price data via geometric Brownian motion (reproducible with seed)
  • Technical indicators: EMA-9/21/50, MACD+Signal, RSI, ATR, Bollinger Bands
  • ATR-based stop-loss levels and 1:2 reward targets in every observation
  • Strict 1% risk-per-trade constraint with proportional penalties
  • 0.1% transaction fee on every trade
  • Multi-dimensional reward: PnL + risk compliance + R-multiple tracking
  • Episode metrics: Expectancy, Avg R-Multiple, Sharpe Ratio, Win Rate
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from server.models import Action, ActionType, Observation, Reward

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_BALANCE: float = 100_000.0
TRANSACTION_FEE_RATE: float = 0.001   # 0.1%
RISK_FRACTION: float = 0.01           # 1% of portfolio per trade
SEED_PRICE: float = 50_000.0          # Starting BTC price
RISK_REWARD_TARGET: float = 2.0       # 1:2 risk/reward ratio


# ---------------------------------------------------------------------------
# Synthetic price generator (geometric Brownian motion)
# ---------------------------------------------------------------------------

def _generate_price_series(
    length: int,
    start_price: float = SEED_PRICE,
    volatility: float = 0.02,
    drift: float = 0.0001,
    seed: Optional[int] = None,
) -> List[float]:
    """Generate a synthetic price series using geometric Brownian motion."""
    rng = random.Random(seed)
    prices = [start_price]
    for _ in range(length - 1):
        shock = rng.gauss(0, 1) * volatility
        new_price = prices[-1] * math.exp(drift + shock)
        prices.append(round(new_price, 2))
    return prices


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _ema(prices: List[float], period: int) -> float:
    """Exponential Moving Average over the full history."""
    if len(prices) < period:
        return prices[-1]
    k = 2.0 / (period + 1)
    ema_val = prices[0]
    for p in prices[1:]:
        ema_val = p * k + ema_val * (1 - k)
    return round(ema_val, 2)


def _ema_series(prices: List[float], period: int) -> List[float]:
    """Full EMA series for computing MACD signal line."""
    if len(prices) < period:
        return [prices[-1]] * len(prices)
    k = 2.0 / (period + 1)
    result = [prices[0]]
    for p in prices[1:]:
        result.append(p * k + result[-1] * (1 - k))
    return result


def _rsi(prices: List[float], period: int = 14) -> float:
    """Relative Strength Index."""
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent = deltas[-period:]
    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0001
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


def _macd(prices: List[float]) -> float:
    """MACD line = EMA-12 minus EMA-26."""
    return round(_ema(prices, 12) - _ema(prices, 26), 2)


def _macd_signal(prices: List[float]) -> float:
    """MACD signal line = 9-period EMA of the MACD values."""
    if len(prices) < 26:
        return 0.0
    ema12_series = _ema_series(prices, 12)
    ema26_series = _ema_series(prices, 26)
    macd_values = [round(e12 - e26, 2) for e12, e26 in zip(ema12_series, ema26_series)]
    if len(macd_values) < 9:
        return macd_values[-1] if macd_values else 0.0
    return round(_ema(macd_values, 9), 2)


def _atr(prices: List[float], period: int = 14) -> float:
    """Average True Range (simplified using close prices only)."""
    if len(prices) < 2:
        return 0.0
    trs = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    recent = trs[-period:]
    return round(sum(recent) / len(recent), 2)


def _bollinger_bands(prices: List[float], period: int = 20) -> Tuple[float, float]:
    """Bollinger Bands: mean ± 2 * std over `period`."""
    if len(prices) < period:
        window = prices
    else:
        window = prices[-period:]
    mean = sum(window) / len(window)
    variance = sum((p - mean) ** 2 for p in window) / len(window)
    std = math.sqrt(variance)
    return round(mean + 2 * std, 2), round(mean - 2 * std, 2)


# ---------------------------------------------------------------------------
# Portfolio tracker
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    """Tracks the agent's financial state across the episode."""

    cash: float = INITIAL_BALANCE
    holdings: float = 0.0           # quantity of asset held
    avg_entry_price: float = 0.0    # average cost basis
    total_fees_paid: float = 0.0
    total_trades: int = 0
    total_buys: int = 0
    total_sells: int = 0
    risk_violations: int = 0
    compliant_trades: int = 0
    trade_history: List[Dict[str, Any]] = field(default_factory=list)

    # R-multiple tracking (risk management theory)
    completed_trades: List[Dict[str, Any]] = field(default_factory=list)
    # Stores completed round-trips: {entry, exit, risk_per_share, pnl, r_multiple}

    def value_at(self, price: float) -> float:
        """Total portfolio value at a given price."""
        return self.cash + self.holdings * price

    def position_value(self, price: float) -> float:
        return self.holdings * price

    def position_pct(self, price: float) -> float:
        total = self.value_at(price)
        if total <= 0:
            return 0.0
        return round((self.holdings * price) / total * 100, 2)

    def unrealized_pnl(self, price: float) -> float:
        if self.holdings <= 0 or self.avg_entry_price <= 0:
            return 0.0
        return round(self.holdings * (price - self.avg_entry_price), 2)

    def to_dict(self, price: float) -> Dict[str, Any]:
        return {
            "cash": round(self.cash, 2),
            "holdings_qty": round(self.holdings, 6),
            "holdings_value": round(self.holdings * price, 2),
            "total_value": round(self.value_at(price), 2),
            "unrealized_pnl": self.unrealized_pnl(price),
            "avg_entry_price": round(self.avg_entry_price, 2),
            "total_fees_paid": round(self.total_fees_paid, 2),
            "total_trades": self.total_trades,
            "total_buys": self.total_buys,
            "total_sells": self.total_sells,
            "risk_violations": self.risk_violations,
            "compliant_trades": self.compliant_trades,
            "position_pct": self.position_pct(price),
            "completed_round_trips": len(self.completed_trades),
        }


# ---------------------------------------------------------------------------
# CryptoRiskEnv
# ---------------------------------------------------------------------------

class CryptoRiskEnv:
    """Stateful trading environment implementing the OpenEnv interface.

    Core philosophy from risk management theory:
    - "You don't need to win often; you need to win big when right
       and lose small when wrong."
    - Tests position sizing, R:R ratio, and expectancy.
    """

    def __init__(
        self,
        task_id: str = "easy",
        max_steps: int = 5,
        volatility: float = 0.02,
        drift: float = 0.0001,
        seed: Optional[int] = None,
    ):
        self.task_id = task_id
        self.max_steps = max_steps
        self.volatility = volatility
        self.drift = drift
        self.seed = seed

        # Pre-generate price history for indicators + episode steps
        history_warmup = 60  # for EMA-50 warm-up
        total_length = history_warmup + max_steps + 1
        self._all_prices = _generate_price_series(
            total_length, volatility=volatility, drift=drift, seed=seed
        )
        self._history_offset = history_warmup

        # Mutable state
        self.portfolio = Portfolio()
        self.step_count: int = 0
        self.done: bool = False
        self._cumulative_reward: float = 0.0
        self._actions_taken: List[Dict[str, Any]] = []
        self._step_rewards: List[float] = []
        self._prev_portfolio_value: float = INITIAL_BALANCE

        # Track stop-loss/take-profit set by agent per buy
        self._active_stop_loss: float = 0.0
        self._active_take_profit: float = 0.0

    # ----- OpenEnv public interface ----------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self.portfolio = Portfolio()
        self.step_count = 0
        self.done = False
        self._cumulative_reward = 0.0
        self._actions_taken = []
        self._step_rewards = []
        self._prev_portfolio_value = INITIAL_BALANCE
        self._active_stop_loss = 0.0
        self._active_take_profit = 0.0
        return self._observe()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Execute one step: apply the action, advance the price, return results."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        price = self._current_price()
        info: Dict[str, Any] = {}
        portfolio_val = self.portfolio.value_at(price)
        
        # Calculate risk-compliant trade size (Amount that risks 1% of portfolio)
        history = self._price_history()
        atr_val = _atr(history)
        risk_budget = portfolio_val * RISK_FRACTION
        
        # Stop-loss distance (default to 2*ATR)
        risk_per_share = price - (action.stop_loss if (action.stop_loss and action.stop_loss > 0) else (price - 2 * atr_val if atr_val > 0 else price * 0.98))
        risk_per_share = max(risk_per_share, 0.01)
        
        # Max allowed trade size (USD) to keep risk at 1%
        max_trade_limit = (risk_budget / risk_per_share) * price
        # Also cap by available cash or 100% of portfolio for safety
        max_trade_limit = min(max_trade_limit, portfolio_val * 2.0)

        # ---- Track risk compliance ------------------------------------------
        risk_violated = False
        risk_penalty = 0.0
        compliance_bonus = 0.0
        trade_amount = 0.0

        if action.action == ActionType.BUY:
            desired = action.amount if action.amount is not None else max_trade_limit
            desired = max(0.0, desired)

            # Risk check: amount must not exceed the position size that risks 1%
            if desired > max_trade_limit * 1.05:  # 5% tolerance
                risk_violated = True
                violation_severity = (desired - max_trade_limit) / max_trade_limit
                risk_penalty = -0.3 * min(violation_severity, 3.0)
                self.portfolio.risk_violations += 1
                info["risk_violation"] = True
                info["desired_amount"] = round(desired, 2)
                info["max_trade_limit"] = round(max_trade_limit, 2)
                desired = max_trade_limit

            # Execute buy
            fee = desired * TRANSACTION_FEE_RATE
            total_cost = desired + fee
            if total_cost > self.portfolio.cash:
                desired = self.portfolio.cash / (1 + TRANSACTION_FEE_RATE)
                fee = desired * TRANSACTION_FEE_RATE
                total_cost = desired + fee

            if desired > 0 and price > 0:
                qty = desired / price
                # Update average entry price
                old_value = self.portfolio.holdings * self.portfolio.avg_entry_price
                new_value = qty * price
                total_holdings = self.portfolio.holdings + qty
                if total_holdings > 0:
                    self.portfolio.avg_entry_price = (old_value + new_value) / total_holdings

                self.portfolio.cash -= total_cost
                self.portfolio.holdings += qty
                self.portfolio.total_fees_paid += fee
                self.portfolio.total_trades += 1
                self.portfolio.total_buys += 1
                trade_amount = desired

                # Track stop-loss/take-profit from the agent's action
                atr_val = _atr(self._price_history())
                if action.stop_loss is not None and action.stop_loss > 0:
                    self._active_stop_loss = action.stop_loss
                else:
                    self._active_stop_loss = price - 2 * atr_val  # default ATR-based
                if action.take_profit is not None and action.take_profit > 0:
                    self._active_take_profit = action.take_profit
                else:
                    risk_per = price - self._active_stop_loss
                    self._active_take_profit = price + risk_per * RISK_REWARD_TARGET

                if not risk_violated:
                    self.portfolio.compliant_trades += 1
                    compliance_bonus = 0.05

                self.portfolio.trade_history.append({
                    "step": self.step_count, "type": "BUY",
                    "amount": round(desired, 2), "price": price,
                    "fee": round(fee, 2), "risk_violated": risk_violated,
                    "stop_loss": round(self._active_stop_loss, 2),
                    "take_profit": round(self._active_take_profit, 2),
                })

        elif action.action == ActionType.SELL:
            if self.portfolio.holdings > 0:
                max_sell_value = self.portfolio.holdings * price
                desired_sell = action.amount if action.amount is not None else max_sell_value
                desired_sell = max(0.0, min(desired_sell, max_sell_value))

                if desired_sell > 0:
                    qty_to_sell = desired_sell / price
                    fee = desired_sell * TRANSACTION_FEE_RATE

                    # Record completed round-trip for R-multiple tracking
                    if self.portfolio.avg_entry_price > 0:
                        entry_price = self.portfolio.avg_entry_price
                        exit_price = price
                        risk_per_share = entry_price - self._active_stop_loss if self._active_stop_loss > 0 else _atr(self._price_history()) * 2
                        risk_per_share = max(risk_per_share, 0.01)  # prevent div by 0
                        pnl_per_share = exit_price - entry_price
                        r_multiple = pnl_per_share / risk_per_share

                        self.portfolio.completed_trades.append({
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(exit_price, 2),
                            "risk_per_share": round(risk_per_share, 2),
                            "pnl_per_share": round(pnl_per_share, 2),
                            "r_multiple": round(r_multiple, 4),
                            "qty": round(qty_to_sell, 6),
                            "pnl_dollar": round(pnl_per_share * qty_to_sell, 2),
                            "step": self.step_count,
                        })

                    self.portfolio.holdings -= qty_to_sell
                    self.portfolio.cash += desired_sell - fee
                    self.portfolio.total_fees_paid += fee
                    self.portfolio.total_trades += 1
                    self.portfolio.total_sells += 1
                    trade_amount = desired_sell

                    if not risk_violated:
                        self.portfolio.compliant_trades += 1
                        compliance_bonus = 0.05

                    self.portfolio.trade_history.append({
                        "step": self.step_count, "type": "SELL",
                        "amount": round(desired_sell, 2), "price": price,
                        "fee": round(fee, 2), "risk_violated": risk_violated,
                    })

                    # Reset active stop/take-profit if fully exited
                    if self.portfolio.holdings <= 0.000001:
                        self._active_stop_loss = 0.0
                        self._active_take_profit = 0.0
            else:
                info["warning"] = "No holdings to sell"

        else:
            # Hold — always compliant
            compliance_bonus = 0.02

        # Record action metadata
        self._actions_taken.append({
            "step": self.step_count,
            "action": action.action.value,
            "amount": round(trade_amount, 2),
            "risk_violated": risk_violated,
            "stop_loss": action.stop_loss,
            "take_profit": action.take_profit,
            "reasoning": action.reasoning or "",
        })

        # ---- Advance step ---------------------------------------------------
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        # ---- Compute reward (multi-dimensional) -----------------------------
        new_price = self._current_price()
        new_portfolio_value = self.portfolio.value_at(new_price)

        # PnL reward: normalised portfolio change this step
        pnl_change = (new_portfolio_value - self._prev_portfolio_value) / INITIAL_BALANCE
        pnl_reward = pnl_change * 10  # scale to meaningful range
        self._prev_portfolio_value = new_portfolio_value

        # Total step reward (raw)
        raw_step_reward = round(pnl_reward + risk_penalty + compliance_bonus, 6)
        
        # Squash rewards to strictly [0.001, 0.999] bounds
        def _squash(x: float) -> float:
            v = 1.0 / (1.0 + math.exp(-x))
            return max(0.001, min(0.999, v))

        # Update cumulative with raw reward before squashing
        self._cumulative_reward += raw_step_reward
        self._step_rewards.append(raw_step_reward)

        step_reward = round(_squash(raw_step_reward), 6)
        cumulative_reward_squashed = round(_squash(self._cumulative_reward), 6)

        reward = Reward(
            step_reward=step_reward,
            cumulative_reward=cumulative_reward_squashed,
            risk_penalty=round(_squash(risk_penalty), 6),
            pnl_reward=round(_squash(pnl_reward), 6),
            compliance_bonus=round(_squash(compliance_bonus), 6),
        )

        info["portfolio_value"] = round(new_portfolio_value, 2)
        info["step"] = self.step_count
        info["done"] = self.done

        obs = self._observe()
        return obs, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Return a full snapshot of the environment state."""
        price = self._current_price()
        obs = self._observe()
        return {
            "observation": obs.model_dump(),
            "portfolio": self.portfolio.to_dict(price),
            "step_count": self.step_count,
            "done": self.done,
            "task_id": self.task_id,
            "episode_metrics": self._episode_metrics(),
            "info": {
                "actions_taken": self._actions_taken,
                "initial_balance": INITIAL_BALANCE,
                "risk_fraction": RISK_FRACTION,
                "transaction_fee_rate": TRANSACTION_FEE_RATE,
                "risk_reward_target": RISK_REWARD_TARGET,
            },
        }

    # ----- internal helpers --------------------------------------------------

    def _current_price(self) -> float:
        idx = self._history_offset + self.step_count
        return self._all_prices[idx]

    def _prev_price(self) -> float:
        idx = self._history_offset + max(0, self.step_count - 1)
        return self._all_prices[idx]

    def _price_history(self) -> List[float]:
        """Return prices up to and including the current step."""
        end = self._history_offset + self.step_count + 1
        return self._all_prices[:end]

    def _observe(self) -> Observation:
        """Build an observation with risk management context.

        Includes ATR-based stop-loss, position sizing, and 1:2 R:R target
        so the agent has all the information a professional trader would use.
        """
        history = self._price_history()
        price = history[-1]
        prev_price = history[-2] if len(history) >= 2 else price
        price_change = ((price - prev_price) / prev_price * 100) if prev_price else 0.0

        bb_upper, bb_lower = _bollinger_bands(history, 20)
        atr_val = _atr(history)

        portfolio_val = self.portfolio.value_at(price)
        risk_budget = portfolio_val * RISK_FRACTION  # 1% of portfolio

        # Position sizing from risk management theory:
        # Formula: (Account × Risk%) / (Entry - StopLoss)
        # StopLoss = Entry - 2 × ATR (standard ATR-based stop)
        stop_loss_price = price - 2 * atr_val if atr_val > 0 else price * 0.98
        risk_per_share = price - stop_loss_price
        risk_per_share = max(risk_per_share, 0.01)  # safety

        # Optimal shares = risk_budget / risk_per_share
        # Then position size in USD = shares × price
        optimal_shares = risk_budget / risk_per_share
        suggested_position = optimal_shares * price

        # 1:2 Risk/Reward target
        reward_target = price + risk_per_share * RISK_REWARD_TARGET

        # Max trade size = amount that risks 1% of portfolio
        max_trade = suggested_position

        return Observation(
            current_price=price,
            price_change_pct=round(price_change, 4),
            ema_9=_ema(history, 9),
            ema_21=_ema(history, 21),
            ema_50=_ema(history, 50),
            macd=_macd(history),
            macd_signal=_macd_signal(history),
            rsi=_rsi(history),
            atr=atr_val,
            bollinger_upper=bb_upper,
            bollinger_lower=bb_lower,
            suggested_stop_loss=round(stop_loss_price, 2),
            risk_per_share=round(risk_per_share, 2),
            suggested_position_size=round(suggested_position, 2),
            reward_target=round(reward_target, 2),
            portfolio_value=round(portfolio_val, 2),
            cash_balance=round(self.portfolio.cash, 2),
            position_size=round(self.portfolio.position_value(price), 2),
            position_pct=self.portfolio.position_pct(price),
            unrealized_pnl=self.portfolio.unrealized_pnl(price),
            risk_budget_remaining=round(risk_budget, 2),
            max_trade_size=round(max_trade, 2),
            step_number=self.step_count,
            total_steps=self.max_steps,
        )

    def _episode_metrics(self) -> Dict[str, Any]:
        """Compute summary metrics aligned with risk management theory.

        Key metrics:
        - Expectancy = (WinRate × AvgWin) - (LossRate × AvgLoss)
        - Average R-Multiple: how many R the agent earns per trade
        - Sharpe Ratio: risk-adjusted return
        - Win Rate: % of profitable round-trips
        """
        price = self._current_price()
        portfolio_val = self.portfolio.value_at(price)
        total_return = (portfolio_val - INITIAL_BALANCE) / INITIAL_BALANCE

        # Compute Sharpe-like risk-adjusted return
        if len(self._step_rewards) >= 2:
            mean_r = sum(self._step_rewards) / len(self._step_rewards)
            variance = sum((r - mean_r) ** 2 for r in self._step_rewards) / len(self._step_rewards)
            std_r = math.sqrt(variance) if variance > 0 else 0.001
            sharpe = mean_r / std_r
        else:
            sharpe = 0.0

        # R-multiple metrics from completed trades
        completed = self.portfolio.completed_trades
        if completed:
            r_multiples = [t["r_multiple"] for t in completed]
            avg_r = sum(r_multiples) / len(r_multiples)
            winners = [t for t in completed if t["pnl_dollar"] > 0]
            losers = [t for t in completed if t["pnl_dollar"] <= 0]
            win_rate = len(winners) / len(completed)

            avg_win = sum(t["pnl_dollar"] for t in winners) / len(winners) if winners else 0.0
            avg_loss = abs(sum(t["pnl_dollar"] for t in losers) / len(losers)) if losers else 0.0
            loss_rate = 1.0 - win_rate

            # Expectancy formula: (WinRate × AvgWin) - (LossRate × AvgLoss)
            expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        else:
            avg_r = 0.0
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            expectancy = 0.0

        return {
            "total_return_pct": round(total_return * 100, 4),
            "portfolio_value": round(portfolio_val, 2),
            "total_trades": self.portfolio.total_trades,
            "completed_round_trips": len(completed),
            "risk_violations": self.portfolio.risk_violations,
            "compliant_trades": self.portfolio.compliant_trades,
            "compliance_rate": round(
                self.portfolio.compliant_trades / max(1, self.portfolio.total_trades), 4
            ),
            "total_fees": round(self.portfolio.total_fees_paid, 2),
            "sharpe_ratio": round(sharpe, 4),
            "cumulative_reward": round(self._cumulative_reward, 4),
            # Risk management theory metrics
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_r_multiple": round(avg_r, 4),
            "expectancy": round(expectancy, 2),
            "expectancy_label": "POSITIVE (profitable system)" if expectancy > 0 else "NEGATIVE (losing system)",
        }
