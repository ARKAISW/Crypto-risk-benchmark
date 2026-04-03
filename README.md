---
title: Crypto Risk Env
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags: [openenv]
---
# CryptoRiskEnv — OpenEnv for LLM Risk Management Evaluation

<p align="center">
  <strong>Can your AI agent trade profitably while obeying strict risk management rules?</strong>
  <br>
  An OpenEnv-compliant benchmark for evaluating LLM agents on real-world<br>
  cryptocurrency risk management discipline, position sizing, and market analysis.
</p>

---

## 🎯 Why This Matters

Financial institutions lose billions annually from poor risk management — and as LLMs are integrated into trading systems, the stakes only grow. A model that can analyze markets but ignores position-sizing rules is **dangerous** in production.

**CryptoRiskEnv** fills a critical gap: it tests whether AI agents can simultaneously:

| Capability | Real-World Importance |
|---|---|
| **Parse complex market data** | Trading systems require exact numerical interpretation — a misread indicator can trigger wrong trades |
| **Obey risk constraints under pressure** | The 1% risk-per-trade rule is an industry standard. Breaking it once can wipe out months of gains |
| **Make profitable decisions** | Using EMA crossovers, MACD, RSI, Bollinger Bands — the same tools human traders use |
| **Balance aggression with discipline** | The hardest skill: knowing when to trade big vs. when to sit out |

Unlike static benchmarks, CryptoRiskEnv tests agents in a **multi-step, stateful loop** where each decision affects future outcomes — just like real trading.

---

## 🧠 Theory & Motivation: The Math of Risk Management

This environment evaluates whether agents understand that **trading is math, not magic or gambling**. Professional traders don't need a 90% win rate; in fact, many legends (like Mark Minervini or Jesse Livermore) win less than 40% of their trades. They succeed through strict adherence to three risk management formulas:

### 1. Risk/Reward Ratio
Amateur traders focus on how *often* they win. Professionals focus on how *much* they make when they are right. 
- If your risk/reward is 1:2 (risking $1 to make $2), you only need to be right **33% of the time** to break even. 
- **The Agent's Test:** Can the agent target a 2:1 R:R, setting take-profits that are 2x the distance of their stop-losses?

### 2. Expectancy
Expectancy reveals if a trading system has a true statistical edge over hundreds of trades.
- **Formula:** `(Win Rate × Average Win) - (Loss Rate × Average Loss)`
- **The Agent's Test:** In the Hard task, the agent must achieve a positive mathematical expectancy in a volatile market with zero drift.

### 3. Position Sizing
This is the most critical formula for survival. If you risk 10% per trade, 7 losses cut your account in half (requiring a 100% gain to recover). At 1% risk, a 10-trade losing streak only costs ~10%.
- **Formula:** `Position Size = (Account × 1%) / (Entry - StopLoss)`
- **The Agent's Test:** The environment enforces a strict 1% risk limit. The agent must dynamically size its trades based on ATR (Average True Range) volatility without violating this constraint.

> **"You don't need to win often; you need to win big when right and lose small when wrong."**

---

## 📐 OpenEnv Specification

CryptoRiskEnv implements the full [OpenEnv](https://github.com/meta-pytorch/OpenEnv) standard:

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Initialize an episode for a specific task |
| `/step` | POST | Submit an action and advance the simulation |
| `/state` | GET | Retrieve a full state snapshot |
| `/tasks` | GET | List available evaluation tasks |
| `/grade` | POST | Grade a completed episode |
| `/health` | GET | Health check |

All schemas are defined with **strict Pydantic models** and documented in `openenv.yaml`.

---

## 📊 Observation Space

Each step returns a rich JSON observation with **17 fields** across 4 categories:

### Price Data
| Field | Type | Description |
|---|---|---|
| `current_price` | float | Current asset price (USD) |
| `price_change_pct` | float | Price change from previous step (%) |

### Technical Indicators
| Field | Type | Description |
|---|---|---|
| `ema_9` | float | 9-period EMA (fast signal) |
| `ema_21` | float | 21-period EMA (medium signal) |
| `ema_50` | float | 50-period EMA (slow trend) |
| `macd` | float | MACD line (EMA-12 minus EMA-26) |
| `macd_signal` | float | MACD signal line (9-period EMA of MACD) |
| `rsi` | float | Relative Strength Index (0–100) |
| `atr` | float | Average True Range (volatility) |
| `bollinger_upper` | float | Upper Bollinger Band (+2σ) |
| `bollinger_lower` | float | Lower Bollinger Band (-2σ) |

### Portfolio Context
| Field | Type | Description |
|---|---|---|
| `portfolio_value` | float | Total portfolio value (USD) |
| `cash_balance` | float | Available cash (USD) |
| `position_size` | float | Current position value (USD) |
| `position_pct` | float | Position as % of portfolio |
| `unrealized_pnl` | float | Unrealized P&L on open position |

### Risk Context
| Field | Type | Description |
|---|---|---|
| `risk_budget_remaining` | float | Max allowed trade size (USD) |
| `max_trade_size` | float | = 1% of portfolio value |
| `step_number` | int | Current step in episode |
| `total_steps` | int | Total steps in episode |

### Example Observation
```json
{
  "current_price": 50432.15,
  "price_change_pct": -0.85,
  "ema_9": 50102.33,
  "ema_21": 49876.45,
  "ema_50": 49345.67,
  "macd": 225.88,
  "macd_signal": 180.42,
  "rsi": 42.3,
  "atr": 1245.89,
  "bollinger_upper": 52100.50,
  "bollinger_lower": 48200.30,
  "portfolio_value": 100000.00,
  "cash_balance": 100000.00,
  "position_size": 0.00,
  "position_pct": 0.0,
  "unrealized_pnl": 0.0,
  "risk_budget_remaining": 1000.00,
  "max_trade_size": 1000.00,
  "step_number": 0,
  "total_steps": 20
}
```

---

## 🎮 Action Space

Actions are discrete, submitted as JSON:

```json
{"action": "Buy",  "amount": 950.00, "reasoning": "RSI oversold, EMA bullish cross"}
{"action": "Sell", "amount": 500.00, "reasoning": "RSI overbought, taking profit"}
{"action": "Hold", "amount": null,   "reasoning": "No clear signal, preserving capital"}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `action` | string | ✅ | `"Buy"`, `"Sell"`, or `"Hold"` |
| `amount` | float/null | ✅ | Dollar amount to trade (must be ≤ 1% of portfolio) |
| `reasoning` | string | Optional | Agent's reasoning (for evaluation, not scoring) |

### Risk Rules
- **1% Rule**: No single trade can risk more than 1% of portfolio value
- **Transaction Fee**: 0.1% applied to every Buy/Sell
- **Violation Penalty**: Exceeding the risk limit incurs proportional penalties

---

## 🏋️ Reward Function

The reward provides **rich, multi-dimensional signals** at every step:

| Component | Description |
|---|---|
| `pnl_reward` | Portfolio change scaled to meaningful range |
| `risk_penalty` | Proportional penalty for risk violations (up to -0.9) |
| `compliance_bonus` | Small bonus for risk-compliant trades (+0.05) or holds (+0.02) |

```json
{
  "step_reward": 0.052,
  "cumulative_reward": 0.318,
  "risk_penalty": 0.0,
  "pnl_reward": 0.032,
  "compliance_bonus": 0.02
}
```

This design ensures the agent receives **continuous feedback** — not just a binary signal at episode end.

---

## 📋 Tasks & Grading

### Task 1 — Easy: Market Data Parsing & Hold
| Property | Value |
|---|---|
| **Steps** | 5 |
| **Volatility** | Low (1%) |
| **Objective** | Parse observation data correctly and hold every step |
| **Grader** | Binary — `1.0` if all actions are Hold, `0.0` otherwise |
| **Expected Baseline** | `1.0` (trivial for any capable model) |

### Task 2 — Medium: Risk-Constrained Active Trading
| Property | Value |
|---|---|
| **Steps** | 20 |
| **Volatility** | Moderate (2.5%) |
| **Objective** | Trade actively while strictly obeying the 1% risk limit |
| **Grader** | Multi-factor: 40% risk compliance + 30% trading activity + 30% PnL |
| **Expected Baseline** | `0.55–0.75` |

The medium grader **penalizes pure holders** — agents must actually trade to score well. This ensures the task genuinely tests risk-constrained trading, not avoidance.

### Task 3 — Hard: Profitable Trading Under Extreme Volatility
| Property | Value |
|---|---|
| **Steps** | 30 |
| **Volatility** | High (4.5%) |
| **Objective** | Maximize risk-adjusted returns in extreme conditions |
| **Grader** | Multi-factor: 30% Sharpe ratio + 30% PnL + 25% compliance + 15% trade quality |
| **Expected Baseline** | `0.35–0.55` |

The hard task uses a **Sharpe-ratio-based evaluation** — rewarding consistent, risk-adjusted returns over volatile raw gains. This genuinely challenges frontier models.

---

## 🚀 Quick Start

### 1. Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 7860

# Server is now at http://localhost:7860
```

### 2. Run with Docker

```bash
# Build
docker build -t crypto-risk-env .

# Run the server
docker run -p 7860:7860 crypto-risk-env

# Server is now at http://localhost:7860
```

### 3. Run the Baseline Inference

```bash
# Set environment variables
export OPENAI_API_KEY="your-groq-or-openai-key"
export API_BASE_URL="https://api.groq.com/openai/v1"   # Groq (free tier)
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your-hf-token"
export ENV_BASE_URL="http://localhost:7860"

# Run inference
python inference.py
```

### Expected Baseline Output (Llama 3.3 70B via Groq)
```
============================================================
  FINAL RESULTS
============================================================
  [Easy  ] Market Data Parsing & Hold                  → 1.0000
  [Medium] Risk-Constrained Active Trading             → 0.6500
  [Hard  ] Profitable Trading Under Extreme Volatility → 0.4200

  Average Score: 0.6900
  Total Time:   35.2s
============================================================
```

---

## 📁 Project Structure

```
├── app/                    # Core environment package
│   ├── __init__.py
│   ├── main.py             # FastAPI endpoints (OpenEnv API)
│   ├── models.py           # Pydantic schemas (Observation, Action, Reward)
│   ├── env.py              # CryptoRiskEnv simulation engine
│   └── tasks.py            # Task configs & multi-factor graders
├── inference.py            # Baseline LLM inference script
├── openenv.yaml            # OpenEnv specification manifest
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build (HF Spaces ready)
└── README.md               # This file
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | API key for the LLM provider |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Model identifier |
| `HF_TOKEN` | *(optional)* | Hugging Face token (fallback for API key) |
| `ENV_BASE_URL` | `http://localhost:7860` | CryptoRiskEnv server URL |

---

## 🧪 API Examples

### Reset an episode
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

### Take a step
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action": "Hold", "amount": null}}'
```

### Get current state
```bash
curl http://localhost:7860/state
```

### Grade the episode
```bash
curl -X POST http://localhost:7860/grade
```

### List tasks
```bash
curl http://localhost:7860/tasks
```

---

## 🏗️ Architecture & Design Decisions

### Why Risk Management as the Core Focus?

Most trading benchmarks evaluate **profitability** alone. But in institutional finance, the primary evaluation criterion is **risk-adjusted returns** — a strategy that makes 50% but risks blowing up is worse than one making 10% consistently.

CryptoRiskEnv uniquely tests:
1. **Can the agent read its own risk budget?** (observation includes `max_trade_size`)
2. **Will it obey constraints even when "obvious" opportunities exist?**
3. **Can it balance exploration (trading) with constraint satisfaction?**

### Why Crypto?

- Crypto markets exhibit **higher volatility** than equities, making risk management harder
- The 1% risk rule is more frequently tested when prices swing 4-5% per step
- Technical indicators behave differently under extreme volatility, testing true comprehension

### Reward Design Philosophy

The reward is **dense** (every step), **multi-dimensional** (PnL + compliance + penalty), and **proportional** (bigger violations = bigger penalties). This gives training algorithms rich gradient signal while remaining interpretable.

---

## 📜 License

MIT — Built for the Scaler School of Technology × Meta OpenEnv Hackathon 2026.
