"""Judge test script — validates all endpoints and grading behavior."""
import requests
import json
import sys

base = "http://localhost:7860"

print("=" * 60)
print("  JUDGE VALIDATION TESTS")
print("=" * 60)

# 1. Health
print("\n=== 1. Health Check ===")
r = requests.get(f"{base}/health")
print(f"  Status: {r.status_code}, Body: {r.json()}")

# 2. Tasks
print("\n=== 2. Tasks List ===")
r = requests.get(f"{base}/tasks")
tasks = r.json()["tasks"]
for t in tasks:
    print(f"  {t['task_id']}: {t['name']} ({t['max_steps']} steps)")
print(f"  Total tasks: {len(tasks)}")

# 3. Reset empty body (validator ping)
print("\n=== 3. Reset (empty body — validator ping) ===")
r = requests.post(f"{base}/reset")
print(f"  Status: {r.status_code}, Has observation: {'current_price' in r.json()}")

# 4. Reset easy
print("\n=== 4. Reset Easy ===")
r = requests.post(f"{base}/reset", json={"task_id": "easy"})
obs = r.json()
print(f"  Price: {obs['current_price']}, RSI: {obs['rsi']}")
print(f"  MaxTrade: {obs['max_trade_size']}, StopLoss: {obs['suggested_stop_loss']}")
print(f"  Observation fields: {len(obs)} fields")
print(f"  Fields: {list(obs.keys())}")

# 5. Step Hold
print("\n=== 5. Step Hold ===")
r = requests.post(f"{base}/step", json={"action": "Hold", "amount": None})
data = r.json()
print(f"  Reward: {data['reward']['step_reward']}, Done: {data['done']}")
print(f"  Reward fields: {list(data['reward'].keys())}")

# 6. State
print("\n=== 6. State ===")
r = requests.get(f"{base}/state")
state = r.json()
print(f"  Step: {state['step_count']}, Portfolio: {state['portfolio']['total_value']}")
print(f"  State fields: {list(state.keys())}")

# 7. Easy full run + grade -> score 1.0
print("\n=== 7. Easy Full Run (All Hold -> 1.0) ===")
requests.post(f"{base}/reset", json={"task_id": "easy"})
for i in range(5):
    requests.post(f"{base}/step", json={"action": "Hold", "amount": None})
g = requests.post(f"{base}/grade").json()
print(f"  Score: {g['score']} (expected ~0.9999)")
assert 0.99 <= g["score"] < 1.0, f"FAIL: Expected ~0.9999, got {g['score']}"

# 8. Easy fail case -> score 0.0
print("\n=== 8. Easy Fail (One Buy -> 0.0) ===")
requests.post(f"{base}/reset", json={"task_id": "easy"})
for i in range(5):
    if i == 2:
        act = {"action": "Buy", "amount": 100}
    else:
        act = {"action": "Hold", "amount": None}
    requests.post(f"{base}/step", json=act)
g = requests.post(f"{base}/grade").json()
print(f"  Score: {g['score']} (expected ~0.0001)")
assert 0.0 < g["score"] <= 0.01, f"FAIL: Expected ~0.0001, got {g['score']}"

# 9. Medium task - compliant trading
print("\n=== 9. Medium Task (Compliant Trading) ===")
requests.post(f"{base}/reset", json={"task_id": "medium"})
for i in range(20):
    if i % 3 == 0:
        act = {"action": "Buy", "amount": 800.0}
    elif i % 3 == 1:
        act = {"action": "Hold", "amount": None}
    else:
        act = {"action": "Sell", "amount": 500.0}
    requests.post(f"{base}/step", json=act)
g = requests.post(f"{base}/grade").json()
print(f"  Score: {g['score']:.4f}")
print(f"  Breakdown: {json.dumps(g['breakdown'], indent=4)}")
assert 0.0 < g["score"] < 1.0

# 10. Medium task - with violations
print("\n=== 10. Medium Task (With Violations) ===")
requests.post(f"{base}/reset", json={"task_id": "medium"})
violations = 0
for i in range(20):
    if i % 4 == 0:
        act = {"action": "Buy", "amount": 50000.0}  # huge amount - should violate
    elif i % 4 == 2:
        act = {"action": "Sell", "amount": 500.0}
    else:
        act = {"action": "Hold", "amount": None}
    r2 = requests.post(f"{base}/step", json=act)
    info = r2.json().get("info", {})
    if info.get("risk_violation"):
        violations += 1
g = requests.post(f"{base}/grade").json()
print(f"  Score: {g['score']:.4f}, Violations: {violations}")
assert 0.0 < g["score"] < 1.0

# 11. Hard task
print("\n=== 11. Hard Task ===")
requests.post(f"{base}/reset", json={"task_id": "hard"})
for i in range(30):
    if i % 5 == 0:
        act = {"action": "Buy", "amount": 900.0}
    elif i % 5 == 3:
        act = {"action": "Sell", "amount": 700.0}
    else:
        act = {"action": "Hold", "amount": None}
    requests.post(f"{base}/step", json=act)
g = requests.post(f"{base}/grade").json()
print(f"  Score: {g['score']:.4f}")
print(f"  Breakdown: {json.dumps(g['breakdown'], indent=4)}")
assert 0.0 < g["score"] < 1.0

# 12. Scores vary (not constant)
print("\n=== 12. Graders Produce Different Scores ===")
scores = set()

requests.post(f"{base}/reset", json={"task_id": "easy"})
for _ in range(5):
    requests.post(f"{base}/step", json={"action": "Hold", "amount": None})
s1 = requests.post(f"{base}/grade").json()["score"]
scores.add(s1)

requests.post(f"{base}/reset", json={"task_id": "easy"})
for _ in range(5):
    requests.post(f"{base}/step", json={"action": "Buy", "amount": 100})
s2 = requests.post(f"{base}/grade").json()["score"]
scores.add(s2)
print(f"  Scores: {scores} (should have >1 distinct value)")
assert len(scores) > 1, f"FAIL: Grader always returns same score: {scores}"

# 13. Reproducibility test — same seed should give same results
print("\n=== 13. Reproducibility Test ===")
requests.post(f"{base}/reset", json={"task_id": "easy"})
obs1 = requests.get(f"{base}/state").json()["observation"]["current_price"]
requests.post(f"{base}/reset", json={"task_id": "easy"})
obs2 = requests.get(f"{base}/state").json()["observation"]["current_price"]
print(f"  Run 1 price: {obs1}, Run 2 price: {obs2}")
assert obs1 == obs2, f"FAIL: Not reproducible! {obs1} != {obs2}"
print("  REPRODUCIBLE ✓")

# 14. Score range check on medium
print("\n=== 14. Score Range Validation ===")
# Pure hold on medium should score low (activity penalized)
requests.post(f"{base}/reset", json={"task_id": "medium"})
for _ in range(20):
    requests.post(f"{base}/step", json={"action": "Hold", "amount": None})
g = requests.post(f"{base}/grade").json()
hold_score = g["score"]
print(f"  Medium (pure hold): {hold_score:.4f}")

# Active compliant trading on medium should score higher
requests.post(f"{base}/reset", json={"task_id": "medium"})
for i in range(20):
    if i % 3 == 0:
        act = {"action": "Buy", "amount": 800.0}
    elif i % 3 == 2:
        act = {"action": "Sell", "amount": 500.0}
    else:
        act = {"action": "Hold", "amount": None}
    requests.post(f"{base}/step", json=act)
g = requests.post(f"{base}/grade").json()
active_score = g["score"]
print(f"  Medium (active): {active_score:.4f}")
print(f"  Active > Hold? {active_score > hold_score} ✓" if active_score > hold_score else f"  WARNING: Hold scored >= Active")

# 15. Test reset clears state
print("\n=== 15. Reset Produces Clean State ===")
requests.post(f"{base}/reset", json={"task_id": "medium"})
for _ in range(5):
    requests.post(f"{base}/step", json={"action": "Buy", "amount": 800.0})
state_before = requests.get(f"{base}/state").json()
requests.post(f"{base}/reset", json={"task_id": "medium"})
state_after = requests.get(f"{base}/state").json()
print(f"  Before reset - Step: {state_before['step_count']}, Portfolio: {state_before['portfolio']['total_value']:.2f}")
print(f"  After reset  - Step: {state_after['step_count']}, Portfolio: {state_after['portfolio']['total_value']:.2f}")
assert state_after["step_count"] == 0, "FAIL: Step count not reset"
assert state_after["portfolio"]["total_value"] == 100000.0, "FAIL: Portfolio not reset"
print("  CLEAN RESET ✓")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED ✓")
print("=" * 60)
