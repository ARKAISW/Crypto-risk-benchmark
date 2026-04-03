"""Test script for CryptoRiskEnv — validates all OpenEnv endpoints."""
import requests
import json
import sys

base = "http://localhost:7860"
errors = []

def test(name, fn):
    try:
        fn()
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        errors.append(name)

print("=" * 60)
print("  CryptoRiskEnv — Automated Endpoint Tests")
print("=" * 60)

# 1. Health check
def t1():
    r = requests.get(f"{base}/health")
    assert r.status_code == 200, f"Status {r.status_code}"
    assert r.json()["status"] == "ok"
test("GET /health", t1)

# 2. List tasks
def t2():
    r = requests.get(f"{base}/tasks")
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    assert len(tasks) == 3, f"Expected 3 tasks, got {len(tasks)}"
    ids = [t["task_id"] for t in tasks]
    assert "easy" in ids and "medium" in ids and "hard" in ids
    for t in tasks:
        print(f"      {t['task_id']}: {t['name']} ({t['max_steps']} steps)")
test("GET /tasks (3 tasks)", t2)

# 3. Reset easy
def t3():
    r = requests.post(f"{base}/reset", json={"task_id": "easy"})
    assert r.status_code == 200
    data = r.json()
    obs = data
    assert "current_price" in obs
    assert "rsi" in obs
    assert "max_trade_size" in obs
    assert "portfolio_value" in obs
    print(f"      Price: ${obs['current_price']:,.2f}, RSI: {obs['rsi']}")
test("POST /reset (easy)", t3)

# 4. Step Hold
def t4():
    requests.post(f"{base}/reset", json={"task_id": "easy"})
    r = requests.post(f"{base}/step", json={"action": "Hold", "amount": None})
    assert r.status_code == 200
    data = r.json()
    assert "observation" in data
    assert "reward" in data
    assert "done" in data
    reward = data["reward"]
    assert "step_reward" in reward
    assert "cumulative_reward" in reward
    assert "risk_penalty" in reward
    print(f"      Reward: {reward['step_reward']:.4f}, Done: {data['done']}")
test("POST /step (Hold)", t4)

# 5. State
def t5():
    requests.post(f"{base}/reset", json={"task_id": "easy"})
    requests.post(f"{base}/step", json={"action": "Hold", "amount": None})
    r = requests.get(f"{base}/state")
    assert r.status_code == 200
    state = r.json()
    assert "observation" in state
    assert "portfolio" in state
    assert "step_count" in state
    assert "done" in state
    assert "task_id" in state
    assert state["step_count"] == 1
    print(f"      Portfolio: ${state['portfolio']['total_value']:,.2f}")
test("GET /state", t5)

# 6. Easy full run + grade
def t6():
    requests.post(f"{base}/reset", json={"task_id": "easy"})
    for i in range(5):
        r = requests.post(f"{base}/step", json={"action": "Hold", "amount": None})
        assert r.status_code == 200
    r = requests.post(f"{base}/grade")
    assert r.status_code == 200
    grade = r.json()
    assert grade["score"] == 1.0, f"Expected 1.0, got {grade['score']}"
    assert 0.0 <= grade["score"] <= 1.0
    print(f"      Score: {grade['score']} — {grade['reason'][:60]}")
test("Easy task: all Hold → score 1.0", t6)

# 7. Easy with wrong action → score 0.0
def t7():
    requests.post(f"{base}/reset", json={"task_id": "easy"})
    for i in range(5):
        if i == 2:
            action = {"action": "Buy", "amount": 100}
        else:
            action = {"action": "Hold", "amount": None}
        r = requests.post(f"{base}/step", json=action)
    r = requests.post(f"{base}/grade")
    grade = r.json()
    assert grade["score"] == 0.0, f"Expected 0.0, got {grade['score']}"
    print(f"      Score: {grade['score']} (correct: non-hold penalized)")
test("Easy task: one Buy → score 0.0", t7)

# 8. Medium with violations
def t8():
    requests.post(f"{base}/reset", json={"task_id": "medium"})
    violations = 0
    for i in range(20):
        if i % 4 == 0:
            action = {"action": "Buy", "amount": 5000.0}  # over limit
        elif i % 4 == 2:
            action = {"action": "Sell", "amount": 500.0}
        else:
            action = {"action": "Hold", "amount": None}
        r = requests.post(f"{base}/step", json=action)
        info = r.json().get("info", {})
        if info.get("risk_violation"):
            violations += 1
    r = requests.post(f"{base}/grade")
    grade = r.json()
    assert 0.0 <= grade["score"] <= 1.0
    print(f"      Score: {grade['score']:.4f}, Violations: {violations}")
    print(f"      Breakdown: {json.dumps(grade.get('breakdown', {}))}")
test("Medium task: mixed + violations", t8)

# 9. Medium with perfect compliance
def t9():
    requests.post(f"{base}/reset", json={"task_id": "medium"})
    for i in range(20):
        if i % 3 == 0:
            action = {"action": "Buy", "amount": 800.0}  # under limit
        elif i % 3 == 1:
            action = {"action": "Hold", "amount": None}
        else:
            action = {"action": "Sell", "amount": 500.0}
        r = requests.post(f"{base}/step", json=action)
    r = requests.post(f"{base}/grade")
    grade = r.json()
    assert 0.0 <= grade["score"] <= 1.0
    print(f"      Score: {grade['score']:.4f} (compliant trading)")
test("Medium task: compliant trading", t9)

# 10. Hard task run
def t10():
    requests.post(f"{base}/reset", json={"task_id": "hard"})
    for i in range(30):
        if i % 5 == 0:
            action = {"action": "Buy", "amount": 900.0}
        elif i % 5 == 3:
            action = {"action": "Sell", "amount": 700.0}
        else:
            action = {"action": "Hold", "amount": None}
        r = requests.post(f"{base}/step", json=action)
    r = requests.post(f"{base}/grade")
    grade = r.json()
    assert 0.0 <= grade["score"] <= 1.0
    print(f"      Score: {grade['score']:.4f}")
    print(f"      Breakdown: {json.dumps(grade.get('breakdown', {}))}")
test("Hard task: mixed trading", t10)

# 11. Empty reset (validator ping)
def t11():
    r = requests.post(f"{base}/reset")
    assert r.status_code == 200
    assert "current_price" in r.json()
test("POST /reset (empty body — validator)", t11)

# 12. Grader scores are different (not constant)
def t12():
    scores = set()
    # Run easy with hold
    requests.post(f"{base}/reset", json={"task_id": "easy"})
    for _ in range(5):
        requests.post(f"{base}/step", json={"action": "Hold", "amount": None})
    s1 = requests.post(f"{base}/grade").json()["score"]
    scores.add(s1)
    
    # Run easy with buy
    requests.post(f"{base}/reset", json={"task_id": "easy"})
    for _ in range(5):
        requests.post(f"{base}/step", json={"action": "Buy", "amount": 100})
    s2 = requests.post(f"{base}/grade").json()["score"]
    scores.add(s2)
    
    assert len(scores) > 1, f"Grader always returns same score: {scores}"
    print(f"      Scores vary: {scores}")
test("Graders produce different scores", t12)

# Summary
print()
print("=" * 60)
if errors:
    print(f"  FAILED: {len(errors)} test(s)")
    for e in errors:
        print(f"    ✗ {e}")
    sys.exit(1)
else:
    print("  ALL 12 TESTS PASSED ✓")
print("=" * 60)
