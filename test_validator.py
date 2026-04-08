"""
Simulate what the hackathon validator does:
  1. Reset each task
  2. Run steps (simple agent)
  3. Grade
  4. Check score is STRICTLY in (0, 1)
  
Tests against the deployed HF space.
"""
import requests
import json
import sys

BASE = "https://arkaisw-cryptoriskenv.hf.space"
# BASE = "http://localhost:7860"

def test_task(task_id, max_steps, strategy="hold"):
    print(f"\n=== Testing {task_id} ({strategy}) ===")
    
    # Reset
    r = requests.post(f"{BASE}/reset", json={"task_id": task_id}, timeout=30)
    if r.status_code != 200:
        print(f"  RESET FAILED: {r.status_code} {r.text}")
        return None
    obs = r.json()
    print(f"  Reset OK, price={obs['current_price']}")
    
    # Steps
    for i in range(max_steps):
        if strategy == "hold":
            act = {"action": "Hold", "amount": None}
        elif strategy == "buy_all":
            act = {"action": "Buy", "amount": obs.get("suggested_position_size", 1000),
                   "stop_loss": obs.get("suggested_stop_loss"),
                   "take_profit": obs.get("reward_target")}
        elif strategy == "mixed":
            if i % 3 == 0:
                act = {"action": "Buy", "amount": 800.0}
            elif i % 3 == 2:
                act = {"action": "Sell", "amount": 500.0}
            else:
                act = {"action": "Hold", "amount": None}
        else:
            act = {"action": "Hold", "amount": None}
            
        r = requests.post(f"{BASE}/step", json=act, timeout=30)
        if r.status_code != 200:
            print(f"  STEP {i+1} FAILED: {r.status_code} {r.text}")
            return None
        data = r.json()
        obs = data["observation"]
        done = data["done"]
        if done:
            break
    
    # Grade
    r = requests.post(f"{BASE}/grade", timeout=30)
    if r.status_code != 200:
        print(f"  GRADE FAILED: {r.status_code} {r.text}")
        print(f"  Response body: {r.text[:500]}")
        return None
    
    grade = r.json()
    score = grade["score"]
    print(f"  Score: {score}")
    print(f"  Valid (0 < score < 1): {0.0 < score < 1.0}")
    print(f"  Score == 0.0: {score == 0.0}")
    print(f"  Score == 1.0: {score == 1.0}")
    
    if not (0.0 < score < 1.0):
        print(f"  *** FAIL: Score {score} is OUT OF RANGE! ***")
    
    return score

print("=" * 60)
print("  VALIDATOR SIMULATION - Testing Deployed HF Space")
print("=" * 60)
print(f"  URL: {BASE}")

scores = []

# Test all tasks with different strategies
for task_id, max_steps in [("easy", 5), ("medium", 20), ("hard", 30)]:
    for strategy in ["hold", "buy_all", "mixed"]:
        score = test_task(task_id, max_steps, strategy)
        if score is not None:
            scores.append((task_id, strategy, score))

print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
all_valid = True
for task_id, strategy, score in scores:
    valid = 0.0 < score < 1.0
    if not valid:
        all_valid = False
    status = "✓" if valid else "✗ FAIL"
    print(f"  {task_id:8s} ({strategy:8s}): {score:.6f}  {status}")

print(f"\n  ALL VALID: {all_valid}")
if not all_valid:
    print("  *** SOME SCORES ARE OUT OF RANGE ***")
    sys.exit(1)
