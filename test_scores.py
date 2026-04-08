"""Quick test: verify all grader scores are strictly in (0, 1)."""

from server.tasks import grade_easy, grade_medium, grade_hard, create_env_for_task
from server.models import Action

results = []

# Test Easy - all hold (should get max score)
env = create_env_for_task("easy")
env.reset()
for _ in range(5):
    env.step(Action(action="Hold", amount=None))
r = grade_easy(env)
print(f"Easy (all hold):  score={r['score']}  ok={0.0 < r['score'] < 1.0}")
results.append(r["score"])

# Test Easy - not holding (should get min score)
env = create_env_for_task("easy")
env.reset()
obs, _, _, _ = env.step(Action(action="Buy", amount=100))
for _ in range(4):
    env.step(Action(action="Hold", amount=None))
r = grade_easy(env)
print(f"Easy (buy first):  score={r['score']}  ok={0.0 < r['score'] < 1.0}")
results.append(r["score"])

# Test Medium - no trades (worst case)
env = create_env_for_task("medium")
env.reset()
for _ in range(20):
    env.step(Action(action="Hold", amount=None))
r = grade_medium(env)
print(f"Medium (hold only): score={r['score']}  ok={0.0 < r['score'] < 1.0}")
results.append(r["score"])

# Test Medium - buy every step (potential upper bound)
env = create_env_for_task("medium")
obs = env.reset()
for i in range(20):
    amt = obs.suggested_position_size
    sl = obs.suggested_stop_loss
    tp = obs.reward_target
    obs, _, done, _ = env.step(Action(action="Buy", amount=amt, stop_loss=sl, take_profit=tp))
    if done:
        break
r = grade_medium(env)
print(f"Medium (buy all):  score={r['score']}  ok={0.0 < r['score'] < 1.0}")
results.append(r["score"])

# Test Hard - no trades
env = create_env_for_task("hard")
env.reset()
for _ in range(30):
    env.step(Action(action="Hold", amount=None))
r = grade_hard(env)
print(f"Hard (hold only):  score={r['score']}  ok={0.0 < r['score'] < 1.0}")
results.append(r["score"])

# Test Hard - buy every step
env = create_env_for_task("hard")
obs = env.reset()
for i in range(30):
    amt = obs.suggested_position_size
    sl = obs.suggested_stop_loss
    tp = obs.reward_target
    obs, _, done, _ = env.step(Action(action="Buy", amount=amt, stop_loss=sl, take_profit=tp))
    if done:
        break
r = grade_hard(env)
print(f"Hard (buy all):    score={r['score']}  ok={0.0 < r['score'] < 1.0}")
results.append(r["score"])

print()
all_ok = all(0.0 < s < 1.0 for s in results)
print(f"ALL SCORES STRICTLY IN (0, 1): {all_ok}")
if not all_ok:
    for s in results:
        if not (0.0 < s < 1.0):
            print(f"  FAILED: {s}")
