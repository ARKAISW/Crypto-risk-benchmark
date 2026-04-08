"""Test edge cases for score clamping."""
import math

SCORE_MIN = 0.0001
SCORE_MAX = 0.9999

def _clamp(v):
    return max(SCORE_MIN, min(SCORE_MAX, float(v)))

# If all sub-scores are SCORE_MAX
s = round(0.35*SCORE_MAX + 0.25*SCORE_MAX + 0.25*SCORE_MAX + 0.15*SCORE_MAX, 4)
print(f"All max sub-scores: {s} -> clamped: {_clamp(s)}")

# If all sub-scores are SCORE_MIN
s = round(0.35*SCORE_MIN + 0.25*SCORE_MIN + 0.25*SCORE_MIN + 0.15*SCORE_MIN, 4)
print(f"All min sub-scores: {s} -> clamped: {_clamp(s)}")

# NaN test
print(f"NaN: {_clamp(float('nan'))}")
print(f"Inf: {_clamp(float('inf'))}")
print(f"-Inf: {_clamp(float('-inf'))}")

# Check: what does max/min do with NaN?
print(f"max(0.0001, nan): {max(0.0001, float('nan'))}")
print(f"min(0.9999, nan): {min(0.9999, float('nan'))}")
