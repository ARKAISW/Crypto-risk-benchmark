"""
FastAPI application exposing CryptoRiskEnv as an OpenEnv-compliant API.

Endpoints (OpenEnv standard):
  POST /reset    — Reset the environment for a given task
  POST /step     — Submit an action and advance one step
  GET  /state    — Retrieve the full environment state

Additional endpoints:
  GET  /tasks    — List available evaluation tasks
  POST /grade    — Grade the completed episode
  GET  /health   — Health check (returns 200)
  GET  /         — Root endpoint (redirects to docs)
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.env import CryptoRiskEnv
from app.models import (
    Action,
    GradeResponse,
    ResetRequest,
    StateResponse,
    StepResponse,
    TaskListResponse,
    Observation,
)
from app.tasks import create_env_for_task, get_task_list, grade_task

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CryptoRiskEnv — OpenEnv",
    description=(
        "An OpenEnv-compliant environment for evaluating LLM agents on "
        "cryptocurrency risk management discipline. Tests market-data parsing, "
        "risk-constrained trading, and profitable decision-making under volatility."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state (single-session for hackathon simplicity)
# ---------------------------------------------------------------------------

_env: CryptoRiskEnv | None = None


def _get_env() -> CryptoRiskEnv:
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call /reset first.",
        )
    return _env


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Root endpoint — redirect to API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "environment": "CryptoRiskEnv", "version": "1.0.0"}


@app.get("/tasks", response_model=TaskListResponse)
def list_tasks():
    """List all available evaluation tasks."""
    return TaskListResponse(tasks=get_task_list())


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest | None = Body(default=None)):
    """Reset the environment for the specified task and return the initial observation.

    If no request body is provided, defaults to the 'easy' task.
    This ensures compatibility with automated validators that ping /reset.
    """
    global _env
    # Handle empty body (validator ping)
    task_id = request.task_id if request else "easy"

    try:
        _env = create_env_for_task(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    obs = _env.reset()
    return obs


@app.post("/step", response_model=StepResponse)
def step(action: Action = Body(...)):
    """Submit an action and advance the environment by one step."""
    env = _get_env()
    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode.",
        )
    try:
        obs, reward, done, info = env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
def state():
    """Retrieve the full current environment state."""
    env = _get_env()
    s = env.state()

    return StateResponse(
        observation=Observation(**s["observation"]),
        portfolio=s["portfolio"],
        step_count=s["step_count"],
        done=s["done"],
        task_id=s["task_id"],
        episode_metrics=s["episode_metrics"],
        info=s["info"],
    )


@app.post("/grade", response_model=GradeResponse)
def grade():
    """Grade the current (completed) episode."""
    env = _get_env()
    if not env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is not done yet. Complete all steps before grading.",
        )
    result = grade_task(env)
    return GradeResponse(**result)


# ---------------------------------------------------------------------------
# Uvicorn entry point for HF Spaces and Docker
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
