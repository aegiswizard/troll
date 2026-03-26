"""
troll/interfaces/rest.py

Troll REST API — one-shot callable by any agent stack.

Endpoints:
  POST /run           — start a full run (async, returns job_id)
  GET  /run/{job_id}  — poll run status / result
  POST /run/sync      — blocking synchronous run (for short budgets)
  GET  /health        — liveness check
  GET  /models        — list available providers/models
  GET  /docs          — auto-docs (via FastAPI)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Troll 🧌 REST API",
    description=(
        "ARC-AGI-3 agent harness by Aegis Wizard 🧙‍♂️\n\n"
        "One-shot callable by any agent stack. MIT License."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)
_jobs: Dict[str, Dict[str, Any]] = {}   # in-memory job store (use Redis for production)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    env_id: str = "default"
    provider: str = "openai"
    model_tier: str = "mid"
    model_id: Optional[str] = None
    mode: str = "benchmark"
    max_steps: int = 50
    max_budget_usd: float = 2.0
    search_mode: str = "balanced"
    arc_local: bool = True
    arc_server_url: str = "http://localhost:8080"
    competition_mode: bool = False
    artifacts_dir: str = "./troll_artifacts"
    tags: List[str] = []
    seed: Optional[int] = None
    verbose: bool = False


class JobStatus(BaseModel):
    job_id: str
    status: str   # "queued" | "running" | "done" | "error"
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[str] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "troll", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Models listing
# ---------------------------------------------------------------------------

@app.get("/models", tags=["info"])
def list_models() -> Dict[str, Any]:
    return {
        "providers": {
            "openai": {
                "tiers": {
                    "budget": "gpt-4o-mini",
                    "mid":    "gpt-4o",
                    "max":    "o3",
                }
            },
            "anthropic": {
                "tiers": {
                    "budget": "claude-haiku-4-5-20251001",
                    "mid":    "claude-sonnet-4-6",
                    "max":    "claude-opus-4-6",
                }
            },
            "openrouter": {
                "tiers": {
                    "budget": "google/gemini-flash-1.5-8b",
                    "mid":    "google/gemini-2.0-flash-001",
                    "max":    "deepseek/deepseek-r1",
                }
            },
        }
    }


# ---------------------------------------------------------------------------
# Async run (non-blocking)
# ---------------------------------------------------------------------------

@app.post("/run", tags=["run"], response_model=JobStatus, status_code=202)
def submit_run(req: RunRequest, background_tasks: BackgroundTasks) -> JobStatus:
    """Submit a run. Returns a job_id to poll."""
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    now = time.time()
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
        "progress": None,
    }
    background_tasks.add_task(_run_job, job_id, req)
    return JobStatus(**_jobs[job_id])


@app.get("/run/{job_id}", tags=["run"], response_model=JobStatus)
def get_run_status(job_id: str) -> JobStatus:
    """Poll run status. Result available when status == 'done'."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatus(**_jobs[job_id])


# ---------------------------------------------------------------------------
# Sync run (blocking — for small budgets / testing)
# ---------------------------------------------------------------------------

@app.post("/run/sync", tags=["run"])
def sync_run(req: RunRequest) -> Dict[str, Any]:
    """
    Blocking run. Returns full result when complete.
    Use only for small max_steps / budget. For long runs use POST /run.
    """
    try:
        result = _execute_run(req)
        return {"status": "done", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# List jobs
# ---------------------------------------------------------------------------

@app.get("/runs", tags=["run"])
def list_runs() -> Dict[str, Any]:
    return {
        "jobs": [
            {"job_id": j["job_id"], "status": j["status"], "created_at": j["created_at"]}
            for j in _jobs.values()
        ]
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_config(req: RunRequest):
    from troll.core.config import RunConfig, SearchConfig
    return RunConfig(
        arc_env_id=req.env_id,
        provider=req.provider,
        model_tier=req.model_tier,
        model_id=req.model_id,
        mode=req.mode,
        max_steps=req.max_steps,
        max_budget_usd=req.max_budget_usd,
        search=SearchConfig(mode=req.search_mode),
        arc_local=req.arc_local,
        arc_server_url=req.arc_server_url,
        arc_competition_mode=req.competition_mode,
        artifacts_dir=req.artifacts_dir,
        tags=req.tags,
        seed=req.seed,
        verbose=req.verbose,
    )


def _execute_run(req: RunRequest) -> Dict[str, Any]:
    from troll.core.orchestrator import Orchestrator
    cfg = _build_config(req)
    return Orchestrator(cfg).run()


def _run_job(job_id: str, req: RunRequest) -> None:
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = time.time()
    try:
        result = _execute_run(req)
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = result
    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)
    finally:
        _jobs[job_id]["finished_at"] = time.time()
