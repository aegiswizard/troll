"""
troll/interfaces/agent_wrapper.py

Minimal one-shot API for external agent stacks.

Usage:
    from troll import TrollAgent
    result = TrollAgent(provider="anthropic", tier="mid").run("default")

Or via the REST API:
    POST http://localhost:7474/run/sync
    {"env_id": "default", "provider": "anthropic", "model_tier": "mid"}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


class TrollAgent:
    """
    Minimal wrapper — the "one shot get and use" interface.

    Parameters
    ----------
    provider     : "openai" | "anthropic" | "openrouter"
    tier         : "budget" | "mid" | "max" | "auto"
    model_id     : explicit model string (overrides tier)
    mode         : "sandbox" | "benchmark" | "competition" | "agent"
    max_steps    : maximum agent steps
    budget_usd   : maximum spend in USD
    search_mode  : "greedy" | "balanced" | "deep" | "competition"
    artifacts_dir: where to save scorecards/reports/checkpoints
    tags         : list of string tags for this run
    verbose      : print per-step output
    """

    def __init__(
        self,
        provider: str = "openai",
        tier: str = "mid",
        model_id: Optional[str] = None,
        mode: str = "benchmark",
        max_steps: int = 100,
        budget_usd: float = 5.0,
        search_mode: str = "balanced",
        artifacts_dir: str = "./troll_artifacts",
        tags: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> None:
        self._provider = provider
        self._tier = tier
        self._model_id = model_id
        self._mode = mode
        self._max_steps = max_steps
        self._budget_usd = budget_usd
        self._search_mode = search_mode
        self._artifacts_dir = artifacts_dir
        self._tags = tags or []
        self._verbose = verbose

    def run(self, env_id: str = "default", **kwargs: Any) -> Dict[str, Any]:
        """
        Run a full episode against the given ARC environment.

        Returns a dict with:
          run_id, success, final_score, steps, budget, report, scorecard
        """
        from troll.core.config import RunConfig, SearchConfig
        from troll.core.orchestrator import Orchestrator

        cfg = RunConfig(
            arc_env_id=env_id,
            provider=self._provider,
            model_tier=self._tier,
            model_id=self._model_id,
            mode=self._mode,
            max_steps=self._max_steps,
            max_budget_usd=self._budget_usd,
            search=SearchConfig(mode=self._search_mode),
            artifacts_dir=self._artifacts_dir,
            tags=self._tags,
            verbose=self._verbose,
            **{k: v for k, v in kwargs.items() if k in RunConfig.model_fields},
        )
        return Orchestrator(cfg).run()

    def run_remote(
        self,
        env_id: str = "default",
        troll_server: str = "http://localhost:7474",
        sync: bool = True,
    ) -> Dict[str, Any]:
        """
        Delegate execution to a remote Troll REST server.
        Useful when Troll runs as a sidecar service.
        """
        import httpx

        payload = {
            "env_id": env_id,
            "provider": self._provider,
            "model_tier": self._tier,
            "model_id": self._model_id,
            "mode": self._mode,
            "max_steps": self._max_steps,
            "max_budget_usd": self._budget_usd,
            "search_mode": self._search_mode,
            "tags": self._tags,
            "verbose": self._verbose,
        }

        client = httpx.Client(timeout=600)
        endpoint = "/run/sync" if sync else "/run"
        resp = client.post(f"{troll_server}{endpoint}", json=payload)
        resp.raise_for_status()
        return resp.json()
