"""
troll/core/config.py
All Pydantic configuration models for Troll 🧌.
Load from YAML, env vars, or code. Single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Model / Provider config
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    id: str
    display_name: str
    multimodal: bool = False
    context_window: int = 128_000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    supports_thinking: bool = False
    thinking_budget: Optional[int] = None
    tier: Literal["budget", "mid", "max"] = "mid"
    extra: Dict[str, Any] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    name: str
    api_key_env: str
    base_url: Optional[str] = None
    models: Dict[str, ModelConfig] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Search config
# ---------------------------------------------------------------------------

class SearchConfig(BaseModel):
    mode: Literal["greedy", "balanced", "deep", "competition"] = "balanced"
    max_branches: int = 3
    novelty_weight: float = 0.4
    progress_weight: float = 0.6
    loop_detection_window: int = 5
    max_depth: int = 3


# ---------------------------------------------------------------------------
# Run config (the main knob users touch)
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    # Identity
    run_id: Optional[str] = None          # auto-generated if None
    tags: List[str] = Field(default_factory=list)

    # Mode
    mode: Literal["sandbox", "benchmark", "competition", "agent"] = "benchmark"

    # Provider / model selection
    provider: str = "openai"
    model_tier: Literal["budget", "mid", "max", "auto"] = "mid"
    model_id: Optional[str] = None         # overrides tier selection

    # Budget
    max_steps: int = 100
    max_budget_usd: float = 5.0
    max_budget_per_step_usd: float = 0.20

    # Search
    search: SearchConfig = Field(default_factory=SearchConfig)

    # ARC environment
    arc_env_id: Optional[str] = None       # specific env/level id
    arc_local: bool = True                  # True = local SDK, False = remote REST
    arc_server_url: str = "http://localhost:8080"
    arc_competition_mode: bool = False

    # Paths
    artifacts_dir: Path = Path("./troll_artifacts")
    resume_checkpoint: Optional[Path] = None

    # Misc
    seed: Optional[int] = None
    verbose: bool = False

    @field_validator("artifacts_dir", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        return Path(v).expanduser().resolve()

    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts_dir / "checkpoints"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def replays_dir(self) -> Path:
        return self.artifacts_dir / "replays"


# ---------------------------------------------------------------------------
# Global Troll settings (env-aware)
# ---------------------------------------------------------------------------

class TrollSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TROLL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys (read from env directly in providers, but expose here for info)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")

    # Default run settings (overridden by CLI or code)
    default_provider: str = "openai"
    default_model_tier: str = "mid"
    default_artifacts_dir: Path = Path("./troll_artifacts")

    model_config = SettingsConfigDict(
        env_prefix="TROLL_",
        env_file=".env",
        extra="ignore",
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_run_config(path: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None) -> RunConfig:
    """Load RunConfig from YAML file + optional overrides dict."""
    data: Dict[str, Any] = {}

    # 1. Default config file
    default_cfg = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    if default_cfg.exists():
        with open(default_cfg) as f:
            data = yaml.safe_load(f) or {}

    # 2. User-supplied config file
    if path and path.exists():
        with open(path) as f:
            user_data = yaml.safe_load(f) or {}
        data.update(user_data)

    # 3. Overrides from CLI / code
    if overrides:
        data.update(overrides)

    return RunConfig(**data)


def load_providers_config(path: Optional[Path] = None) -> Dict[str, ProviderConfig]:
    """Load provider/model config from YAML."""
    default_cfg = path or (Path(__file__).parent.parent.parent / "configs" / "providers.yaml")
    if not default_cfg.exists():
        return {}
    with open(default_cfg) as f:
        raw = yaml.safe_load(f) or {}
    return {k: ProviderConfig(**v) for k, v in raw.get("providers", {}).items()}
