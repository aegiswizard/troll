"""
troll/interfaces/cli.py

Troll CLI — the primary power-user interface.

Commands:
  troll run       — run a full benchmark / sandbox episode
  troll serve     — start REST API server
  troll info      — show available providers and models
  troll report    — render a past run's report
  troll version   — show version
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()
VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(VERSION, prog_name="troll")
def main() -> None:
    """
    Troll 🧌 — ARC-AGI-3 agent harness by Aegis Wizard 🧙‍♂️

    A reproducible, model-agnostic benchmark + agent operating system.
    MIT License — https://github.com/aegiswizard/troll
    """


# ---------------------------------------------------------------------------
# troll run
# ---------------------------------------------------------------------------

@main.command()
@click.option("--env",        default="default",   help="ARC environment ID", show_default=True)
@click.option("--provider",   default="openai",    help="LLM provider (openai/anthropic/openrouter)", show_default=True)
@click.option("--tier",       default="mid",       type=click.Choice(["budget","mid","max","auto"]), help="Model tier", show_default=True)
@click.option("--model-id",   default=None,        help="Explicit model ID (overrides --tier)")
@click.option("--mode",       default="benchmark", type=click.Choice(["sandbox","benchmark","competition","agent"]), show_default=True)
@click.option("--steps",      default=100,         help="Max steps per run", show_default=True)
@click.option("--budget",     default=5.0,         help="Max spend in USD", show_default=True)
@click.option("--search",     default="balanced",  type=click.Choice(["greedy","balanced","deep","competition"]), show_default=True)
@click.option("--local/--remote", default=True,    help="Use local ARC SDK (--local) or remote server (--remote)")
@click.option("--server-url", default="http://localhost:8080", help="ARC REST server URL (for --remote)")
@click.option("--competition-mode", is_flag=True,  help="Enable ARC competition semantics")
@click.option("--config",     default=None,        type=click.Path(exists=False), help="Path to YAML config file")
@click.option("--artifacts",  default="./troll_artifacts", help="Output directory", show_default=True)
@click.option("--resume",     default=None,        type=click.Path(exists=True), help="Resume from checkpoint path")
@click.option("--seed",       default=None,        type=int, help="Random seed")
@click.option("--tags",       default=None,        help="Comma-separated tags")
@click.option("--verbose",    is_flag=True,        help="Print per-step details")
def run(
    env: str,
    provider: str,
    tier: str,
    model_id: Optional[str],
    mode: str,
    steps: int,
    budget: float,
    search: str,
    local: bool,
    server_url: str,
    competition_mode: bool,
    config: Optional[str],
    artifacts: str,
    resume: Optional[str],
    seed: Optional[int],
    tags: Optional[str],
    verbose: bool,
) -> None:
    """Run a full Troll episode against an ARC environment."""
    from troll.core.config import RunConfig, SearchConfig, load_run_config

    overrides = {
        "arc_env_id":          env,
        "provider":            provider,
        "model_tier":          tier,
        "mode":                mode,
        "max_steps":           steps,
        "max_budget_usd":      budget,
        "arc_local":           local,
        "arc_server_url":      server_url,
        "arc_competition_mode": competition_mode,
        "artifacts_dir":       artifacts,
        "verbose":             verbose,
        "search":              {"mode": search},
    }
    if model_id:
        overrides["model_id"] = model_id
    if seed is not None:
        overrides["seed"] = seed
    if resume:
        overrides["resume_checkpoint"] = resume
    if tags:
        overrides["tags"] = [t.strip() for t in tags.split(",")]

    cfg = load_run_config(Path(config) if config else None, overrides)

    from troll.core.orchestrator import Orchestrator
    result = Orchestrator(cfg).run()

    console.print("\n[bold]Summary:[/bold]")
    console.print_json(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# troll serve
# ---------------------------------------------------------------------------

@main.command()
@click.option("--host", default="0.0.0.0",  help="Bind host", show_default=True)
@click.option("--port", default=7474,        help="Bind port", show_default=True)
@click.option("--reload", is_flag=True,      help="Auto-reload on code changes")
def serve(host: str, port: int, reload: bool) -> None:
    """Start the Troll REST API server."""
    try:
        import uvicorn  # type: ignore[import]
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[standard][/red]")
        sys.exit(1)

    console.print(f"[green]Starting Troll REST API at http://{host}:{port}[/green]")
    uvicorn.run(
        "troll.interfaces.rest:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# troll info
# ---------------------------------------------------------------------------

@main.command()
def info() -> None:
    """Show available providers, models, and environment info."""
    from troll.providers.router import PROVIDER_REGISTRY, OpenAIProvider, AnthropicProvider

    table = Table(title="Troll 🧌 Providers & Models", show_lines=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Tier", style="green")
    table.add_column("Model ID", style="white")
    table.add_column("Notes", style="dim")

    rows = [
        ("openai",     "budget", "gpt-4o-mini",             "Fast, cheap"),
        ("openai",     "mid",    "gpt-4o",                  "Balanced"),
        ("openai",     "max",    "o3",                      "Best reasoning"),
        ("anthropic",  "budget", "claude-haiku-4-5-20251001","Fast Haiku"),
        ("anthropic",  "mid",    "claude-sonnet-4-6",       "Sonnet — recommended"),
        ("anthropic",  "max",    "claude-opus-4-6",         "Opus — strongest"),
        ("openrouter", "budget", "google/gemini-flash-1.5-8b","Cheapest"),
        ("openrouter", "mid",    "google/gemini-2.0-flash-001","Good value"),
        ("openrouter", "max",    "deepseek/deepseek-r1",    "Strong reasoning"),
    ]
    for provider, tier, model, note in rows:
        table.add_row(provider, tier, model, note)

    console.print(table)
    console.print("\n[dim]Set API keys in env: OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY[/dim]")
    console.print("[dim]Full docs: https://github.com/aegiswizard/troll[/dim]")


# ---------------------------------------------------------------------------
# troll report
# ---------------------------------------------------------------------------

@main.command()
@click.argument("run_id_or_path")
@click.option("--artifacts", default="./troll_artifacts", help="Artifacts directory")
def report(run_id_or_path: str, artifacts: str) -> None:
    """Print or render a past run's report."""
    path = Path(run_id_or_path)

    # Try as a direct file path first
    if path.exists() and path.suffix == ".md":
        console.print(path.read_text())
        return

    # Try as a run ID
    reports_dir = Path(artifacts) / "reports"
    candidates = list(reports_dir.glob(f"{run_id_or_path}*_report.md"))
    if candidates:
        console.print(candidates[0].read_text())
    else:
        console.print(f"[red]No report found for: {run_id_or_path}[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
