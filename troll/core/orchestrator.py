"""
troll/core/orchestrator.py

The Orchestrator runs the full agent loop for one episode.

Lifecycle:
  1. bind scorecard + env + memory
  2. reset env → initial state pack
  3. council → plan → search → execute → verify → checkpoint
  4. repeat until done / budget / steps exhausted
  5. finalise scorecard + emit report
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from troll.artifacts.manager import CheckpointManager, Reporter, Scorecard, write_manifest
from troll.core.budgets import BudgetTracker
from troll.core.config import RunConfig
from troll.env.arc_adapter import TrollEnv, make_env
from troll.memory.store import (
    EpisodicEntry,
    EpisodicMemory,
    HypothesisMemory,
    ProceduralMemory,
    RunMetadata,
)
from troll.perception.frame_parser import FrameParser, Summarizer
from troll.providers.router import ProviderRouter
from troll.reasoning.council import ReasoningCouncil
from troll.search.engine import SearchEngine

console = Console()


class Orchestrator:
    """
    Top-level run conductor.
    One Orchestrator = one run = one scorecard.
    """

    def __init__(self, config: RunConfig) -> None:
        self._cfg = config
        self._run_id = config.run_id or f"troll_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        # Ensure output dirs exist
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        config.reports_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute a full agent run. Returns summary dict."""
        cfg = self._cfg

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Initialising Troll 🧌…", total=cfg.max_steps)

            # ----- Setup -------------------------------------------------
            budget = BudgetTracker(
                max_budget_usd=cfg.max_budget_usd,
                max_per_step_usd=cfg.max_budget_per_step_usd,
                max_steps=cfg.max_steps,
            )
            router = ProviderRouter(
                provider_name=cfg.provider,
                model_tier=cfg.model_tier,
                model_id=cfg.model_id,
                budget=budget,
            )
            episodic = EpisodicMemory()
            hypotheses = HypothesisMemory()
            procedural = ProceduralMemory()
            parser = FrameParser()
            summarizer = Summarizer()
            council = ReasoningCouncil(
                router=router,
                episodic=episodic,
                hypotheses=hypotheses,
                procedural=procedural,
                compress_every=20,
            )
            search = SearchEngine(
                mode=cfg.search.mode,
                max_branches=cfg.search.max_branches,
                novelty_weight=cfg.search.novelty_weight,
                progress_weight=cfg.search.progress_weight,
                loop_detection_window=cfg.search.loop_detection_window,
            )
            checkpoint_mgr = CheckpointManager(self._run_id, cfg.checkpoints_dir)
            env_id = cfg.arc_env_id or "default"
            scorecard = Scorecard(self._run_id, env_id, cfg.artifacts_dir)
            reporter = Reporter(self._run_id, cfg.reports_dir)
            run_meta = RunMetadata(
                run_id=self._run_id,
                env_id=env_id,
                provider=cfg.provider,
                model_id=router.model_id,
                mode=cfg.mode,
                max_budget_usd=cfg.max_budget_usd,
                max_steps=cfg.max_steps,
                search_mode=cfg.search.mode,
                scorecard_id=self._run_id,
                seed=cfg.seed,
                tags=cfg.tags,
            )

            console.print(Panel(
                f"[bold green]Troll 🧌 v1.0.0[/bold green]\n"
                f"Made with ❤️ by Aegis Wizard 🧙‍♂️\n\n"
                f"Run ID:   [cyan]{self._run_id}[/cyan]\n"
                f"Env:      [yellow]{env_id}[/yellow]\n"
                f"Provider: [magenta]{cfg.provider}[/magenta] / [white]{router.model_id}[/white]\n"
                f"Mode:     [blue]{cfg.mode}[/blue]\n"
                f"Budget:   ${cfg.max_budget_usd:.2f} / {cfg.max_steps} steps",
                title="[bold]Starting Run[/bold]",
            ))

            # ----- Resume from checkpoint? --------------------------------
            start_step = 0
            if cfg.resume_checkpoint:
                cp = checkpoint_mgr.load_latest()
                if cp:
                    start_step = cp.get("step", 0)
                    console.print(f"[yellow]Resuming from step {start_step}[/yellow]")

            # ----- Build environment ------------------------------------
            env: TrollEnv = make_env(
                env_id=env_id,
                local=cfg.arc_local,
                server_url=cfg.arc_server_url,
                competition_mode=cfg.arc_competition_mode,
            )

            # ----- Reset ------------------------------------------------
            obs = env.reset()
            parser.reset()
            pack = parser.parse(obs)
            pack = summarizer.summarize(pack)

            final_score = 0.0
            success = False
            done = False
            step = start_step

            # ----- Main loop --------------------------------------------
            try:
                while not done:
                    if budget.over_total():
                        console.print("[red]Budget exhausted — stopping run.[/red]")
                        break
                    if budget.over_steps():
                        console.print("[yellow]Step limit reached — stopping run.[/yellow]")
                        break

                    progress.update(task, description=f"Step {step} | ${budget._total_cost:.4f}", advance=1)

                    # Council reasoning
                    council_output = council.run(pack, step)
                    plan = council_output.plan

                    # Search selects final action
                    final_action = search.select_action(
                        planner_action=plan.chosen_action,
                        planner_alternatives=plan.alternatives,
                        available_actions=pack.action_space,
                        episodic=episodic,
                        reward=pack.reward,
                    )

                    if cfg.verbose:
                        console.print(
                            f"[dim]Step {step}: action={final_action} "
                            f"confidence={plan.confidence:.0%} | {plan.reasoning[:60]}…[/dim]"
                        )

                    # Execute
                    step_result = env.step(final_action, reasoning=plan.reasoning)
                    pack = parser.parse(step_result.observation)
                    pack = summarizer.summarize(pack)
                    done = step_result.done
                    reward = step_result.reward

                    # Track state hash for novelty
                    state_hash = hashlib.md5(
                        str(pack.grid).encode()
                    ).hexdigest()
                    search.record_step(final_action, state_hash)

                    # Was there progress?
                    prev_entries = episodic.recent(1)
                    prev_reward = prev_entries[0].reward if prev_entries else 0.0
                    improved = reward > prev_reward

                    # Record episodic memory
                    episodic.record(EpisodicEntry(
                        step=step,
                        action=final_action,
                        action_params={},
                        state_summary=pack.current_summary,
                        delta_summary=pack.delta_summary,
                        reward=reward,
                        progress_improved=improved,
                        reasoning_note=plan.reasoning[:100],
                        cost_usd=0.0,  # per-step cost from budget tracker
                        latency_s=step_result.latency_s,
                    ))

                    # Scorecard
                    scorecard.record_step(step, reward, done, final_action, plan.reasoning[:60])

                    # Checkpoint every 10 steps
                    if step % 10 == 0:
                        checkpoint_mgr.save(step, episodic, hypotheses, budget)

                    if done and reward > 0:
                        success = True
                        final_score = float(reward)

                    budget.mark_step_complete()
                    step += 1

            except KeyboardInterrupt:
                console.print("[yellow]\nInterrupted by user.[/yellow]")
            finally:
                env.close()

            # ----- Finalise ------------------------------------------
            run_meta.finish()
            scorecard.finalise(final_score, success, meta=budget.summary())
            for tag in cfg.tags:
                scorecard.add_tag(tag)

            report_path = reporter.write(
                run_meta=run_meta,
                scorecard=scorecard,
                budget=budget,
                episodic=episodic,
                hypotheses=hypotheses,
                search_stats=search.stats(),
            )
            manifest_path = write_manifest(
                run_id=self._run_id,
                artifacts_dir=cfg.artifacts_dir,
                meta={
                    "env_id": env_id,
                    "provider": cfg.provider,
                    "model": router.model_id,
                    "mode": cfg.mode,
                    "final_score": final_score,
                    "success": success,
                    "steps": step,
                    "report": str(report_path),
                    "scorecard": str(scorecard.path),
                },
            )

            result = {
                "run_id": self._run_id,
                "success": success,
                "final_score": final_score,
                "steps": step,
                "budget": budget.summary(),
                "search": search.stats(),
                "report": str(report_path),
                "scorecard": str(scorecard.path),
                "manifest": str(manifest_path),
            }

            if success:
                console.print(Panel(
                    f"[bold green]✅ SUCCESS![/bold green]\n"
                    f"Score: {final_score:.3f}  Steps: {step}  "
                    f"Cost: ${budget._total_cost:.4f}",
                    title="Run Complete",
                ))
            else:
                console.print(Panel(
                    f"[yellow]Run finished.[/yellow]\n"
                    f"Steps: {step}  Cost: ${budget._total_cost:.4f}\n"
                    f"Report: {report_path}",
                    title="Run Complete",
                ))

            return result
