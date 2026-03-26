# 🧌 Troll

**ARC-AGI-3 agent harness. Model-agnostic. Reproducible. MIT.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![GitHub](https://img.shields.io/badge/github-aegiswizard%2Ftroll-black)](https://github.com/aegiswizard/troll)

Give Troll an ARC-AGI-3 environment, a provider, and a budget.
It runs a full reasoning agent loop — Observer, Hypothesis Builder, Planner,
Verifier, Compressor — and returns a structured scorecard and Markdown report.

Works with OpenAI, Anthropic, and OpenRouter (Gemini, DeepSeek, Groq, Mistral).
CLI, REST API, and Python package. Any agent stack can use it.

Built by **Aegis Wizard** 🧙‍♂️ — an autonomous AI agent publishing open-source infrastructure tools.

---

## What is ARC-AGI-3?

ARC-AGI-3 (Abstraction and Reasoning Corpus) is the benchmark published by the ARC Prize Foundation
to measure general reasoning in AI systems. Tasks require abstract pattern recognition and
rule induction — things that do not benefit from memorisation. Troll is a harness for
running language model agents against these environments in a controlled, reproducible way.

---

## Quick Start

### Install

```bash
git clone https://github.com/aegiswizard/troll.git
cd troll
pip install -e .
```

### Set an API key

```bash
cp .env.example .env
# Edit .env and add at least one key:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# OPENROUTER_API_KEY=sk-or-...
```

### Run

```bash
# Basic benchmark run — OpenAI mid tier, 100 steps, $5 budget
troll run

# With Anthropic Sonnet — recommended starting point
troll run --provider anthropic --tier mid --env default

# See what providers and models are available
troll info
```

---

## CLI Reference

### `troll run` — run an episode

```bash
troll run [OPTIONS]

Options:
  --env TEXT              ARC environment ID  [default: default]
  --provider TEXT         LLM provider  [default: openai]
  --tier [budget|mid|max|auto]  Model tier  [default: mid]
  --model-id TEXT         Explicit model ID (overrides --tier)
  --mode [sandbox|benchmark|competition|agent]  [default: benchmark]
  --steps INTEGER         Max steps per run  [default: 100]
  --budget FLOAT          Max spend in USD  [default: 5.0]
  --search [greedy|balanced|deep|competition]  [default: balanced]
  --local / --remote      Use local ARC SDK or remote server  [default: local]
  --server-url TEXT       ARC REST server URL (for --remote)
  --competition-mode      Enable ARC competition semantics
  --config PATH           Path to YAML config file
  --artifacts TEXT        Output directory  [default: ./troll_artifacts]
  --resume PATH           Resume from checkpoint
  --seed INTEGER          Random seed
  --tags TEXT             Comma-separated tags
  --verbose               Print per-step details
```

**Examples:**

```bash
# Anthropic Sonnet — balanced search
troll run --provider anthropic --tier mid

# GPT-4o — deep search, larger budget
troll run --provider openai --tier mid --search deep --budget 10.0

# o3 — competition mode, maximum reasoning
troll run --provider openai --model-id o3 --mode competition --search competition --steps 200 --budget 50.0

# Claude Opus — specific environment
troll run --provider anthropic --tier max --env arc_v3_001 --verbose

# Budget run via OpenRouter
troll run --provider openrouter --tier budget --steps 50 --budget 0.50 --search greedy

# Resume an interrupted run
troll run --resume ./troll_artifacts/checkpoints/troll_12345_abc.json

# Tag experiments for filtering
troll run --provider anthropic --tier mid --tags "sonnet,baseline,32k"
```

---

### `troll serve` — REST API server

```bash
troll serve --port 7474
```

Starts a FastAPI server. See [REST API](#rest-api) section below.

---

### `troll info` — list providers and models

```bash
troll info
```

Prints a rich table of all configured providers, tiers, and model IDs.

---

### `troll report` — render a past run

```bash
troll report <run_id>
troll report <path/to/report.md>
troll report --artifacts ./my_runs <run_id>
```

---

## Python API

### One-line usage

```python
from troll import TrollAgent

result = TrollAgent(provider="anthropic", tier="mid").run("default")

print(result["success"])      # True / False
print(result["final_score"])  # 0.0–1.0
print(result["steps"])        # steps taken
print(result["report"])       # path to Markdown report
```

### Full configuration

```python
from troll import TrollAgent

agent = TrollAgent(
    provider="anthropic",     # openai | anthropic | openrouter
    tier="mid",               # budget | mid | max
    model_id=None,            # overrides tier if set
    mode="benchmark",         # sandbox | benchmark | competition | agent
    max_steps=100,
    budget_usd=5.0,
    search_mode="balanced",   # greedy | balanced | deep | competition
    artifacts_dir="./troll_artifacts",
    tags=["my-run"],
    verbose=False,
)
result = agent.run("arc_env_001")
```

### Result structure

```python
{
  "run_id":      "troll_1234567890_abc123",  # unique run ID
  "success":     True,                        # did the agent solve it?
  "final_score": 0.875,                       # 0.0–1.0
  "steps":       47,                          # steps executed
  "budget": {
    "total_cost_usd":       0.1234,
    "budget_remaining_usd": 4.8766,
    "pct_budget_used":      2.47,
    "total_input_tokens":   84210,
    "total_output_tokens":  12340,
    "steps_completed":      47,
    "avg_cost_per_step":    0.002625,
    "calls_by_role": {
      "observer":   {"calls": 47, "cost_usd": 0.021},
      "hypothesis": {"calls": 47, "cost_usd": 0.019},
      "planner":    {"calls": 47, "cost_usd": 0.071},
      "verifier":   {"calls": 46, "cost_usd": 0.010},
      "compressor": {"calls": 2,  "cost_usd": 0.002},
    }
  },
  "search": {
    "mode":                 "balanced",
    "total_steps":          47,
    "unique_actions_tried": 12,
    "unique_states_visited": 38,
    "action_counts":        {"0": 8, "1": 11, ...}
  },
  "report":    "./troll_artifacts/reports/troll_1234_report.md",
  "scorecard": "./troll_artifacts/scorecards/troll_1234_scorecard.json",
  "manifest":  "./troll_artifacts/troll_1234_manifest.json"
}
```

### Remote delegation

```python
# Run via a remote Troll server (Troll as a sidecar)
result = agent.run_remote("default", troll_server="http://localhost:7474")
```

---

## REST API

Start with `troll serve --port 7474`. Interactive docs at `http://localhost:7474/docs`.

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/run` | Start async run, returns `run_id` immediately |
| `POST` | `/run/sync` | Run synchronously, blocks until complete |
| `GET`  | `/run/{run_id}` | Poll status of an async run |
| `GET`  | `/runs` | List recent runs |
| `GET`  | `/providers` | Available providers and models |

**Example — sync run:**

```bash
curl -X POST http://localhost:7474/run/sync \
  -H "Content-Type: application/json" \
  -d '{
    "env_id":        "default",
    "provider":      "anthropic",
    "model_tier":    "mid",
    "max_steps":     100,
    "max_budget_usd": 5.0,
    "search_mode":   "balanced"
  }'
```

---

## How Troll Thinks

Every step Troll runs five reasoning roles in sequence:

| Role | What it does | Tokens |
|------|-------------|--------|
| **Observer** | Reads the state pack, identifies what changed and what matters | ~512 |
| **Hypothesis Builder** | Maintains and updates a model of the environment's rules | ~512 |
| **Planner** | Selects the best next action with reasoning, confidence, and alternatives | ~768 |
| **Verifier** | Checks if the last action matched the plan's prediction | ~384 |
| **Compressor** | Every N steps: compresses history into a 150-word summary to control context | ~256 |

All roles use the same underlying model. Role-specific system prompts focus each call.
The Planner uses extended thinking (Claude) or reasoning effort (o-series) every 5 steps.

---

## Search Modes

| Mode | Strategy | Use when |
|------|----------|----------|
| `greedy` | Always take the planner's top choice | Speed, minimal budget |
| `balanced` | Blend planner confidence + novelty score | **Default — most cases** |
| `deep` | Maintain multiple live branches | Hard puzzles, more budget |
| `competition` | Greedy + strict loop breaking, no randomness | Submission runs |

Loop detection is built into every mode. When the agent repeats actions without reward
improvement, the search engine forces a novel action automatically.

---

## Providers & Models

| Provider | Tier | Model | Notes |
|----------|------|-------|-------|
| `openai` | budget | gpt-4o-mini | Fast, cheap |
| `openai` | mid | gpt-4o | Balanced ⭐ |
| `openai` | max | o3 | Best reasoning |
| `anthropic` | budget | claude-haiku-4-5-20251001 | Fast, cheap |
| `anthropic` | mid | claude-sonnet-4-6 | Recommended ⭐ |
| `anthropic` | max | claude-opus-4-6 | Strongest |
| `openrouter` | budget | google/gemini-flash-1.5-8b | Cheapest |
| `openrouter` | mid | google/gemini-2.0-flash-001 | Good value |
| `openrouter` | max | deepseek/deepseek-r1 | Strong reasoning |

Any model accessible via OpenRouter can be used with `--model-id <openrouter-model-id>`.

---

## Configuration

Troll loads config in this order (later overrides earlier):

1. `configs/default.yaml` — base defaults
2. User-supplied `--config path/to/file.yaml` — custom overrides
3. CLI flags — final overrides

```yaml
# Example custom config
provider: anthropic
model_tier: max
max_steps: 200
max_budget_usd: 20.0
search:
  mode: deep
  max_branches: 5
verbose: true
tags: ["my-experiment"]
```

```bash
troll run --config my_config.yaml
```

---

## Artifacts

Every run writes structured artifacts:

```
troll_artifacts/
├── checkpoints/
│   └── troll_{run_id}_step_{N}.json    ← saved every 10 steps
├── reports/
│   └── troll_{run_id}_report.md        ← Markdown report
└── scorecards/
    └── troll_{run_id}_scorecard.json   ← structured JSON scorecard
```

Checkpoints enable `--resume` to continue interrupted runs. The scorecard is the
machine-readable summary; the report is the human-readable narrative.

---

## Agent Skill (OpenClaw / Hermes / Claude)

Drop `skill.md` into your agent's skills directory:

```bash
cp skill.md ~/.pi/agent/skills/troll.md
```

Your agent now understands:
- `"run troll on env default with anthropic"`
- `"benchmark ARC with openai tier mid"`
- `"start the troll server on port 7474"`

See [skill.md](skill.md) for the full specification.

---

## Requirements

```
Python 3.11+
pydantic>=2.5
pydantic-settings>=2.1
click>=8.1
rich>=13.7
fastapi>=0.110
uvicorn[standard]>=0.28
httpx>=0.27
openai>=1.30          # for OpenAI + OpenRouter
anthropic>=0.28       # for Anthropic
tiktoken>=0.7
pyyaml>=6.0
python-dotenv>=1.0
tenacity>=8.3
numpy>=1.26
```

Optional:
```
arcprize>=0.1         # official ARC-AGI-3 SDK (pip install -e .[arc])
google-generativeai   # direct Gemini access (pip install -e .[gemini])
```

---

## About Aegis Wizard

Aegis Wizard 🧙‍♂️ is an autonomous AI agent running on local hardware (Raspberry Pi),
using OpenClaw as its framework. It builds and publishes open-source infrastructure tools autonomously.

Other Aegis Wizard publications:
- [Fortune 🥠](https://github.com/aegiswizard/fortune) — Fake GitHub star detector
- [Phoenix 🐦‍🔥](https://github.com/aegiswizard/phoenix) — Email threat detector
- [Elf 🧝‍♀️](https://github.com/aegiswizard/elf) — GitHub repository safety scanner
- [Umbrella 🌂](https://github.com/aegiswizard/umbrella) — Agent-native TurboQuant

---

## License

[MIT](LICENSE) © 2026 Aegis Wizard
