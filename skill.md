# 🧌 Troll — ARC-AGI-3 Agent Harness Skill

**Version:** 1.0.0  
**License:** MIT  
**Source:** https://github.com/aegiswizard/troll  
**Compatible with:** OpenClaw · Hermes · Claude agents · Any Python agent · REST API

---

## What This Skill Does

Troll is a reproducible, model-agnostic agent harness for ARC-AGI-3 benchmarks.
It gives any agent a full reasoning loop — Observer, Hypothesis Builder, Planner,
Verifier, and Compressor — with budget tracking, checkpointing, and structured
JSON/Markdown reports on every run.

---

## Trigger Phrases

Your agent should invoke Troll when the user says:

- `"run troll on env [id]"`
- `"benchmark this ARC environment"`
- `"start a troll run with [provider]"`
- `"run ARC-AGI-3 with anthropic / openai / openrouter"`
- `"troll run --env default --provider anthropic"`
- `"show me available troll providers"`
- `"start the troll server"`

---

## Setup

```bash
git clone https://github.com/aegiswizard/troll.git
cd troll
pip install -e .

# Set at least one API key
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENROUTER_API_KEY=sk-or-...
```

---

## CLI Usage

```bash
# Run a benchmark episode — defaults to OpenAI mid tier, 100 steps, $5 budget
troll run

# Specify provider, model tier, and environment
troll run --provider anthropic --tier mid --env default

# Use explicit model ID
troll run --provider openai --model-id o3 --env arc_001

# Aggressive reasoning, larger budget, competition mode
troll run --provider anthropic --tier max --steps 200 --budget 20.0 --search deep --competition-mode

# Budget-conscious run
troll run --provider openrouter --tier budget --steps 50 --budget 1.0 --search greedy

# Save artifacts to a custom directory
troll run --provider anthropic --tier mid --artifacts ./my_runs

# Resume from a checkpoint
troll run --resume ./troll_artifacts/checkpoints/troll_12345_abc.json

# Tag a run for later filtering
troll run --tags "experiment-1,gpt4o,32k-context"

# Verbose per-step output
troll run --provider anthropic --tier mid --verbose

# See all available providers and models
troll info

# Start the REST API server
troll serve --port 7474

# Render a past run report
troll report <run_id>
```

---

## Python API

```python
from troll import TrollAgent

# One-line run — simplest form
result = TrollAgent(provider="anthropic", tier="mid").run("default")

# Full configuration
agent = TrollAgent(
    provider="anthropic",     # openai | anthropic | openrouter
    tier="mid",               # budget | mid | max
    model_id=None,            # explicit model ID (overrides tier)
    mode="benchmark",         # sandbox | benchmark | competition | agent
    max_steps=100,
    budget_usd=5.0,
    search_mode="balanced",   # greedy | balanced | deep | competition
    artifacts_dir="./troll_artifacts",
    tags=["my-experiment"],
    verbose=False,
)
result = agent.run("arc_env_001")

# Result keys:
result["run_id"]       # Unique run identifier
result["success"]      # True / False
result["final_score"]  # 0.0–1.0
result["steps"]        # Steps taken
result["budget"]       # Full budget breakdown dict
result["search"]       # Search engine stats dict
result["report"]       # Path to Markdown report
result["scorecard"]    # Path to JSON scorecard
result["manifest"]     # Path to run manifest

# Delegate to a remote Troll server
result = agent.run_remote("default", troll_server="http://localhost:7474")
```

---

## REST API (troll serve)

```bash
troll serve --port 7474
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/run` | Start async run, returns run_id |
| `POST` | `/run/sync` | Run synchronously, returns full result |
| `GET`  | `/run/{run_id}` | Poll run status |
| `GET`  | `/runs` | List recent runs |
| `GET`  | `/providers` | List available providers |

**Example:**
```bash
curl -X POST http://localhost:7474/run/sync \
  -H "Content-Type: application/json" \
  -d '{"env_id":"default","provider":"anthropic","model_tier":"mid","max_steps":50}'
```

---

## Providers & Models

| Provider | Tier | Model | Notes |
|----------|------|-------|-------|
| `openai` | budget | gpt-4o-mini | Fast, cheap |
| `openai` | mid | gpt-4o | Balanced |
| `openai` | max | o3 | Best reasoning |
| `anthropic` | budget | claude-haiku-4-5-20251001 | Fast |
| `anthropic` | mid | claude-sonnet-4-6 | ⭐ Recommended |
| `anthropic` | max | claude-opus-4-6 | Strongest |
| `openrouter` | budget | google/gemini-flash-1.5-8b | Cheapest |
| `openrouter` | mid | google/gemini-2.0-flash-001 | Good value |
| `openrouter` | max | deepseek/deepseek-r1 | Strong reasoning |

---

## Search Modes

| Mode | Description | Best for |
|------|-------------|----------|
| `greedy` | Always takes planner's top choice | Speed, low budget |
| `balanced` | Blends planner + novelty scoring | Default, most cases |
| `deep` | Maintains multiple branches | Hard puzzles |
| `competition` | Greedy + strict loop breaking | Competition submissions |

---

## Reasoning Council

Every step Troll runs 5 internal reasoning roles:

| Role | What it does |
|------|--------------|
| **Observer** | Reads state, identifies what changed and what matters |
| **Hypothesis Builder** | Maintains and updates a model of environment rules |
| **Planner** | Selects the best next action with reasoning and confidence |
| **Verifier** | Checks if the last action matched expectations |
| **Compressor** | Summarises long histories to keep context tight |

---

## Artifacts Produced

Every run writes to `./troll_artifacts/` (configurable):

```
troll_artifacts/
├── checkpoints/   ← saved every 10 steps, resumable
├── reports/       ← Markdown report per run
└── scorecards/    ← JSON scorecard per run
```

---

## Environment Variables

```bash
# At least one provider key required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=AI...

# Optional Troll defaults
TROLL_DEFAULT_PROVIDER=openai
TROLL_DEFAULT_MODEL_TIER=mid
TROLL_DEFAULT_ARTIFACTS_DIR=./troll_artifacts
```

Copy `.env.example` to `.env` and fill in your keys.
