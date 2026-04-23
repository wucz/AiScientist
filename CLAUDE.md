# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Install dependencies
uv sync --dev

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_store.py

# Run a single test
uv run pytest tests/test_store.py::test_create_job

# Health checks
AISCI_PAPER_DOCTOR_PROFILE=gpt-5.4 uv run aisci paper doctor
uv run aisci mle doctor

# Build Docker images (required before running jobs)
bash docker/build_paper_image.sh   # → aisci-paper:latest
bash docker/build_mle_image.sh     # → aisci-mle:test

# Run a paper reproduction job
uv run aisci --env-file .env paper run \
  --paper-md /abs/path/to/paper.md \
  --image aisci-paper:latest \
  --llm-profile gpt-5.4 \
  --time-limit 24h --wait --tui

# Run an MLE competition job
uv run aisci --env-file .env mle run \
  --zip /abs/path/to/competition.zip \
  --name <slug> \
  --image aisci-mle:test \
  --llm-profile gpt-5.4 \
  --time-limit 12h --wait --tui

# Inspect jobs
uv run aisci jobs list
uv run aisci jobs show <job_id>
uv run aisci logs tail <job_id> --kind main
uv run aisci logs tail <job_id> --kind conversation
uv run aisci artifacts ls <job_id>
uv run aisci tui job <job_id>
uv run aisci export <job_id>
```

## Architecture

### Core Design: File-as-Bus

The system coordinates agents through workspace files on disk rather than in-memory message passing. All plans, code, experiments, and logs are written to `jobs/<job_id>/workspace/agent/`. This makes state persistent, resumable, and auditable.

### Two Tracks

| Track | Input | Goal |
|-------|-------|------|
| `paper` | `--paper-md` / `--zip` | Reproduce an ML paper end-to-end |
| `mle` | `--zip` / `--name` / `--data-dir` | Improve a metric via iterative ML engineering |

Both tracks share the same host-side infrastructure (CLI, job store, Docker runtime) but have separate domain adapters and embedded engines.

### Execution Model

```
CLI (aisci run)
  → JobService.create_job()       # write JobRecord to SQLite
  → JobService.spawn_worker()     # subprocess: python -m aisci_app.worker_main
      → JobRunner.run_job()
          → PaperDomainAdapter.run()  OR  MLEDomainAdapter.run()
              → DockerRuntimeManager  (start container, bind workspace)
              → EmbeddedPaperEngine / EmbeddedMLEEngine  (agent main loop)
```

### Module Map

| Package | Responsibility |
|---------|---------------|
| `aisci_app` | CLI (Typer), JobService, TUI dashboard, worker entry point |
| `aisci_core` | JobSpec/JobRecord models, SQLite store, JobRunner, paths, exporter |
| `aisci_agent_runtime` | LLM client (OpenAI + Azure), token counting, retry logic, tracing |
| `aisci_runtime_docker` | Container lifecycle, workspace mounting, shell execution in containers |
| `aisci_domain_paper` | Paper track adapter, EmbeddedPaperEngine, subagents, Jinja prompts |
| `aisci_domain_mle` | MLE track adapter, EmbeddedMLEEngine, subagents, candidate registry |

### Workspace Layout (File-as-Bus Bus)

```
jobs/<job_id>/
├── workspace/
│   ├── paper/          (paper track: input paper)
│   ├── data/           (mle track: competition data)
│   ├── code/           (mle track: solution code, git-tracked)
│   ├── submission/     (paper track: output code, git-tracked)
│   └── agent/          ← coordination bus
│       ├── paper_analysis/   or  analysis/
│       ├── prioritized_tasks.md
│       ├── plan.md
│       ├── impl_log.md
│       ├── exp_log.md
│       └── final_self_check.{md,json}
├── logs/
├── artifacts/
├── state/
└── export/
```

### LLM Client

`aisci_agent_runtime/llm_client.py` supports two API modes:
- **`ResponsesLLMClient`** — OpenAI Responses API (used for `gpt-5.4`)
- **`CompletionsLLMClient`** — Chat Completions API (used for `glm-5`, `gemini-3-flash`)

Retry budget: up to 2 hours total with exponential backoff (1s–300s). Non-retryable: `ContentPolicyError`, `ContextLengthError`.

### Configuration Files

All YAML configs live in `config/`:
- `llm_profiles.yaml` — LLM backends (`openai` / `azure-openai`), model limits, API mode
- `image_profiles.yaml` — Docker image tags and pull policies
- `paper_subagents.yaml` — Per-subagent step limits and time budgets

Override at runtime via env vars `AISCI_LLM_PROFILE_FILE` / `AISCI_IMAGE_PROFILE_FILE` or CLI flags.

### Key Data Models (`aisci_core/models.py`)

- `JobSpec` — immutable job definition (type, objective, LLM profile, runtime profile, domain-specific spec)
- `RuntimeProfile` — Docker resource limits, GPU assignment, network policy, validation strategy
- `PaperSpec` / `MLESpec` — domain-specific inputs
- `RunPhase` — `ingest → analyze → prioritize → implement → validate → finalize`
- `JobStatus` — `pending → running → succeeded | failed | cancelled`

### Environment Setup

Copy `.env.example` to `.env` and populate:
- **Required**: `OPENAI_API_KEY` (OpenAI) or `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` + `OPENAI_API_VERSION` (Azure)
- **Optional**: `AISCI_OUTPUT_ROOT`, `AISCI_MAX_STEPS`, `AISCI_REMINDER_FREQ`, `HTTP_PROXY`, `HF_TOKEN`
