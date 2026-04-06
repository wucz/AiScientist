<p align="center">
  <img src="assets/readme/logo.svg" alt="AiScientist logo" width="104" />
</p>

<h1 align="center">AiScientist: A File-as-Bus Research Lab</h1>

<p align="center">
  Automate paper reproduction and Kaggle-style MLE competitions with one host-side console,<br />
  Docker sandboxes, durable workspace artifacts, and job state you can actually inspect.
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-Local_Setup-0F766E?style=for-the-badge" alt="Quick Start" /></a>
  <a href="#paper-track"><img src="https://img.shields.io/badge/Paper-Reproduce_Papers-D86A42?style=for-the-badge" alt="Paper Track" /></a>
  <a href="#mle-track"><img src="https://img.shields.io/badge/MLE-Solve_Competitions-F59E0B?style=for-the-badge" alt="MLE Track" /></a>
  <a href="#what-lands-on-disk"><img src="https://img.shields.io/badge/Inspectability-Files_%2B_Artifacts-10233F?style=for-the-badge" alt="Inspectability" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+" />
  <img src="https://img.shields.io/badge/runtime-Docker-2496ED?logo=docker&logoColor=white" alt="Docker runtime" />
  <img src="https://img.shields.io/badge/interface-CLI%20%2B%20TUI-0F766E" alt="CLI and TUI" />
  <img src="https://img.shields.io/badge/domains-Paper%20%2B%20MLE-C2410C" alt="Paper and MLE" />
  <img src="https://img.shields.io/badge/orchestration-File--as--Bus-111827" alt="File-as-bus orchestration" />
</p>

<p align="center">
  <img src="assets/readme/overview.svg" alt="AiScientist overview" />
</p>

> AiScientist is built for people who want the agent to leave a lab notebook behind, not just a final answer.

AiScientist is a dual-domain research automation framework with one shared control plane:

- `paper`: reproduce a paper from `paper.md` or a bundle, then keep the analysis, priorities, implementation log, experiment log, and final self-check.
- `mle`: solve a Kaggle-style competition from a zip, cached dataset, or prepared data directory, then keep candidate history, submission state, and validation artifacts.

The "file-as-bus" part is literal. Agents coordinate through durable workspace files such as `prioritized_tasks.md`, `impl_log.md`, `exp_log.md`, `submission_registry.jsonl`, and validation reports instead of hiding all state in transient prompts.

## One-Line Setup for Coding Agents

If you use Codex, Claude Code, or another coding agent, you can hand it this prompt:

```text
Help me clone AiScientist, adapt the Dockerfiles if this machine cannot use the current base images, fill .env from .env.example, build the paper and mle images, run doctor for both modes, and then launch either a paper or mle job.
```

## Why It Feels Different

<table>
<tr>
<td width="25%" valign="top">

### Two real workflows

One repo, one CLI, two specialized engines: paper reproduction and competition-style MLE.

</td>
<td width="25%" valign="top">

### File-native coordination

Planning, implementation, experiments, and candidate selection all leave durable files behind.

</td>
<td width="25%" valign="top">

### Inspectable runs

Every job records logs, conversation traces, sandbox metadata, artifacts, and an export bundle.

</td>
<td width="25%" valign="top">

### Operator control

Choose LLM profiles, image refs, GPU binding, time limits, output roots, and validation behavior explicitly.

</td>
</tr>
</table>

## Two Tracks

| Track | Primary inputs | Working outputs | Validation endpoint |
| --- | --- | --- | --- |
| `paper` | `--paper-md`, `--zip` | paper analysis notes, prioritized task list, implementation log, experiment log, self-check artifacts | final self-check plus `validation_report.json` |
| `mle` | `--zip`, `--name`, `--data-dir`, `--workspace-zip`, `--competition-bundle-zip` | dataset analysis, prioritized task list, candidate snapshots, `submission.csv`, `submission_registry.jsonl` | submission-format or grading validation |

The domain implementations live in [`src/aisci_domain_paper`](src/aisci_domain_paper) and [`src/aisci_domain_mle`](src/aisci_domain_mle). Shared host infrastructure lives in [`src/aisci_app`](src/aisci_app), [`src/aisci_core`](src/aisci_core), and [`src/aisci_runtime_docker`](src/aisci_runtime_docker).

## Quick Start

> [!IMPORTANT]
> The current Dockerfiles are tuned for our operator environment. Both [`docker/paper-agent.Dockerfile`](docker/paper-agent.Dockerfile) and [`docker/mle-agent.Dockerfile`](docker/mle-agent.Dockerfile) reference internal Ubuntu images and package mirrors. If you are outside that environment, replace those base-image and mirror lines before the first build.

> [!TIP]
> The shipped LLM defaults are not symmetric: `paper=glm-5`, `mle=gpt-5.4` in [`config/llm_profiles.yaml`](config/llm_profiles.yaml). If you only have `OPENAI_API_KEY`, run paper commands with `--llm-profile gpt-5.4` and use `AISCI_PAPER_DOCTOR_PROFILE=gpt-5.4` for `paper doctor`, or update the default profile locally.

### 1. Configure the host

```bash
git clone <your-fork-or-origin-url> AiScientist
cd AiScientist

cp .env.example .env
# Fill either OpenAI or Azure OpenAI credentials.

uv sync --dev
```

Host-side requirements:

- Python 3.12+
- Docker with a reachable daemon
- `uv`
- API credentials for at least one configured LLM backend
- Optional NVIDIA GPUs if you want GPU-bound runs

### 2. Build the default runtime images

If you are not supplying your own runtime images, these are the intended defaults:

```bash
bash docker/build_paper_image.sh
bash docker/build_mle_image.sh
```

The public quick starts below use the locally built tags directly:

- `aisci-paper:latest`
- `aisci-mle:test`

That avoids relying on profile defaults you may want to change later.

### 3. Run the built-in health checks

```bash
AISCI_PAPER_DOCTOR_PROFILE=gpt-5.4 uv run aisci paper doctor
uv run aisci mle doctor
```

If you use the shipped Azure-backed `glm-5` paper profile, you can drop the `AISCI_PAPER_DOCTOR_PROFILE` override.

## Paper Track

Canonical Markdown-first paper reproduction flow:

```bash
uv run aisci --env-file .env paper run \
  --paper-md /abs/path/to/paper.md \
  --image aisci-paper:latest \
  --llm-profile gpt-5.4 \
  --gpu-ids 0 \
  --time-limit 24h \
  --wait \
  --tui
```

Other paper entrypoints:

- `--zip /abs/path/to/paper_bundle_or_context.zip`
- `--submission-seed-repo-zip /abs/path/to/starter_repo.zip`
- `--supporting-materials /abs/path/to/extra_note.md`

Paper zip note:

- `paper run` accepts one primary `--zip`; if you previously staged multiple archives into `/home/paper`, combine them locally before launching the job.

Expected paper outputs:

- `workspace/agent/paper_analysis/summary.md`
- `workspace/agent/paper_analysis/structure.md`
- `workspace/agent/prioritized_tasks.md`
- `workspace/agent/impl_log.md`
- `workspace/agent/exp_log.md`
- `workspace/agent/final_self_check.md`
- `artifacts/validation_report.json`
- `export/<job_id>.zip`

## MLE Track

Canonical self-contained MLE flow with a local competition zip:

```bash
uv run aisci --env-file .env mle run \
  --zip /abs/path/to/detecting-insults-in-social-commentary.zip \
  --name detecting-insults-in-social-commentary \
  --image aisci-mle:test \
  --llm-profile gpt-5.4 \
  --gpu-ids 0 \
  --time-limit 12h \
  --wait \
  --tui
```

Input selection notes:

- Prefer `--zip` for an offline, self-contained entrypoint.
- Use `--name` alone when you already have a prepared MLE-Bench cache.
- Use `--data-dir`, `--workspace-zip`, or `--competition-bundle-zip` for operator or migration flows.
- `--name` is the canonical competition slug used for prepared-cache lookup, runtime planning, and grading metadata.
- If the zip stem and competition slug differ, keep `--name` explicit so the run keeps the correct registry id.

Expected MLE outputs:

- `workspace/agent/analysis/summary.md`
- `workspace/agent/prioritized_tasks.md`
- `workspace/agent/impl_log.md`
- `workspace/agent/exp_log.md`
- `workspace/submission/submission.csv`
- `workspace/submission/submission_registry.jsonl`
- `artifacts/validation_report.json` when final validation is enabled
- `export/<job_id>.zip`

## What Lands On Disk

Every run gets a concrete workspace under `jobs/<job_id>/`:

```text
jobs/<job_id>/
├── input/
├── workspace/
│   ├── paper/ or data/
│   ├── code/               # MLE
│   ├── submission/
│   └── agent/
├── logs/
├── artifacts/
├── export/
└── state/
```

The files inside that tree are the bus:

- planning becomes `prioritized_tasks.md`
- implementation and experiments become `impl_log.md` and `exp_log.md`
- MLE candidate history becomes `submission_registry.jsonl`
- paper self-check becomes `final_self_check.md` and `final_self_check.json`
- runtime metadata becomes `sandbox_session.json` and `resolved_llm_config.json`

That makes runs easier to inspect, resume, diff, export, and audit after the model has already moved on.

## How It Works

1. The host CLI resolves env vars, LLM profiles, image refs, runtime options, and job state paths.
2. AiScientist stages a job workspace under `jobs/<job_id>/`.
3. A Docker sandbox starts with the workspace mounted into canonical paths under `/home`.
4. The domain engine runs, and subagents coordinate through durable files in the workspace.
5. Validation artifacts, logs, and an export bundle are written back to the job directory.

## Operate And Inspect

Core job inspection commands:

```bash
uv run aisci jobs list
uv run aisci jobs show <job_id>
uv run aisci logs list <job_id>
uv run aisci logs tail <job_id> --kind conversation
uv run aisci artifacts ls <job_id>
uv run aisci export <job_id>
```

Terminal-native monitoring:

```bash
uv run aisci tui
uv run aisci tui job <job_id>
```

Lifecycle helpers:

```bash
uv run aisci paper validate <job_id> --wait
uv run aisci paper resume <job_id> --wait
uv run aisci mle validate <job_id> --wait
uv run aisci mle resume <job_id> --wait
```

## Configuration Surface

- [`.env.example`](.env.example): backend credentials, optional proxy variables, optional Hugging Face token, output-root overrides, and doctor flags.
- [`config/llm_profiles.yaml`](config/llm_profiles.yaml): shared model registry and per-domain defaults.
- [`config/image_profiles.yaml`](config/image_profiles.yaml): runtime image registry and pull policy defaults.
- [`config/paper_subagents.yaml`](config/paper_subagents.yaml): paper-mode subagent step budgets and bash timeouts.

Useful host-side knobs:

- `--env-file /path/to/.env`
- `--output-root /abs/path/to/runtime_root`
- `--llm-profile-file /abs/path/to/llm_profiles.yaml`
- `--image-profile-file /abs/path/to/image_profiles.yaml`
- `--gpu-ids 0,1` or `--gpus 2`

## Example Scripts

Operator examples already live in the repo:

- [`examples/paper/example_run_paper_md.sh`](examples/paper/example_run_paper_md.sh)
- [`examples/mle/example_run_mle.sh`](examples/mle/example_run_mle.sh)

The paper example uses the Markdown-first path, and the MLE example vendors a real local sample zip under `examples/mle/`.

## Repo Map

```text
config/                   shared LLM, image, and paper-subagent registries
docker/                   default paper and MLE runtime image recipes
examples/                 operator examples and sample assets
src/aisci_app/            CLI, job service, presentation, TUI
src/aisci_core/           job models, paths, store, exporter, runner
src/aisci_runtime_docker/ Docker session manager and image profile resolver
src/aisci_domain_paper/   paper reproduction workflow
src/aisci_domain_mle/     Kaggle-style competition workflow
tests/                    host-side regression tests
```

## Current Caveats

- The control plane is generic Python plus Docker orchestration, but the bundled image recipes still carry internal infrastructure assumptions.
- The default paper image path is local-first after you build `aisci-paper:latest`.
- The current shared MLE image profile is remote-first, so public quick starts should pass `--image aisci-mle:test` explicitly or update [`config/image_profiles.yaml`](config/image_profiles.yaml).
- Job state lives under the repo root by default. Use `AISCI_OUTPUT_ROOT` or `--output-root` if you want `jobs/` and `.aisci/` somewhere else.

AiScientist is opinionated enough to run real work, but still transparent enough that you can inspect every file the lab leaves behind.
