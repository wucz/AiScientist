from __future__ import annotations

import os
import re
from pathlib import Path

from aisci_agent_runtime.llm_client import LLMConfig, create_llm_client
from aisci_agent_runtime.llm_profiles import resolve_llm_profile
from aisci_agent_runtime.shell_interface import ShellInterface
from aisci_agent_runtime.trace import AgentTraceWriter
from aisci_core.models import ArtifactRecord, JobRecord, PaperSpec, RunPhase
from aisci_domain_paper.engine import EmbeddedPaperEngine, PaperRuntimeConfig
from aisci_domain_paper.runtime import build_workspace


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_duration_to_seconds(text: str) -> int:
    value = (text or "24h").strip().lower()
    total = 0
    for amount, unit in re.findall(r"(\d+)([smhd])", value):
        n = int(amount)
        total += {
            "s": n,
            "m": n * 60,
            "h": n * 3600,
            "d": n * 86400,
        }[unit]
    return total or 24 * 3600


def _has_llm_credentials() -> bool:
    return any(
        os.environ.get(key)
        for key in (
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
        )
    )


def _build_llm(profile_name: str, *, enable_online_research: bool):
    if not _has_llm_credentials():
        raise RuntimeError(
            "Paper mode requires OPENAI_API_KEY or AZURE_OPENAI_API_KEY. "
            "No local fallback loop is available."
        )
    profile = resolve_llm_profile(profile_name)
    return create_llm_client(
        LLMConfig(
            model=profile.model,
            api_mode=profile.api_mode,
            reasoning_effort=profile.reasoning_effort,
            web_search=enable_online_research and profile.api_mode == "responses",
            max_tokens=profile.max_tokens,
            context_window=profile.context_window,
            use_phase=profile.use_phase,
        )
    )


def _artifact_phase(path: Path) -> RunPhase:
    suffix = path.name
    if suffix in {"summary.md", "structure.md", "algorithm.md", "experiments.md", "baseline.md"}:
        return RunPhase.ANALYZE
    if suffix in {"prioritized_tasks.md"}:
        return RunPhase.PRIORITIZE
    if suffix in {"plan.md", "impl_log.md"}:
        return RunPhase.IMPLEMENT
    if suffix in {"exp_log.md", "final_self_check.md", "final_self_check.json"}:
        return RunPhase.VALIDATE
    return RunPhase.FINALIZE


def _artifact_type(path: Path) -> str:
    name = path.name
    mapping = {
        "summary.md": "paper_analysis",
        "structure.md": "paper_structure",
        "algorithm.md": "paper_algorithm",
        "experiments.md": "paper_experiments",
        "baseline.md": "paper_baseline",
        "prioritized_tasks.md": "prioritized_tasks",
        "plan.md": "plan",
        "impl_log.md": "impl_log",
        "exp_log.md": "exp_log",
        "reproduce.sh": "reproduce_script",
        "capabilities.json": "capabilities",
        "paper_main_prompt.md": "prompt",
        "final_self_check.md": "self_check_report",
        "final_self_check.json": "self_check_report_json",
        "agent.log": "agent_log",
        "conversation.jsonl": "conversation_log",
        "paper_session_state.json": "orchestrator_state",
    }
    return mapping.get(name, path.stem)


class PaperOrchestrator:
    def run(self, job: JobRecord, job_paths) -> list[ArtifactRecord]:
        assert isinstance(job.mode_spec, PaperSpec)
        workspace = build_workspace(job_paths)
        trace = AgentTraceWriter(workspace.logs_dir)
        engine = EmbeddedPaperEngine(
            config=PaperRuntimeConfig(
                job_id=job.id,
                objective=job.objective,
                llm_profile_name=job.llm_profile,
                time_limit_seconds=parse_duration_to_seconds(job.runtime_profile.time_limit),
                max_steps=int(os.environ.get("AISCI_MAX_STEPS", "80")),
                reminder_freq=int(os.environ.get("AISCI_REMINDER_FREQ", "5")),
                enable_online_research=job.mode_spec.enable_online_research,
                enable_github_research=job.mode_spec.enable_github_research,
            ),
            shell=workspace.shell,
            llm=_build_llm(
                job.llm_profile,
                enable_online_research=job.mode_spec.enable_online_research,
            ),
            paper_dir=workspace.paper_dir,
            submission_dir=workspace.submission_dir,
            agent_dir=workspace.agent_dir,
            logs_dir=workspace.logs_dir,
            trace=trace,
        )
        engine.run()
        return [
            ArtifactRecord(
                artifact_type=_artifact_type(path),
                path=str(path),
                phase=_artifact_phase(path),
                size_bytes=path.stat().st_size,
                metadata={},
            )
            for path in engine.collect_artifacts()
        ]


def main() -> None:
    logs_dir = Path("/home/logs")
    paper_dir = Path("/home/paper")
    submission_dir = Path("/home/submission")
    agent_dir = Path("/home/agent")
    trace = AgentTraceWriter(logs_dir)
    if not _has_llm_credentials():
        raise RuntimeError(
            "Paper mode requires OPENAI_API_KEY or AZURE_OPENAI_API_KEY. "
            "No local fallback loop is available."
        )
    engine = EmbeddedPaperEngine(
        config=PaperRuntimeConfig(
            job_id=os.environ.get("AISCI_JOB_ID", "paper-job"),
            objective=os.environ.get("AISCI_OBJECTIVE", "paper reproduction job"),
            llm_profile_name=os.environ.get("AISCI_LLM_PROFILE", "gpt-5.4-responses"),
            time_limit_seconds=int(os.environ.get("TIME_LIMIT_SECS", str(24 * 3600))),
            max_steps=int(os.environ.get("AISCI_MAX_STEPS", "80")),
            reminder_freq=int(os.environ.get("AISCI_REMINDER_FREQ", "5")),
            enable_online_research=_bool_env("AISCI_ENABLE_ONLINE_RESEARCH", True),
            enable_github_research=_bool_env("AISCI_ENABLE_GITHUB_RESEARCH", True),
        ),
        shell=ShellInterface("/home"),
        llm=_build_llm(
            os.environ.get("AISCI_LLM_PROFILE", "gpt-5.4-responses"),
            enable_online_research=_bool_env("AISCI_ENABLE_ONLINE_RESEARCH", True),
        ),
        paper_dir=paper_dir,
        submission_dir=submission_dir,
        agent_dir=agent_dir,
        logs_dir=logs_dir,
        trace=trace,
    )
    engine.run()


if __name__ == "__main__":
    main()
