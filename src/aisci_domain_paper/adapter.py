from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

from aisci_agent_runtime.llm_profiles import llm_env
from aisci_core.logging_utils import append_log
from aisci_core.models import ArtifactRecord, JobRecord, PaperSpec, RunPhase, ValidationReport, WorkspaceLayout
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_runtime_docker.profiles import default_paper_profile
from aisci_runtime_docker.runtime import DockerRuntimeError, DockerRuntimeManager


class PaperDomainAdapter:
    def __init__(self, runtime: DockerRuntimeManager):
        self.runtime = runtime

    def run(self, job: JobRecord) -> dict[str, object]:
        assert isinstance(job.mode_spec, PaperSpec)
        job_paths = ensure_job_dirs(resolve_job_paths(job.id))
        append_log(job_paths.logs_dir / "job.log", "paper job started")
        self.runtime.ensure_layout(job_paths, WorkspaceLayout.PAPER)
        self._stage_inputs(job.mode_spec, job_paths)

        submission_dir = job_paths.workspace_dir / "submission"
        if not (submission_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=submission_dir, check=False, capture_output=True)

        self._ensure_runtime_ready(job, job_paths)
        self._run_real_loop(job, job_paths)

        artifacts = self._collect_artifacts(job_paths)
        validation = self._maybe_validate(job, job_paths)
        return {"artifacts": artifacts, "validation_report": validation}

    def _ensure_runtime_ready(self, job: JobRecord, job_paths) -> None:
        if not self.runtime.can_use_docker():
            message = "Paper mode requires a reachable Docker daemon. No local fallback loop is available."
            append_log(job_paths.logs_dir / "job.log", message)
            raise DockerRuntimeError(message)
        env = llm_env(job.llm_profile)
        if not any(env.get(key) for key in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")):
            message = (
                "Paper mode requires OPENAI_API_KEY or AZURE_OPENAI_API_KEY in the host environment. "
                "No local fallback loop is available."
            )
            append_log(job_paths.logs_dir / "job.log", message)
            raise RuntimeError(message)

    def _stage_inputs(self, spec: PaperSpec, job_paths) -> None:
        paper_dir = job_paths.workspace_dir / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)

        if spec.paper_bundle_zip:
            archive = Path(spec.paper_bundle_zip).resolve()
            shutil.copy2(archive, job_paths.input_dir / "paper_bundle.zip")
            self._extract_zip(archive, paper_dir)
        if spec.context_bundle_zip:
            archive = Path(spec.context_bundle_zip).resolve()
            shutil.copy2(archive, job_paths.input_dir / "context_bundle.zip")
            self._extract_zip(archive, paper_dir)
        if spec.pdf_path:
            src = Path(spec.pdf_path).resolve()
            shutil.copy2(src, paper_dir / "paper.pdf")
            shutil.copy2(src, job_paths.input_dir / "paper.pdf")
        if spec.paper_md_path:
            src = Path(spec.paper_md_path).resolve()
            shutil.copy2(src, paper_dir / "paper.md")
        self._copy_optional(spec.rubric_path, paper_dir / "rubric.json")
        self._copy_optional(spec.blacklist_path, paper_dir / "blacklist.txt")
        self._copy_optional(spec.addendum_path, paper_dir / "addendum.md")
        for material in spec.supporting_materials:
            src = Path(material).resolve()
            if src.is_file():
                shutil.copy2(src, paper_dir / src.name)
        if spec.submission_seed_repo_zip:
            self._extract_zip(Path(spec.submission_seed_repo_zip).resolve(), job_paths.workspace_dir / "submission")

    def _extract_zip(self, archive_path: Path, destination: Path) -> None:
        with ZipFile(archive_path) as zf:
            zf.extractall(destination)

    def _copy_optional(self, source: str | None, destination: Path) -> None:
        if source:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(source).resolve(), destination)

    def _run_real_loop(self, job: JobRecord, job_paths) -> None:
        profile = default_paper_profile()
        image_tag = self.runtime.build_profile(profile, job.runtime_profile)
        spec = self.runtime.create_session_spec(
            job.id,
            job_paths,
            profile,
            job.runtime_profile,
            layout=WorkspaceLayout.PAPER,
            workdir="/home/submission",
            env=self._session_env(job),
        )
        session = self.runtime.start_session(spec, image_tag)
        keep_session = False
        try:
            result = self.runtime.exec(
                session,
                "python -m aisci_domain_paper.orchestrator",
                workdir="/home/submission",
                check=False,
            )
            if result.stdout:
                append_log(job_paths.logs_dir / "job.log", result.stdout)
            if result.stderr:
                append_log(job_paths.logs_dir / "job.log", result.stderr)
            if result.exit_code != 0:
                if job.runtime_profile.keep_container_on_failure:
                    keep_session = True
                raise DockerRuntimeError(
                    result.combined_output or "Paper orchestrator exited with a non-zero status."
                )
        finally:
            if not keep_session:
                self.runtime.cleanup(session)

    def _session_env(self, job: JobRecord) -> dict[str, str]:
        env = llm_env(job.llm_profile)
        env.update(
            {
                "AISCI_JOB_ID": job.id,
                "AISCI_OBJECTIVE": job.objective,
                "AISCI_LLM_PROFILE": job.llm_profile,
                "AISCI_ENABLE_ONLINE_RESEARCH": "true" if job.mode_spec.enable_online_research else "false",
                "AISCI_ENABLE_GITHUB_RESEARCH": "true" if job.mode_spec.enable_github_research else "false",
                "TIME_LIMIT_SECS": str(self._parse_duration(job.runtime_profile.time_limit)),
                "AISCI_MAX_STEPS": os.environ.get("AISCI_MAX_STEPS", "80"),
                "AISCI_REMINDER_FREQ": os.environ.get("AISCI_REMINDER_FREQ", "5"),
                "LOGS_DIR": "/home/logs",
            }
        )
        if os.environ.get("GITHUB_TOKEN"):
            env["GITHUB_TOKEN"] = os.environ["GITHUB_TOKEN"]
        if job.mode_spec.enable_online_research and env.get("AISCI_API_MODE") == "responses":
            env["AISCI_WEB_SEARCH"] = "true"
        return env

    def _collect_artifacts(self, job_paths) -> list[ArtifactRecord]:
        known: list[tuple[str, Path, RunPhase]] = [
            ("paper_analysis", job_paths.workspace_dir / "agent" / "paper_analysis" / "summary.md", RunPhase.ANALYZE),
            ("paper_structure", job_paths.workspace_dir / "agent" / "paper_analysis" / "structure.md", RunPhase.ANALYZE),
            ("paper_algorithm", job_paths.workspace_dir / "agent" / "paper_analysis" / "algorithm.md", RunPhase.ANALYZE),
            ("paper_experiments", job_paths.workspace_dir / "agent" / "paper_analysis" / "experiments.md", RunPhase.ANALYZE),
            ("paper_baseline", job_paths.workspace_dir / "agent" / "paper_analysis" / "baseline.md", RunPhase.ANALYZE),
            ("prioritized_tasks", job_paths.workspace_dir / "agent" / "prioritized_tasks.md", RunPhase.PRIORITIZE),
            ("plan", job_paths.workspace_dir / "agent" / "plan.md", RunPhase.IMPLEMENT),
            ("impl_log", job_paths.workspace_dir / "agent" / "impl_log.md", RunPhase.IMPLEMENT),
            ("exp_log", job_paths.workspace_dir / "agent" / "exp_log.md", RunPhase.VALIDATE),
            ("reproduce_script", job_paths.workspace_dir / "submission" / "reproduce.sh", RunPhase.FINALIZE),
            ("capabilities", job_paths.workspace_dir / "agent" / "capabilities.json", RunPhase.ANALYZE),
            ("self_check_report", job_paths.workspace_dir / "agent" / "final_self_check.md", RunPhase.VALIDATE),
            ("self_check_report_json", job_paths.workspace_dir / "agent" / "final_self_check.json", RunPhase.VALIDATE),
            ("prompt", job_paths.workspace_dir / "agent" / "paper_main_prompt.md", RunPhase.INGEST),
            ("agent_log", job_paths.logs_dir / "agent.log", RunPhase.FINALIZE),
            ("conversation_log", job_paths.logs_dir / "conversation.jsonl", RunPhase.FINALIZE),
            ("orchestrator_state", job_paths.logs_dir / "paper_session_state.json", RunPhase.FINALIZE),
        ]
        artifacts: list[ArtifactRecord] = []
        for artifact_type, path, phase in known:
            if path.exists():
                artifacts.append(
                    ArtifactRecord(
                        artifact_type=artifact_type,
                        path=str(path),
                        phase=phase,
                        size_bytes=path.stat().st_size,
                        metadata={},
                    )
                )
        subagent_logs = job_paths.logs_dir / "subagent_logs"
        if subagent_logs.exists():
            for path in sorted(subagent_logs.rglob("*")):
                if path.is_file():
                    artifacts.append(
                        ArtifactRecord(
                            artifact_type="subagent_log",
                            path=str(path),
                            phase=RunPhase.FINALIZE,
                            size_bytes=path.stat().st_size,
                            metadata={},
                        )
                    )
        return artifacts

    def _maybe_validate(self, job: JobRecord, job_paths) -> ValidationReport:
        if not job.runtime_profile.run_final_validation:
            return ValidationReport(
                status="skipped",
                summary="Final validation disabled for this job.",
                runtime_profile_hash="disabled",
                container_image="not-built",
            )
        if not self.runtime.can_use_docker():
            return ValidationReport(
                status="failed",
                summary="Docker is not available on this machine.",
                details={"reason": "docker_not_found"},
                runtime_profile_hash="docker-missing",
                container_image="not-built",
            )
        reproduce_path = job_paths.workspace_dir / "submission" / "reproduce.sh"
        if not reproduce_path.exists():
            return ValidationReport(
                status="failed",
                summary="No reproduce.sh was staged for validation.",
                details={"reason": "reproduce_missing"},
                runtime_profile_hash="reproduce-missing",
                container_image="not-built",
            )
        try:
            profile = default_paper_profile()
            image_tag = self.runtime.build_profile(profile, job.runtime_profile)
            spec = self.runtime.create_session_spec(
                job.id,
                job_paths,
                profile,
                job.runtime_profile,
                layout=WorkspaceLayout.PAPER,
                workdir="/home/submission",
            )
            return self.runtime.run_validation(
                spec,
                image_tag,
                "bash reproduce.sh",
                workdir="/home/submission",
            )
        except DockerRuntimeError as exc:
            return ValidationReport(
                status="failed",
                summary="Final validation failed while preparing the Docker runtime.",
                details={"reason": str(exc)},
                runtime_profile_hash="docker-build-failed",
                container_image="build-failed",
            )

    def _parse_duration(self, text: str) -> int:
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
