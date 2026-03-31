from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from pypdf import PdfWriter

from aisci_core.models import (
    JobRecord,
    JobStatus,
    JobType,
    MLESpec,
    PaperSpec,
    RunPhase,
    RuntimeProfile,
    WorkspaceLayout,
)
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_domain_mle.adapter import MLEDomainAdapter
from aisci_domain_paper.adapter import PaperDomainAdapter
from aisci_runtime_docker.profiles import default_mle_profile
from aisci_runtime_docker.runtime import DockerRuntimeError, DockerRuntimeManager


def _make_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)


def _paper_job(tmp_path: Path, pdf_path: Path) -> JobRecord:
    now = __import__("datetime").datetime.now().astimezone()
    return JobRecord(
        id="paper-job",
        job_type=JobType.PAPER,
        status=JobStatus.PENDING,
        phase=RunPhase.INGEST,
        objective="paper objective",
        llm_profile="test",
        runtime_profile=RuntimeProfile(
            run_final_validation=False,
            workspace_layout=WorkspaceLayout.PAPER,
        ),
        mode_spec=PaperSpec(pdf_path=str(pdf_path)),
        created_at=now,
        updated_at=now,
    )


def _mle_job(tmp_path: Path, bundle_path: Path) -> JobRecord:
    now = __import__("datetime").datetime.now().astimezone()
    return JobRecord(
        id="mle-job",
        job_type=JobType.MLE,
        status=JobStatus.PENDING,
        phase=RunPhase.INGEST,
        objective="mle objective",
        llm_profile="test",
        runtime_profile=RuntimeProfile(
            run_final_validation=False,
            workspace_layout=WorkspaceLayout.MLE,
        ),
        mode_spec=MLESpec(workspace_bundle_zip=str(bundle_path)),
        created_at=now,
        updated_at=now,
    )


def test_paper_adapter_stages_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (tmp_path / "docker").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docker" / "paper.Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)
    runtime = DockerRuntimeManager()
    monkeypatch.setattr(runtime, "can_use_docker", lambda: True)

    def fake_run_real_loop(job, job_paths) -> None:  # noqa: ANN001
        analysis_dir = job_paths.workspace_dir / "agent" / "paper_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        for name in ("summary.md", "structure.md", "algorithm.md", "experiments.md", "baseline.md"):
            (analysis_dir / name).write_text(f"# {name}\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "prioritized_tasks.md").write_text("# priorities\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "plan.md").write_text("# plan\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "capabilities.json").write_text("{}", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "final_self_check.md").write_text("# self-check\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "final_self_check.json").write_text("{}", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "paper_main_prompt.md").write_text("# prompt\n", encoding="utf-8")
        (job_paths.workspace_dir / "submission" / "reproduce.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (job_paths.logs_dir / "agent.log").write_text("agent log\n", encoding="utf-8")
        (job_paths.logs_dir / "conversation.jsonl").write_text("{}\n", encoding="utf-8")
        (job_paths.logs_dir / "paper_session_state.json").write_text("{}", encoding="utf-8")

    adapter = PaperDomainAdapter(runtime)
    monkeypatch.setattr(adapter, "_run_real_loop", fake_run_real_loop)
    result = adapter.run(_paper_job(tmp_path, pdf_path))
    assert result["validation_report"].status == "skipped"
    artifact_types = {item.artifact_type for item in result["artifacts"]}
    assert "paper_analysis" in artifact_types
    assert "prioritized_tasks" in artifact_types
    assert "capabilities" in artifact_types
    assert "self_check_report" in artifact_types
    job_paths = ensure_job_dirs(resolve_job_paths("paper-job"))
    assert (job_paths.workspace_dir / "submission" / "reproduce.sh").exists()
    assert (job_paths.workspace_dir / "agent" / "paper_analysis" / "summary.md").exists()
    assert (job_paths.workspace_dir / "agent" / "capabilities.json").exists()
    assert (job_paths.workspace_dir / "agent" / "final_self_check.md").exists()


def test_paper_adapter_requires_docker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)
    runtime = DockerRuntimeManager()
    monkeypatch.setattr(runtime, "can_use_docker", lambda: False)
    adapter = PaperDomainAdapter(runtime)

    try:
        adapter.run(_paper_job(tmp_path, pdf_path))
    except DockerRuntimeError as exc:
        assert "No local fallback loop is available" in str(exc)
    else:
        raise AssertionError("Expected DockerRuntimeError when Docker is unavailable")


def test_mle_adapter_stages_summary_and_submission(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (tmp_path / "docker").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docker" / "mle.Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    bundle = tmp_path / "workspace.zip"
    with ZipFile(bundle, "w") as zf:
        zf.writestr("description.md", "# task")
        zf.writestr("sample_submission.csv", "id,target\n1,0\n")
    runtime = DockerRuntimeManager()
    monkeypatch.setattr(runtime, "can_use_docker", lambda: True)

    def fake_run_real_loop(job, job_paths) -> None:  # noqa: ANN001
        analysis_dir = job_paths.workspace_dir / "agent" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        submission_dir = job_paths.workspace_dir / "submission"
        candidates_dir = submission_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "summary.md").write_text("# analysis\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "prioritized_tasks.md").write_text("# priorities\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "impl_log.md").write_text("# impl\n", encoding="utf-8")
        (job_paths.workspace_dir / "agent" / "exp_log.md").write_text("# exp\n", encoding="utf-8")
        (submission_dir / "submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
        candidate = candidates_dir / "submission_001_final.csv"
        candidate.write_text("id,target\n1,0\n", encoding="utf-8")
        (submission_dir / "submission_registry.jsonl").write_text(
            '{"event":"system_snapshot"}\n{"event":"candidate_detail"}\n{"event":"champion_selected"}\n',
            encoding="utf-8",
        )
        (job_paths.logs_dir / "agent.log").write_text("agent log\n", encoding="utf-8")
        (job_paths.logs_dir / "conversation.jsonl").write_text("{}\n", encoding="utf-8")
        (job_paths.logs_dir / "summary.json").write_text("{}", encoding="utf-8")
        (job_paths.artifacts_dir / "champion_report.md").write_text("# Champion Report\n", encoding="utf-8")

    adapter = MLEDomainAdapter(runtime)
    monkeypatch.setattr(adapter, "_run_real_loop", fake_run_real_loop)
    result = adapter.run(_mle_job(tmp_path, bundle))
    assert result["validation_report"].status == "skipped"
    artifact_types = {item.artifact_type for item in result["artifacts"]}
    assert "champion_report" in artifact_types
    assert "submission_registry" in artifact_types
    assert "candidate_snapshot" in artifact_types
    job_paths = ensure_job_dirs(resolve_job_paths("mle-job"))
    registry_text = (job_paths.workspace_dir / "submission" / "submission_registry.jsonl").read_text(
        encoding="utf-8"
    )
    assert "champion_selected" in registry_text
    assert (job_paths.workspace_dir / "submission" / "submission.csv").exists()


def test_mle_adapter_requires_docker(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bundle = tmp_path / "workspace.zip"
    with ZipFile(bundle, "w") as zf:
        zf.writestr("description.md", "# task")
        zf.writestr("sample_submission.csv", "id,target\n1,0\n")
    runtime = DockerRuntimeManager()
    monkeypatch.setattr(runtime, "can_use_docker", lambda: False)
    adapter = MLEDomainAdapter(runtime)

    try:
        adapter.run(_mle_job(tmp_path, bundle))
    except DockerRuntimeError as exc:
        assert "No local fallback loop is available" in str(exc)
    else:
        raise AssertionError("Expected DockerRuntimeError when Docker is unavailable")


def test_runtime_session_spec_uses_canonical_mounts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    (tmp_path / "docker").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docker" / "mle.Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    runtime = DockerRuntimeManager()
    job_paths = ensure_job_dirs(resolve_job_paths("layout-job"))
    runtime.ensure_layout(job_paths, WorkspaceLayout.MLE)
    spec = runtime.create_session_spec(
        "layout-job",
        job_paths,
        default_mle_profile(),
        RuntimeProfile(workspace_layout=WorkspaceLayout.MLE),
        layout=WorkspaceLayout.MLE,
        workdir="/home/code",
    )
    targets = {mount.target for mount in spec.mounts}
    assert "/home/data" in targets
    assert "/home/code" in targets
    assert "/home/submission" in targets
    assert spec.workdir == "/home/code"
