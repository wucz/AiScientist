from __future__ import annotations

import json
from pathlib import Path

from aisci_app.cli import _log_targets_for_kind
from aisci_app.presentation import paper_doctor_report, paper_job_summary
from aisci_core.models import JobSpec, JobType, PaperSpec, RuntimeProfile, WorkspaceLayout
from aisci_core.paths import ensure_job_dirs, resolve_job_paths
from aisci_core.store import JobStore
from aisci_app.service import JobService


def _create_paper_job(tmp_path: Path):
    store = JobStore()
    service = JobService(store=store)
    job = service.create_job(
        JobSpec(
            job_type=JobType.PAPER,
            objective="paper console",
            llm_profile="gpt-5.4-responses",
            runtime_profile=RuntimeProfile(
                workspace_layout=WorkspaceLayout.PAPER,
                run_final_validation=True,
            ),
            mode_spec=PaperSpec(pdf_path=str(tmp_path / "paper.pdf")),
        )
    )
    paths = ensure_job_dirs(resolve_job_paths(job.id))
    (paths.logs_dir / "job.log").write_text("main log line\n", encoding="utf-8")
    (paths.logs_dir / "conversation.jsonl").write_text(
        json.dumps({"event_type": "model_response", "phase": "analyze", "message": "hello"}) + "\n",
        encoding="utf-8",
    )
    (paths.logs_dir / "agent.log").write_text("agent log line\n", encoding="utf-8")
    (paths.logs_dir / "subagent_logs").mkdir(parents=True, exist_ok=True)
    (paths.logs_dir / "subagent_logs" / "implement_001_20260331_120000").mkdir(parents=True, exist_ok=True)
    (paths.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)
    (paths.workspace_dir / "agent" / "paper_analysis").mkdir(parents=True, exist_ok=True)
    (paths.workspace_dir / "agent" / "paper_analysis" / "summary.md").write_text("# summary\n", encoding="utf-8")
    (paths.workspace_dir / "agent" / "prioritized_tasks.md").write_text("# tasks\n", encoding="utf-8")
    (paths.workspace_dir / "agent" / "plan.md").write_text("# plan\n", encoding="utf-8")
    (paths.workspace_dir / "agent" / "impl_log.md").write_text("# impl\n", encoding="utf-8")
    (paths.workspace_dir / "agent" / "exp_log.md").write_text("# exp\n", encoding="utf-8")
    (paths.workspace_dir / "agent" / "paper_main_prompt.md").write_text("# prompt\n", encoding="utf-8")
    (paths.workspace_dir / "agent" / "capabilities.json").write_text(
        json.dumps(
            {
                "online_research": {"available": True},
                "github_research": {"available": False},
                "linter": {"available": True},
            }
        ),
        encoding="utf-8",
    )
    (paths.workspace_dir / "submission" / "reproduce.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    return job, paths


def test_paper_job_summary_exposes_product_signals(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    job, _ = _create_paper_job(tmp_path)
    summary = paper_job_summary(job)
    assert summary["paper_capabilities"]["online_research"] == "available"
    assert summary["paper_capabilities"]["github_research"] == "requested"
    assert summary["paper_capabilities"]["final_self_check"] == "enabled"
    assert any(item["label"] == "main job log" for item in summary["paper_log_targets"])
    assert any(item["label"].startswith("session: ") for item in summary["paper_log_targets"])
    assert any(item["label"] == "paper summary" for item in summary["paper_artifacts"])
    assert any(item["label"] == "main prompt snapshot" for item in summary["paper_artifacts"])


def test_log_target_helper_lists_paper_logs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AISCI_REPO_ROOT", str(tmp_path))
    job, _ = _create_paper_job(tmp_path)
    targets = _log_targets_for_kind(job.id, "all")
    labels = {label for label, _ in targets}
    assert "main job log" in labels
    assert "conversation log" in labels
    assert "subagent logs" in labels


def test_paper_doctor_reports_console_tip() -> None:
    report = paper_doctor_report()
    assert any(check.name == "paper console" for check in report)
    assert any(check.name == "github_token" for check in report)
    assert any(check.name == "online_research" for check in report)
