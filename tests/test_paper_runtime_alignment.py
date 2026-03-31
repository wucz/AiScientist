from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from aisci_agent_runtime.subagents.base import SubagentOutput, SubagentStatus
from aisci_domain_paper.configs import (
    DEFAULT_PAPER_READER_CONFIG,
    DEFAULT_PAPER_STRUCTURE_CONFIG,
    DEFAULT_PAPER_SYNTHESIS_CONFIG,
    DEFAULT_PRIORITIZATION_CONFIG,
)
from aisci_domain_paper.subagents.paper_reader import PaperReaderCoordinator
from aisci_domain_paper.subagents.prioritization import PaperPrioritizationSubagent, PrioritizationRunner
from aisci_domain_paper.state_manager import PaperStateManager
from aisci_domain_paper.tools.basic_tool import (
    build_explore_tools,
    build_general_tools,
    build_implementation_tools,
    build_main_direct_tools,
    build_reader_tools,
    build_plan_tools,
    build_search_executor_tools,
    build_search_strategist_tools,
)
from aisci_domain_paper.tools.spawn_subagent_tool import build_main_tools, build_spawn_subagent_tool


class _StubEngine:
    def _capabilities(self):
        return {
            "online_research": {"available": True},
            "github_research": {"available": True},
        }

    def read_paper(self, refresh: bool = False):  # noqa: ARG002
        return "read"

    def prioritize_tasks(self, refresh: bool = False):  # noqa: ARG002
        return "prioritize"

    def run_implementation(self, **kwargs):
        return f"implement:{kwargs['task']}"

    def run_experiment(self, **kwargs):
        return f"experiment:{kwargs['task']}"

    def run_named_subagent(self, **kwargs):
        return f"subagent:{kwargs['subagent_type']}"

    def run_clean_validation(self, refresh: bool = False):  # noqa: ARG002
        return "clean"


class _TraceRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str | None, dict | None]] = []

    def event(self, name: str, message: str, phase: str | None = None, payload: dict | None = None) -> None:
        self.events.append((name, message, phase, payload))


def test_main_tools_expose_direct_solver_tools() -> None:
    tools = build_main_tools(_StubEngine())
    names = {tool.name() for tool in tools}
    assert {"bash", "python", "read_file_chunk", "search_file"} <= names
    assert {"read_paper", "prioritize_tasks", "implement", "run_experiment", "clean_reproduce_validation", "submit"} <= names


def test_reader_and_prioritization_configs_match_upstream_scale() -> None:
    assert DEFAULT_PAPER_STRUCTURE_CONFIG is not DEFAULT_PAPER_READER_CONFIG
    assert DEFAULT_PAPER_READER_CONFIG is not DEFAULT_PAPER_SYNTHESIS_CONFIG
    assert DEFAULT_PAPER_STRUCTURE_CONFIG is not DEFAULT_PRIORITIZATION_CONFIG
    assert DEFAULT_PAPER_STRUCTURE_CONFIG.max_steps == 500
    assert DEFAULT_PAPER_READER_CONFIG.max_steps == 500
    assert DEFAULT_PAPER_SYNTHESIS_CONFIG.max_steps == 500
    assert DEFAULT_PRIORITIZATION_CONFIG.max_steps == 500
    assert DEFAULT_PAPER_STRUCTURE_CONFIG.time_limit == 36_000
    assert DEFAULT_PAPER_READER_CONFIG.time_limit == 36_000
    assert DEFAULT_PAPER_SYNTHESIS_CONFIG.time_limit == 36_000
    assert DEFAULT_PRIORITIZATION_CONFIG.time_limit == 36_000


def test_spawn_subagent_contract_matches_upstream_scope() -> None:
    schema = build_spawn_subagent_tool(_StubEngine()).get_tool_schema()
    values = schema["function"]["parameters"]["properties"]["subagent_type"]["enum"]
    assert values == ["explore", "plan", "general"]


def test_generic_subagent_tool_permissions_are_split() -> None:
    explore = {tool.name() for tool in build_explore_tools({"online_research": {"available": True}})}
    plan = {tool.name() for tool in build_plan_tools({"online_research": {"available": True}})}
    general = {tool.name() for tool in build_general_tools({"online_research": {"available": True}})}
    implementation = {tool.name() for tool in build_implementation_tools({"online_research": {"available": True}, "github_research": {"available": True}})}
    reader = {tool.name() for tool in build_reader_tools({"online_research": {"available": True}})}
    search_strategist = {tool.name() for tool in build_search_strategist_tools()}
    search_executor = {tool.name() for tool in build_search_executor_tools()}

    assert "edit_file" not in explore
    assert "write_plan" in plan
    assert "edit_file" not in general
    assert "python" in general
    assert "github" in implementation
    assert "linter" in implementation
    main_direct = {tool.name() for tool in build_main_direct_tools({"online_research": {"available": True}})}
    assert {"web_search", "link_summary"} <= main_direct
    assert "python" not in reader
    assert reader >= {"read_file_chunk", "search_file", "bash", "subagent_complete"}
    assert search_strategist == {"read_file_chunk", "subagent_complete"}
    assert search_executor == {"read_file_chunk", "search_file", "subagent_complete"}


def test_state_manager_tracks_session_boundaries_and_recent_history(tmp_path: Path) -> None:
    state = PaperStateManager(
        agent_dir=tmp_path / "agent",
        logs_dir=tmp_path / "logs",
        subagent_logs_dir=tmp_path / "logs" / "subagent_logs",
    )
    state.ensure_logs()
    session = state.create_session("implementation")
    state.append_separator(session)
    state.append_session_note("implementation", "Implemented core loop", "Changed reproduce.sh and training entrypoint.")
    recent = state.recent_impl_history()
    assert "Implement Session 1" in recent
    assert "Implemented core loop" in recent


def test_reader_coordinator_returns_navigation_summary(tmp_path: Path, monkeypatch) -> None:
    class Engine:
        def __init__(self) -> None:
            self.analysis_dir = tmp_path / "agent" / "paper_analysis"
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            self.shell = object()
            self.llm = object()
            self.trace = _TraceRecorder()

        def reader_context(self) -> str:
            return "Objective:\nReproduce the paper."

        def subagent_config(self, kind: str, base):  # noqa: ARG002
            return base

        def _capabilities(self) -> dict[str, dict[str, bool]]:
            return {
                "online_research": {"available": True},
                "github_research": {"available": True},
            }

        def constraints(self) -> dict[str, list[str]]:
            return {"blacklist": []}

    engine = Engine()
    calls: list[tuple[str, int, int]] = []

    def fake_run_stage(self, subagent_cls, *, objective, context, config, phase, label):  # noqa: ANN001
        calls.append((label, config.max_steps, config.time_limit))
        return SubagentOutput(
            status=SubagentStatus.COMPLETED,
            content=f"{label} findings",
            num_steps=3,
            runtime_seconds=1.5,
            log_path=f"/tmp/{label}.jsonl",
        )

    monkeypatch.setattr(PaperReaderCoordinator, "_run_stage", fake_run_stage, raising=False)

    result = PaperReaderCoordinator(engine).run()

    assert [label for label, _, _ in calls] == [
        "paper_structure",
        "paper_algorithm",
        "paper_experiments",
        "paper_baseline",
        "paper_synthesis",
    ]
    assert all(max_steps == 500 for _, max_steps, _ in calls)
    assert all(time_limit == 36_000 for _, _, time_limit in calls)
    assert result.all_success is True
    assert result.failed_subagents == []
    assert result.executive_summary == "paper_synthesis findings"
    assert "Detailed Analysis Files" in result.summary_with_navigation
    assert "/home/agent/paper_analysis/structure.md" in result.summary_with_navigation
    assert (engine.analysis_dir / "summary.md").exists()
    assert "paper_structure findings" in (engine.analysis_dir / "structure.md").read_text(encoding="utf-8")


def test_prioritization_runner_builds_structured_task_description(tmp_path: Path, monkeypatch) -> None:
    class Engine:
        def __init__(self) -> None:
            self.paper_dir = tmp_path / "paper"
            self.analysis_dir = tmp_path / "agent" / "paper_analysis"
            self.prioritized_path = tmp_path / "agent" / "prioritized_tasks.md"
            self.plan_path = tmp_path / "agent" / "plan.md"
            self.paper_dir.mkdir(parents=True, exist_ok=True)
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            self.prioritized_path.parent.mkdir(parents=True, exist_ok=True)
            self.plan_path.parent.mkdir(parents=True, exist_ok=True)
            self.shell = object()
            self.llm = object()
            self.config = SimpleNamespace(objective="Reproduce the paper")
            self.trace = _TraceRecorder()
            (self.paper_dir / "rubric.json").write_text('{"name": "core", "weight": 10}', encoding="utf-8")
            (self.paper_dir / "addendum.md").write_text("Use official datasets only.", encoding="utf-8")
            (self.paper_dir / "blacklist.txt").write_text("http://blocked.example", encoding="utf-8")
            (self.analysis_dir / "summary.md").write_text("# Summary", encoding="utf-8")
            (self.analysis_dir / "structure.md").write_text("# Structure", encoding="utf-8")
            (self.analysis_dir / "algorithm.md").write_text("# Algorithm", encoding="utf-8")
            (self.analysis_dir / "experiments.md").write_text("# Experiments", encoding="utf-8")
            (self.analysis_dir / "baseline.md").write_text("# Baseline", encoding="utf-8")

        def _capabilities(self) -> dict[str, dict[str, bool]]:
            return {
                "online_research": {"available": True},
                "github_research": {"available": True},
            }

        def _read_text(self, path: Path, limit: int = 8_000) -> str:  # noqa: ARG002
            return path.read_text(encoding="utf-8") if path.exists() else ""

        def read_paper(self, refresh: bool = False):  # noqa: ARG002
            return "# Summary\n"

        def subagent_config(self, kind: str, base):  # noqa: ARG002
            return base

        def _capabilities(self) -> dict[str, dict[str, bool]]:
            return {
                "online_research": {"available": True},
                "github_research": {"available": True},
            }

        def constraints(self) -> dict[str, list[str]]:
            return {"blacklist": []}

    engine = Engine()

    def fake_run(self, context: str = ""):  # noqa: ANN001
        return SubagentOutput(
            status=SubagentStatus.COMPLETED,
            content="# Prioritized Implementation Plan\n\n## Executive Summary\n- P0: core method.\n",
            num_steps=6,
            runtime_seconds=2.0,
            log_path="/tmp/prioritization.jsonl",
        )

    monkeypatch.setattr(PaperPrioritizationSubagent, "run", fake_run, raising=False)

    result = PrioritizationRunner(engine).run()

    assert "parse_rubric" in result.task_description
    assert "write_priorities" in result.task_description
    assert "Treat each model variant as a separate task." in result.task_description
    assert result.prioritized_path.exists()
    assert result.plan_path.exists()
    assert "Prioritized tasks saved to" in result.summary
    assert "- P0: core method." in result.summary
