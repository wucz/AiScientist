from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from aisci_agent_runtime.subagents.base import SubagentOutput, SubagentStatus
from aisci_domain_paper.configs import (
    DEFAULT_PAPER_READER_CONFIG,
    DEFAULT_PAPER_STRUCTURE_CONFIG,
    DEFAULT_PAPER_SYNTHESIS_CONFIG,
)
from aisci_domain_paper.prompts.templates import (
    ALGORITHM_SYSTEM_PROMPT,
    BASELINE_SYSTEM_PROMPT,
    EXPERIMENTS_SYSTEM_PROMPT,
    STRUCTURE_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)
from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.tools import build_reader_tools


class PaperReaderSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_reader"

    def system_prompt(self) -> str:
        return self.engine.render_subagent_prompt("paper_reader")

    def get_tools(self):
        return build_reader_tools(self.capabilities)


class StructureSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_structure"

    def system_prompt(self) -> str:
        return STRUCTURE_SYSTEM_PROMPT

    def get_tools(self):
        return build_reader_tools(self.capabilities)


class AlgorithmSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_algorithm"

    def system_prompt(self) -> str:
        return ALGORITHM_SYSTEM_PROMPT

    def get_tools(self):
        return build_reader_tools(self.capabilities)


class ExperimentsSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_experiments"

    def system_prompt(self) -> str:
        return EXPERIMENTS_SYSTEM_PROMPT

    def get_tools(self):
        return build_reader_tools(self.capabilities)


class BaselineSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_baseline"

    def system_prompt(self) -> str:
        return BASELINE_SYSTEM_PROMPT

    def get_tools(self):
        return build_reader_tools(self.capabilities)


class SynthesisSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "paper_synthesis"

    def system_prompt(self) -> str:
        return SYNTHESIS_SYSTEM_PROMPT

    def get_tools(self):
        return build_reader_tools(self.capabilities)


@dataclass(frozen=True)
class PaperAnalysisSection:
    name: str
    filename: str
    content: str
    description: str
    line_count: int = 0

    def __post_init__(self) -> None:
        if self.line_count == 0:
            object.__setattr__(self, "line_count", len(self.content.splitlines()) or 1)


@dataclass
class PaperAnalysisResult:
    executive_summary: str
    sections: dict[str, PaperAnalysisSection] = field(default_factory=dict)
    outputs: dict[str, SubagentOutput] = field(default_factory=dict)
    all_success: bool = True
    failed_subagents: list[str] = field(default_factory=list)
    total_runtime_seconds: float = 0.0

    @property
    def summary_with_navigation(self) -> str:
        nav_table = self._build_navigation_table()
        return "\n\n".join(
            [
                self.executive_summary.strip(),
                "---",
                "## Detailed Analysis Files",
                "",
                nav_table,
                "",
                "**How to Access Details:**",
                "- Use `read_file_chunk(file=\"/home/agent/paper_analysis/<section>.md\")` to read specific sections.",
                "- Use `search_file(file=\"/home/agent/paper_analysis/<section>.md\", query=\"...\")` to search within a section.",
                "- The executive summary above is the quick reference; the detailed files contain the full phase outputs.",
            ]
        ).strip()

    def _build_navigation_table(self) -> str:
        lines = [
            "| Section | File | Lines | Description |",
            "|---------|------|-------|-------------|",
        ]
        for section in self.sections.values():
            filepath = f"/home/agent/paper_analysis/{section.filename}"
            lines.append(
                f"| {section.name} | `{filepath}` | {section.line_count} | {section.description} |"
            )
        return "\n".join(lines)


class PaperReaderCoordinator:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.structure_config = engine.subagent_config("paper_reader", DEFAULT_PAPER_STRUCTURE_CONFIG)
        self.reader_config = engine.subagent_config("paper_reader", DEFAULT_PAPER_READER_CONFIG)
        self.synthesis_config = engine.subagent_config("paper_reader", DEFAULT_PAPER_SYNTHESIS_CONFIG)

    def run(self) -> PaperAnalysisResult:
        self.engine.analysis_dir.mkdir(parents=True, exist_ok=True)
        start = perf_counter()

        structure_out = self._run_stage(
            StructureSubagent,
            objective="Extract the paper structure, metadata, and constraints.",
            context=self.engine.reader_context(),
            config=self.structure_config,
            phase="analyze",
            label="paper_structure",
        )
        algorithm_out = self._run_stage(
            AlgorithmSubagent,
            objective="Extract the implementation-facing algorithm and architecture details.",
            context=self._stage_context(structure_out.content),
            config=self.reader_config,
            phase="analyze",
            label="paper_algorithm",
        )
        experiments_out = self._run_stage(
            ExperimentsSubagent,
            objective="Extract the paper's datasets, metrics, tables, and evaluation procedure.",
            context=self._stage_context(structure_out.content),
            config=self.reader_config,
            phase="analyze",
            label="paper_experiments",
        )
        baseline_out = self._run_stage(
            BaselineSubagent,
            objective="Identify baselines, comparison methods, and model variants that must be reproduced.",
            context=self._stage_context(structure_out.content),
            config=self.reader_config,
            phase="analyze",
            label="paper_baseline",
        )
        synthesis_out = self._run_stage(
            SynthesisSubagent,
            objective="Synthesize the paper analysis into a concise overview for downstream implementation agents.",
            context=self._synthesis_context(
                structure_out.content,
                algorithm_out.content,
                experiments_out.content,
                baseline_out.content,
            ),
            config=self.synthesis_config,
            phase="analyze",
            label="paper_synthesis",
        )

        sections = {
            "structure": self._section("Structure", "structure.md", structure_out.content, "Paper structure, metadata, and constraints."),
            "algorithm": self._section("Algorithm", "algorithm.md", algorithm_out.content, "Implementation-facing algorithm and architecture details."),
            "experiments": self._section("Experiments", "experiments.md", experiments_out.content, "Datasets, metrics, tables, and evaluation procedure."),
            "baseline": self._section("Baseline", "baseline.md", baseline_out.content, "Baselines, comparison methods, and model variants."),
            "summary": self._section("Executive Summary", "summary.md", synthesis_out.content, "Quick reference for the main agent."),
        }

        result = PaperAnalysisResult(
            executive_summary=synthesis_out.content.strip() or self._fallback_summary(sections),
            sections=sections,
            outputs={
                "structure": structure_out,
                "algorithm": algorithm_out,
                "experiments": experiments_out,
                "baseline": baseline_out,
                "synthesis": synthesis_out,
            },
            all_success=all(out.status == SubagentStatus.COMPLETED for out in (structure_out, algorithm_out, experiments_out, baseline_out, synthesis_out)),
            failed_subagents=[
                name
                for name, output in {
                    "structure": structure_out,
                    "algorithm": algorithm_out,
                    "experiments": experiments_out,
                    "baseline": baseline_out,
                    "synthesis": synthesis_out,
                }.items()
                if output.status != SubagentStatus.COMPLETED
            ],
            total_runtime_seconds=perf_counter() - start,
        )

        self._write_outputs(result)
        return result

    def _run_stage(
        self,
        subagent_cls,
        *,
        objective: str,
        context: str,
        config,
        phase: str,
        label: str,
    ) -> SubagentOutput:
        subagent = subagent_cls(
            self.engine,
            self.engine.shell,
            self.engine.llm,
            config,
            objective=objective,
            context=context,
        )
        self.engine.trace.event(
            "subagent_start",
            f"{label} subagent started.",
            phase=phase,
            payload={"objective": objective},
        )
        output = subagent.run(context=subagent.build_context())
        self.engine.trace.event(
            "subagent_finish",
            f"{label} subagent finished with status={output.status.value}.",
            phase=phase,
            payload={"status": output.status.value, "log_path": output.log_path},
        )
        return output

    def _write_outputs(self, result: PaperAnalysisResult) -> None:
        for section in result.sections.values():
            (self.engine.analysis_dir / section.filename).write_text(section.content.rstrip() + "\n", encoding="utf-8")
        (self.engine.analysis_dir / "summary.md").write_text(result.summary_with_navigation.rstrip() + "\n", encoding="utf-8")

    def _stage_context(self, structure_content: str) -> str:
        return "\n\n".join(
            [
                self.engine.reader_context(),
                "Existing structure analysis:",
                structure_content.strip() or "(no structure analysis available)",
            ]
        ).strip()

    def _synthesis_context(self, structure: str, algorithm: str, experiments: str, baseline: str) -> str:
        return "\n\n".join(
            [
                "Paper analysis inputs:",
                f"Structure:\n{structure.strip()}",
                f"Algorithm:\n{algorithm.strip()}",
                f"Experiments:\n{experiments.strip()}",
                f"Baseline:\n{baseline.strip()}",
            ]
        ).strip()

    def _section(self, name: str, filename: str, content: str, description: str) -> PaperAnalysisSection:
        return PaperAnalysisSection(name=name, filename=filename, content=content, description=description)

    def _fallback_summary(self, sections: dict[str, PaperAnalysisSection]) -> str:
        lines = [
            "# Paper Analysis: Executive Summary",
            "",
            "*Synthesis agent did not complete. Showing section previews.*",
            "",
        ]
        for key in ("structure", "algorithm", "experiments", "baseline"):
            section = sections[key]
            preview = "\n".join(section.content.splitlines()[:12]).strip()
            lines.extend(
                [
                    f"## {section.name}",
                    "",
                    preview or "(empty)",
                    "",
                ]
            )
        return "\n".join(lines).strip()


__all__ = [
    "AlgorithmSubagent",
    "BaselineSubagent",
    "ExperimentsSubagent",
    "PaperAnalysisResult",
    "PaperAnalysisSection",
    "PaperReaderCoordinator",
    "PaperReaderSubagent",
    "StructureSubagent",
    "SynthesisSubagent",
]
