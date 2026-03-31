from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aisci_agent_runtime.subagents.base import SubagentOutput, SubagentStatus
from aisci_domain_paper.configs import DEFAULT_PRIORITIZATION_CONFIG
from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.tools import build_prioritization_tools


class PaperPrioritizationSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "prioritization"

    def system_prompt(self) -> str:
        return self.engine.render_subagent_prompt("prioritization")

    def get_tools(self):
        return build_prioritization_tools(self.capabilities)


@dataclass(frozen=True)
class PrioritizationResult:
    prioritized_path: Path
    plan_path: Path
    summary: str
    output: SubagentOutput
    task_description: str


class PrioritizationRunner:
    def __init__(self, engine) -> None:
        self.engine = engine

    def run(self) -> PrioritizationResult:
        summary = self.engine.read_paper(refresh=False)
        rubric = self.engine._read_text(self.engine.paper_dir / "rubric.json", limit=12_000)
        addendum = self.engine._read_text(self.engine.paper_dir / "addendum.md", limit=8_000)
        blacklist = self.engine._read_text(self.engine.paper_dir / "blacklist.txt", limit=4_000)
        task_description = self._build_task_description(summary, rubric, addendum, blacklist)
        config = self.engine.subagent_config("prioritization", DEFAULT_PRIORITIZATION_CONFIG)
        subagent = PaperPrioritizationSubagent(
            self.engine,
            self.engine.shell,
            self.engine.llm,
            config,
            objective="Create a prioritized implementation plan for reproducing the paper.",
            context=task_description,
        )
        self.engine.trace.event("subagent_start", "prioritize_tasks started.", phase="prioritize", payload={})
        output = subagent.run(context=subagent.build_context())
        self.engine.trace.event(
            "subagent_finish",
            "prioritize_tasks completed.",
            phase="prioritize",
            payload={"status": output.status.value, "log_path": output.log_path},
        )

        prioritized_path = self.engine.prioritized_path
        plan_path = self.engine.plan_path
        if not prioritized_path.exists():
            prioritized_path.write_text(self._fallback_priorities(summary, rubric, addendum, blacklist), encoding="utf-8")
        if not plan_path.exists():
            plan_path.write_text(
                "# Plan\n\nRefer to `prioritized_tasks.md` for the authoritative execution order.\n",
                encoding="utf-8",
            )

        return PrioritizationResult(
            prioritized_path=prioritized_path,
            plan_path=plan_path,
            summary=self._build_agent_summary(output, prioritized_path, plan_path),
            output=output,
            task_description=task_description,
        )

    def _build_task_description(self, summary: str, rubric: str, addendum: str, blacklist: str) -> str:
        return "\n\n".join(
            [
                "Analyze the paper analysis and any staged rubric hints to create a prioritized implementation plan.",
                "",
                "## Paper Analysis Context",
                summary.strip() or "(paper summary missing)",
                "",
                "## Detailed Files",
                "- /home/agent/paper_analysis/summary.md",
                "- /home/agent/paper_analysis/structure.md",
                "- /home/agent/paper_analysis/algorithm.md",
                "- /home/agent/paper_analysis/experiments.md",
                "- /home/agent/paper_analysis/baseline.md",
                "",
                "## Other Files to Check",
                "- /home/paper/rubric.json",
                "- /home/paper/addendum.md",
                "- /home/paper/blacklist.txt",
                "",
                "## Rubric Hints",
                rubric.strip() or "No rubric.json staged.",
                "",
                "## Addendum Constraints",
                addendum.strip() or "No addendum.md staged.",
                "",
                "## Blacklist Constraints",
                blacklist.strip() or "No blacklist.txt staged.",
                "",
                "## Required Workflow",
                "1. Use `parse_rubric` when rubric.json exists.",
                "2. Read paper_analysis files with read_file_chunk/search_file when needed.",
                "3. Treat baselines in main-text tables as P0.",
                "4. Treat each model variant as a separate task.",
                "5. Use `write_priorities` to write /home/agent/prioritized_tasks.md and /home/agent/plan.md.",
                "6. Return concise findings with `subagent_complete` when done.",
                "",
                "## Output Requirements",
                "The prioritized_tasks.md file should contain:",
                "- Executive summary",
                "- P0/P1/P2/P3 breakdown",
                "- Dependency graph",
                "- Risk assessment",
                "- Recommended execution order",
                "- Time allocation guidance",
            ]
        ).strip()

    def _fallback_priorities(self, summary: str, rubric: str, addendum: str, blacklist: str) -> str:
        return "\n".join(
            [
                "# Prioritized Implementation Plan",
                "",
                "## Executive Summary",
                "",
                "- P0: reproduce.sh, core method, main experiments, baselines in main tables.",
                "- P1: important secondary experiments and hardening work.",
                "- P2: appendix-only or lower-leverage work.",
                "",
                "## Analysis Inputs",
                "",
                "### Paper Summary",
                "",
                summary.strip() or "No paper summary available.",
                "",
                "### Rubric",
                "",
                rubric.strip() or "No rubric.json staged.",
                "",
                "### Addendum",
                "",
                addendum.strip() or "No addendum.md staged.",
                "",
                "### Blacklist",
                "",
                blacklist.strip() or "No blacklist.txt staged.",
                "",
                "## Recommended Execution Order",
                "1. Stabilize reproduce.sh.",
                "2. Implement core method.",
                "3. Run experiments.",
                "4. Run clean validation.",
                "",
                "## Time Allocation Strategy",
                "- 40% P0 tasks",
                "- 35% P1 tasks",
                "- 20% P2 tasks",
                "- 5% debugging buffer",
            ]
        ).strip() + "\n"

    def _build_agent_summary(self, output: SubagentOutput, prioritized_path: Path, plan_path: Path) -> str:
        status_icon = "✓" if output.status == SubagentStatus.COMPLETED else "✗"
        lines = [
            f"[Prioritization {status_icon}] ({output.num_steps} steps, {output.runtime_seconds:.1f}s)",
            "",
            f"**Prioritized tasks saved to**: `{prioritized_path}`",
            f"**Plan saved to**: `{plan_path}`",
            "",
            "## Summary",
            output.content.strip() or "(no subagent output)",
            "",
            "---",
            "",
            "**Next Steps**:",
            "1. Review `/home/agent/prioritized_tasks.md`.",
            "2. Start with P0-Critical tasks.",
            "3. Use `spawn_subagent(type='plan')` if you need task decomposition.",
        ]
        return "\n".join(lines).strip()


__all__ = ["PaperPrioritizationSubagent", "PrioritizationResult", "PrioritizationRunner"]
