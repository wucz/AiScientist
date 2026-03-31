from __future__ import annotations

from typing import Any

from aisci_agent_runtime.tools.base import Tool
from aisci_domain_paper.subagents.prioritization import PrioritizationRunner


class PrioritizeTasksTool(Tool):
    def __init__(self, engine) -> None:
        self.engine = engine

    def name(self) -> str:
        return "prioritize_tasks"

    def execute(self, shell, refresh: bool = False, **kwargs: Any) -> str:  # noqa: ARG002
        self.engine._ensure_workspace()
        if self.engine.prioritized_path.exists() and self.engine.plan_path.exists() and not refresh:
            prioritized_text = self.engine.prioritized_path.read_text(encoding="utf-8")
            plan_text = self.engine.plan_path.read_text(encoding="utf-8")
            return "\n\n".join(
                [
                    "Prioritized plan already exists at /home/agent/prioritized_tasks.md.",
                    "",
                    prioritized_text,
                    "",
                    plan_text,
                ]
            ).strip()

        result = PrioritizationRunner(self.engine).run()
        self.engine.trace.event(
            "subagent_finish",
            "prioritize_tasks completed.",
            phase="prioritize",
            payload={"priorities": str(self.engine.prioritized_path)},
        )
        return result.summary

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "prioritize_tasks",
                "description": "Prioritize the paper implementation work and write a ranked plan to /home/agent/prioritized_tasks.md.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "refresh": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
            },
        }


def build_prioritize_tasks_tool(engine):
    return PrioritizeTasksTool(engine)


__all__ = ["PrioritizeTasksTool", "build_prioritize_tasks_tool"]
