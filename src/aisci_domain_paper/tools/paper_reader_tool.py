from __future__ import annotations

from typing import Any

from aisci_agent_runtime.tools.base import Tool
from aisci_domain_paper.subagents.paper_reader import PaperReaderCoordinator


class ReadPaperTool(Tool):
    def __init__(self, engine) -> None:
        self.engine = engine

    def name(self) -> str:
        return "read_paper"

    def execute(self, shell, refresh: bool = False, **kwargs: Any) -> str:  # noqa: ARG002
        self.engine._ensure_workspace()
        if self._analysis_ready() and not refresh:
            summary_path = self.engine.analysis_dir / "summary.md"
            return summary_path.read_text(encoding="utf-8") if summary_path.exists() else "Paper analysis already exists at /home/agent/paper_analysis/summary.md."

        self.engine.trace.event("subagent_start", "read_paper started.", phase="analyze", payload={})
        result = PaperReaderCoordinator(self.engine).run()
        summary_path = self.engine.analysis_dir / "summary.md"
        self.engine.trace.event(
            "subagent_finish",
            "read_paper completed.",
            phase="analyze",
            payload={"summary": str(summary_path)},
        )
        return result.summary_with_navigation

    def _analysis_ready(self) -> bool:
        required = [
            self.engine.analysis_dir / "summary.md",
            self.engine.analysis_dir / "structure.md",
            self.engine.analysis_dir / "algorithm.md",
            self.engine.analysis_dir / "experiments.md",
            self.engine.analysis_dir / "baseline.md",
        ]
        return all(path.exists() for path in required)

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_paper",
                "description": "Read the staged paper bundle and write structured analysis artifacts to /home/agent/paper_analysis/.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "refresh": {
                            "type": "boolean",
                            "description": "Regenerate analysis even if files already exist.",
                        }
                    },
                    "additionalProperties": False,
                },
            },
        }


def build_read_paper_tool(engine):
    return ReadPaperTool(engine)


__all__ = ["ReadPaperTool", "build_read_paper_tool"]
