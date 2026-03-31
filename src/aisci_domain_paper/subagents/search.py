from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from aisci_domain_paper.prompts.templates import SEARCH_EXECUTOR_PROMPT, SEARCH_STRATEGIST_PROMPT
from aisci_agent_runtime.subagents.base import SubagentConfig
from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.tools.basic_tool import build_search_executor_tools, build_search_strategist_tools


@dataclass(frozen=True)
class SearchTask:
    focus: str
    files: list[str]
    keywords: list[str]
    sections: str = ""


DEFAULT_SEARCH_STRATEGY_CONFIG = SubagentConfig(max_steps=50, time_limit=900, reminder_freq=10)
DEFAULT_SEARCH_EXECUTOR_CONFIG = SubagentConfig(max_steps=100, time_limit=1800, reminder_freq=10)


class SearchStrategistSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "search_strategist"

    def system_prompt(self) -> str:
        return SEARCH_STRATEGIST_PROMPT

    def get_tools(self):
        return build_search_strategist_tools(self.capabilities)


class SearchExecutorSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "search_executor"

    def system_prompt(self) -> str:
        return SEARCH_EXECUTOR_PROMPT

    def get_tools(self):
        return build_search_executor_tools(self.capabilities)


class SearchPaperTool:
    def __init__(self, engine) -> None:
        self.engine = engine

    def name(self) -> str:
        return "search_paper"

    def execute(
        self,
        shell,  # noqa: ARG002
        query: str,
        search_strategy: str = "auto",
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        if self.engine.llm is None:
            return "search_paper requires a configured LLM client."

        if search_strategy == "simple":
            return self._run_simple_search(query)

        strategy_text = self.engine.run_subagent_instance(
            SearchStrategistSubagent,
            objective=f"Create a focused search plan for: {query}",
            context=self._strategy_context(query),
            config=DEFAULT_SEARCH_STRATEGY_CONFIG,
            phase="analyze",
            label="search_strategist",
        )
        tasks = self._parse_strategy(strategy_text, query)
        if not tasks:
            return self._run_simple_search(query)

        search_outputs: list[str] = []
        for index, task in enumerate(tasks[:3], start=1):
            search_outputs.append(
                self.engine.run_subagent_instance(
                    SearchExecutorSubagent,
                    objective=self._executor_objective(task),
                    context=self._executor_context(task),
                    config=DEFAULT_SEARCH_EXECUTOR_CONFIG,
                    phase="analyze",
                    label=f"search_executor_{index}",
                )
            )

        return "\n\n".join(
            [
                "# Search Results",
                "",
                "## Query",
                query,
                "",
                "## Strategy",
                strategy_text,
                "",
                "## Findings",
                *search_outputs,
            ]
        ).strip()

    def _run_simple_search(self, query: str) -> str:
        result = self.engine.run_subagent_instance(
            SearchExecutorSubagent,
            objective=f"Search for information relevant to: {query}",
            context=self._executor_context(SearchTask(focus=query, files=["/home/paper/paper.md"], keywords=[])),
            config=DEFAULT_SEARCH_EXECUTOR_CONFIG,
            phase="analyze",
            label="search_executor",
        )
        return "\n".join(["# Search Results", "", result]).strip()

    def _strategy_context(self, query: str) -> str:
        return "\n".join(
            [
                f"Query: {query}",
                "",
                "Available files:",
                "- /home/paper/paper.md",
                "- /home/paper/addendum.md",
                "- /home/paper/blacklist.txt",
            ]
        )

    def _executor_context(self, task: SearchTask) -> str:
        files = "\n".join(f"- {path}" for path in task.files) or "- /home/paper/paper.md"
        keywords = ", ".join(task.keywords) if task.keywords else "(none)"
        parts = [
            f"Focus: {task.focus}",
            "",
            "Files to inspect:",
            files,
            "",
            f"Keywords: {keywords}",
        ]
        if task.sections:
            parts.extend(["", f"Sections: {task.sections}"])
        return "\n".join(parts)

    def _executor_objective(self, task: SearchTask) -> str:
        return f"Find concrete evidence for: {task.focus}"

    def _parse_strategy(self, strategy_text: str, original_query: str) -> list[SearchTask]:
        match = re.search(r"\{[\s\S]*\}", strategy_text)
        if not match:
            return [SearchTask(focus=original_query, files=["/home/paper/paper.md"], keywords=[])]
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [SearchTask(focus=original_query, files=["/home/paper/paper.md"], keywords=[])]
        searches = payload.get("searches", [])
        tasks: list[SearchTask] = []
        for item in searches:
            if not isinstance(item, dict):
                continue
            tasks.append(
                SearchTask(
                    focus=str(item.get("focus", original_query)),
                    files=[str(path) for path in item.get("files", ["/home/paper/paper.md"]) if path],
                    keywords=[str(keyword) for keyword in item.get("keywords", []) if keyword],
                    sections=str(item.get("sections", "")),
                )
            )
        return tasks or [SearchTask(focus=original_query, files=["/home/paper/paper.md"], keywords=[])]

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_paper",
                "description": "Coordinate a targeted paper search using a strategist and one or more search executors.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "search_strategy": {"type": "string", "enum": ["auto", "simple"]},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }


def build_search_paper_tool(engine):
    return SearchPaperTool(engine)


__all__ = [
    "SearchExecutorSubagent",
    "SearchPaperTool",
    "SearchStrategistSubagent",
    "build_search_paper_tool",
]
