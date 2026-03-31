from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aisci_agent_runtime.tools.base import SubagentCompleteSignal, Tool
from aisci_agent_runtime.tools.research_tools import GithubTool, LinkSummaryTool, LinterTool, WebSearchTool
from aisci_agent_runtime.tools.shell_tools import (
    AddExpLogTool,
    AddImplLogTool,
    BashToolWithTimeout,
    ExecCommandTool,
    PythonTool,
    ReadFileChunkTool,
    SearchFileTool,
)
from aisci_domain_paper.configs import (
    EXPERIMENT_BASH_DEFAULT_TIMEOUT,
    EXPERIMENT_COMMAND_TIMEOUT,
    IMPLEMENTATION_BASH_DEFAULT_TIMEOUT,
    MAIN_AGENT_BASH_DEFAULT_TIMEOUT,
    MAIN_AGENT_BASH_MAX_TIMEOUT,
)


@dataclass(frozen=True)
class _ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


def _capability_enabled(capabilities: dict[str, Any] | None, key: str) -> bool:
    if not capabilities:
        return False
    value = capabilities.get(key)
    if isinstance(value, dict):
        return bool(value.get("available"))
    return bool(value)


class CallbackTool(Tool):
    def __init__(self, spec: _ToolSpec, callback: Callable[..., Any]):
        self._spec = spec
        self._callback = callback

    def name(self) -> str:
        return self._spec.name

    def execute(self, shell, **kwargs) -> str:  # noqa: ANN001
        return str(self._callback(shell=shell, **kwargs))

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._spec.name,
                "description": self._spec.description,
                "parameters": self._spec.parameters,
            },
        }


def callback_tool(name: str, description: str, parameters: dict[str, Any], callback) -> CallbackTool:
    return CallbackTool(_ToolSpec(name=name, description=description, parameters=parameters), callback)


class MappedFileEditTool(Tool):
    def name(self) -> str:
        return "edit_file"

    def execute(
        self,
        shell,
        command: str,
        path: str,
        file_text: str = "",
        old_str: str = "",
        new_str: str = "",
        insert_line: int = 0,
        **kwargs: Any,
    ) -> str:
        if command == "create":
            shell.write_file(path, file_text)
            lines = file_text.count("\n") + 1 if file_text else 0
            return f"Created {path} ({lines} lines)"
        if command == "str_replace":
            if not old_str:
                return "Error: old_str is required for str_replace"
            if not shell.file_exists(path):
                return f"Error: {path} does not exist"
            content = shell.read_file(path)
            count = content.count(old_str)
            if count == 0:
                return f"Error: old_str not found in {path}. Use read_file_chunk first."
            if count > 1:
                return f"Error: old_str appears {count} times in {path}. Provide more context."
            shell.write_file(path, content.replace(old_str, new_str, 1))
            return f"Replaced in {path}"
        if command == "insert":
            if not shell.file_exists(path):
                return f"Error: {path} does not exist"
            content = shell.read_file(path)
            lines = content.split("\n")
            idx = max(0, min(insert_line, len(lines)))
            lines.insert(idx, new_str)
            shell.write_file(path, "\n".join(lines))
            return f"Inserted at line {idx} in {path}"
        return f"Error: unknown command '{command}'. Use create / str_replace / insert."

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Create or edit files with create, str_replace, or insert modes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "enum": ["create", "str_replace", "insert"]},
                        "path": {"type": "string"},
                        "file_text": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                        "insert_line": {"type": "integer"},
                    },
                    "required": ["command", "path"],
                    "additionalProperties": False,
                },
            },
        }


class PaperGitCommitTool(Tool):
    def name(self) -> str:
        return "git_commit"

    def execute(self, shell, message: str, **kwargs: Any) -> str:
        shell.send_command("cd /home/submission && git init 2>/dev/null || true", timeout=10)
        gitignore_path = "/home/submission/.gitignore"
        if not shell.file_exists(gitignore_path):
            shell.write_file(
                gitignore_path,
                "\n".join(
                    [
                        "# Auto-managed by AiScientist paper mode",
                        "venv/",
                        ".venv/",
                        "__pycache__/",
                        "*.pyc",
                        ".cache/",
                        "data/",
                        "models/",
                        "checkpoints/",
                        "",
                    ]
                ),
            )
        shell.write_file("/tmp/_paper_commit_msg.txt", message)
        result = shell.send_command(
            "cd /home/submission && "
            "git config user.email 'aiscientist@local' && "
            "git config user.name 'AiScientist' && "
            "git add -A && (git diff --cached --quiet || git commit -F /tmp/_paper_commit_msg.txt) 2>&1",
            timeout=90,
        )
        return result.output.strip() or "Nothing to commit."

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": "Stage and commit the /home/submission repository with the provided message.",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
        }


class SubmitTool(Tool):
    def name(self) -> str:
        return "submit"

    def execute(self, shell, summary: str, **kwargs: Any) -> str:  # noqa: ARG002
        raise SubagentCompleteSignal(summary, kwargs)

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Finish the paper run after reading, prioritization, implementation, experiments, and self-check are complete.",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        }


class FinishRunTool(SubmitTool):
    def name(self) -> str:
        return "finish_run"

    def get_tool_schema(self) -> dict[str, Any]:
        schema = super().get_tool_schema()
        schema["function"]["name"] = "finish_run"
        schema["function"]["description"] = "Compatibility alias for submit(). Prefer submit()."
        return schema


class PlanWriteTool(Tool):
    PATH = Path("/home/agent/plan.md")

    def name(self) -> str:
        return "write_plan"

    def execute(self, shell, content: str, **kwargs: Any) -> str:
        shell.write_file(str(self.PATH), content)
        return f"Wrote {self.PATH}"

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "write_plan",
                "description": "Write a planning note to /home/agent/plan.md.",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
        }


class ParseRubricTool(Tool):
    PATH = Path("/home/paper/rubric.json")

    def name(self) -> str:
        return "parse_rubric"

    def execute(self, shell, **kwargs: Any) -> str:  # noqa: ARG002
        if not self.PATH.exists():
            return "No rubric.json staged."
        try:
            payload = json.loads(self.PATH.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return f"Failed to parse rubric.json: {exc}"
        lines: list[str] = []
        self._collect(payload, lines, prefix="")
        return "\n".join(lines[:80]) if lines else "Rubric parsed but no weighted tasks were found."

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "parse_rubric",
                "description": "Parse /home/paper/rubric.json into a concise task and weight summary.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }

    def _collect(self, value: Any, lines: list[str], prefix: str) -> None:
        if isinstance(value, dict):
            name = value.get("name") or value.get("title") or prefix
            weight = value.get("weight")
            if name:
                suffix = f" (weight={weight})" if weight is not None else ""
                lines.append(f"- {name}{suffix}")
            for key, nested in value.items():
                if key in {"name", "title", "weight", "description"}:
                    continue
                self._collect(nested, lines, key)
        elif isinstance(value, list):
            for item in value:
                self._collect(item, lines, prefix)


class PriorityWriteTool(Tool):
    PRIORITY_PATH = Path("/home/agent/prioritized_tasks.md")
    PLAN_PATH = Path("/home/agent/plan.md")

    def name(self) -> str:
        return "write_priorities"

    def execute(self, shell, content: str, plan_content: str = "", **kwargs: Any) -> str:
        shell.write_file(str(self.PRIORITY_PATH), content)
        if plan_content:
            shell.write_file(str(self.PLAN_PATH), plan_content)
        return f"Wrote {self.PRIORITY_PATH}"

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "write_priorities",
                "description": "Write the prioritized implementation plan to /home/agent/prioritized_tasks.md and optionally /home/agent/plan.md.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "plan_content": {"type": "string"},
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
        }


def build_main_direct_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    tools: list[Tool] = [
        ReadFileChunkTool(),
        SearchFileTool(),
        BashToolWithTimeout(default_timeout=MAIN_AGENT_BASH_DEFAULT_TIMEOUT, max_timeout=MAIN_AGENT_BASH_MAX_TIMEOUT),
        PythonTool(default_timeout=600, max_timeout=7200),
    ]
    if _capability_enabled(capabilities, "online_research"):
        tools.append(WebSearchTool())
        tools.append(LinkSummaryTool())
    return tools


def build_shared_file_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    tools = build_main_direct_tools(capabilities)
    tools.append(MappedFileEditTool())
    return tools


def build_reader_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [
        ReadFileChunkTool(),
        SearchFileTool(),
        BashToolWithTimeout(
            default_timeout=MAIN_AGENT_BASH_DEFAULT_TIMEOUT,
            max_timeout=MAIN_AGENT_BASH_MAX_TIMEOUT,
        ),
        SubagentCompleteTool(),
    ]


def build_search_strategist_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [ReadFileChunkTool(), SubagentCompleteTool()]


def build_search_executor_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [SearchFileTool(), ReadFileChunkTool(), SubagentCompleteTool()]


def build_prioritization_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [
        *build_main_direct_tools(capabilities),
        ParseRubricTool(),
        PriorityWriteTool(),
        SubagentCompleteTool(),
    ]


def build_explore_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [*build_main_direct_tools(capabilities), SubagentCompleteTool()]


def build_plan_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [*build_main_direct_tools(capabilities), PlanWriteTool(), SubagentCompleteTool()]


def build_general_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [*build_main_direct_tools(capabilities), SubagentCompleteTool()]


def build_generic_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    return build_general_tools(capabilities)


def build_implementation_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    tools: list[Tool] = [
        ReadFileChunkTool(),
        SearchFileTool(),
        BashToolWithTimeout(default_timeout=IMPLEMENTATION_BASH_DEFAULT_TIMEOUT, max_timeout=36_000),
        PythonTool(default_timeout=1_800, max_timeout=36_000),
        MappedFileEditTool(),
        PaperGitCommitTool(),
        AddImplLogTool(),
        LinterTool(),
    ]
    if _capability_enabled(capabilities, "online_research"):
        tools.extend([WebSearchTool(), LinkSummaryTool()])
    if _capability_enabled(capabilities, "github_research"):
        tools.append(GithubTool())
    tools.append(SubagentCompleteTool())
    return tools


def build_experiment_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    tools: list[Tool] = [
        ReadFileChunkTool(),
        SearchFileTool(),
        BashToolWithTimeout(default_timeout=EXPERIMENT_BASH_DEFAULT_TIMEOUT, max_timeout=36_000),
        PythonTool(default_timeout=3_600, max_timeout=36_000),
        MappedFileEditTool(),
        ExecCommandTool(default_timeout=EXPERIMENT_COMMAND_TIMEOUT, max_timeout=18_000),
        PaperGitCommitTool(),
        AddExpLogTool(),
        LinterTool(),
    ]
    if _capability_enabled(capabilities, "online_research"):
        tools.extend([WebSearchTool(), LinkSummaryTool()])
    tools.append(SubagentCompleteTool())
    return tools


def build_env_setup_tools() -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    return [
        ReadFileChunkTool(),
        BashToolWithTimeout(default_timeout=600, max_timeout=3600),
        MappedFileEditTool(),
        SubagentCompleteTool(),
    ]


def build_resource_download_tools(capabilities: dict[str, Any] | None = None) -> list[Tool]:
    from aisci_agent_runtime.tools.base import SubagentCompleteTool

    tools: list[Tool] = [
        ReadFileChunkTool(),
        BashToolWithTimeout(default_timeout=900, max_timeout=7200),
        PythonTool(default_timeout=900, max_timeout=7200),
        MappedFileEditTool(),
    ]
    if _capability_enabled(capabilities, "online_research"):
        tools.extend([WebSearchTool(), LinkSummaryTool()])
    tools.append(SubagentCompleteTool())
    return tools
