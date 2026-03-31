from __future__ import annotations

import base64
import json
import os
import re
import urllib.parse
import urllib.request
from html import unescape
from typing import Any

from aisci_agent_runtime.tools.base import Tool


def _blocked_by_constraints(candidate: str, constraints: dict[str, Any] | None) -> str | None:
    if not constraints:
        return None
    blacklist = constraints.get("blacklist") or constraints.get("blocked_resources") or []
    for blocked in blacklist:
        if not blocked:
            continue
        if str(blocked).lower() in candidate.lower():
            return f"Blocked by blacklist constraint: {blocked}"
    return None


def _fetch_text(url: str, headers: dict[str, str] | None = None, max_chars: int = 20_000) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AiScientist/1.0",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
        body = response.read()
    text = body.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(re.sub(r"\s+", " ", text)).strip()
    return text[:max_chars]


class WebSearchTool(Tool):
    def name(self) -> str:
        return "web_search"

    def supports_constraints(self) -> bool:
        return True

    def execute_with_constraints(self, shell, constraints: dict[str, Any] | None = None, **kwargs) -> str:
        return self.execute(shell, constraints=constraints, **kwargs)

    def execute(
        self,
        shell,  # noqa: ARG002
        query: str,
        max_results: int = 5,
        constraints: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        blocked = _blocked_by_constraints(query, constraints)
        if blocked:
            return blocked
        encoded = urllib.parse.quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={encoded}"
        try:
            html = _fetch_text(url, max_chars=25_000)
        except Exception as exc:  # noqa: BLE001
            return f"web_search failed: {exc}"
        lines = [line.strip() for line in html.split(" Result ") if line.strip()]
        results = []
        for line in lines:
            results.append(line[:500])
            if len(results) >= max(1, min(max_results, 10)):
                break
        if not results:
            return "No search results extracted."
        return "\n\n".join(f"[{idx}] {item}" for idx, item in enumerate(results, start=1))

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for documentation, installation guidance, dataset pages, or debugging hints.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }


class LinkSummaryTool(Tool):
    def name(self) -> str:
        return "link_summary"

    def supports_constraints(self) -> bool:
        return True

    def execute_with_constraints(self, shell, constraints: dict[str, Any] | None = None, **kwargs) -> str:
        return self.execute(shell, constraints=constraints, **kwargs)

    def execute(
        self,
        shell,  # noqa: ARG002
        url: str,
        focus: str = "",
        constraints: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        blocked = _blocked_by_constraints(url, constraints)
        if blocked:
            return blocked
        try:
            text = _fetch_text(url)
        except Exception as exc:  # noqa: BLE001
            return f"link_summary failed: {exc}"
        summary_lines = [f"URL: {url}"]
        if focus:
            summary_lines.append(f"Focus: {focus}")
        summary_lines.extend(
            [
                "",
                "Content preview:",
                text[:4_000] or "(empty response)",
            ]
        )
        return "\n".join(summary_lines)

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "link_summary",
                "description": "Fetch a URL and return a concise text preview for targeted documentation lookup.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "focus": {"type": "string"},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        }


class GithubTool(Tool):
    def name(self) -> str:
        return "github"

    def supports_constraints(self) -> bool:
        return True

    def execute_with_constraints(self, shell, constraints: dict[str, Any] | None = None, **kwargs) -> str:
        return self.execute(shell, constraints=constraints, **kwargs)

    def execute(
        self,
        shell,  # noqa: ARG002
        action: str,
        query: str = "",
        repo: str = "",
        path: str = "",
        ref: str = "HEAD",
        max_results: int = 5,
        constraints: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            return "github tool unavailable: GITHUB_TOKEN is not set."
        candidate = " ".join(part for part in [query, repo, path] if part)
        blocked = _blocked_by_constraints(candidate, constraints)
        if blocked:
            return blocked
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            if action == "search_repositories":
                payload = self._get_json(
                    f"https://api.github.com/search/repositories?q={urllib.parse.quote_plus(query)}&per_page={max(1, min(max_results, 10))}",
                    headers=headers,
                )
                items = payload.get("items", [])
                return "\n".join(
                    f"- {item['full_name']}: {item.get('description') or ''}".strip()
                    for item in items
                ) or "No repositories found."
            if action == "search_code":
                payload = self._get_json(
                    f"https://api.github.com/search/code?q={urllib.parse.quote_plus(query)}&per_page={max(1, min(max_results, 10))}",
                    headers=headers,
                )
                items = payload.get("items", [])
                return "\n".join(
                    f"- {item['repository']['full_name']}:{item['path']}"
                    for item in items
                ) or "No code matches found."
            if action == "read_file":
                payload = self._get_json(
                    f"https://api.github.com/repos/{repo}/contents/{path}?ref={urllib.parse.quote_plus(ref)}",
                    headers=headers,
                )
                if payload.get("encoding") == "base64":
                    data = base64.b64decode(payload.get("content", "")).decode("utf-8", errors="replace")
                    return data[:12_000]
                return json.dumps(payload, indent=2)[:12_000]
        except Exception as exc:  # noqa: BLE001
            return f"github tool failed: {exc}"
        return "Unsupported github action. Use search_repositories, search_code, or read_file."

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "github",
                "description": "Search GitHub repositories/code or fetch a file from a repository when GITHUB_TOKEN is available.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["search_repositories", "search_code", "read_file"],
                        },
                        "query": {"type": "string"},
                        "repo": {"type": "string"},
                        "path": {"type": "string"},
                        "ref": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))


class LinterTool(Tool):
    def name(self) -> str:
        return "linter"

    def execute(
        self,
        shell,
        path: str = "/home/submission",
        command: str = "",
        timeout: int = 120,
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        lint_command = command.strip() or f"python -m compileall {path}"
        result = shell.send_command(lint_command, timeout=timeout)
        return result.output.strip() or f"exit_code={result.exit_code}"

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "linter",
                "description": "Run a lightweight lint or syntax-validation command over the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "additionalProperties": False,
                },
            },
        }
