from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from aisci_agent_runtime.subagents.base import SubagentOutput
from aisci_agent_runtime.tools.base import Tool
from aisci_domain_paper.configs import DEFAULT_EXPERIMENT_CONFIG, EXPERIMENT_VALIDATE_TIME_LIMIT


CLEANUP_COMMANDS = [
    "cd /home/submission && git add -A && (git diff --cached --quiet || git commit -m 'auto-commit before clean validation')",
    "cd /home/submission && git clean -fd",
    "rm -rf /home/submission/venv /home/submission/.venv",
    "rm -rf ~/.cache/huggingface/datasets/",
    "rm -rf ~/.cache/torch/hub/",
    "rm -rf /home/submission/.hf_cache /home/submission/.cache /home/submission/huggingface",
    "echo '--- Hardcoded path check ---' && grep -rn '/home/submission' /home/submission/src/ /home/submission/*.py /home/submission/scripts/*.sh 2>/dev/null | grep -v '.pyc' | grep -v '__pycache__' || true",
    "cd /home/submission && python3 -m venv venv",
]


class CleanReproduceValidationTool(Tool):
    def __init__(self, engine) -> None:
        self.engine = engine

    def name(self) -> str:
        return "clean_reproduce_validation"

    def execute(self, shell, refresh: bool = False, time_budget: int | None = None, **kwargs: Any) -> str:  # noqa: ARG002
        self.engine._ensure_workspace()
        if self.engine.self_check_path.exists() and not refresh:
            return "Final self-check report already exists at /home/agent/final_self_check.md."

        self.engine.trace.event(
            "clean_validation_started",
            "Final self-check started.",
            phase="validate",
            payload={},
        )

        cleanup_lines: list[str] = []
        for command in CLEANUP_COMMANDS:
            result = shell.send_command(command, timeout=300)
            status = "ok" if result.exit_code == 0 else f"exit={result.exit_code}"
            cleanup_lines.append(f"$ {command}\n[{status}]\n{result.output.strip()}".strip())
        cleanup_summary = "\n\n".join(cleanup_lines).strip()

        hardcoded_result = shell.send_command(
            "grep -rn '/home/submission' /home/submission/src/ /home/submission/*.py /home/submission/scripts/*.sh 2>/dev/null || true",
            timeout=20,
        )
        hardcoded_hits = [line for line in hardcoded_result.output.splitlines() if line.strip()]

        result = self._run_experiment_subagent(cleanup_summary, time_budget=time_budget)
        experiment_result = self._format_subagent_result(result)

        passed = "failed" not in experiment_result.lower() and "error" not in experiment_result.lower()
        if hardcoded_hits:
            passed = False
        status = "passed" if passed else "failed"

        report = {
            "status": status,
            "cleanup_summary": cleanup_summary,
            "hardcoded_path_hits": hardcoded_hits,
            "result": experiment_result,
        }
        self.engine.self_check_path.write_text(
            "\n".join(
                [
                    "# Final Self-Check",
                    "",
                    f"- Status: **{status}**",
                    f"- Hardcoded `/home/submission` Hits: {len(hardcoded_hits)}",
                    "",
                    "## Cleanup Summary",
                    "",
                    "```text",
                    cleanup_summary or "(empty)",
                    "```",
                    "",
                    "## Validation Result",
                    "",
                    experiment_result,
                ]
            ).rstrip()
            + "\n",
            encoding="utf-8",
        )
        self.engine.self_check_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.engine.mark_validation_run()
        self.engine.trace.event(
            "clean_validation_finished",
            "Final self-check completed.",
            phase="validate",
            payload={"status": status, "hardcoded_hits": len(hardcoded_hits)},
        )
        return f"Final self-check {status}. Report: /home/agent/final_self_check.md"

    def _run_experiment_subagent(self, cleanup_summary: str, *, time_budget: int | None) -> SubagentOutput:
        session = self.engine.state_manager.create_session("clean_val")
        self.engine.state_manager.append_separator(session)
        task = self._task_description(cleanup_summary)
        cfg = self.engine.subagent_config("experiment", DEFAULT_EXPERIMENT_CONFIG)
        cfg = replace(
            cfg,
            time_limit=time_budget or EXPERIMENT_VALIDATE_TIME_LIMIT,
            log_dir=str(session.directory),
        )
        return self.engine.run_subagent_output(
            self.engine.subagent_class("experiment"),
            objective=task,
            context="",
            config=cfg,
            session=session,
            phase="validate",
            label="experiment",
        )

    def _task_description(self, cleanup_summary: str) -> str:
        parts = [
            "## Clean Environment Validation (Pre-Submit)",
            "",
            "### Environment Cleanup Performed",
            "The following cleanup was run to simulate a fresh execution environment:",
            "```text",
            cleanup_summary or "(empty)",
            "```",
            "",
            "### Your Task",
            "1. Run `cd /home/submission && bash reproduce.sh` from a clean state.",
            "2. Treat hardcoded `/home/submission` paths as a reproducibility failure.",
            "3. Diagnose missing dependencies, dataset download failures, or files lost by `git clean -fd`.",
            "4. If validation succeeds, confirm the output artifacts are coherent.",
        ]
        injected = self.engine.state_manager.recent_impl_history()
        if injected:
            parts.extend(["", "### Recent Implementation History", "", injected])
        return "\n".join(parts).strip()

    def _format_subagent_result(self, result: SubagentOutput) -> str:
        header = f"[Clean Validation Experiment | {result.status.value}] ({result.num_steps} steps, {result.runtime_seconds:.1f}s)"
        body = result.content.strip() or "(no output)"
        if result.log_path:
            return f"{header}\n\nLog: {result.log_path}\n\n{body}"
        return f"{header}\n\n{body}"

    def get_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "clean_reproduce_validation",
                "description": "Run the final self-check workflow and write a validation report.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "refresh": {"type": "boolean"},
                        "time_budget": {"type": "integer"},
                    },
                    "additionalProperties": False,
                },
            },
        }


def build_clean_validation_tool(engine):
    return CleanReproduceValidationTool(engine)


__all__ = ["CLEANUP_COMMANDS", "CleanReproduceValidationTool", "build_clean_validation_tool"]
