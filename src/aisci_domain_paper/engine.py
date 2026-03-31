from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from openai import BadRequestError

from aisci_agent_runtime.llm_client import ContextLengthError, ContentPolicyError, LLMClient
from aisci_agent_runtime.log_utils import log_messages_to_file, log_model_response_event, log_tool_result_event
from aisci_agent_runtime.subagents.base import (
    SubagentConfig,
    SubagentOutput,
    SubagentStatus,
    fix_message_consistency,
    prune_messages,
    prune_messages_individual,
)
from aisci_agent_runtime.summary_utils import SummaryConfig, summarize_messages
from aisci_agent_runtime.tools.base import SubagentCompleteSignal
from aisci_domain_paper.configs import (
    DEFAULT_DOWNLOAD_CONFIG,
    DEFAULT_ENV_SETUP_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_EXPLORE_SUBAGENT_CONFIG,
    DEFAULT_GENERAL_SUBAGENT_CONFIG,
    DEFAULT_IMPLEMENTATION_CONFIG,
    DEFAULT_PLAN_SUBAGENT_CONFIG,
)
from aisci_domain_paper.constants import MAIN_AGENT_WORKSPACE_REFERENCE
from aisci_domain_paper.prompts import (
    render_experiment_system_prompt,
    render_explore_system_prompt,
    render_general_system_prompt,
    render_implementation_system_prompt,
    render_main_agent_system_prompt,
    render_paper_reader_system_prompt,
    render_plan_system_prompt,
    render_prioritization_system_prompt,
)
from aisci_domain_paper.runtime import (
    build_reproduce_scaffold_script,
    ensure_submission_repo,
    list_files,
)
from aisci_domain_paper.state_manager import PaperStateManager, SessionInfo
from aisci_domain_paper.subagents import subagent_class_for_kind
from aisci_domain_paper.tools import build_main_tools


@dataclass(frozen=True)
class PaperRuntimeConfig:
    job_id: str
    objective: str
    llm_profile_name: str
    time_limit_seconds: int = 24 * 3600
    max_steps: int = 80
    reminder_freq: int = 5
    enable_online_research: bool = True
    enable_github_research: bool = True


class EmbeddedPaperEngine:
    def __init__(
        self,
        *,
        config: PaperRuntimeConfig,
        shell,
        llm: LLMClient | None,
        paper_dir: Path,
        submission_dir: Path,
        agent_dir: Path,
        logs_dir: Path,
        trace,
    ) -> None:
        self.config = config
        self.shell = shell
        self.llm = llm
        self.paper_dir = paper_dir
        self.submission_dir = submission_dir
        self.agent_dir = agent_dir
        self.logs_dir = logs_dir
        self.trace = trace
        self.analysis_dir = agent_dir / "paper_analysis"
        self.subagent_logs_dir = logs_dir / "subagent_logs"
        self.agent_log_path = logs_dir / "agent.log"
        self.conversation_path = logs_dir / "conversation.jsonl"
        self.capability_path = agent_dir / "capabilities.json"
        self.self_check_path = agent_dir / "final_self_check.md"
        self.self_check_json_path = agent_dir / "final_self_check.json"
        self.prompt_path = agent_dir / "paper_main_prompt.md"
        self.state_path = logs_dir / "paper_session_state.json"
        self.prioritized_path = agent_dir / "prioritized_tasks.md"
        self.plan_path = agent_dir / "plan.md"
        self.reproduce_path = submission_dir / "reproduce.sh"
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self._impl_runs = 0
        self._exp_runs = 0
        self._validate_runs = 0
        self._submit_attempts = 0
        self._impl_exp_sequence: list[str] = []
        self._clean_validation_called = False
        self._main_summary_config = SummaryConfig()
        self.state_manager = PaperStateManager(
            agent_dir=agent_dir,
            logs_dir=logs_dir,
            subagent_logs_dir=self.subagent_logs_dir,
        )

    def run(self) -> str:
        self._ensure_workspace()
        self._write_capability_report()
        if self.llm is None:
            raise RuntimeError("Paper mode requires a configured LLM client. No product fallback loop is available.")
        return self.run_main_loop()

    def run_main_loop(self) -> str:
        prompt = self.render_main_prompt()
        self.prompt_path.write_text(prompt, encoding="utf-8")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": self._initial_user_prompt()},
        ]
        log_messages_to_file(messages, str(self.agent_log_path))

        tools = build_main_tools(self)
        tool_map = {tool.name(): tool for tool in tools}
        tool_schemas = [tool.get_tool_schema() for tool in tools]
        start_time = time.time()
        last_summary: str | None = None

        for step in range(1, self.config.max_steps + 1):
            elapsed = max(
                time.time() - start_time - (self.llm.total_retry_time if self.llm else 0.0),
                0.0,
            )
            if elapsed >= self.config.time_limit_seconds:
                break
            if step > 1 and step % self.config.reminder_freq == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": self._build_reminder(
                            step,
                            elapsed,
                            reproduce_sh_exists=self._reproduce_script_exists(),
                        ),
                    }
                )

            try:
                assert self.llm is not None
                response = self.llm.chat(messages, tools=tool_schemas)
            except ContentPolicyError as exc:
                summary = f"Paper run stopped by content policy: {exc}"
                if exc.dump_path:
                    summary += f"\n\nSafety dump: {exc.dump_path}"
                self._write_state(summary=summary, mode="content_policy_stop")
                return summary
            except ContextLengthError as exc:
                if self._main_summary_config.enabled:
                    messages, last_summary, summarized = summarize_messages(
                        llm=self.llm,
                        messages=messages,
                        config=self._main_summary_config,
                        task_description=self.config.objective,
                        last_summary=last_summary,
                    )
                    if not summarized:
                        messages = self._reduce_messages(messages, exc)
                else:
                    messages = self._reduce_messages(messages, exc)
                continue
            except BadRequestError:
                messages = fix_message_consistency(messages)
                continue

            log_model_response_event(
                str(self.conversation_path),
                self.session_id,
                step,
                len(messages),
                response.text_content,
                [{"id": call.call_id, "name": call.name, "arguments": call.arguments} for call in response.tool_calls],
                response.usage,
                response.reasoning_content,
            )

            assistant_message: dict[str, Any] = {"role": "assistant", "content": response.text_content}
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in response.tool_calls
                ]
            messages.append(assistant_message)
            log_messages_to_file(messages, str(self.agent_log_path))

            if not response.tool_calls:
                messages.append({"role": "user", "content": self._continue_instruction()})
                continue

            for call in response.tool_calls:
                if call.name in {"submit", "finish_run"}:
                    submit_feedback = self._handle_submit_precheck(elapsed)
                    if submit_feedback is not None:
                        log_tool_result_event(
                            str(self.conversation_path),
                            self.session_id,
                            step,
                            call.name,
                            call.call_id,
                            submit_feedback,
                        )
                        messages.append({"role": "tool", "tool_call_id": call.call_id, "content": submit_feedback})
                        log_messages_to_file(messages, str(self.agent_log_path))
                        continue
                tool = tool_map.get(call.name)
                if tool is None:
                    tool_result = f"Unknown tool: {call.name}"
                else:
                    try:
                        if tool.supports_constraints():
                            tool_result = str(
                                tool.execute_with_constraints(
                                    self.shell,
                                    constraints=self.constraints(),
                                    **call.arguments,
                                )
                            )
                        else:
                            tool_result = str(tool.execute(self.shell, **call.arguments))
                    except SubagentCompleteSignal as signal:
                        messages.append({"role": "tool", "tool_call_id": call.call_id, "content": signal.content})
                        log_messages_to_file(messages, str(self.agent_log_path))
                        log_tool_result_event(
                            str(self.conversation_path),
                            self.session_id,
                            step,
                            call.name,
                            call.call_id,
                            signal.content,
                        )
                        self._write_state(summary=signal.content, mode="submitted")
                        self.trace.event(
                            "agent_step",
                            "Paper engine finished via submit.",
                            phase="finalize",
                            payload={"summary": signal.content},
                        )
                        return signal.content
                    except Exception as exc:  # noqa: BLE001
                        tool_result = f"Tool {call.name} failed: {exc}"

                self._track_main_tool_usage(call.name)

                log_tool_result_event(
                    str(self.conversation_path),
                    self.session_id,
                    step,
                    call.name,
                    call.call_id,
                    tool_result,
                )
                messages.append({"role": "tool", "tool_call_id": call.call_id, "content": tool_result})
                log_messages_to_file(messages, str(self.agent_log_path))

        final_summary = self._auto_finalize_summary()
        self._write_state(summary=final_summary, mode="auto_finalize")
        return final_summary

    def read_paper(self, refresh: bool = False) -> str:
        from aisci_domain_paper.tools.paper_reader_tool import ReadPaperTool

        return str(ReadPaperTool(self).execute(self.shell, refresh=refresh))

    def prioritize_tasks(self, refresh: bool = False) -> str:
        from aisci_domain_paper.tools.prioritization_tool import PrioritizeTasksTool

        return str(PrioritizeTasksTool(self).execute(self.shell, refresh=refresh))

    def run_implementation(
        self,
        *,
        task: str,
        mode: str = "full",
        context: str = "",
        max_steps: int | None = None,
        time_budget: int | None = None,
    ) -> str:
        from aisci_domain_paper.tools.implementation_tool import ImplementationTool

        return str(
            ImplementationTool(self).execute(
                self.shell,
                task=task,
                mode=mode,
                context=context,
                time_budget=time_budget,
                max_steps=max_steps,
            )
        )

    def run_experiment(
        self,
        *,
        task: str,
        mode: str = "full",
        context: str = "",
        max_steps: int | None = None,
        time_budget: int | None = None,
        session_kind: str = "experiment",
    ) -> str:
        from aisci_domain_paper.tools.experiment_tool import ExperimentTool

        return str(
            ExperimentTool(self).execute(
                self.shell,
                task=task,
                mode=mode,
                context=context,
                time_budget=time_budget,
                max_steps=max_steps,
                session_kind=session_kind,
            )
        )

    def execute_named_subagent(
        self,
        *,
        subagent_type: str,
        objective: str,
        context: str = "",
        max_steps: int | None = None,
        time_limit: int | None = None,
        config: SubagentConfig | None = None,
        session: SessionInfo | None = None,
        phase: str | None = None,
    ) -> str:
        cls = subagent_class_for_kind(subagent_type)
        cfg = config or self.subagent_config(subagent_type, self._default_config_for_kind(subagent_type))
        if max_steps is not None or time_limit is not None or session is not None:
            cfg = replace(
                cfg,
                max_steps=max_steps or cfg.max_steps,
                time_limit=time_limit or cfg.time_limit,
                log_dir=str(session.directory if session else Path(cfg.log_dir)),
            )
        return self.run_subagent_instance(
            cls,
            objective=objective,
            context=context,
            config=cfg,
            phase=phase or self._phase_for_subagent(subagent_type),
            label=subagent_type,
            session=session,
        )

    def run_named_subagent(
        self,
        *,
        subagent_type: str,
        objective: str,
        context: str = "",
        max_steps: int | None = None,
        time_limit: int | None = None,
        config: SubagentConfig | None = None,
        session: SessionInfo | None = None,
        phase: str | None = None,
    ) -> str:
        return self.execute_named_subagent(
            subagent_type=subagent_type,
            objective=objective,
            context=context,
            max_steps=max_steps,
            time_limit=time_limit,
            config=config,
            session=session,
            phase=phase,
        )

    def run_subagent_output(
        self,
        cls,
        *,
        objective: str,
        context: str,
        config: SubagentConfig,
        phase: str,
        label: str,
        session: SessionInfo | None = None,
    ) -> SubagentOutput:
        current_session = session or self.state_manager.create_session(label)
        cfg = replace(config, log_dir=str(current_session.directory))
        self.trace.event(
            "subagent_start",
            f"{label} subagent started.",
            phase=phase,
            payload={"objective": objective, "session_dir": str(current_session.directory)},
        )
        started = time.time()
        subagent = cls(self, self.shell, self.llm, cfg, objective=objective, context=context)
        result: SubagentOutput = subagent.run(context=subagent.build_context())
        if result.runtime_seconds <= 0:
            result.runtime_seconds = max(time.time() - started, 0.0)
        self.trace.event(
            "subagent_finish",
            f"{label} subagent finished with status={result.status.value}.",
            phase=phase,
            payload={"log_path": result.log_path, "session_dir": str(current_session.directory)},
        )
        return result

    def run_subagent_instance(
        self,
        cls,
        *,
        objective: str,
        context: str,
        config: SubagentConfig,
        phase: str,
        label: str,
        session: SessionInfo | None = None,
    ) -> str:
        result = self.run_subagent_output(
            cls,
            objective=objective,
            context=context,
            config=config,
            phase=phase,
            label=label,
            session=session,
        )
        return self._format_subagent_result(label, result)

    def run_clean_validation(self, refresh: bool = False) -> str:
        from aisci_domain_paper.tools.clean_validation_tool import CleanReproduceValidationTool

        return str(CleanReproduceValidationTool(self).execute(self.shell, refresh=refresh))

    def collect_artifacts(self) -> list[Path]:
        candidates = [
            self.analysis_dir / "summary.md",
            self.analysis_dir / "structure.md",
            self.analysis_dir / "algorithm.md",
            self.analysis_dir / "experiments.md",
            self.analysis_dir / "baseline.md",
            self.prioritized_path,
            self.plan_path,
            self.state_manager.impl_log_path,
            self.state_manager.exp_log_path,
            self.reproduce_path,
            self.capability_path,
            self.prompt_path,
            self.self_check_path,
            self.self_check_json_path,
            self.agent_log_path,
            self.conversation_path,
            self.state_path,
        ]
        return [path for path in candidates if path.exists()]

    def render_main_prompt(self) -> str:
        return render_main_agent_system_prompt(self._capabilities())

    def subagent_class(self, kind: str):
        return subagent_class_for_kind(kind)

    def constraints(self) -> dict[str, Any]:
        blacklist_path = self.paper_dir / "blacklist.txt"
        if not blacklist_path.exists():
            return {"blacklist": []}
        lines = [
            line.strip()
            for line in blacklist_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return {"blacklist": lines}

    def reader_context(self) -> str:
        files = list_files(self.paper_dir)
        return "\n".join(
            [
                f"Objective:\n{self.config.objective}",
                "",
                "Canonical paper inputs:",
                *(files or ["- No staged paper inputs detected."]),
                "",
                MAIN_AGENT_WORKSPACE_REFERENCE.strip(),
            ]
        ).strip()

    def subagent_config(self, kind: str, base: SubagentConfig) -> SubagentConfig:
        return replace(base)

    def _ensure_workspace(self) -> None:
        for path in (self.paper_dir, self.submission_dir, self.agent_dir, self.logs_dir, self.analysis_dir, self.subagent_logs_dir):
            path.mkdir(parents=True, exist_ok=True)
        ensure_submission_repo(self.submission_dir)
        self.prompt_path.write_text(self.render_main_prompt(), encoding="utf-8")
        self.state_manager.ensure_logs()
        self._ensure_reproduce_script()
        if not (self.submission_dir / "README.md").exists():
            (self.submission_dir / "README.md").write_text(
                "# Paper Reproduction Workspace\n\nManaged by AiScientist paper mode.\n",
                encoding="utf-8",
            )

    def _ensure_reproduce_script(self) -> None:
        if not self.reproduce_path.exists():
            self.reproduce_path.write_text(
                build_reproduce_scaffold_script(
                    self.config.objective,
                    extra_notes="Replace this scaffold with the paper-specific reproduction workflow.",
                ),
                encoding="utf-8",
            )
        try:
            self.reproduce_path.chmod(0o755)
        except OSError:
            pass

    def _write_capability_report(self) -> None:
        self.capability_path.write_text(json.dumps(self._capabilities(), indent=2), encoding="utf-8")

    def _capabilities(self) -> dict[str, Any]:
        llm_enabled = self.llm is not None
        online_available = llm_enabled and self.config.enable_online_research and bool(
            os.environ.get("AISCI_WEB_SEARCH", "").strip().lower() in {"1", "true", "yes", "on"}
            or (self.llm and getattr(self.llm.config, "web_search", False))
        )
        github_available = llm_enabled and self.config.enable_github_research and bool(os.environ.get("GITHUB_TOKEN"))
        return {
            "llm_enabled": llm_enabled,
            "online_research": {
                "requested": self.config.enable_online_research,
                "available": online_available,
                "mode": "model_native_web_search" if online_available else "disabled_or_unavailable",
            },
            "github_research": {
                "requested": self.config.enable_github_research,
                "available": github_available,
                "mode": "github_token" if github_available else "disabled_or_unavailable",
            },
            "linter": {"available": True},
        }

    def _initial_user_prompt(self) -> str:
        return (
            f"You are running a paper reproduction job for objective: {self.config.objective}\n\n"
            "Read the paper, prioritize the work, implement the core method, validate with experiments, "
            "run clean validation, then submit when the workspace is coherent.\n\n"
            f"Capability status:\n{json.dumps(self._capabilities(), indent=2)}\n\n"
            f"{MAIN_AGENT_WORKSPACE_REFERENCE.strip()}"
        )

    def _build_reminder(self, step: int, elapsed: float, *, reproduce_sh_exists: bool) -> str:
        remaining = max(0, self.config.time_limit_seconds - elapsed)
        lines = [
            f"Reminder: step {step}/{self.config.max_steps}. Elapsed {int(elapsed)}s, remaining {int(remaining)}s.",
            f"Artifacts ready: read_paper={'yes' if (self.analysis_dir / 'summary.md').exists() else 'no'}, "
            f"priorities={'yes' if self.prioritized_path.exists() else 'no'}, "
            f"impl_runs={self._impl_runs}, exp_runs={self._exp_runs}, self_checks={self._validate_runs}.",
        ]
        exp_impl_gap = self._impl_exp_sequence.count("exp") - self._impl_exp_sequence.count("impl")
        impl_exp_gap = self._impl_exp_sequence.count("impl") - self._impl_exp_sequence.count("exp")
        if exp_impl_gap >= 4:
            lines.append(
                "Experiment calls are far ahead of implementation calls. Do not rerun experiments without making fixes; "
                "use implement(mode='fix', context='<diagnosis>') or move on."
            )
        elif exp_impl_gap >= 2:
            lines.append("Experiments are outpacing fixes. If results are failing, implement a targeted fix before validating again.")
        if impl_exp_gap >= 3:
            lines.append("Implementation is outpacing validation. Run run_experiment(mode='validate') before starting more major changes.")

        time_ratio = elapsed / self.config.time_limit_seconds if self.config.time_limit_seconds else 0.0
        if time_ratio >= 0.85:
            if reproduce_sh_exists:
                lines.append("Less than 15% time remains. Stabilize reproduce.sh, commit all meaningful work, and avoid starting large new tasks.")
            else:
                lines.append("Less than 15% time remains and reproduce.sh is still missing. Creating and committing reproduce.sh is the top priority.")
        elif time_ratio >= 0.70:
            lines.append("More than 70% of time is used. Focus on validating in-flight work and keeping reproduce.sh current.")
        elif time_ratio >= 0.50:
            lines.append("More than 50% of time is used. Keep pushing breadth across P0/P1 tasks before considering submit.")

        lines.append("Keep reproduce.sh committed and do not submit before final self-check or a concrete failure diagnosis.")
        return "\n".join(lines)

    def _continue_instruction(self) -> str:
        return (
            "Continue the paper workflow. Use direct tools for quick inspection, use implement/run_experiment for heavy work, "
            "and call submit only after reading, prioritization, implementation, experiments, and self-check are done."
        )

    def _reduce_messages(self, messages: list[dict], exc: ContextLengthError) -> list[dict]:
        if exc.prune_individual:
            messages = prune_messages_individual(messages, max_tokens_per_message=self.llm.config.context_window if self.llm else None)
        return prune_messages(messages)

    def _default_config_for_kind(self, kind: str) -> SubagentConfig:
        mapping = {
            "implementation": DEFAULT_IMPLEMENTATION_CONFIG,
            "experiment": DEFAULT_EXPERIMENT_CONFIG,
            "env_setup": DEFAULT_ENV_SETUP_CONFIG,
            "resource_download": DEFAULT_DOWNLOAD_CONFIG,
            "explore": DEFAULT_EXPLORE_SUBAGENT_CONFIG,
            "plan": DEFAULT_PLAN_SUBAGENT_CONFIG,
            "general": DEFAULT_GENERAL_SUBAGENT_CONFIG,
        }
        return mapping.get(kind, DEFAULT_GENERAL_SUBAGENT_CONFIG)

    def _phase_for_subagent(self, kind: str) -> str:
        mapping = {
            "implementation": "implement",
            "experiment": "validate",
            "env_setup": "implement",
            "resource_download": "implement",
            "explore": "analyze",
            "plan": "prioritize",
            "general": "implement",
        }
        return mapping.get(kind, "implement")

    def _write_state(self, *, summary: str, mode: str) -> None:
        payload = {
            "job_id": self.config.job_id,
            "mode": mode,
            "summary": summary,
            "impl_runs": self._impl_runs,
            "exp_runs": self._exp_runs,
            "self_checks": self._validate_runs,
            "submit_attempts": self._submit_attempts,
            "clean_validation_called": self._clean_validation_called,
            "impl_exp_sequence": self._impl_exp_sequence,
            "updated_at": time.time(),
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _auto_finalize_summary(self) -> str:
        parts = ["Paper loop reached the step limit or time limit."]
        if (self.analysis_dir / "summary.md").exists():
            parts.append("paper_analysis ready")
        if self.prioritized_path.exists():
            parts.append("prioritized_tasks ready")
        if self.reproduce_path.exists():
            parts.append("reproduce.sh present")
        if self.self_check_path.exists():
            parts.append("final self-check written")
        return ". ".join(parts) + "."

    def _read_text(self, path: Path, *, limit: int = 8_000) -> str:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:limit]

    def _extract_headings(self, markdown: str) -> list[str]:
        headings = []
        for idx, line in enumerate(markdown.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                headings.append(f"- line {idx}: {line.strip()}")
        return headings[:200]

    def _summarize_paper_text(self, paper_md: str, pdf_excerpt: str) -> str:
        source = paper_md or pdf_excerpt or "No paper text could be extracted."
        paras = [part.strip() for part in re.split(r"\n\s*\n", source) if part.strip()]
        return "\n\n".join(paras[:4])[:4_000]

    def _pick_excerpt(self, source: str, pattern: str, default_label: str) -> str:
        if not source:
            return f"No {default_label} excerpt available."
        match = re.search(pattern, source)
        if not match:
            return self._truncate_text(source, 1_600)
        start = max(0, match.start() - 400)
        end = min(len(source), match.end() + 1_200)
        return self._truncate_text(source[start:end], 1_600)

    def _truncate_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]

    def _format_subagent_result(self, label: str, result: SubagentOutput) -> str:
        header = f"[{label} subagent: {result.status.value}]"
        if result.log_path:
            header += f" log={result.log_path}"
        body = result.content.strip()
        if body:
            return header + "\n\n" + body
        return header

    def mark_implementation_run(self) -> None:
        self._impl_runs += 1

    def mark_experiment_run(self, session_kind: str = "experiment") -> None:
        if session_kind == "clean_val":
            return
        self._exp_runs += 1

    def mark_validation_run(self) -> None:
        self._validate_runs += 1
        self._clean_validation_called = True

    def render_subagent_prompt(self, kind: str) -> str:
        capabilities = self._capabilities()
        mapping = {
            "implementation": render_implementation_system_prompt,
            "experiment": render_experiment_system_prompt,
            "explore": render_explore_system_prompt,
            "plan": render_plan_system_prompt,
            "general": render_general_system_prompt,
            "paper_reader": render_paper_reader_system_prompt,
            "prioritization": render_prioritization_system_prompt,
        }
        renderer = mapping.get(kind)
        if renderer is None:
            raise KeyError(kind)
        return renderer(capabilities)

    def _track_main_tool_usage(self, tool_name: str) -> None:
        if tool_name == "implement":
            self._impl_exp_sequence.append("impl")
        elif tool_name == "run_experiment":
            self._impl_exp_sequence.append("exp")
        elif tool_name == "clean_reproduce_validation":
            self._clean_validation_called = True

    def _reproduce_script_exists(self) -> bool:
        return self.reproduce_path.exists()

    def _reproduce_script_tracked(self) -> bool:
        if not self._reproduce_script_exists():
            return False
        result = self.shell.send_command(
            "cd /home/submission && git ls-files reproduce.sh | grep -q reproduce.sh && echo TRACKED || echo UNTRACKED",
            timeout=20,
        )
        return "TRACKED" in result.output

    def _submit_syntax_warnings(self) -> list[str]:
        warnings: list[str] = []
        if not self._reproduce_script_exists():
            return warnings
        result = self.shell.send_command(
            "cd /home/submission && "
            "bash -n reproduce.sh 2>&1 && echo SYNTAX_OK || echo SYNTAX_ERR; "
            "if [ -f requirements.txt ]; then echo REQS_FOUND; else echo REQS_MISSING; fi",
            timeout=30,
        )
        output = result.output
        if "SYNTAX_ERR" in output:
            warnings.append("reproduce.sh has syntax errors. Fix it before submitting.")
        if "REQS_MISSING" in output:
            warnings.append("requirements.txt is missing. If reproduce.sh relies on it, fresh-environment setup will fail.")
        return warnings

    def _handle_submit_precheck(self, elapsed: float) -> str | None:
        self._submit_attempts += 1
        hard_errors: list[str] = []
        warnings: list[str] = []
        time_ratio = elapsed / self.config.time_limit_seconds if self.config.time_limit_seconds else 1.0
        remaining_hours = max(self.config.time_limit_seconds - elapsed, 0.0) / 3600.0 if self.config.time_limit_seconds else 0.0

        if self._submit_attempts == 1 and time_ratio < 0.5:
            warnings.append(
                f"Early submission: only {time_ratio * 100:.0f}% of the budget is used ({remaining_hours:.1f}h remaining). "
                "Review prioritized_tasks.md for remaining coverage before finishing."
            )

        if not self._reproduce_script_exists():
            hard_errors.append(
                "reproduce.sh is missing at /home/submission/reproduce.sh. Submission is blocked until it exists."
            )
        elif not self._reproduce_script_tracked():
            hard_errors.append(
                "reproduce.sh exists but is not tracked by git. git clean -fd will remove it, so submission remains blocked."
            )

        if hard_errors:
            self._submit_attempts -= 1
            message = "SUBMISSION BLOCKED:\n\n" + "\n\n".join(f"BLOCKED -- {item}" for item in hard_errors)
            self.trace.event(
                "submit_precheck",
                "Paper submit blocked by hard pre-check failures.",
                phase="finalize",
                payload={"hard_errors": hard_errors},
            )
            return message

        warnings.extend(self._submit_syntax_warnings())
        if not self._clean_validation_called and self._submit_attempts == 1:
            warnings.append(
                "clean_reproduce_validation() has not been run yet. Run it before submit to catch clean-environment failures."
            )

        if warnings:
            self.trace.event(
                "submit_precheck",
                "Paper submit issued warnings.",
                phase="finalize",
                payload={"warnings": warnings, "submit_attempt": self._submit_attempts},
            )
            return "SUBMIT PRE-CHECK WARNINGS:\n\n" + "\n\n".join(f"- {item}" for item in warnings) + "\n\nIf you still want to finish, call submit() again."

        return None
