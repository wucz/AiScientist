"""
LocalShellInterface — a ComputerInterface implementation for local (no-Docker) execution.

Wraps MappedShellInterface from aisci_domain_paper.paper_compat with the canonical
path mapping used by both paper and MLE tracks:

    /home/paper      → jobs/<id>/workspace/paper
    /home/submission → jobs/<id>/workspace/submission
    /home/agent      → jobs/<id>/workspace/agent
    /home/code       → jobs/<id>/workspace/code
    /home/data       → jobs/<id>/workspace/data
    /home/logs       → jobs/<id>/logs
    /workspace/logs  → jobs/<id>/logs

Commands are executed directly via subprocess on the host (no container).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from aisci_core.models import JobPaths
from aisci_domain_paper.paper_compat import MappedShellInterface, PathMapper, ShellResult

# Extra buffer beyond the command timeout for subprocess to clean up.
_TIMEOUT_BUFFER = 30

# Prefer GNU timeout if available (e.g. installed via `brew install coreutils`).
_TIMEOUT_CMD = shutil.which("gtimeout") or shutil.which("timeout")


def _build_mapper(job_paths: JobPaths) -> PathMapper:
    workspace = job_paths.workspace_dir
    return PathMapper(
        {
            "/home/paper": workspace / "paper",
            "/home/submission": workspace / "submission",
            "/home/agent": workspace / "agent",
            "/home/code": workspace / "code",
            "/home/data": workspace / "data",
            "/home/logs": job_paths.logs_dir,
            "/workspace/logs": job_paths.logs_dir,
        }
    )


class LocalShellInterface(MappedShellInterface):
    """
    Drop-in replacement for DockerShellInterface that runs commands locally.

    Satisfies the ComputerInterface Protocol defined in
    aisci_agent_runtime/shell_interface.py.
    """

    def __init__(self, job_paths: JobPaths, working_dir: str = "/home/submission"):
        mapper = _build_mapper(job_paths)
        super().__init__(working_dir=working_dir, mapper=mapper)
        # Ensure mapped directories exist so agents can write immediately.
        for real_path in mapper.canonical_to_real.values():
            Path(real_path).mkdir(parents=True, exist_ok=True)

    def send_shell_command(self, cmd: str, timeout: int = 300) -> ShellResult:
        breakpoint()  # DEBUG: shell 命令执行 — 查看 cmd / rewritten / 执行结果
        from aisci_agent_runtime.shell_interface import _refuse_broad_python_kill, _shell_quote  # noqa: PLC0415

        refusal = _refuse_broad_python_kill(cmd)
        if refusal is not None:
            return ShellResult(output=refusal, exit_code=1)
        rewritten = self.mapper.rewrite_command(cmd)

        if _TIMEOUT_CMD:
            # GNU timeout available — wrap as usual.
            inner = f"{_TIMEOUT_CMD} --signal=KILL {timeout} bash --noprofile --norc -lc {_shell_quote(rewritten)}"
        else:
            # macOS without coreutils: rely on Python subprocess timeout.
            inner = f"bash --noprofile --norc -lc {_shell_quote(rewritten)}"

        try:
            completed = subprocess.run(
                ["bash", "--noprofile", "--norc", "-lc", inner],
                capture_output=True,
                text=True,
                cwd=str(self.working_dir),
                timeout=timeout + _TIMEOUT_BUFFER,
            )
        except subprocess.TimeoutExpired:
            return ShellResult(output=f"ERROR: command timed out after {timeout}s", exit_code=137)
        output = (completed.stdout or "") + (completed.stderr or "")
        return ShellResult(output=output.strip(), exit_code=completed.returncode)
