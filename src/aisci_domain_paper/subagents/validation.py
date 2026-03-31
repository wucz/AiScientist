from __future__ import annotations

from aisci_domain_paper.subagents.base import PaperSubagent
from aisci_domain_paper.tools import build_general_tools


VALIDATION_SYSTEM_PROMPT = """You are a Validation Agent for an AI paper reproduction project.

Your job is to inspect the current state, verify that `reproduce.sh` exists and is runnable, look for fragile paths or missing artifacts, and return a concise diagnosis.

Use the available shell and file tools to inspect the repository and workspace. Do not do major code rewrites here; focus on validation and concise reporting.
"""


class PaperValidationSubagent(PaperSubagent):
    @property
    def name(self) -> str:
        return "validation"

    def system_prompt(self) -> str:
        return VALIDATION_SYSTEM_PROMPT

    def get_tools(self):
        return build_general_tools(self.capabilities)
