from __future__ import annotations

from aisci_agent_runtime.subagents.base import SubagentConfig
from aisci_agent_runtime.summary_utils import SummaryConfig


DEFAULT_IMPLEMENTATION_CONFIG = SubagentConfig(
    max_steps=250,
    time_limit=28_800,
    reminder_freq=20,
    summary_config=SummaryConfig(),
)
DEFAULT_EXPERIMENT_CONFIG = SubagentConfig(
    max_steps=250,
    time_limit=36_000,
    reminder_freq=25,
    summary_config=SummaryConfig(),
)
DEFAULT_ENV_SETUP_CONFIG = SubagentConfig(max_steps=150, time_limit=7_200, reminder_freq=15)
DEFAULT_DOWNLOAD_CONFIG = SubagentConfig(max_steps=150, time_limit=7_200, reminder_freq=15)
DEFAULT_PAPER_STRUCTURE_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36_000,
    reminder_freq=15,
    summary_config=SummaryConfig(),
)
DEFAULT_PAPER_READER_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36_000,
    reminder_freq=15,
    summary_config=SummaryConfig(),
)
DEFAULT_PAPER_SYNTHESIS_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36_000,
    reminder_freq=15,
    summary_config=SummaryConfig(),
)
DEFAULT_PRIORITIZATION_CONFIG = SubagentConfig(
    max_steps=500,
    time_limit=36_000,
    reminder_freq=15,
    summary_config=SummaryConfig(),
)
DEFAULT_SEARCH_SIMPLE_CONFIG = SubagentConfig(
    max_steps=100,
    time_limit=1_800,
    reminder_freq=10,
    summary_config=SummaryConfig(),
)
DEFAULT_SEARCH_STRATEGY_CONFIG = SubagentConfig(
    max_steps=50,
    time_limit=900,
    reminder_freq=10,
    summary_config=SummaryConfig(),
)
DEFAULT_SEARCH_EXECUTOR_CONFIG = SubagentConfig(
    max_steps=100,
    time_limit=1_800,
    reminder_freq=10,
    summary_config=SummaryConfig(),
)
DEFAULT_EXPLORE_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=150,
    time_limit=7_200,
    reminder_freq=15,
    summary_config=SummaryConfig(),
)
DEFAULT_PLAN_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=120,
    time_limit=5_400,
    reminder_freq=15,
    summary_config=SummaryConfig(),
)
DEFAULT_GENERAL_SUBAGENT_CONFIG = SubagentConfig(
    max_steps=160,
    time_limit=10_800,
    reminder_freq=15,
    summary_config=SummaryConfig(),
)

MAIN_AGENT_BASH_DEFAULT_TIMEOUT = 36_000
MAIN_AGENT_BASH_MAX_TIMEOUT = 86_400
IMPLEMENTATION_BASH_DEFAULT_TIMEOUT = 36_000
EXPERIMENT_BASH_DEFAULT_TIMEOUT = 36_000
EXPERIMENT_COMMAND_TIMEOUT = 36_000
EXPERIMENT_VALIDATE_TIME_LIMIT = 18_000
EXPLORE_BASH_DEFAULT_TIMEOUT = 36_000
PLAN_BASH_DEFAULT_TIMEOUT = 36_000
GENERAL_BASH_DEFAULT_TIMEOUT = 36_000
