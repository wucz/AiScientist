from __future__ import annotations

from aisci_domain_paper.prompts import (
    ENV_SETUP_SYSTEM_PROMPT,
    EXPERIMENT_SYSTEM_PROMPT,
    EXPLORE_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    IMPLEMENTATION_SYSTEM_PROMPT,
    MAIN_AGENT_SYSTEM_PROMPT,
    PAPER_READER_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    PRIORITIZATION_SYSTEM_PROMPT,
    RESOURCE_DOWNLOAD_SYSTEM_PROMPT,
    SEARCH_EXECUTOR_PROMPT,
    SEARCH_STRATEGIST_PROMPT,
    render_implementation_system_prompt,
    render_main_agent_system_prompt,
    render_paper_reader_system_prompt,
    render_plan_system_prompt,
    render_prioritization_system_prompt,
)
from aisci_domain_paper.subagents import GeneralSubagent, subagent_class_for_kind


def test_main_prompt_contains_upstream_paper_workflow() -> None:
    assert "reproduce.sh" in MAIN_AGENT_SYSTEM_PROMPT
    assert "read_paper" in MAIN_AGENT_SYSTEM_PROMPT
    assert "prioritize_tasks" in MAIN_AGENT_SYSTEM_PROMPT
    assert "implement" in MAIN_AGENT_SYSTEM_PROMPT
    assert "run_experiment" in MAIN_AGENT_SYSTEM_PROMPT
    assert "clean_reproduce_validation" in MAIN_AGENT_SYSTEM_PROMPT
    assert "submit" in MAIN_AGENT_SYSTEM_PROMPT


def test_subagent_prompts_are_no_longer_placeholder_role_blurbs() -> None:
    assert "Implementation Specialist" in IMPLEMENTATION_SYSTEM_PROMPT
    assert "Experiment Agent" in EXPERIMENT_SYSTEM_PROMPT
    assert "Prioritization Strategist" in PRIORITIZATION_SYSTEM_PROMPT
    assert "Paper Reader Specialist" in PAPER_READER_SYSTEM_PROMPT
    assert "Environment Setup Specialist" in ENV_SETUP_SYSTEM_PROMPT
    assert "Resource Download Specialist" in RESOURCE_DOWNLOAD_SYSTEM_PROMPT
    assert "Search Strategist" in SEARCH_STRATEGIST_PROMPT
    assert "Search Executor" in SEARCH_EXECUTOR_PROMPT


def test_live_prompt_renderers_hide_unavailable_optional_tools() -> None:
    capabilities = {
        "online_research": {"available": False},
        "github_research": {"available": False},
        "linter": {"available": True},
    }
    main_prompt = render_main_agent_system_prompt(capabilities)
    implementation_prompt = render_implementation_system_prompt(capabilities)
    paper_reader_prompt = render_paper_reader_system_prompt(capabilities)
    prioritization_prompt = render_prioritization_system_prompt(capabilities)
    plan_prompt = render_plan_system_prompt(capabilities)

    assert "web_search" not in main_prompt
    assert "link_summary" not in implementation_prompt
    assert "github" not in implementation_prompt
    assert "edit_file" not in paper_reader_prompt
    assert "python" not in paper_reader_prompt
    assert "parse_rubric" in prioritization_prompt
    assert "write_priorities" in prioritization_prompt
    assert "write_plan" in plan_prompt
    assert "edit_file" not in plan_prompt


def test_full_prompts_match_current_runtime_tool_contract() -> None:
    assert "web_search" in MAIN_AGENT_SYSTEM_PROMPT
    assert "link_summary" in MAIN_AGENT_SYSTEM_PROMPT
    assert "github" in IMPLEMENTATION_SYSTEM_PROMPT
    assert "git_commit" not in GENERAL_SYSTEM_PROMPT
    assert "edit_file" not in GENERAL_SYSTEM_PROMPT
    assert "write_plan" in PLAN_SYSTEM_PROMPT
    assert "edit_file" not in PAPER_READER_SYSTEM_PROMPT
    assert "python" not in PAPER_READER_SYSTEM_PROMPT
    assert "write_priorities" in PRIORITIZATION_SYSTEM_PROMPT
    assert "parse_rubric" in PRIORITIZATION_SYSTEM_PROMPT
    assert "web_search" in EXPLORE_SYSTEM_PROMPT
    assert "search_file" in SEARCH_EXECUTOR_PROMPT
    assert "read_file_chunk" in SEARCH_STRATEGIST_PROMPT


def test_generic_alias_maps_to_general_subagent() -> None:
    assert subagent_class_for_kind("generic") is GeneralSubagent
