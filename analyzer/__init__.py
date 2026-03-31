"""
analyzer/__init__.py

Public API re-exports — all existing imports from 'analyzer' continue to work
without changes in main.py or tests.py.
"""

# Core analysis
from analyzer.llm import (
    analyze_match,
    call_anthropic_once,
    call_ollama_once,
    # backward compat aliases
    analyze_with_anthropic,
    analyze_with_ollama,
    _call_anthropic_once,
    _call_ollama_once,
)

# Salary
from analyzer.salary import (
    estimate_salary,
    extract_salary,
    _job_has_salary,
    _build_salary_prompt,
    _parse_salary_response,
)

# Config
from analyzer.config import (
    get_mode_config,
    ollama_model as _ollama_model,
    ollama_base_url as _ollama_base_url,
    ANTHROPIC_MODEL,
    MAX_RETRIES,
    MODE_CONFIG,
    MODE_ESTIMATES,
    BLOCKER_KEYWORDS,
)

# Prompts
from analyzer.prompts import (
    build_system_prompt,
    build_user_prompt,
    _build_system_prompt,
    _build_user_prompt,
)

# Parsers
from analyzer.parsers import (
    parse_response,
    sanitize_json,
    repair_truncated_json,
    _parse_response,
    _sanitize_json,
    _repair_truncated_json,
)

# Skills helpers
from analyzer.skills_helpers import (
    parse_matched_skills,
    parse_missing_skills,
    parse_suggestions,
    keyword_boost,
    normalize_match_type,
    normalize_severity,
    normalize_requirement_type,
    _parse_matched_skills,
    _parse_missing_skills,
    _parse_suggestions,
    _keyword_boost,
    _normalize_match_type,
    _normalize_severity,
    _normalize_requirement_type,
)

# Penalties
from analyzer.penalties import (
    penalty_for_skill,
    compute_adjusted_score,
    auto_correct_llm_output,
    validate_llm_output,
    partial_fallback_analysis,
    _compute_adjusted_score,
    _partial_fallback_analysis,
)

__all__ = [
    # Core
    "analyze_match",
    "call_anthropic_once", "call_ollama_once",
    "analyze_with_anthropic", "analyze_with_ollama",
    # Salary
    "estimate_salary", "extract_salary",
    "_job_has_salary", "_build_salary_prompt", "_parse_salary_response",
    # Config
    "get_mode_config", "_ollama_model", "_ollama_base_url",
    "ANTHROPIC_MODEL", "MAX_RETRIES", "MODE_CONFIG", "MODE_ESTIMATES", "BLOCKER_KEYWORDS",
    # Prompts
    "build_system_prompt", "build_user_prompt",
    # Parsers
    "parse_response", "sanitize_json", "repair_truncated_json",
    # Skills
    "parse_matched_skills", "parse_missing_skills", "parse_suggestions", "keyword_boost",
    "normalize_match_type", "normalize_severity", "normalize_requirement_type",
    # Penalties
    "penalty_for_skill", "compute_adjusted_score", "auto_correct_llm_output", "validate_llm_output", "partial_fallback_analysis",
]
