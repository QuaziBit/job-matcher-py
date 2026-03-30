"""
analyzer/skills_helpers.py — Skill normalization, parsing, and keyword boost.
"""

import logging
import re

from analyzer.config import BLOCKER_KEYWORDS
from skills import normalize_skill, get_skill_category

logger = logging.getLogger("analyzer.skills")


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_match_type(raw: str) -> str:
    """Normalize non-standard match_type values from models."""
    s = (raw or "").strip().lower()
    if s in ("exact", "direct", "verbatim", "full"):
        return "exact"
    if s in ("partial", "related", "similar", "close"):
        return "partial"
    return "inferred"


def normalize_severity(raw: str) -> str:
    """Normalize non-standard severity labels from models."""
    s = (raw or "").strip().lower()
    if s in ("blocker", "critical", "must", "required", "mandatory"):
        return "blocker"
    if s in ("major", "high", "significant", "important"):
        return "major"
    return "minor"


def normalize_requirement_type(raw: str) -> str:
    """Normalize non-standard requirement_type labels."""
    s = (raw or "").strip().lower()
    if s in ("hard", "required", "mandatory", "must"):
        return "hard"
    if s in ("bonus", "optional", "nice-to-have", "nice to have", "plus"):
        return "bonus"
    return "preferred"


# ── Skill parsers ─────────────────────────────────────────────────────────────

def parse_matched_skills(raw_matched: list) -> list:
    """
    Parse matched_skills from LLM response.
    Handles both new object format and old plain string format.
    """
    result = []
    for item in raw_matched:
        if isinstance(item, dict):
            skill_name = normalize_skill(item.get("skill", ""))
            result.append({
                "skill":          skill_name,
                "match_type":     normalize_match_type(item.get("match_type", "exact")),
                "jd_snippet":     (item.get("jd_snippet", "") or "")[:100],
                "resume_snippet": (item.get("resume_snippet", "") or "")[:100],
                "category":       get_skill_category(skill_name),
            })
        elif isinstance(item, str) and item.strip():
            skill_name = normalize_skill(item.strip())
            result.append({
                "skill":          skill_name,
                "match_type":     "exact",
                "jd_snippet":     "",
                "resume_snippet": "",
                "category":       get_skill_category(skill_name),
            })
    return result


def parse_missing_skills(raw_missing: list) -> list:
    """
    Parse missing_skills from LLM response.
    Handles both new object format and old plain string format.
    """
    result = []
    for item in raw_missing:
        if isinstance(item, dict):
            skill_name = normalize_skill(item.get("skill", str(item)))
            result.append({
                "skill":            skill_name,
                "severity":         normalize_severity(item.get("severity", "minor")),
                "requirement_type": normalize_requirement_type(item.get("requirement_type", "preferred")),
                "jd_snippet":       (item.get("jd_snippet", "") or "")[:100],
                "cluster_group":    get_skill_category(skill_name),
            })
        else:
            skill_name = normalize_skill(str(item))
            result.append({
                "skill":            skill_name,
                "severity":         "minor",
                "requirement_type": "preferred",
                "jd_snippet":       "",
                "cluster_group":    get_skill_category(skill_name),
            })
    return result


def parse_suggestions(raw_suggestions: list) -> list:
    """
    Parse suggestions from LLM response.
    Handles both object format and plain string format.
    """
    result = []
    for item in raw_suggestions:
        if isinstance(item, dict) and item.get("detail"):
            result.append({
                "title":           item.get("title", "Suggestion"),
                "detail":          item.get("detail", ""),
                "job_requirement": item.get("job_requirement", ""),
            })
        elif isinstance(item, str) and item.strip():
            result.append({
                "title":           "Suggestion",
                "detail":          item.strip(),
                "job_requirement": "",
            })
        else:
            logger.info(f"→ skipping unparseable suggestion: {item!r}")
    return result[:3]


# ── Keyword boost ─────────────────────────────────────────────────────────────

def keyword_boost(missing: list, job_description: str) -> list:
    """
    Upgrade severity of missing skills that match known hard-blocker
    patterns in the job description.
    """
    jd_lower       = job_description.lower()
    jd_has_blocker = any(kw in jd_lower for kw in BLOCKER_KEYWORDS)
    jd_years_match = re.search(r"(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)", jd_lower)

    result = []
    for item in missing:
        skill_lower = item["skill"].lower()
        severity    = item["severity"]

        if any(kw in skill_lower for kw in BLOCKER_KEYWORDS):
            severity = "blocker"
        if re.search(r"\d+\s*years?", skill_lower) and jd_years_match:
            severity = "blocker"
        if severity == "major" and jd_has_blocker and any(
            kw in skill_lower for kw in ["required", "must"]
        ):
            severity = "blocker"

        result.append({**item, "severity": severity})
    return result


# ── Backward compat aliases ───────────────────────────────────────────────────
_normalize_match_type     = normalize_match_type
_normalize_severity       = normalize_severity
_normalize_requirement_type = normalize_requirement_type
_parse_matched_skills     = parse_matched_skills
_parse_missing_skills     = parse_missing_skills
_parse_suggestions        = parse_suggestions
_keyword_boost            = keyword_boost
