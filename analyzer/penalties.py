"""
analyzer/penalties.py — Penalty pipeline, score adjustment, and validation.
"""

import logging
from collections import defaultdict

from analyzer.config import MAX_RETRIES
from skills import get_skill_category, cluster_penalty_cap

logger = logging.getLogger("analyzer.penalties")


# ── Penalty pipeline ──────────────────────────────────────────────────────────

def penalty_for_skill(skill: dict) -> int:
    """Return the raw penalty for a single missing skill."""
    req_type = skill.get("requirement_type", "preferred")
    severity = skill.get("severity", "minor")

    if req_type == "bonus":
        return 0
    if severity == "blocker":
        return 2
    elif severity == "major":
        return 1
    return 0  # minor — handled by count threshold


def compute_adjusted_score(raw_score: int, missing: list) -> tuple:
    """
    Cluster-aware penalty pipeline.
    Groups missing skills by category, caps penalty per cluster,
    never adjusts below 1.
    Returns (adjusted_score, penalty_breakdown).
    """
    for skill in missing:
        if not skill.get("cluster_group"):
            skill["cluster_group"] = get_skill_category(skill["skill"])

    clusters = defaultdict(list)
    for skill in missing:
        clusters[skill["cluster_group"]].append(skill)

    total_penalty     = 0
    cluster_penalties = {}

    for group, skills in clusters.items():
        group_penalty = sum(penalty_for_skill(s) for s in skills)
        cap           = cluster_penalty_cap(group)
        group_penalty = min(group_penalty, cap)
        cluster_penalties[group] = group_penalty
        total_penalty += group_penalty

    adjusted = max(1, raw_score - total_penalty)

    blockers = [s for s in missing if s.get("severity") == "blocker"]
    majors   = [s for s in missing if s.get("severity") == "major"]
    minors   = [s for s in missing if s.get("severity") == "minor"]

    breakdown = {
        "blockers":        len(blockers),
        "majors":          len(majors),
        "minors":          len(minors),
        "blocker_penalty": sum(v for k, v in cluster_penalties.items() if k == "security"),
        "major_penalty":   sum(v for k, v in cluster_penalties.items() if k != "security"),
        "minor_penalty":   0,
        "count_penalty":   0,
        "total_penalty":   total_penalty,
        "clusters":        cluster_penalties,
    }
    return adjusted, breakdown


# ── Validation ────────────────────────────────────────────────────────────────

def validate_llm_output(result: dict, jd: str, resume: str) -> dict:
    """Validate LLM output. Returns {"valid": bool, "errors": list[str]}."""
    errors = []

    score = result.get("score", 0)
    if not (1 <= score <= 5):
        errors.append(f"score {score} out of range 1-5")

    matched = result.get("matched_skills", [])
    if len(jd) > 500 and len(matched) == 0:
        errors.append("no matched skills despite rich job description")

    matched_names = {
        (s["skill"] if isinstance(s, dict) else s).lower()
        for s in matched
    }
    for skill in result.get("missing_skills", []):
        name = (skill["skill"] if isinstance(skill, dict) else skill).lower()
        if name in matched_names:
            errors.append(f"skill '{name}' appears in both matched and missing")

    if not result.get("reasoning", "").strip():
        errors.append("empty reasoning")

    return {"valid": len(errors) == 0, "errors": errors}


def partial_fallback_analysis() -> dict:
    """Return a minimal valid result when all retries fail."""
    return {
        "score":             1,
        "adjusted_score":    1,
        "penalty_breakdown": {},
        "matched_skills":    [],
        "missing_skills":    [],
        "reasoning":         "Analysis could not be completed reliably. "
                             "Please try again or switch providers.",
        "suggestions":       [],
        "retry_count":       MAX_RETRIES,
        "used_fallback":     True,
        "validation_errors": "",
    }


# ── Backward compat aliases ───────────────────────────────────────────────────
_compute_adjusted_score  = compute_adjusted_score
_partial_fallback_analysis = partial_fallback_analysis
