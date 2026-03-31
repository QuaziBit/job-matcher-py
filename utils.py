"""
utils.py — Shared helper functions for main.py.
Extracted to keep main.py focused on route handlers only.
"""

import re


# ── Text cleaning ───────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Remove or normalize control characters from pasted or scraped text.
    PDF copy-paste and resume formatting often embed literal tabs, form feeds,
    and other control characters that break JSON serialization when the LLM
    copies them verbatim into snippet fields.

    - Tabs replaced with a single space (preserve readability)
    - Other non-printable control chars (except newline) are removed
    - Consecutive spaces collapsed to one
    """
    if not text:
        return text
    text = text.replace("\t", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" {2,}", " ", text)
    return text





def format_duration(seconds: int) -> str:
    """Format seconds as '1:23' or '45s'. Returns '' for zero/None."""
    if not seconds:
        return ""
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}" if m > 0 else f"{s}s"


# ── Resume comparison helpers ─────────────────────────────────────────────────

def has_blocker(missing_skills: list) -> bool:
    """Return True if any missing skill has severity='blocker'."""
    return any(
        (s.get("severity") if isinstance(s, dict) else "") == "blocker"
        for s in missing_skills
    )


def determine_better_fit(a: dict, b: dict) -> tuple:
    """Compare two analyses and return (better_resume_label, reason)."""
    a_has_blocker = has_blocker(a.get("missing_skills", []))
    b_has_blocker = has_blocker(b.get("missing_skills", []))

    if a_has_blocker and not b_has_blocker:
        return b["resume_label"], f"No hard blockers vs {a['resume_label']} which has blockers"
    if b_has_blocker and not a_has_blocker:
        return a["resume_label"], f"No hard blockers vs {b['resume_label']} which has blockers"

    a_score = a.get("adjusted_score", 0)
    b_score = b.get("adjusted_score", 0)
    if a_score > b_score:
        return a["resume_label"], f"Higher adjusted score ({a_score} vs {b_score})"
    if b_score > a_score:
        return b["resume_label"], f"Higher adjusted score ({b_score} vs {a_score})"
    return "Tie", "Both resumes score equally for this role"


def build_comparison(analyses: list) -> dict | None:
    """
    Build a side-by-side comparison of the two most recent unique-resume analyses.
    Returns None if fewer than 2 different resumes have been analyzed.
    """
    seen: dict = {}
    for a in analyses:
        rid = a.get("resume_id")
        if rid not in seen:
            seen[rid] = a
        if len(seen) == 2:
            break
    if len(seen) < 2:
        return None

    ids = list(seen.keys())
    ra, rb = seen[ids[0]], seen[ids[1]]
    better, reason = determine_better_fit(ra, rb)
    return {
        "resume_a":      ra,
        "resume_b":      rb,
        "better_fit":    better,
        "better_reason": reason,
    }


# ── Backward compat aliases (main.py used underscore-prefixed names) ──────────
_format_duration    = format_duration
_has_blocker        = has_blocker
_determine_better_fit = determine_better_fit
_build_comparison   = build_comparison
_clean_text         = clean_text
