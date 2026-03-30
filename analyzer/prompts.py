"""
analyzer/prompts.py — LLM prompt builders for job analysis.
"""

import os

from analyzer.config import get_mode_config

# ── Shared prompt constants ───────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = """You are an expert technical recruiter and career coach specializing in software engineering,
DevSecOps, and cloud infrastructure roles. You evaluate how well a candidate's resume matches a job description.

You MUST respond with ONLY valid JSON — no prose, no markdown, no code fences."""

_SCORING_RUBRIC = """
Scoring rubric:
  1 = Poor match — major gaps, different domain entirely
  2 = Weak match — some overlap but significant missing requirements
  3 = Moderate match — meets roughly half the requirements
  4 = Strong match — meets most requirements with minor gaps
  5 = Excellent match — highly aligned, apply immediately"""

_SEVERITY_DEFS = """
Severity definitions for missing_skills:
  blocker = eliminates candidacy entirely (e.g. required clearance, mandatory cert, minimum years not met)
  major   = significant gap that will hurt chances substantially
  minor   = nice-to-have or learnable gap that is unlikely to disqualify

requirement_type definitions for missing_skills:
  hard      = job uses "required", "must have", "must hold", "mandatory", eligibility-blocking
  preferred = job uses "preferred", "desired", "strong plus", "ideally"
  bonus     = job uses "nice to have", "is a plus", "familiarity with"
  If unclear, use "preferred" as default.

match_type definitions for matched_skills:
  exact    = skill name appears verbatim in both JD and resume
  partial  = related term found (e.g. "REST" matches "REST APIs")
  inferred = implied by context, no direct phrase found"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_system_prompt(cfg: dict | None = None) -> str:
    """Build mode-appropriate system prompt."""
    if cfg is None:
        cfg = get_mode_config()

    mode     = os.getenv("ANALYSIS_MODE", "standard").lower().strip()
    slen     = cfg["snippet_len"]
    mmatched = cfg["max_matched"]
    mmissing = cfg["max_missing"]
    do_sugg  = cfg["suggestions"]

    if mode == "fast":
        json_shape = "\n".join([
            "{",
            '  "score": <integer 1-5>,',
            '  "matched_skills": [',
            f'    {{"skill": "name", "match_type": "exact|partial|inferred", "jd_snippet": "<{slen} chars>"}}',
            "  ],",
            '  "missing_skills": [',
            f'    {{"skill": "name", "severity": "blocker|major|minor",',
            f'      "requirement_type": "hard|preferred|bonus", "jd_snippet": "<{slen} chars>"}}',
            "  ],",
            '  "reasoning": "<1-2 sentence honest assessment>"',
            "}",
        ])
        sections = [
            _SYSTEM_PROMPT_BASE,
            f"Return at most {mmatched} matched skills and at most {mmissing} missing skills — only the most significant ones.\n"
            f"Snippets must be verbatim phrases, max {slen} characters. Do NOT fabricate snippets.",
            "Exactly this JSON shape:",
            json_shape,
            _SEVERITY_DEFS,
            _SCORING_RUBRIC,
        ]
        return "\n".join(sections)

    # standard or detailed
    matched_item = f'    {{"skill": "name", "match_type": "exact|partial|inferred", "jd_snippet": "<{slen} chars>", "resume_snippet": "<{slen} chars>"}}'
    missing_item = f'    {{"skill": "name", "severity": "blocker|major|minor", "requirement_type": "hard|preferred|bonus", "jd_snippet": "<{slen} chars>"}}'
    sugg_item    = '    {"title": "short label", "detail": "specific actionable text", "job_requirement": "verbatim JD phrase this addresses"}'

    json_lines = [
        "{",
        '  "score": <integer 1-5>,',
        '  "matched_skills": [',
        matched_item,
        "  ],",
        '  "missing_skills": [',
        missing_item,
        "  ],",
        '  "reasoning": "<2-4 sentence honest assessment>"',
    ]

    sugg_rules = ""
    if do_sugg:
        json_lines[-1] += ","   # add comma after reasoning
        json_lines += [
            '  "suggestions": [',
            sugg_item,
            "  ]",
        ]
        sugg_rules = "\n".join([
            "Suggestion rules — you MUST follow these exactly:",
            "  - Generate exactly 3 resume improvement suggestions",
            "  - ONLY suggest clarifying, repositioning, or expanding EXISTING resume content",
            "  - NEVER suggest adding skills the candidate does not already have",
            "  - Each suggestion must cite the specific job requirement it addresses",
        ])

    json_lines.append("}")
    json_shape = "\n".join(json_lines)

    sections = [
        _SYSTEM_PROMPT_BASE,
        f"Return at most {mmatched} matched skills and at most {mmissing} missing skills.\n"
        f"Snippets must be verbatim phrases copied from the provided text, max {slen} characters.\n"
        'Do NOT fabricate or paraphrase snippets. If no direct phrase exists, set match_type to "inferred" and omit resume_snippet.',
        "Exactly this JSON shape:",
        json_shape,
    ]
    if sugg_rules:
        sections.append(sugg_rules)
    sections += [_SEVERITY_DEFS, _SCORING_RUBRIC]

    return "\n".join(sections)


def build_user_prompt(resume: str, job_description: str) -> str:
    sections = [
        "## RESUME",
        resume,
        "---",
        "## JOB DESCRIPTION",
        job_description,
        "---",
        "Evaluate the match and return ONLY the JSON object described in your instructions.",
    ]
    return "\n\n".join(sections)


# ── Backward compat aliases ───────────────────────────────────────────────────
_build_system_prompt = build_system_prompt
_build_user_prompt   = build_user_prompt
