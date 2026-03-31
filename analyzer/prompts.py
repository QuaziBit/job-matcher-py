"""
analyzer/prompts.py — LLM prompt builders for job analysis.

Mode complexity is tuned to model capability:
  fast     — minimal schema, lite prompt, no snippets, reasoning first
  standard — medium schema, lite prompt, jd_snippet only, reasoning first
  detailed — full schema, full prompt with definitions and rubric
"""

import os

from analyzer.config import get_mode_config


# ── System prompt bases ───────────────────────────────────────────────────────

# Used for fast + standard modes (smaller models).
_BASE_LITE = (
    "You are a technical recruiter. Evaluate how well a resume matches a job description.\n"
    "Respond with ONLY a valid JSON object. No text before or after the JSON. No markdown."
)

# Used for detailed mode (capable models only).
_BASE_FULL = (
    "You are an expert technical recruiter and career coach specializing in software engineering,\n"
    "DevSecOps, and cloud infrastructure roles. You evaluate how well a candidate's resume\n"
    "matches a job description.\n\n"
    "You MUST respond with ONLY valid JSON — no prose, no markdown, no code fences."
)


# ── Definition blocks (detailed mode only) ────────────────────────────────────

_SCORING_RUBRIC = """\
Scoring rubric:
  1 = Poor match    — major gaps, different domain entirely
  2 = Weak match    — some overlap but significant missing requirements
  3 = Moderate match — meets roughly half the requirements
  4 = Strong match  — meets most requirements with minor gaps
  5 = Excellent match — highly aligned, apply immediately"""

_SEVERITY_DEFS = """\
Severity definitions for missing_skills:
  blocker   = eliminates candidacy entirely (required clearance, mandatory cert, minimum years not met)
  major     = significant gap that will hurt chances substantially
  minor     = nice-to-have or learnable gap unlikely to disqualify

requirement_type definitions:
  hard      = "required", "must have", "must hold", "mandatory", eligibility-blocking
  preferred = "preferred", "desired", "strong plus", "ideally"
  bonus     = "nice to have", "is a plus", "familiarity with"
  Default: "preferred"

match_type definitions:
  exact     = skill name appears verbatim in both JD and resume
  partial   = related term found (e.g. "REST" matches "REST APIs")
  inferred  = implied by context, no direct phrase found"""

_SUGGESTION_RULES = """\
Suggestion rules — follow exactly:
  - Generate exactly 3 resume improvement suggestions
  - ONLY suggest clarifying, repositioning, or expanding EXISTING resume content
  - NEVER suggest adding skills the candidate does not already have
  - Each suggestion must cite the specific job requirement it addresses"""


# ── JSON schema templates ─────────────────────────────────────────────────────
# Defined as plain strings — no f-string escaping noise.
# build_system_prompt() calls .format() on these with the mode config values.

_SCHEMA_FAST = """\
{
  "score": <1=poor 2=weak 3=moderate 4=strong 5=excellent>,
  "reasoning": "<1-2 sentence assessment>",
  "matched_skills": [
    {"skill": "<short name, max 4 words>", "match_type": "exact|partial|inferred"}
  ],
  "missing_skills": [
    {"skill": "<short name, max 4 words>", "severity": "blocker|major|minor", "requirement_type": "hard|preferred|bonus"}
  ]
}"""

_SCHEMA_STANDARD = """\
{{
  "score": <1=poor 2=weak 3=moderate 4=strong 5=excellent>,
  "reasoning": "<2-3 sentence assessment>",
  "matched_skills": [
    {{"skill": "name", "match_type": "exact|partial|inferred", "jd_snippet": "<{slen} chars or empty string>"}}
  ],
  "missing_skills": [
    {{"skill": "name", "severity": "blocker|major|minor", "requirement_type": "hard|preferred|bonus", "jd_snippet": "<{slen} chars or empty string>"}}
  ]
}}"""

_SCHEMA_DETAILED = """\
{{
  "score": <integer 1-5>,
  "reasoning": "<2-4 sentence honest assessment>",
  "matched_skills": [
    {{"skill": "name", "match_type": "exact|partial|inferred", "jd_snippet": "<{slen} chars>", "resume_snippet": "<{slen} chars>"}}
  ],
  "missing_skills": [
    {{"skill": "name", "severity": "blocker|major|minor", "requirement_type": "hard|preferred|bonus", "jd_snippet": "<{slen} chars>"}}
  ]{suggestions_slot}
}}"""

_SCHEMA_SUGGESTIONS_SLOT = """,
  "suggestions": [
    {"title": "short label", "detail": "specific actionable text", "job_requirement": "verbatim JD phrase this addresses"}
  ]"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_system_prompt(cfg: dict | None = None, mode: str | None = None, resume_snippet: bool = True) -> str:
    """
    Build mode-appropriate system prompt.

    The mode parameter takes priority over the ANALYSIS_MODE env var when provided.
    llm.py passes the effective (capped) mode so the correct schema is always built
    regardless of what the env var says.

    resume_snippet controls whether matched_skills includes resume_snippet in standard mode.
    On retry, llm.py passes False to reduce output complexity for struggling models.
    """
    if cfg is None:
        cfg = get_mode_config()

    mode     = (mode or os.getenv("ANALYSIS_MODE", "standard")).lower().strip()
    slen     = cfg["snippet_len"]
    mmatched = cfg["max_matched"]
    mmissing = cfg["max_missing"]
    do_sugg  = cfg["suggestions"]

    if mode == "fast":
        return "\n\n".join([
            _BASE_LITE,
            (
                f"Return at most {mmatched} matched skills and {mmissing} missing skills.\n"
                "Skill names must be short — maximum 4 words (e.g. \"Python\", \"AWS Lambda\", \"REST APIs\").\n"
                "Do NOT write descriptions or sentences as skill names."
            ),
            "Exactly this JSON shape:",
            _SCHEMA_FAST,
        ])

    if mode == "standard":
        if resume_snippet:
            matched_item = (
                '    {"skill": "name", "match_type": "exact|partial|inferred",'
                ' "jd_snippet": "<SLEN chars or empty>", "resume_snippet": "<SLEN chars or empty>"}'
            ).replace("SLEN", str(slen))
            snippet_instr = (
                f"For jd_snippet and resume_snippet: copy short verbatim phrases from the "
                f"job description and resume respectively (max {slen} chars each). "
                f"Use empty string if no direct phrase exists."
            )
        else:
            matched_item = (
                '    {"skill": "name", "match_type": "exact|partial|inferred",'
                ' "jd_snippet": "<SLEN chars or empty>"}'
            ).replace("SLEN", str(slen))
            snippet_instr = (
                f"For jd_snippet: copy a short phrase verbatim from the job description "
                f"(max {slen} chars). If no direct phrase exists, use an empty string."
            )
        missing_item = (
            '    {"skill": "name", "severity": "blocker|major|minor",'
            ' "requirement_type": "hard|preferred|bonus", "jd_snippet": "<SLEN chars or empty>"}'
        ).replace("SLEN", str(slen))
        schema = "\n".join([
            "{",
            '  "score": <1=poor 2=weak 3=moderate 4=strong 5=excellent>,',
            '  "reasoning": "<2-3 sentence assessment>",',
            '  "matched_skills": [',
            matched_item,
            "  ],",
            '  "missing_skills": [',
            missing_item,
            "  ]",
            "}",
        ])
        return "\n\n".join([
            _BASE_LITE,
            f"Return at most {mmatched} matched skills and {mmissing} missing skills.\n" + snippet_instr,
            "Exactly this JSON shape:",
            schema,
        ])

    # detailed

    # detailed
    suggestions_slot = _SCHEMA_SUGGESTIONS_SLOT if do_sugg else ""
    schema = _SCHEMA_DETAILED.format(slen=slen, suggestions_slot=suggestions_slot)

    sections = [
        _BASE_FULL,
        (
            f"Return at most {mmatched} matched skills and at most {mmissing} missing skills.\n"
            f"Snippets must be verbatim phrases copied from the provided text, max {slen} characters.\n"
            'Do NOT fabricate or paraphrase snippets. If no direct phrase exists, '
            'set match_type to "inferred" and omit resume_snippet.'
        ),
        "Exactly this JSON shape:",
        schema,
    ]
    if do_sugg:
        sections.append(_SUGGESTION_RULES)
    sections.append(_SEVERITY_DEFS)
    sections.append(_SCORING_RUBRIC)

    return "\n\n".join(sections)


def build_user_prompt(resume: str, job_description: str) -> str:
    return (
        "## RESUME\n"
        f"{resume}\n\n"
        "---\n\n"
        "## JOB DESCRIPTION\n"
        f"{job_description}\n\n"
        "---\n\n"
        "Evaluate the match and return ONLY the JSON object described in your instructions."
    )


# ── Backward compat aliases ───────────────────────────────────────────────────
_build_system_prompt = build_system_prompt
_build_user_prompt   = build_user_prompt
