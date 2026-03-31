"""
analyzer/prompts.py — LLM prompt builders for job analysis.

All JSON schema fragments are defined as plain string constants at the top.
Builder functions just select and assemble the right pieces — no messy
string construction inside function bodies.

Mode complexity is tuned to model capability:
  fast     — minimal schema, lite prompt, no snippets, reasoning first
  standard — medium schema, lite prompt, jd_snippet + optional resume_snippet
  detailed — full schema, full prompt with definitions and rubric

Chunked mode splits the request into 3-4 tiny focused calls:
  chunk 1 — score + reasoning  (~60 tokens output)
  chunk 2 — matched_skills     (~200 tokens output)
  chunk 3 — missing_skills     (~200 tokens output)
  chunk 4 — suggestions        (~150 tokens, detailed mode only)
"""

import os

from analyzer.config import get_mode_config


# ── System prompt bases ───────────────────────────────────────────────────────

_BASE_LITE = """\
You are a technical recruiter. Evaluate how well a resume matches a job description.
Respond with ONLY a valid JSON object. No text before or after the JSON. No markdown."""

_BASE_FULL = """\
You are an expert technical recruiter and career coach specializing in software engineering,
DevSecOps, and cloud infrastructure roles. You evaluate how well a candidate's resume
matches a job description.

You MUST respond with ONLY valid JSON — no prose, no markdown, no code fences."""


# ── Definition blocks (detailed mode only) ────────────────────────────────────

_SCORING_RUBRIC = """\
Scoring rubric:
  1 = Poor match     — major gaps, different domain entirely
  2 = Weak match     — some overlap but significant missing requirements
  3 = Moderate match — meets roughly half the requirements
  4 = Strong match   — meets most requirements with minor gaps
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


# ── Single-shot schema fragments ──────────────────────────────────────────────
# Used by build_system_prompt() for non-chunked (single-request) analysis.
# Templates that need config values use {placeholders} filled via .format().

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

_SCHEMA_STANDARD_MATCHED_WITH_RESUME = """\
    {{"skill": "name", "match_type": "exact|partial|inferred", "jd_snippet": "<{slen} chars or empty>", "resume_snippet": "<{slen} chars or empty>"}}"""

_SCHEMA_STANDARD_MATCHED_NO_RESUME = """\
    {{"skill": "name", "match_type": "exact|partial|inferred", "jd_snippet": "<{slen} chars or empty>"}}"""

_SCHEMA_STANDARD_MISSING = """\
    {{"skill": "name", "severity": "blocker|major|minor", "requirement_type": "hard|preferred|bonus", "jd_snippet": "<{slen} chars or empty>"}}"""

_SCHEMA_STANDARD_WRAPPER = """\
{{
  "score": <1=poor 2=weak 3=moderate 4=strong 5=excellent>,
  "reasoning": "<2-3 sentence assessment>",
  "matched_skills": [
{matched_item}
  ],
  "missing_skills": [
{missing_item}
  ]
}}"""

_SCHEMA_DETAILED_MATCHED = """\
    {{"skill": "name", "match_type": "exact|partial|inferred", "jd_snippet": "<{slen} chars>", "resume_snippet": "<{slen} chars>"}}"""

_SCHEMA_DETAILED_MISSING = """\
    {{"skill": "name", "severity": "blocker|major|minor", "requirement_type": "hard|preferred|bonus", "jd_snippet": "<{slen} chars>"}}"""

_SCHEMA_DETAILED_SUGGESTIONS = """\
  "suggestions": [
    {{"title": "short label", "detail": "specific actionable text", "job_requirement": "verbatim JD phrase this addresses"}}
  ]"""

_SCHEMA_DETAILED_WRAPPER_NO_SUGGESTIONS = """\
{{
  "score": <integer 1-5>,
  "reasoning": "<2-4 sentence honest assessment>",
  "matched_skills": [
{matched_item}
  ],
  "missing_skills": [
{missing_item}
  ]
}}"""

_SCHEMA_DETAILED_WRAPPER_WITH_SUGGESTIONS = """\
{{
  "score": <integer 1-5>,
  "reasoning": "<2-4 sentence honest assessment>",
  "matched_skills": [
{matched_item}
  ],
  "missing_skills": [
{missing_item}
  ],
{suggestions}
}}"""


# ── Chunked schema fragments ──────────────────────────────────────────────────
# Used by build_chunkN_prompt() — one schema per chunk, all tiny.

_CHUNK1_SCHEMA = """\
{
  "score": <1=poor 2=weak 3=moderate 4=strong 5=excellent>,
  "reasoning": "<2-3 sentence honest assessment of the overall match>"
}"""

_CHUNK1_SCORING = """\
Scoring:
  1=poor  2=weak  3=moderate  4=strong  5=excellent"""

_CHUNK2_SCHEMA_FAST = """\
{
  "matched_skills": [
    {"skill": "<short name max 4 words>", "match_type": "exact|partial|inferred"}
  ]
}"""

_CHUNK2_SCHEMA_STANDARD = """\
{{
  "matched_skills": [
    {{"skill": "name", "match_type": "exact|partial|inferred", "jd_snippet": "<{slen} chars or empty>"}}
  ]
}}"""

_CHUNK2_MATCH_TYPES = """\
match_type:
  exact    = skill appears verbatim in both JD and resume
  partial  = related term (e.g. "REST" matches "REST APIs")
  inferred = implied by context"""

_CHUNK3_SCHEMA_FAST = """\
{
  "missing_skills": [
    {"skill": "<short name max 4 words>", "severity": "blocker|major|minor", "requirement_type": "hard|preferred|bonus"}
  ]
}"""

_CHUNK3_SCHEMA_STANDARD = """\
{{
  "missing_skills": [
    {{"skill": "name", "severity": "blocker|major|minor", "requirement_type": "hard|preferred|bonus", "jd_snippet": "<{slen} chars or empty>"}}
  ]
}}"""

_CHUNK3_SEVERITY = """\
severity:
  blocker = eliminates candidacy (required clearance, mandatory cert, minimum years)
  major   = significant gap that will hurt chances
  minor   = nice-to-have, unlikely to disqualify

requirement_type:
  hard      = "required", "must have", "mandatory"
  preferred = "preferred", "desired", "ideally"
  bonus     = "nice to have", "is a plus"
  Default: preferred"""

_CHUNK4_SCHEMA = """\
{
  "suggestions": [
    {"title": "short label", "detail": "specific actionable advice", "job_requirement": "verbatim JD phrase this addresses"}
  ]
}"""


# ── Single-shot prompt builders ───────────────────────────────────────────────

def build_system_prompt(cfg: dict | None = None, mode: str | None = None, resume_snippet: bool = True) -> str:
    """
    Build mode-appropriate system prompt for single-shot (non-chunked) analysis.

    mode          — takes priority over ANALYSIS_MODE env var when provided
    resume_snippet — standard mode only: include resume_snippet field on first attempt,
                     drop it on retry to reduce output complexity
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
        matched_item = (
            _SCHEMA_STANDARD_MATCHED_WITH_RESUME if resume_snippet
            else _SCHEMA_STANDARD_MATCHED_NO_RESUME
        ).format(slen=slen)
        missing_item = _SCHEMA_STANDARD_MISSING.format(slen=slen)
        schema       = _SCHEMA_STANDARD_WRAPPER.format(
            matched_item=matched_item,
            missing_item=missing_item,
        )
        snippet_instr = (
            f"For jd_snippet and resume_snippet: copy short verbatim phrases from the "
            f"job description and resume respectively (max {slen} chars each). "
            f"Use empty string if no direct phrase exists."
            if resume_snippet else
            f"For jd_snippet: copy a short phrase verbatim from the job description "
            f"(max {slen} chars). Use empty string if no direct phrase exists."
        )
        return "\n\n".join([
            _BASE_LITE,
            f"Return at most {mmatched} matched skills and {mmissing} missing skills.\n{snippet_instr}",
            "Exactly this JSON shape:",
            schema,
        ])

    # detailed
    matched_item = _SCHEMA_DETAILED_MATCHED.format(slen=slen)
    missing_item = _SCHEMA_DETAILED_MISSING.format(slen=slen)
    if do_sugg:
        suggestions = _SCHEMA_DETAILED_SUGGESTIONS.format(slen=slen)
        schema = _SCHEMA_DETAILED_WRAPPER_WITH_SUGGESTIONS.format(
            matched_item=matched_item,
            missing_item=missing_item,
            suggestions=suggestions,
        )
    else:
        schema = _SCHEMA_DETAILED_WRAPPER_NO_SUGGESTIONS.format(
            matched_item=matched_item,
            missing_item=missing_item,
        )

    sections = [
        _BASE_FULL,
        (
            f"Return at most {mmatched} matched skills and at most {mmissing} missing skills.\n"
            f"Snippets must be verbatim phrases copied from the provided text, max {slen} characters.\n"
            "Do NOT fabricate or paraphrase snippets. If no direct phrase exists, "
            "set match_type to \"inferred\" and omit resume_snippet."
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
    return "\n\n".join([
        f"## RESUME\n{resume}",
        "---",
        f"## JOB DESCRIPTION\n{job_description}",
        "---",
        "Evaluate the match and return ONLY the JSON object described in your instructions.",
    ])


# ── Chunked prompt builders ───────────────────────────────────────────────────

def build_chunk1_prompt(cfg: dict, mode: str) -> str:
    """Chunk 1 — score + reasoning. Shared by all modes (~60 token output)."""
    return "\n\n".join([
        _BASE_LITE,
        "Return ONLY a JSON object with score and reasoning.",
        _CHUNK1_SCHEMA,
        _CHUNK1_SCORING,
    ])


def build_chunk2_prompt(cfg: dict, mode: str) -> str:
    """Chunk 2 — matched_skills only. Schema scales with mode."""
    slen     = cfg["snippet_len"]
    mmatched = cfg["max_matched"]

    if mode == "fast":
        schema      = _CHUNK2_SCHEMA_FAST
        instruction = (
            f"Return at most {mmatched} matched skills — skills found in BOTH the resume and JD.\n"
            "Skill names must be short — maximum 4 words. Do NOT write sentences as skill names."
        )
    else:
        schema      = _CHUNK2_SCHEMA_STANDARD.format(slen=slen)
        instruction = (
            f"Return at most {mmatched} matched skills — skills found in BOTH the resume and JD.\n"
            f"For jd_snippet: copy a short phrase verbatim from the JD (max {slen} chars). "
            f"Use empty string if no direct phrase exists."
        )

    return "\n\n".join([
        _BASE_LITE,
        instruction,
        "Return ONLY this JSON shape:",
        schema,
        _CHUNK2_MATCH_TYPES,
    ])


def build_chunk3_prompt(cfg: dict, mode: str) -> str:
    """Chunk 3 — missing_skills only. Schema scales with mode."""
    slen     = cfg["snippet_len"]
    mmissing = cfg["max_missing"]

    if mode == "fast":
        schema      = _CHUNK3_SCHEMA_FAST
        instruction = (
            f"Return at most {mmissing} missing skills — skills required by the JD but NOT in the resume.\n"
            "Skill names must be short — maximum 4 words."
        )
    else:
        schema      = _CHUNK3_SCHEMA_STANDARD.format(slen=slen)
        instruction = (
            f"Return at most {mmissing} missing skills — skills required by the JD but NOT in the resume.\n"
            f"For jd_snippet: copy a short phrase verbatim from the JD (max {slen} chars). "
            f"Use empty string if no direct phrase exists."
        )

    return "\n\n".join([
        _BASE_LITE,
        instruction,
        "Return ONLY this JSON shape:",
        schema,
        _CHUNK3_SEVERITY,
    ])


def build_chunk4_prompt(cfg: dict) -> str:
    """Chunk 4 — suggestions only. Detailed mode only."""
    return "\n\n".join([
        _BASE_LITE,
        (
            "Generate exactly 3 resume improvement suggestions.\n"
            "ONLY suggest clarifying or repositioning EXISTING resume content.\n"
            "NEVER suggest adding skills the candidate does not already have."
        ),
        _CHUNK4_SCHEMA,
    ])


def build_chunk_user_prompt(resume: str, job_description: str) -> str:
    """User prompt shared by all chunks."""
    return "\n\n".join([
        f"## RESUME\n{resume}",
        "---",
        f"## JOB DESCRIPTION\n{job_description}",
        "---",
        "Return ONLY the JSON object. No other text.",
    ])


# ── Backward compat aliases ───────────────────────────────────────────────────
_build_system_prompt = build_system_prompt
_build_user_prompt   = build_user_prompt
