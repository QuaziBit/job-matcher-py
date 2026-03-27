import json
import logging
import os
import re
from collections import defaultdict

import httpx
import anthropic

from skills import normalize_skill, get_skill_category, cluster_penalty_cap

# Experimental
from ollama_utils import safe_num_predict

logger = logging.getLogger("analyzer")

ANTHROPIC_MODEL = "claude-opus-4-5"
MAX_RETRIES = 3

# Hard-blocker keyword patterns for the keyword-based detector
BLOCKER_KEYWORDS = [
    "clearance", "ts/sci", "top secret", "secret", "polygraph",
    "citizenship", "citizen only", "usc only",
]


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1")


SYSTEM_PROMPT = """You are an expert technical recruiter and career coach specializing in software engineering,
DevSecOps, and cloud infrastructure roles. You evaluate how well a candidate's resume matches a job description.

You MUST respond with ONLY valid JSON — no prose, no markdown, no code fences. Exactly this shape:
{
  "score": <integer 1-5>,
  "matched_skills": [
    {"skill": "skill name", "match_type": "exact|partial|inferred",
     "jd_snippet": "verbatim phrase from JD (max 100 chars)",
     "resume_snippet": "verbatim phrase from resume (max 100 chars)"}
  ],
  "missing_skills": [
    {"skill": "skill name", "severity": "blocker|major|minor",
     "requirement_type": "hard|preferred|bonus",
     "jd_snippet": "verbatim phrase from JD (max 100 chars)"}
  ],
  "reasoning": "<2-4 sentence honest assessment>",
  "suggestions": [
    {"title": "short label", "detail": "specific actionable text",
     "job_requirement": "verbatim JD phrase this addresses"}
  ]
}

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
  inferred = implied by context, no direct phrase found

Snippet rules — you MUST follow these exactly:
  - Snippets must be verbatim phrases copied from the provided text, max 100 characters
  - Do NOT fabricate or paraphrase snippets
  - If you cannot find a direct phrase for a matched skill, set match_type to "inferred"
    and omit resume_snippet

Suggestion rules — you MUST follow these exactly:
  - Generate exactly 3 resume improvement suggestions
  - ONLY suggest clarifying, repositioning, or expanding EXISTING resume content
  - NEVER suggest adding skills, certifications, or experience the candidate does not already have
  - Each suggestion must cite the specific job requirement it addresses
  - If the resume is already strong for a requirement, skip it (fewer than 3 is acceptable)

Scoring rubric:
  1 = Poor match — major gaps, different domain entirely
  2 = Weak match — some overlap but significant missing requirements
  3 = Moderate match — meets roughly half the requirements
  4 = Strong match — meets most requirements with minor gaps
  5 = Excellent match — highly aligned, apply immediately
"""


def _build_user_prompt(resume: str, job_description: str) -> str:
    return f"""## RESUME
{resume}

---

## JOB DESCRIPTION
{job_description}

---

Evaluate the match and return ONLY the JSON object described in your instructions."""


# ── Skill parsing helpers ─────────────────────────────────────────────────────

def _normalize_match_type(raw: str) -> str:
    """Normalize non-standard match_type values from models."""
    s = (raw or "").strip().lower()
    if s in ("exact", "direct", "verbatim", "full"):
        return "exact"
    if s in ("partial", "related", "similar", "close"):
        return "partial"
    return "inferred"  # default for anything else


def _parse_matched_skills(raw_matched: list) -> list:
    """
    Parse matched_skills from LLM response.
    Handles both new object format and old plain string format.
    Ollama often returns plain strings — we accept both.
    Normalizes non-standard match_type values.
    """
    result = []
    for item in raw_matched:
        if isinstance(item, dict):
            skill_name = normalize_skill(item.get("skill", ""))
            result.append({
                "skill":          skill_name,
                "match_type":     _normalize_match_type(item.get("match_type", "exact")),
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


def _normalize_severity(raw: str) -> str:
    """
    Normalize non-standard severity labels from models that don't follow instructions.
    e.g. nemotron uses "high", "critical", "required" instead of "blocker", "major", "minor"
    """
    s = (raw or "").strip().lower()
    if s in ("blocker", "critical", "must", "required", "mandatory"):
        return "blocker"
    if s in ("major", "high", "significant", "important"):
        return "major"
    return "minor"  # default for "minor", "low", "nice-to-have", anything else


def _normalize_requirement_type(raw: str) -> str:
    """
    Normalize non-standard requirement_type labels.
    e.g. nemotron uses "required" instead of "hard"
    """
    s = (raw or "").strip().lower()
    if s in ("hard", "required", "mandatory", "must"):
        return "hard"
    if s in ("bonus", "optional", "nice-to-have", "nice to have", "plus"):
        return "bonus"
    return "preferred"  # default


def _parse_missing_skills(raw_missing: list) -> list:
    """
    Parse missing_skills from LLM response.
    Handles both new object format and old plain string format.
    Normalizes non-standard severity and requirement_type values.
    """
    result = []
    for item in raw_missing:
        if isinstance(item, dict):
            skill_name = normalize_skill(item.get("skill", str(item)))
            result.append({
                "skill":            skill_name,
                "severity":         _normalize_severity(item.get("severity", "minor")),
                "requirement_type": _normalize_requirement_type(item.get("requirement_type", "preferred")),
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


def _parse_suggestions(raw_suggestions: list) -> list:
    """
    Parse suggestions from LLM response.
    Handles both object format and plain string format (Ollama often returns strings).
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

def _keyword_boost(missing: list, job_description: str) -> list:
    """
    Keyword detector: upgrade severity of missing skills that match
    known hard-blocker patterns in the job description.
    """
    jd_lower = job_description.lower()
    jd_has_blocker = any(kw in jd_lower for kw in BLOCKER_KEYWORDS)
    jd_years_match = re.search(r"(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)", jd_lower)

    result = []
    for item in missing:
        skill_lower = item["skill"].lower()
        severity = item["severity"]

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


# ── Penalty pipeline ──────────────────────────────────────────────────────────

def penalty_for_skill(skill: dict) -> int:
    """Return the raw penalty for a single missing skill."""
    req_type = skill.get("requirement_type", "preferred")
    severity = skill.get("severity", "minor")

    # Bonus = zero penalty regardless of severity
    if req_type == "bonus":
        return 0

    if severity == "blocker":
        return 2
    elif severity == "major":
        return 1
    return 0  # minor — handled by count threshold


def _compute_adjusted_score(raw_score: int, missing: list) -> tuple:
    """
    Cluster-aware penalty pipeline:
      - Groups missing skills by category (cluster)
      - Caps penalty per cluster (security=2, others=1)
      - Never adjusts below 1
    Returns (adjusted_score, penalty_breakdown)
    """
    # Ensure cluster_group is set on all skills
    for skill in missing:
        if not skill.get("cluster_group"):
            skill["cluster_group"] = get_skill_category(skill["skill"])

    # Group by cluster
    clusters = defaultdict(list)
    for skill in missing:
        clusters[skill["cluster_group"]].append(skill)

    total_penalty = 0
    cluster_penalties = {}

    for group, skills in clusters.items():
        group_penalty = sum(penalty_for_skill(s) for s in skills)
        cap = cluster_penalty_cap(group)
        group_penalty = min(group_penalty, cap)
        cluster_penalties[group] = group_penalty
        total_penalty += group_penalty

    adjusted = max(1, raw_score - total_penalty)

    # Build breakdown for backward compat + new cluster info
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
    """
    Validate LLM output. Returns {"valid": bool, "errors": list[str]}.
    """
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

    # Note: severity and requirement_type are normalized in _parse_missing_skills
    # before validation runs, so non-standard values are already mapped to valid ones.

    if not result.get("reasoning", "").strip():
        errors.append("empty reasoning")

    return {"valid": len(errors) == 0, "errors": errors}


def _partial_fallback_analysis() -> dict:
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


# ── Response parser ───────────────────────────────────────────────────────────

def _sanitize_json(raw: str) -> str:
    """
    Fix unescaped double quotes inside JSON string values produced by LLMs.
    LLMs frequently include unescaped quotes inside snippet/reasoning fields, e.g.:
      "jd_snippet": "experience with "modern" frameworks"
    Uses a JSON-aware state machine to identify and escape inner quotes.
    """
    
    # Temp for Debugging
    # =========================================================== #
    # =========================================================== #
    
    # Normalize curly/smart quotes first
    raw = raw.replace("\u201c", '\\"').replace("\u201d", '\\"')
    raw = raw.replace("\u2018", "'").replace("\u2019", "'")

    out = []
    i = 0
    n = len(raw)

    while i < n:
        c = raw[i]

        if c == '"':
            # Read a quoted token (key or value)
            out.append(c)
            i += 1
            while i < n:
                c = raw[i]
                if c == '\\':
                    out.append(c)
                    i += 1
                    if i < n:
                        out.append(raw[i])
                        i += 1
                    continue
                if c == '"':
                    out.append(c)
                    i += 1
                    break
                out.append(c)
                i += 1

            # Check if this token is a key (followed by colon then a string value)
            j = i
            while j < n and raw[j] in ' \t\r\n':
                j += 1

            if j < n and raw[j] == ':':
                # Consume colon and whitespace
                out.append(raw[i:j + 1])
                i = j + 1
                while i < n and raw[i] in ' \t\r\n':
                    out.append(raw[i])
                    i += 1

                # Consume the string value with inner-quote fixing
                if i < n and raw[i] == '"':
                    out.append('"')
                    i += 1
                    value_chars = []
                    while i < n:
                        c = raw[i]
                        if c == '\\':
                            value_chars.append(c)
                            i += 1
                            if i < n:
                                value_chars.append(raw[i])
                                i += 1
                            continue
                        if c == '"':
                            # Determine if this is the real closing quote
                            k = i + 1
                            while k < n and raw[k] in ' \t\r\n':
                                k += 1
                            next_ch = raw[k] if k < n else ''
                            if next_ch in (',', '}', ']', '', '"'):
                                break  # real closing quote
                            else:
                                # Unescaped inner quote — escape it
                                value_chars.append('\\"')
                                i += 1
                                continue
                        value_chars.append(c)
                        i += 1
                    out.extend(value_chars)
                    out.append('"')
                    i += 1  # skip real closing quote
                    continue
        else:
            out.append(c)
            i += 1

    return ''.join(out)

def _repair_truncated_json(raw: str) -> str:
    """
    Attempt to close a truncated JSON object produced by a model that hit its
    token limit mid-response. Strategy: count unclosed brackets/braces and
    append the necessary closing characters.
    """
    open_braces   = raw.count('{') - raw.count('}')
    open_brackets = raw.count('[') - raw.count(']')

    if open_braces <= 0 and open_brackets <= 0:
        return raw  # not truncated

    # Strip trailing commas, partial keys, or incomplete strings before closing
    # Find the last complete value — walk backwards to last } ] " or digit
    stripped = raw.rstrip()
    # Remove trailing comma or partial line
    while stripped and stripped[-1] in (',', ':', '{', '['):
        stripped = stripped[:-1].rstrip()

    # Close any open arrays first, then objects
    closing = (']' * open_brackets) + ('}' * open_braces)
    repaired = stripped + closing
    logger.info(f"→ repaired truncated JSON: appended {repr(closing)}")
    return repaired


def _parse_response(raw: str, job_description: str = "") -> dict:
    """Extract and validate JSON from LLM response, apply full penalty pipeline."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    json_str = match.group()

    # Try 1: parse as-is
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try 2: repair truncation then parse
        try:
            data = json.loads(_repair_truncated_json(json_str))
        except json.JSONDecodeError:
            # Try 3: sanitize quotes then parse
            try:
                data = json.loads(_sanitize_json(json_str))
            except json.JSONDecodeError:
                # Try 4: repair + sanitize
                try:
                    data = json.loads(_sanitize_json(_repair_truncated_json(json_str)))
                except json.JSONDecodeError as e:
                    raise ValueError(f"failed to parse LLM JSON: {e}")

    score = int(data.get("score", 0))
    if not 1 <= score <= 5:
        raise ValueError(f"Score out of range: {score}")

    matched  = _parse_matched_skills(data.get("matched_skills", []))
    missing  = _parse_missing_skills(data.get("missing_skills", []))
    suggestions = _parse_suggestions(data.get("suggestions", []))

    # Keyword detector pass — upgrade severities based on JD content
    if job_description:
        missing = _keyword_boost(missing, job_description)

    adjusted_score, penalty_breakdown = _compute_adjusted_score(score, missing)

    return {
        "score":             score,
        "adjusted_score":    adjusted_score,
        "penalty_breakdown": penalty_breakdown,
        "matched_skills":    matched,
        "missing_skills":    missing,
        "reasoning":         data.get("reasoning", ""),
        "suggestions":       suggestions,
        "retry_count":       0,
        "used_fallback":     False,
        "validation_errors": "",
    }


# ── LLM call helpers ──────────────────────────────────────────────────────────

async def _call_anthropic_once(resume: str, job_description: str) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "analysis failed: Anthropic API key is not set — add it in the launcher or config.json"
        )
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096, # old max_tokens: 1024
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(resume, job_description)}],
    )
    raw = message.content[0].text
    result = _parse_response(raw, job_description)
    result["llm_provider"] = "anthropic"
    result["llm_model"]    = ANTHROPIC_MODEL
    return result


async def _call_ollama_once(resume: str, job_description: str) -> dict:
    model = _ollama_model()
    full_prompt = SYSTEM_PROMPT + _build_user_prompt(resume, job_description)
    num_predict = safe_num_predict(full_prompt, model_name=model, desired_output=4096)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(resume, job_description)},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": num_predict},
    }


    timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))


    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(f"{_ollama_base_url()}/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.ConnectError:
            raise ValueError(
                f"Cannot connect to Ollama at {_ollama_base_url()}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except httpx.HTTPStatusError as e:
            raise ValueError(f"Ollama error: {e.response.status_code} — {e.response.text}")

    raw = resp.json()["message"]["content"]
    result = _parse_response(raw, job_description)
    result["llm_provider"] = "ollama"
    result["llm_model"]    = model
    return result


# ── Public entry point with retry loop ───────────────────────────────────────

async def analyze_match(resume: str, job_description: str, provider: str = "anthropic") -> dict:
    """
    Entry point. provider = 'anthropic' | 'ollama'
    Retries up to MAX_RETRIES times, validates output each attempt.
    Falls back to partial analysis if all retries fail.
    """
    last_error = None
    last_validation: dict = {"valid": False, "errors": []}

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            logger.info(
                f"→ LLM retry {attempt}/{MAX_RETRIES - 1} "
                f"(prev errors: {last_validation['errors']})"
            )

        try:
            if provider == "ollama":
                result = await _call_ollama_once(resume, job_description)
            else:
                result = await _call_anthropic_once(resume, job_description)
        except Exception as e:
            last_error = e
            logger.error(f"✗ LLM attempt {attempt + 1} failed: {e}")
            continue

        last_validation = validate_llm_output(result, job_description, resume)
        if last_validation["valid"]:
            result["retry_count"]       = attempt
            result["used_fallback"]     = False
            result["validation_errors"] = ""
            return result

        logger.error(f"✗ LLM output validation failed: {last_validation['errors']}")
        last_error = Exception(f"validation: {last_validation['errors']}")

    logger.error(f"✗ All {MAX_RETRIES} attempts failed, using fallback analysis")
    fallback = _partial_fallback_analysis()
    fallback["validation_errors"] = "; ".join(last_validation["errors"])
    fallback["llm_provider"] = provider
    fallback["llm_model"]    = _ollama_model() if provider == "ollama" else ANTHROPIC_MODEL
    return fallback


# ── Backward compat aliases (used by existing tests) ─────────────────────────
analyze_with_anthropic = _call_anthropic_once
analyze_with_ollama    = _call_ollama_once
