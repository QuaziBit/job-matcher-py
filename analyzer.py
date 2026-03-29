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

# ── Analysis Mode configuration ───────────────────────────────────────────────

MODE_CONFIG = {
    "fast": {
        "snippet_len":  40,
        "max_matched":   5,
        "max_missing":   4,
        "suggestions": False,
        "num_predict": 800,
    },
    "standard": {
        "snippet_len":  70,
        "max_matched":   8,
        "max_missing":   6,
        "suggestions": True,
        "num_predict": 1800,
    },
    "detailed": {
        "snippet_len":  100,
        "max_matched":  15,
        "max_missing":  10,
        "suggestions": True,
        "num_predict": 4096,
    },
}

# Estimated response time in seconds per mode (used by frontend progress bar)
MODE_ESTIMATES = {
    "fast":     30,
    "standard": 90,
    "detailed": 240,
}


def get_mode_config() -> dict:
    """Read ANALYSIS_MODE from env and return the corresponding config dict."""
    mode = os.getenv("ANALYSIS_MODE", "standard").lower().strip()
    return MODE_CONFIG.get(mode, MODE_CONFIG["standard"])

# Hard-blocker keyword patterns for the keyword-based detector
BLOCKER_KEYWORDS = [
    "clearance", "ts/sci", "top secret", "secret", "polygraph",
    "citizenship", "citizen only", "usc only",
]


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1")


# ── System prompts per mode ───────────────────────────────────────────────────

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


def _build_system_prompt(cfg: dict | None = None) -> str:
    """Build mode-appropriate system prompt."""
    if cfg is None:
        cfg = get_mode_config()

    mode      = os.getenv("ANALYSIS_MODE", "standard").lower().strip()
    slen      = cfg["snippet_len"]
    mmatched  = cfg["max_matched"]
    mmissing  = cfg["max_missing"]
    do_sugg   = cfg["suggestions"]

    if mode == "fast":
        return f"""{_SYSTEM_PROMPT_BASE}

Return at most {mmatched} matched skills and at most {mmissing} missing skills — only the most significant ones.
Snippets must be verbatim phrases, max {slen} characters. Do NOT fabricate snippets.

Exactly this JSON shape:
{{
  "score": <integer 1-5>,
  "matched_skills": [
    {{"skill": "name", "match_type": "exact|partial|inferred", "jd_snippet": "<{slen} chars>"}}
  ],
  "missing_skills": [
    {{"skill": "name", "severity": "blocker|major|minor",
      "requirement_type": "hard|preferred|bonus", "jd_snippet": "<{slen} chars>"}}
  ],
  "reasoning": "<1-2 sentence honest assessment>"
}}
{_SEVERITY_DEFS}
{_SCORING_RUBRIC}"""

    # standard or detailed
    sugg_block = ""
    if do_sugg:
        sugg_block = f"""
  "suggestions": [
    {{"title": "short label", "detail": "specific actionable text",
      "job_requirement": "verbatim JD phrase this addresses"}}
  ]

Suggestion rules — you MUST follow these exactly:
  - Generate exactly 3 resume improvement suggestions
  - ONLY suggest clarifying, repositioning, or expanding EXISTING resume content
  - NEVER suggest adding skills the candidate does not already have
  - Each suggestion must cite the specific job requirement it addresses"""

    return f"""{_SYSTEM_PROMPT_BASE}

Return at most {mmatched} matched skills and at most {mmissing} missing skills.
Snippets must be verbatim phrases copied from the provided text, max {slen} characters.
Do NOT fabricate or paraphrase snippets. If no direct phrase exists, set match_type to "inferred" and omit resume_snippet.

Exactly this JSON shape:
{{
  "score": <integer 1-5>,
  "matched_skills": [
    {{"skill": "name", "match_type": "exact|partial|inferred",
      "jd_snippet": "<{slen} chars>", "resume_snippet": "<{slen} chars>"}}
  ],
  "missing_skills": [
    {{"skill": "name", "severity": "blocker|major|minor",
      "requirement_type": "hard|preferred|bonus", "jd_snippet": "<{slen} chars>"}}
  ],
  "reasoning": "<2-4 sentence honest assessment>"{sugg_block}
}}
{_SEVERITY_DEFS}
{_SCORING_RUBRIC}"""


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


def _parse_response(raw: str, job_description: str = "", cfg: dict | None = None) -> dict:
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

    if cfg is None:
        cfg = get_mode_config()

    # Parse and apply mode-based caps
    matched = _parse_matched_skills(data.get("matched_skills", []))
    missing = _parse_missing_skills(data.get("missing_skills", []))
    matched = matched[:cfg["max_matched"]]
    missing = missing[:cfg["max_missing"]]

    # Suggestions — skip entirely in fast mode
    suggestions = []
    if cfg["suggestions"]:
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
    cfg = get_mode_config()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "analysis failed: Anthropic API key is not set — add it in the launcher or config.json"
        )
    system_prompt = _build_system_prompt(cfg)
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=cfg["num_predict"],
        system=system_prompt,
        messages=[{"role": "user", "content": _build_user_prompt(resume, job_description)}],
    )
    raw = message.content[0].text
    result = _parse_response(raw, job_description, cfg)
    result["llm_provider"] = "anthropic"
    result["llm_model"]    = ANTHROPIC_MODEL
    result["analysis_mode"] = os.getenv("ANALYSIS_MODE", "standard")
    return result


async def _call_ollama_once(resume: str, job_description: str) -> dict:
    cfg   = get_mode_config()
    model = _ollama_model()
    system_prompt = _build_system_prompt(cfg)
    full_prompt = system_prompt + _build_user_prompt(resume, job_description)
    num_predict = safe_num_predict(full_prompt, model_name=model,
                                   desired_output=cfg["num_predict"])

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
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
    result = _parse_response(raw, job_description, cfg)
    result["llm_provider"]  = "ollama"
    result["llm_model"]     = model
    result["analysis_mode"] = os.getenv("ANALYSIS_MODE", "standard")
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


# ── Salary estimation ─────────────────────────────────────────────────────────

# Patterns that indicate explicit salary info in a JD.
# Strategy: require either an actual number/amount OR very specific salary phrases.
# Generic words like "compensation" and "wage" alone are NOT enough — they appear
# in boilerplate like "compensation determined by contract rates" with no real number.
_SALARY_PATTERNS = [
    # Actual currency amounts — strongest signal
    r"\$\s*\d",                      # e.g. $120,000 or $ 45
    r"£\s*\d",                        # GBP amounts
    r"€\s*\d",                        # EUR amounts
    r"\$/yr",                          # e.g. $80k/yr
    r"\bk/year\b",                    # e.g. 120k/year

    # Specific salary phrases that imply a concrete number follows
    r"\bsalary\s+(range|is|of|:\s*\$)",   # "salary range", "salary: $"
    r"\bbase salary\b",               # almost always followed by a number
    r"\bbase pay\b",                  # same
    r"\bannual salary\b",
    r"\bhourly rate\s+(is|of|:\s*\$)",
    r"\bper\s+hour\s*[:\$]",        # "per hour: $" or "per hour $"
    r"\bper\s+year\s*[:\$]",        # "per year: $"
    r"\bote\b",                       # On-target earnings
    r"\bstipend\b",
    r"\busd\b",
    r"\bcad\b",
]

def _job_has_salary(job_description: str) -> bool:
    """
    Return True if the JD contains explicit salary/compensation amounts.
    Generic mentions of 'compensation' or 'wage' without actual numbers
    are intentionally excluded to avoid false positives on boilerplate text
    like 'compensation determined by contract wage rates'.
    """
    import re
    text = job_description.lower()
    for pattern in _SALARY_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def _build_salary_prompt(title: str, company: str, location: str, job_description: str) -> str:
    """Build a concise prompt for salary estimation."""
    context_parts = []
    if title:    context_parts.append(f"Job Title: {title}")
    if company:  context_parts.append(f"Company: {company}")
    if location: context_parts.append(f"Location: {location}")
    context_parts.append(f"\nJob Description (excerpt):\n{job_description[:4000]}")
    context = "\n".join(context_parts)

    json_schema = "\n".join([
        "{",
        '  "min": <integer, annual USD, must be greater than 0>,',
        '  "max": <integer, annual USD, must be greater than 0>,',
        '  "currency": "USD",',
        '  "period": "annual",',
        '  "confidence": "low" | "medium" | "high",',
        '  "signals": ["signals used, e.g. seniority level, required skills, location"]',
        "}",
    ])

    rules = "\n".join([
        "- ALWAYS provide a best-guess range — never use 0 for min or max",
        "- If uncertain, use a wide range with confidence=\"low\" (e.g. $60,000-$120,000)",
        "- Use annual USD even if the role is hourly (convert: hourly x 2080)",
        "- confidence = \"high\" if title + location + seniority are clear",
        "- confidence = \"medium\" if title is clear but location or seniority is vague",
        "- confidence = \"low\" if very little signal available — still provide a range",
        "- If the role is outside the US, still estimate in USD equivalent",
    ])

    sections = [
        "You are a compensation analyst. Based on the job details below, estimate the salary range.",
        context,
        "Respond with ONLY valid JSON — no explanation, no markdown fences:",
        json_schema,
        "Rules:",
        rules,
    ]
    return "\n\n".join(sections)


def _parse_salary_response(raw: str) -> dict:
    """Parse and validate the salary estimation JSON response."""
    import re

    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract a JSON object from surrounding text
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse salary JSON: {raw[:200]}")

    # Validate required fields
    try:
        min_val = int(data.get("min") or 0)
        max_val = int(data.get("max") or 0)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid salary values — min/max must be integers, got: min={data.get('min')!r} max={data.get('max')!r}")
    if min_val <= 0 or max_val <= 0:
        raise ValueError("Could not estimate salary — the model returned no values. Try again or switch to a different provider.")
    if min_val > max_val:
        min_val, max_val = max_val, min_val  # swap if reversed

    confidence = data.get("confidence", "low")
    if confidence not in ("low", "medium", "high"):
        confidence = "low"

    return {
        "min":      min_val,
        "max":      max_val,
        "currency": data.get("currency", "USD"),
        "period":   data.get("period", "annual"),
        "confidence": confidence,
        "signals":  data.get("signals", []),
    }


async def estimate_salary(
    title: str,
    company: str,
    location: str,
    job_description: str,
    provider: str = "anthropic",
    _skip_salary_check: bool = False,
) -> dict:
    """
    Estimate salary for a job using the configured LLM provider.
    Returns a dict with min, max, currency, period, confidence, signals.
    Raises ValueError if the provider is unavailable or response is unparseable.
    """
    if not _skip_salary_check and _job_has_salary(job_description):
        raise ValueError("Job description already contains salary information")

    prompt = _build_salary_prompt(title, company, location, job_description)

    if provider == "ollama":
        model   = _ollama_model()
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a compensation analyst. Always respond with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
            "stream":  False,
            "options": {"temperature": 0.1, "num_predict": 400},
        }
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

    else:  # anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("Anthropic API key is not set")
        client  = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            system="You are a compensation analyst. Always respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text

    model_name = _ollama_model() if provider == "ollama" else ANTHROPIC_MODEL
    last_exc = None
    for attempt in range(2):
        try:
            result = _parse_salary_response(raw)
            result["llm_provider"] = provider
            result["llm_model"]    = model_name
            return result
        except ValueError as e:
            last_exc = e
            if attempt == 0:
                logger.warning(f"→ {model_name} salary parse failed (attempt 1), raw response: {raw[:300]!r}")
                logger.warning(f"→ retrying...")
                # Re-call the LLM for the retry
                try:
                    if provider == "ollama":
                        async with httpx.AsyncClient(timeout=int(os.getenv("OLLAMA_TIMEOUT", "600"))) as _c:
                            _r = await _c.post(f"{_ollama_base_url()}/api/chat", json=payload)
                            raw = _r.json()["message"]["content"]
                    else:
                        _m = client.messages.create(
                            model=ANTHROPIC_MODEL, max_tokens=400,
                            system="You are a compensation analyst. Always respond with valid JSON only.",
                            messages=[{"role": "user", "content": prompt}],
                        )
                        raw = _m.content[0].text
                except Exception as retry_exc:
                    raise ValueError(
                        f"{model_name} could not estimate salary — retry also failed. "
                        "Try again or switch to a different provider."
                    ) from retry_exc
    raise ValueError(
        f"{model_name} could not estimate salary — no valid range returned after 2 attempts. "
        "Try again or switch to a different provider."
    )


async def extract_salary(
    title: str,
    company: str,
    location: str,
    job_description: str,
    provider: str = "anthropic",
) -> dict:
    """
    Extract explicitly posted salary from a job description.
    Returns same shape as estimate_salary but with source='posted'.
    """
    prompt = "\n\n".join([
        "You are a compensation analyst. Extract the salary range explicitly stated in the job description below.",
        f"Job Title: {title}\nCompany: {company}\nLocation: {location}\n\nJob Description:\n{job_description[:3000]}",
        "Respond with ONLY valid JSON — no explanation, no markdown fences:",
        "\n".join([
            "{",
            '  "min": <integer, annual USD>,',
            '  "max": <integer, annual USD>,',
            '  "currency": "USD",',
            '  "period": "annual",',
            '  "confidence": "high",',
            '  "signals": ["exact phrases from JD that state the salary"]',
            "}",
        ]),
        "\n".join([
            "Rules:",
            "- Extract ONLY explicitly stated numbers — do not estimate",
            "- Convert hourly to annual (hourly × 2080) if needed",
            "- If a single number is given use it for both min and max",
            "- Convert non-USD to USD equivalent if needed",
            "- confidence should always be 'high' since this is a posted salary",
        ]),
    ])

    if provider == "ollama":
        model   = _ollama_model()
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a compensation analyst. Always respond with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 400},
        }
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
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("Anthropic API key is not set")
        client  = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            system="You are a compensation analyst. Always respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text

    model_name = _ollama_model() if provider == "ollama" else ANTHROPIC_MODEL
    last_exc = None
    for attempt in range(2):
        try:
            result = _parse_salary_response(raw)
            result["source"]       = "posted"
            result["llm_provider"] = provider
            result["llm_model"]    = model_name
            return result
        except ValueError as e:
            last_exc = e
            if attempt == 0:
                logger.warning(f"→ {model_name} salary extraction parse failed (attempt 1), raw: {raw[:300]!r}")
                logger.warning(f"→ retrying...")
                try:
                    if provider == "ollama":
                        async with httpx.AsyncClient(timeout=int(os.getenv("OLLAMA_TIMEOUT", "600"))) as _c:
                            _r = await _c.post(f"{_ollama_base_url()}/api/chat", json=payload)
                            raw = _r.json()["message"]["content"]
                    else:
                        _m = client.messages.create(
                            model=ANTHROPIC_MODEL, max_tokens=400,
                            system="You are a compensation analyst. Always respond with valid JSON only.",
                            messages=[{"role": "user", "content": prompt}],
                        )
                        raw = _m.content[0].text
                except Exception as retry_exc:
                    raise ValueError(
                        f"{model_name} could not extract salary — retry also failed. "
                        "Try again or switch to a different provider."
                    ) from retry_exc
    raise ValueError(
        f"{model_name} could not extract salary — no valid range found after 2 attempts. "
        "Try again or switch to a different provider."
    )
