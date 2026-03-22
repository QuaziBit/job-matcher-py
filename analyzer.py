import json
import os
import re
import httpx
import anthropic

ANTHROPIC_MODEL = "claude-opus-4-5"

# Hard-blocker keyword patterns for the keyword-based detector
BLOCKER_PATTERNS = [
    r"\b(ts[/\-]sci|top secret|secret clearance|clearance required|active clearance)\b",
    r"\b(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)\b",
    r"\b(must have|required|mandatory)\b",
    r"\b(pmp|cissp|cisa|cism|giac|oscp)\b",
    r"\b(us citizen|citizenship required|usc only)\b",
]

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
  "matched_skills": ["skill1", "skill2", ...],
  "missing_skills": [
    {"skill": "skill name", "severity": "blocker|major|minor"},
    ...
  ],
  "reasoning": "<2-4 sentence honest assessment>"
}

Severity definitions for missing_skills:
  blocker = eliminates candidacy entirely (e.g. required clearance, mandatory cert, minimum years not met)
  major   = significant gap that will hurt chances substantially
  minor   = nice-to-have or learnable gap that is unlikely to disqualify

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


def _parse_missing_skills(raw_missing: list) -> tuple[list, list]:
    """
    Parse missing_skills from LLM response.
    Handles both new format [{skill, severity}] and old format ["skill"].
    Returns: (structured_list, flat_name_list)
    """
    structured = []
    for item in raw_missing:
        if isinstance(item, dict):
            structured.append({
                "skill":    item.get("skill", str(item)),
                "severity": item.get("severity", "minor"),
            })
        else:
            # Old flat string format — default severity to minor
            structured.append({"skill": str(item), "severity": "minor"})
    return structured


def _keyword_boost(structured_missing: list, job_description: str) -> list:
    """
    Keyword detector pass: upgrade severity of any missing skill that
    matches known hard-blocker patterns in the JD.
    """
    jd_lower = job_description.lower()

    # Check if the JD itself contains blocker-level language
    jd_has_blocker = any(kw in jd_lower for kw in BLOCKER_KEYWORDS)
    jd_years_match = re.search(r"(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)", jd_lower)

    result = []
    for item in structured_missing:
        skill_lower = item["skill"].lower()
        severity = item["severity"]

        # Upgrade to blocker if skill name contains clearance/citizenship keywords
        if any(kw in skill_lower for kw in BLOCKER_KEYWORDS):
            severity = "blocker"

        # Upgrade if skill mentions years and JD has a years requirement
        if re.search(r"\d+\s*years?", skill_lower) and jd_years_match:
            severity = "blocker"

        # Upgrade major → blocker if JD explicitly marks it required and it's already major
        if severity == "major" and jd_has_blocker and any(kw in skill_lower for kw in ["required", "must"]):
            severity = "blocker"

        result.append({"skill": item["skill"], "severity": severity})
    return result


def _compute_adjusted_score(raw_score: int, structured_missing: list) -> tuple[int, dict]:
    """
    Penalty pipeline combining all methods:
      - blocker: -2 each (capped at -3 total)
      - major:   -1 each (capped at -2 total)
      - minor:   -0.5 each (capped at -1 total), rounds down
      - simple count: additional -1 if total missing > 6

    Returns (adjusted_score, penalty_breakdown)
    """
    blockers = [s for s in structured_missing if s["severity"] == "blocker"]
    majors   = [s for s in structured_missing if s["severity"] == "major"]
    minors   = [s for s in structured_missing if s["severity"] == "minor"]

    blocker_penalty = min(len(blockers) * 2, 3)
    major_penalty   = min(len(majors) * 1,   2)
    minor_penalty   = min(int(len(minors) * 0.5), 1)
    count_penalty   = 1 if len(structured_missing) > 6 else 0

    total_penalty = blocker_penalty + major_penalty + minor_penalty + count_penalty
    adjusted = max(1, raw_score - total_penalty)

    breakdown = {
        "blockers":       len(blockers),
        "majors":         len(majors),
        "minors":         len(minors),
        "blocker_penalty": blocker_penalty,
        "major_penalty":   major_penalty,
        "minor_penalty":   minor_penalty,
        "count_penalty":   count_penalty,
        "total_penalty":   total_penalty,
    }
    return adjusted, breakdown


def _parse_response(raw: str, job_description: str = "") -> dict:
    """Extract and validate JSON from LLM response, apply full penalty pipeline."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    data = json.loads(match.group())

    score = int(data.get("score", 0))
    if not 1 <= score <= 5:
        raise ValueError(f"Score out of range: {score}")

    raw_missing = data.get("missing_skills", [])

    # Handle both [{skill, severity}] and ["skill"] formats
    structured_missing = _parse_missing_skills(raw_missing)

    # Keyword detector pass — upgrade severities based on JD content
    if job_description:
        structured_missing = _keyword_boost(structured_missing, job_description)

    # Compute adjusted score via full penalty pipeline
    adjusted_score, penalty_breakdown = _compute_adjusted_score(score, structured_missing)

    return {
        "score":             score,
        "adjusted_score":    adjusted_score,
        "penalty_breakdown": penalty_breakdown,
        "matched_skills":    data.get("matched_skills", []),
        "missing_skills":    structured_missing,
        "reasoning":         data.get("reasoning", ""),
    }


async def analyze_with_anthropic(resume: str, job_description: str) -> dict:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(resume, job_description)}],
    )
    raw = message.content[0].text
    result = _parse_response(raw, job_description)
    result["llm_provider"] = "anthropic"
    result["llm_model"]    = ANTHROPIC_MODEL
    return result


async def analyze_with_ollama(resume: str, job_description: str) -> dict:
    model = _ollama_model()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(resume, job_description)},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    # Large models (70B, 27B) can take several minutes to generate a response.
    # Timeout is configurable via OLLAMA_TIMEOUT env var, default 600s (10 min).
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


async def analyze_match(resume: str, job_description: str, provider: str = "anthropic") -> dict:
    """
    Entry point. provider = 'anthropic' | 'ollama'
    Returns: {score, adjusted_score, penalty_breakdown, matched_skills,
              missing_skills, reasoning, llm_provider, llm_model}
    """
    if provider == "ollama":
        return await analyze_with_ollama(resume, job_description)
    return await analyze_with_anthropic(resume, job_description)
