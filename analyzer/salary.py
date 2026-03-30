"""
analyzer/salary.py — Salary detection, estimation and extraction.
"""

import json
import logging
import os
import re

import anthropic
import httpx

from analyzer.config import ANTHROPIC_MODEL, ollama_base_url, ollama_model

logger = logging.getLogger("analyzer.salary")

# ── Salary detection patterns ─────────────────────────────────────────────────

_SALARY_PATTERNS = [
    # Actual currency amounts — strongest signal
    r"\$\s*\d",                       # e.g. $120,000 or $ 45
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
    r"\bper\s+hour\s*[:\$]",         # "per hour: $" or "per hour $"
    r"\bper\s+year\s*[:\$]",         # "per year: $"
    r"\bote\b",                       # On-target earnings
    r"\bstipend\b",
    r"\busd\b",
    r"\bcad\b",
]


def _job_has_salary(job_description: str) -> bool:
    """
    Return True if the JD contains explicit salary/compensation amounts.
    Generic mentions of 'compensation' or 'wage' without actual numbers
    are intentionally excluded to avoid false positives on boilerplate text.
    """
    text = job_description.lower()
    for pattern in _SALARY_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


# ── Prompt builder ────────────────────────────────────────────────────────────

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
        '- If uncertain, use a wide range with confidence="low" (e.g. $60,000-$120,000)',
        "- Use annual USD even if the role is hourly (convert: hourly x 2080)",
        '- confidence = "high" if title + location + seniority are clear',
        '- confidence = "medium" if title is clear but location or seniority is vague',
        '- confidence = "low" if very little signal available — still provide a range',
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


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_salary_response(raw: str) -> dict:
    """Parse and validate the salary estimation JSON response."""
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse salary JSON: {raw[:200]}")

    try:
        min_val = int(data.get("min") or 0)
        max_val = int(data.get("max") or 0)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid salary values — min/max must be integers, "
            f"got: min={data.get('min')!r} max={data.get('max')!r}"
        )

    if min_val <= 0 or max_val <= 0:
        raise ValueError(
            "Could not estimate salary — the model returned no values. "
            "Try again or switch to a different provider."
        )

    if min_val > max_val:
        min_val, max_val = max_val, min_val

    confidence = data.get("confidence", "low")
    if confidence not in ("low", "medium", "high"):
        confidence = "low"

    return {
        "min":        min_val,
        "max":        max_val,
        "currency":   data.get("currency", "USD"),
        "period":     data.get("period", "annual"),
        "confidence": confidence,
        "signals":    data.get("signals", []),
    }


# ── Shared LLM caller ─────────────────────────────────────────────────────────

async def _call_salary_llm(prompt: str, provider: str, temperature: float = 0.1) -> tuple:
    """
    Call the configured LLM for salary prompt.
    Returns (raw_response, payload_or_client) for retry use.
    """
    if provider == "ollama":
        model   = ollama_model()
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a compensation analyst. Always respond with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
            "stream":  False,
            "options": {"temperature": temperature, "num_predict": 400},
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(f"{ollama_base_url()}/api/chat", json=payload)
                resp.raise_for_status()
            except httpx.ConnectError:
                raise ValueError(
                    f"Cannot connect to Ollama at {ollama_base_url()}. "
                    "Make sure Ollama is running: `ollama serve`"
                )
            except httpx.HTTPStatusError as e:
                raise ValueError(f"Ollama error: {e.response.status_code} — {e.response.text}")
        return resp.json()["message"]["content"], payload

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
        return message.content[0].text, client


# ── Public functions ──────────────────────────────────────────────────────────

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

    prompt     = _build_salary_prompt(title, company, location, job_description)
    model_name = ollama_model() if provider == "ollama" else ANTHROPIC_MODEL

    raw, ctx = await _call_salary_llm(prompt, provider, temperature=0.1)

    for attempt in range(2):
        try:
            result = _parse_salary_response(raw)
            result["llm_provider"] = provider
            result["llm_model"]    = model_name
            return result
        except ValueError:
            if attempt == 0:
                logger.warning(f"→ {model_name} salary parse failed (attempt 1), raw: {raw[:300]!r}")
                logger.warning("→ retrying...")
                try:
                    raw, ctx = await _call_salary_llm(prompt, provider, temperature=0.1)
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
            "- Convert hourly to annual (hourly x 2080) if needed",
            "- If a single number is given use it for both min and max",
            "- Convert non-USD to USD equivalent if needed",
            "- confidence should always be 'high' since this is a posted salary",
        ]),
    ])

    model_name = ollama_model() if provider == "ollama" else ANTHROPIC_MODEL
    raw, ctx   = await _call_salary_llm(prompt, provider, temperature=0.0)

    for attempt in range(2):
        try:
            result = _parse_salary_response(raw)
            result["source"]       = "posted"
            result["llm_provider"] = provider
            result["llm_model"]    = model_name
            return result
        except ValueError:
            if attempt == 0:
                logger.warning(f"→ {model_name} salary extraction parse failed (attempt 1), raw: {raw[:300]!r}")
                logger.warning("→ retrying...")
                try:
                    raw, ctx = await _call_salary_llm(prompt, provider, temperature=0.0)
                except Exception as retry_exc:
                    raise ValueError(
                        f"{model_name} could not extract salary — retry also failed. "
                        "Try again or switch to a different provider."
                    ) from retry_exc

    raise ValueError(
        f"{model_name} could not extract salary — no valid range found after 2 attempts. "
        "Try again or switch to a different provider."
    )
