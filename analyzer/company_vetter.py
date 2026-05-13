"""
analyzer/company_vetter.py — LLM-based company legitimacy vetting.

Uses crawled company data (BBB, Glassdoor, LinkedIn) to build a focused
prompt and calls the configured LLM provider for a risk assessment.
Results are stored back in company_meta (llm_* columns) and cached for 7 days.

Only company names and publicly crawled data are sent to the LLM — no PII.

PRIVACY DECISION: LLM recruiter vetting is deliberately NOT implemented.
Sending recruiter name/email/phone to cloud LLM providers would expose
personal data. MX domain checks (mx_validator.py) are the privacy-safe
recruiter signal: only the domain (e.g. abbtech.com) ever leaves the
machine — never the full email address or any personal identifiers.
"""

import json
import logging
import os
import re

import httpx

from analyzer.config import (
    anthropic_model,
    gemini_model,
    ollama_base_url,
    ollama_model,
    openai_model,
    is_thinking_model,
)
from analyzer.llm import _strip_thinking

logger = logging.getLogger("company_vetter")

RISK_LEVELS = ("low", "medium", "high", "unknown")
CACHE_TTL_DAYS = 7


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_company_prompt(company_name: str, meta: dict) -> str:
    """Build a vetting prompt from crawled company data."""
    lines = [f"Company name: {company_name}"]

    # BBB
    bbb_rating = (meta.get("bbb_rating") or "").strip()
    bbb_url    = (meta.get("bbb_url") or "").strip()
    if bbb_rating:
        lines.append(f"BBB rating: {bbb_rating}")
    elif bbb_url:
        lines.append("BBB: listed but no rating available")
    else:
        lines.append("BBB: no listing found")

    # Glassdoor
    gd_rating = meta.get("glassdoor_rating")
    gd_reviews = meta.get("glassdoor_review_count")
    gd_url    = (meta.get("glassdoor_url") or "").strip()
    if gd_rating:
        rev_str = f" ({gd_reviews} reviews)" if gd_reviews else ""
        lines.append(f"Glassdoor rating: {gd_rating}/5{rev_str}")
    elif gd_url:
        lines.append("Glassdoor: listed but no rating available")
    else:
        lines.append("Glassdoor: no listing found")

    # LinkedIn
    li_employees = (meta.get("linkedin_employee_count") or "").strip()
    li_founded   = (meta.get("linkedin_founded") or "").strip()
    li_url       = (meta.get("linkedin_url") or "").strip()
    if li_employees or li_founded:
        li_parts = []
        if li_employees:
            li_parts.append(f"employees: {li_employees}")
        if li_founded:
            li_parts.append(f"founded: {li_founded}")
        lines.append(f"LinkedIn: {', '.join(li_parts)}")
    elif li_url:
        lines.append("LinkedIn: listed but no details available")
    else:
        lines.append("LinkedIn: no company page found")

    data_section = "\n".join(lines)

    return f"""You are a job-search safety analyst. A job seeker wants to know if a company is legitimate before applying.

Analyze the following company data and assess the legitimacy risk for a job applicant.

{data_section}

Respond with ONLY a valid JSON object in this exact format:
{{
  "risk_level": "low" | "medium" | "high" | "unknown",
  "assessment": "2-3 sentence plain-English summary of legitimacy signals",
  "signals": ["key signal 1", "key signal 2"]
}}

Risk level guidance:
- low: established company, good ratings, strong online presence
- medium: limited data, mixed signals, or minor concerns
- high: no online presence, very poor ratings, suspicious patterns
- unknown: insufficient data to make a determination

Respond with JSON only. No prose, no markdown fences."""


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_vetting_response(raw: str, company_name: str) -> dict:
    """Parse LLM JSON response into a structured vetting result."""
    raw = _strip_thinking(raw).strip()

    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

    # Find JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw[:200]!r}")

    data = json.loads(match.group())

    risk_level = str(data.get("risk_level", "unknown")).lower().strip()
    if risk_level not in RISK_LEVELS:
        risk_level = "unknown"

    assessment = str(data.get("assessment", "")).strip()
    if not assessment:
        assessment = "No assessment available."

    signals = data.get("signals", [])
    if not isinstance(signals, list):
        signals = []
    signals = [str(s).strip() for s in signals if s]

    return {
        "risk_level":  risk_level,
        "assessment":  assessment,
        "signals":     signals,
        "company":     company_name,
    }


# ── LLM call ─────────────────────────────────────────────────────────────────

async def _call_vetting_llm(prompt: str, provider: str, model: str = "") -> str:
    """
    Call the configured LLM for company vetting.
    Returns raw response string.
    Supports: anthropic, openai, gemini, ollama.
    """
    system = "You are a job-search safety analyst. Always respond with valid JSON only."

    if provider == "ollama":
        model   = model or ollama_model()
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))

        if is_thinking_model(model):
            system = (
                "CRITICAL: Respond with ONLY a valid JSON object. "
                "No prose, no markdown, no explanations. Start with '{' end with '}'.\n\n"
                + system
            )

        payload = {
            "model":   model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "stream":  False,
            "options": {
                "temperature": 0.1,
                "num_predict": 600 if is_thinking_model(model) else 300,
                "think":       False,
            },
        }
        if is_thinking_model(model):
            payload["format"] = "json"

        logger.info(f"→ vetting ollama request: model={model}")
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
        msg = resp.json().get("message", {})
        raw = msg.get("content") or msg.get("thinking") or ""
        raw = _strip_thinking(raw)
        if os.getenv("SHOW_MORE_LOGS", "").lower() in ("1", "true", "yes"):
            logger.info(f"→ vetting ollama raw body:\n{raw}")
        return raw

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key is not set")
        from openai import AsyncOpenAI
        _model = model or openai_model()
        logger.info(f"→ vetting openai request: model={_model}")
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=_model,
            max_tokens=300,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        if os.getenv("SHOW_MORE_LOGS", "").lower() in ("1", "true", "yes"):
            logger.info(f"→ vetting openai raw body:\n{raw}")
        return raw

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("Gemini API key is not set")
        from google import genai
        from google.genai import types as genai_types
        _model = model or gemini_model()
        logger.info(f"→ vetting gemini request: model={_model}")
        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.1,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            ),
        )
        raw = response.text or ""
        if os.getenv("SHOW_MORE_LOGS", "").lower() in ("1", "true", "yes"):
            logger.info(f"→ vetting gemini raw body:\n{raw}")
        return raw

    else:  # anthropic (default)
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("Anthropic API key is not set")
        import anthropic as ant
        _model = model or anthropic_model()
        logger.info(f"→ vetting anthropic request: model={_model}")
        client  = ant.AsyncAnthropic(api_key=api_key)
        message = await client.messages.create(
            model=_model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
        if os.getenv("SHOW_MORE_LOGS", "").lower() in ("1", "true", "yes"):
            logger.info(f"→ vetting anthropic raw body:\n{raw}")
        return raw


# ── Main entry point ──────────────────────────────────────────────────────────

async def vet_company(
    company_name: str,
    meta: dict,
    provider: str = "anthropic",
    model: str = "",
) -> dict:
    """
    Run LLM vetting for a company using its crawled metadata.

    Args:
        company_name: The company name.
        meta:         Dict from company_meta DB row (crawl results).
        provider:     LLM provider — anthropic, openai, gemini, ollama.
        model:        Optional model override.

    Returns:
        {risk_level, assessment, signals, company, provider, model}
    """
    prompt = build_company_prompt(company_name, meta)
    logger.info(f"→ vetting company={company_name!r} provider={provider}")

    raw = await _call_vetting_llm(prompt, provider, model)
    result = _parse_vetting_response(raw, company_name)

    result["provider"] = provider
    result["model"]    = model or _model_name_for(provider, model)

    logger.info(
        f"✓ vetting complete: company={company_name!r} "
        f"risk={result['risk_level']} provider={provider}"
    )
    return result


def _model_name_for(provider: str, model: str) -> str:
    if model:
        return model
    if provider == "openai":
        return openai_model()
    if provider == "gemini":
        return gemini_model()
    if provider == "ollama":
        return ollama_model()
    return anthropic_model()
