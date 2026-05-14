"""
analyzer/snippet_parser.py — Parse company ratings from pasted Google search snippets.

The user searches Google for e.g. "Techstra Solutions reviews site:glassdoor.com OR site:bbb.org"
and pastes the raw text from the results page. We send that to the LLM to extract
structured rating data without any web scraping.

No network requests to third-party sites — zero bot detection risk.
"""

import json
import logging
import re

from analyzer.llm import _strip_thinking

logger = logging.getLogger("snippet_parser")

SNIPPET_SYSTEM = (
    "You are a data extraction assistant. Extract company rating data from "
    "pasted Google search result text. Always respond with valid JSON only."
)

SNIPPET_PROMPT = """Extract company rating information from the following Google search result text.

Return ONLY a valid JSON object in this exact format:
{{
  "glassdoor_rating": <float or null>,
  "glassdoor_review_count": <integer or null>,
  "glassdoor_url": "<string or null>",
  "indeed_rating": <float or null>,
  "indeed_review_count": <integer or null>,
  "indeed_url": "<string or null>",
  "bbb_rating": "<string grade like A+ or null>",
  "bbb_url": "<string or null>",
  "linkedin_url": "<string or null>",
  "linkedin_employee_count": "<string or null>",
  "linkedin_founded": "<string or null>"
}}

Rules:
- Extract ratings as floats (e.g. 4.3 not "4.3 stars")
- Extract review counts as integers (e.g. 702 not "(702)")
- Extract URLs as full https:// strings when visible
- Use null for any field not found
- If multiple Glassdoor or Indeed results appear, use the one with the most reviews

Search result text:
{text}"""


def _build_snippet_prompt(text: str) -> str:
    # Truncate to avoid token waste — snippets should be short
    text = text.strip()[:3000]
    return SNIPPET_PROMPT.format(text=text)


def _parse_snippet_response(raw: str) -> dict:
    """Parse LLM JSON response into structured company data."""
    raw = _strip_thinking(raw).strip()

    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

    # Find JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {raw[:200]!r}")

    data = json.loads(match.group())

    # Normalize types
    result = {}

    for float_field in ("glassdoor_rating", "indeed_rating"):
        v = data.get(float_field)
        if v is not None:
            try:
                f = float(v)
                if 1.0 <= f <= 5.0:
                    result[float_field] = f
            except (TypeError, ValueError):
                pass

    for int_field in ("glassdoor_review_count", "indeed_review_count"):
        v = data.get(int_field)
        if v is not None:
            try:
                result[int_field] = int(v)
            except (TypeError, ValueError):
                pass

    for str_field in ("glassdoor_url", "indeed_url", "bbb_rating", "bbb_url",
                      "linkedin_url", "linkedin_employee_count", "linkedin_founded"):
        v = data.get(str_field)
        if v and str(v).strip() and str(v).strip().lower() != "null":
            result[str_field] = str(v).strip()

    return result


async def parse_company_snippet(
    text: str,
    provider: str = "anthropic",
    model: str = "",
) -> dict:
    """
    Parse a pasted Google search snippet to extract company ratings.

    Args:
        text:     Raw pasted text from Google search results.
        provider: LLM provider — anthropic, openai, gemini, ollama.
        model:    Optional model override.

    Returns:
        Dict with extracted fields ready for upsert_company_meta().
    """
    import os
    import httpx
    from analyzer.config import (
        anthropic_model, gemini_model, ollama_base_url,
        ollama_model, openai_model, is_thinking_model,
    )

    prompt = _build_snippet_prompt(text)
    logger.info(f"→ parse_company_snippet provider={provider} text_len={len(text)}")

    if provider == "ollama":
        _model = model or ollama_model()
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
        payload = {
            "model":   _model,
            "messages": [
                {"role": "system", "content": SNIPPET_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 400},
        }
        if is_thinking_model(_model):
            payload["format"] = "json"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{ollama_base_url()}/api/chat", json=payload)
            resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key is not set")
        from openai import AsyncOpenAI
        _model = model or openai_model()
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=_model,
            max_tokens=400,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SNIPPET_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = response.choices[0].message.content or ""

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("Gemini API key is not set")
        from google import genai
        from google.genai import types as genai_types
        _model = model or gemini_model()
        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SNIPPET_SYSTEM,
                temperature=0.0,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            ),
        )
        raw = response.text or ""

    else:  # anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("Anthropic API key is not set")
        import anthropic as ant
        _model = model or anthropic_model()
        client = ant.AsyncAnthropic(api_key=api_key)
        message = await client.messages.create(
            model=_model,
            max_tokens=400,
            system=SNIPPET_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text

    if os.getenv("SHOW_MORE_LOGS", "").lower() in ("1", "true", "yes"):
        logger.info(f"→ snippet_parser raw body:\n{raw}")

    result = _parse_snippet_response(raw)
    logger.info(f"✓ snippet_parser extracted {len(result)} fields: {list(result.keys())}")
    return result
