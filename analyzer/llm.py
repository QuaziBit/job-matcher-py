"""
analyzer/llm.py — LLM call helpers and public analyze_match entry point.
"""

import logging
import os

import anthropic
import httpx

from analyzer.config import (
    ANTHROPIC_MODEL, MAX_RETRIES,
    get_mode_config, ollama_base_url, ollama_model,
)
from analyzer.parsers import parse_response
from analyzer.penalties import validate_llm_output, partial_fallback_analysis
from analyzer.prompts import build_system_prompt, build_user_prompt
from ollama_utils import safe_num_predict

logger = logging.getLogger("analyzer.llm")


# ── LLM call helpers ──────────────────────────────────────────────────────────

async def call_anthropic_once(resume: str, job_description: str) -> dict:
    cfg = get_mode_config()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "analysis failed: Anthropic API key is not set — add it in the launcher or config.json"
        )
    system_prompt = build_system_prompt(cfg)
    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=cfg["num_predict"],
        system=system_prompt,
        messages=[{"role": "user", "content": build_user_prompt(resume, job_description)}],
    )
    raw    = message.content[0].text
    result = parse_response(raw, job_description, cfg)
    result["llm_provider"]  = "anthropic"
    result["llm_model"]     = ANTHROPIC_MODEL
    result["analysis_mode"] = os.getenv("ANALYSIS_MODE", "standard")
    return result


async def call_ollama_once(resume: str, job_description: str) -> dict:
    cfg   = get_mode_config()
    model = ollama_model()
    system_prompt = build_system_prompt(cfg)
    full_prompt   = system_prompt + build_user_prompt(resume, job_description)
    num_predict   = safe_num_predict(full_prompt, model_name=model,
                                     desired_output=cfg["num_predict"])
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": build_user_prompt(resume, job_description)},
        ],
        "stream":  False,
        "options": {"temperature": 0.2, "num_predict": num_predict},
    }
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))

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

    raw    = resp.json()["message"]["content"]
    result = parse_response(raw, job_description, cfg)
    result["llm_provider"]  = "ollama"
    result["llm_model"]     = model
    result["analysis_mode"] = os.getenv("ANALYSIS_MODE", "standard")
    return result


# ── Public entry point ────────────────────────────────────────────────────────

async def analyze_match(resume: str, job_description: str, provider: str = "anthropic") -> dict:
    """
    Entry point. provider = 'anthropic' | 'ollama'
    Retries up to MAX_RETRIES times, validates output each attempt.
    Falls back to partial analysis if all retries fail.
    """
    last_error      = None
    last_validation = {"valid": False, "errors": []}

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            logger.info(
                f"→ LLM retry {attempt}/{MAX_RETRIES - 1} "
                f"(prev errors: {last_validation['errors']})"
            )

        try:
            import analyzer.llm as _self
            if provider == "ollama":
                result = await _self.call_ollama_once(resume, job_description)
            else:
                result = await _self.call_anthropic_once(resume, job_description)
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
    fallback = partial_fallback_analysis()
    fallback["validation_errors"] = "; ".join(last_validation["errors"])
    fallback["llm_provider"] = provider
    fallback["llm_model"]    = ollama_model() if provider == "ollama" else ANTHROPIC_MODEL
    return fallback


# ── Backward compat aliases ───────────────────────────────────────────────────
_call_anthropic_once   = call_anthropic_once
_call_ollama_once      = call_ollama_once
analyze_with_anthropic = call_anthropic_once
analyze_with_ollama    = call_ollama_once
