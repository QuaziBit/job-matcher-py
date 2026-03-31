"""
analyzer/llm.py — LLM call helpers and public analyze_match entry point.
"""

import logging
import os

import anthropic
import httpx

from analyzer.config import (
    ANTHROPIC_MODEL, MAX_RETRIES, MODE_CONFIG,
    get_mode_config, ollama_base_url, ollama_model,
    cap_mode_for_model,
)
from analyzer.parsers import parse_response
from ollama_utils import safe_num_predict, estimate_tokens, get_context_window
from analyzer.penalties import validate_llm_output, partial_fallback_analysis
from analyzer.prompts import build_system_prompt, build_user_prompt

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
    logger.debug(f"→ raw Anthropic response ({len(raw)} chars):\n{raw[:2000]}")
    result = parse_response(raw, job_description, cfg)
    result["llm_provider"]  = "anthropic"
    result["llm_model"]     = ANTHROPIC_MODEL
    result["analysis_mode"] = os.getenv("ANALYSIS_MODE", "standard")
    return result


async def call_ollama_once(resume: str, job_description: str, resume_snippet: bool = True) -> dict:
    cfg   = get_mode_config()
    model = ollama_model()

    # Pre-flight: if the combined prompt is too large for the model's context
    # window, trim the job description so the model has enough room to respond.
    # Target: leave at least desired_output + 512 headroom tokens for the response.
    system_prompt  = build_system_prompt(cfg, resume_snippet=resume_snippet)  # rebuilt below if mode is capped
    context_window = get_context_window(model)
    desired_output = cfg["num_predict"]
    MIN_OUTPUT     = int(desired_output * 0.80)  # require at least 80% of desired output tokens

    requested_mode = os.getenv("ANALYSIS_MODE", "standard").lower()
    actual_mode    = cap_mode_for_model(requested_mode, model)

    if actual_mode != requested_mode:
        logger.warning(
            f"⚠ {model} max mode is '{actual_mode}' — "
            f"downgrading from '{requested_mode}' to '{actual_mode}'"
        )
        cfg            = MODE_CONFIG[actual_mode]
        desired_output = cfg["num_predict"]
        MIN_OUTPUT     = int(desired_output * 0.80)
        system_prompt  = build_system_prompt(cfg, mode=actual_mode, resume_snippet=resume_snippet)

    logger.info(
        f"→ mode={actual_mode} model={model} "
        f"num_predict={cfg['num_predict']} "
        f"max_matched={cfg['max_matched']} max_missing={cfg['max_missing']} "
        f"suggestions={cfg['suggestions']}"
    )

    full_prompt   = system_prompt + build_user_prompt(resume, job_description)
    prompt_tokens = int(estimate_tokens(full_prompt) * 1.20)  # 20% buffer

    # Calculate how much room is left for the model to generate output
    available_for_output = context_window - prompt_tokens - 512

    logger.debug(
        f"→ context pre-flight: model={model} context={context_window} "
        f"prompt_chars={len(full_prompt)} prompt_tokens~{prompt_tokens} "
        f"available={available_for_output} min_required={MIN_OUTPUT} "
        f"desired={desired_output}"
    )

    if available_for_output < MIN_OUTPUT:
        overhead         = estimate_tokens(system_prompt + build_user_prompt(resume, ""))
        overhead_buf     = int(overhead * 1.20)
        original_jd_len  = len(job_description)

        # First try: downgrade mode to reduce desired_output before trimming JD
        fallback_modes = {"detailed": ("standard", 1800), "standard": ("fast", 800)}
        if actual_mode in fallback_modes:
            downgraded_mode, downgraded_output = fallback_modes[actual_mode]
            downgraded_available = context_window - overhead_buf - downgraded_output - 512
            if downgraded_available >= int(downgraded_output * 0.80):
                logger.warning(
                    f"⚠ prompt too large for {model} in {actual_mode} mode "
                    f"(available={available_for_output} < min={MIN_OUTPUT}) — "
                    f"downgrading to {downgraded_mode} mode "
                    f"(desired_output {desired_output} → {downgraded_output})"
                )
                cfg            = MODE_CONFIG[downgraded_mode]
                desired_output = downgraded_output
                MIN_OUTPUT     = int(desired_output * 0.80)
                system_prompt  = build_system_prompt(cfg, mode=downgraded_mode, resume_snippet=resume_snippet)
                full_prompt    = system_prompt + build_user_prompt(resume, job_description)
                prompt_tokens  = int(estimate_tokens(full_prompt) * 1.20)
                available_for_output = context_window - prompt_tokens - 512
                actual_mode    = downgraded_mode

        # Second try: trim JD if still too large after mode downgrade
        if available_for_output < MIN_OUTPUT:
            max_jd_tokens   = context_window - overhead_buf - desired_output - 512
            max_jd_chars    = max(500, int(max_jd_tokens * 3.5))
            if original_jd_len > max_jd_chars:
                job_description = job_description[:max_jd_chars] + "\n\n[...truncated]"
                full_prompt     = system_prompt + build_user_prompt(resume, job_description)
                new_prompt_tok  = int(estimate_tokens(full_prompt) * 1.20)
                new_available   = context_window - new_prompt_tok - 512
                logger.warning(
                    f"⚠ still too large after mode adjustment — "
                    f"trimmed JD {original_jd_len} → {max_jd_chars} chars "
                    f"(overhead~{overhead_buf} tokens, new available={new_available})"
                )

    num_predict = safe_num_predict(full_prompt, model_name=model,
                                   desired_output=desired_output)
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

    resp_json = resp.json()
    raw       = resp_json["message"]["content"]

    # Log token usage from Ollama response if available
    eval_count   = resp_json.get("eval_count", "?")
    prompt_eval  = resp_json.get("prompt_eval_count", "?")
    logger.info(
        f"→ raw Ollama response ({len(raw)} chars) "
        f"prompt_tokens={prompt_eval} output_tokens={eval_count}"
    )
    logger.debug(f"→ raw Ollama response body:\n{raw[:2000]}")

    result = parse_response(raw, job_description, cfg)
    result["llm_provider"]  = "ollama"
    result["llm_model"]     = model
    result["analysis_mode"] = actual_mode
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
            logger.debug(
                f"→ attempt {attempt + 1}/{MAX_RETRIES} provider={provider} "
                f"resume={len(resume)} chars jd={len(job_description)} chars"
            )
            import analyzer.llm as _self
            if provider == "ollama":
                # On retry in standard mode, drop resume_snippet to reduce
                # output complexity for models that struggled on attempt 1.
                # Detailed mode always keeps resume_snippet — it's core to the schema.
                current_mode = os.getenv("ANALYSIS_MODE", "standard").lower()
                use_resume_snippet = (attempt == 0) or (current_mode != "standard")
                if not use_resume_snippet:
                    logger.info("→ retry: dropping resume_snippet from standard mode prompt")
                result = await _self.call_ollama_once(resume, job_description,
                                                      resume_snippet=use_resume_snippet)
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
        logger.info(f"→ raw result on validation failure: score={result.get('score')} "
                    f"matched={len(result.get('matched_skills',[]))} "
                    f"missing={len(result.get('missing_skills',[]))} "
                    f"reasoning={repr(result.get('reasoning','')[:100])}")
        last_error = Exception(f"validation: {last_validation['errors']}")

    logger.error(
        f"✗ All {MAX_RETRIES} attempts failed, using fallback analysis. "
        f"Last error: {last_error}"
    )
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
