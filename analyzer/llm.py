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
from analyzer.parsers import parse_response, repair_truncated_json, sanitize_json, _escape_control_chars
from ollama_utils import safe_num_predict, estimate_tokens, get_context_window
from analyzer.penalties import validate_llm_output, partial_fallback_analysis, auto_correct_llm_output
from analyzer.prompts import (
    build_system_prompt, build_user_prompt,
    build_chunk1_prompt, build_chunk2_prompt,
    build_chunk3_prompt, build_chunk4_prompt,
    build_chunk_user_prompt,
)

logger = logging.getLogger("analyzer.llm")


def _verbose() -> bool:
    """Return True if SHOW_MORE_LOGS=true in env — bumps chunk debug to info."""
    return os.getenv("SHOW_MORE_LOGS", "").lower() in ("1", "true", "yes")


def _log_chunk(msg: str) -> None:
    """Log at info if SHOW_MORE_LOGS, otherwise debug."""
    if _verbose():
        logger.info(msg)
    else:
        logger.debug(msg)



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


# ── Chunked Ollama caller ────────────────────────────────────────────────────

async def _call_chunk(
    system_prompt: str,
    user_prompt: str,
    model: str,
    num_predict: int,
    chunk_name: str,
) -> str | None:
    """
    Execute a single chunk call to Ollama.
    Returns raw response string or None on failure.
    Failures are logged but never raise — caller decides how to handle.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": num_predict},
    }
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{ollama_base_url()}/api/chat", json=payload)
            resp.raise_for_status()
        resp_json   = resp.json()
        raw         = resp_json["message"]["content"]
        eval_count  = resp_json.get("eval_count", "?")
        prompt_eval = resp_json.get("prompt_eval_count", "?")
        logger.info(
            f"→ chunk {chunk_name} response ({len(raw)} chars) "
            f"prompt_tokens={prompt_eval} output_tokens={eval_count}"
        )
        _log_chunk(f"→ chunk {chunk_name} raw body:\n{raw[:800]}")
        return raw
    except httpx.ConnectError:
        logger.error(f"✗ chunk {chunk_name}: cannot connect to Ollama")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"✗ chunk {chunk_name}: HTTP {e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"✗ chunk {chunk_name}: unexpected error: {e}")
        return None


def _parse_chunk(raw: str | None, key: str, chunk_name: str) -> list | dict | None:
    """
    Parse a chunk response — extract the expected key from JSON.
    Applies the same repair pipeline as the main parser:
      1. strip markdown fences
      2. escape control chars
      3. repair truncated JSON (missing closing braces)
      4. sanitize quotes
    Returns the value or None if all passes fail.
    """
    import json as _json, re
    if not raw:
        return None

    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    attempts = [
        lambda s: s,
        lambda s: _escape_control_chars(s),
        lambda s: repair_truncated_json(s),
        lambda s: repair_truncated_json(_escape_control_chars(s)),
        lambda s: sanitize_json(repair_truncated_json(s)),
    ]

    for fn in attempts:
        try:
            candidate = fn(cleaned)
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if not match:
                continue
            data = _json.loads(match.group())
            if key not in data:
                continue
            return data[key]
        except Exception:
            continue

    logger.warning(f"✗ chunk {chunk_name}: could not parse JSON after all repair passes")
    _log_chunk(f"→ chunk {chunk_name} raw: {cleaned[:600]}")
    return None


def _parse_score_chunk(raw: str | None) -> tuple[int | None, str]:
    """
    Parse chunk 1 — returns (score, reasoning).
    Applies repair pipeline so truncated JSON (missing closing brace) still parses.
    """
    import json as _json, re
    if not raw:
        return None, ""

    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    attempts = [
        lambda s: s,
        lambda s: _escape_control_chars(s),
        lambda s: repair_truncated_json(s),
        lambda s: repair_truncated_json(_escape_control_chars(s)),
        lambda s: sanitize_json(repair_truncated_json(s)),
    ]

    for fn in attempts:
        try:
            candidate = fn(cleaned)
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if not match:
                continue
            data      = _json.loads(match.group())
            raw_score = data.get("score", 0)
            score     = round(float(raw_score))
            reasoning = data.get("reasoning", "")
            if not 1 <= score <= 5:
                continue
            return score, reasoning
        except Exception:
            continue

    logger.warning("✗ chunk score: could not parse JSON after all repair passes")
    _log_chunk(f"→ chunk score raw:\n{cleaned[:600]}")
    return None, ""


async def call_ollama_chunked(resume: str, job_description: str) -> dict:
    """
    Chunked Ollama analysis — splits the request into focused sub-calls.
    Each chunk asks for one piece of the result with a tiny schema.
    Gracefully degrades: if a chunk fails, we use what we have.

    Chunk layout per mode:
      fast:     chunk1 (score+reasoning) + chunk2 (matched) + chunk3 (missing)
      standard: chunk1 (score+reasoning) + chunk2 (matched) + chunk3 (missing)
      detailed: chunk1 + chunk2 + chunk3 + chunk4 (suggestions)
    """
    cfg   = get_mode_config()
    model = ollama_model()

    requested_mode = os.getenv("ANALYSIS_MODE", "standard").lower()
    actual_mode    = cap_mode_for_model(requested_mode, model)

    if actual_mode != requested_mode:
        logger.warning(
            f"⚠ {model} max mode is '{actual_mode}' — "
            f"downgrading from '{requested_mode}' to '{actual_mode}'"
        )
        cfg = MODE_CONFIG[actual_mode]

    logger.info(
        f"→ chunked mode={actual_mode} model={model} "
        f"max_matched={cfg['max_matched']} max_missing={cfg['max_missing']} "
        f"suggestions={cfg['suggestions']}"
    )

    user_prompt = build_chunk_user_prompt(resume, job_description)

    # ── Chunk 1: score + reasoning ────────────────────────────────────────────
    sys1 = build_chunk1_prompt(cfg, actual_mode)
    raw1 = await _call_chunk(sys1, user_prompt, model, num_predict=200, chunk_name="1/score+reasoning")
    score, reasoning = _parse_score_chunk(raw1)

    if score is None:
        logger.error("✗ chunk 1 failed — cannot produce analysis without score")
        raise ValueError("Chunked analysis failed: could not get score from model")

    logger.info(f"→ chunk 1 OK: score={score}")

    # ── Chunk 2: matched_skills ───────────────────────────────────────────────
    sys2    = build_chunk2_prompt(cfg, actual_mode)
    raw2    = await _call_chunk(sys2, user_prompt, model, num_predict=800, chunk_name="2/matched")
    matched = _parse_chunk(raw2, "matched_skills", "2/matched") or []
    logger.info(f"→ chunk 2 OK: {len(matched)} matched skills")

    # ── Chunk 3: missing_skills ───────────────────────────────────────────────
    sys3    = build_chunk3_prompt(cfg, actual_mode)
    raw3    = await _call_chunk(sys3, user_prompt, model, num_predict=800, chunk_name="3/missing")
    missing = _parse_chunk(raw3, "missing_skills", "3/missing") or []
    logger.info(f"→ chunk 3 OK: {len(missing)} missing skills")

    # ── Chunk 4: suggestions (detailed only) ─────────────────────────────────
    suggestions = []
    if actual_mode == "detailed" and cfg.get("suggestions"):
        sys4        = build_chunk4_prompt(cfg)
        raw4        = await _call_chunk(sys4, user_prompt, model, num_predict=600, chunk_name="4/suggestions")
        suggestions = _parse_chunk(raw4, "suggestions", "4/suggestions") or []
        logger.info(f"→ chunk 4 OK: {len(suggestions)} suggestions")

    # ── Merge and apply penalty pipeline ─────────────────────────────────────
    from analyzer.skills_helpers import parse_matched_skills, parse_missing_skills, parse_suggestions, keyword_boost
    from analyzer.penalties import compute_adjusted_score

    parsed_matched = parse_matched_skills(matched)[:cfg["max_matched"]]
    parsed_missing = parse_missing_skills(missing)[:cfg["max_missing"]]
    parsed_missing = keyword_boost(parsed_missing, job_description)
    parsed_suggestions = parse_suggestions(suggestions) if suggestions else []

    adjusted_score, penalty_breakdown = compute_adjusted_score(score, parsed_missing)

    if not reasoning.strip():
        reasoning = "Analysis completed. Review matched and missing skills above."

    result = {
        "score":             score,
        "adjusted_score":    adjusted_score,
        "penalty_breakdown": penalty_breakdown,
        "matched_skills":    parsed_matched,
        "missing_skills":    parsed_missing,
        "reasoning":         reasoning,
        "suggestions":       parsed_suggestions,
        "llm_provider":      "ollama",
        "llm_model":         model,
        "analysis_mode":     actual_mode,
        "retry_count":       0,
        "used_fallback":     False,
        "validation_errors": "",
    }

    corrections = auto_correct_llm_output(result)
    if corrections:
        logger.info(f"→ auto-corrected chunked result: {corrections}")

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
                result = await _self.call_ollama_chunked(resume, job_description)
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
_call_ollama_chunked   = call_ollama_chunked
analyze_with_anthropic = call_anthropic_once
analyze_with_ollama    = call_ollama_chunked
