"""
analyzer/llm.py — LLM call helpers and public analyze_match entry point.
"""

import logging
import os
import re

import anthropic
import httpx

from analyzer.config import (
    MAX_RETRIES, MODE_CONFIG,
    get_mode_config, ollama_base_url, ollama_model,
    anthropic_model, openai_model, gemini_model,
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


def _get_model_for_provider(provider: str) -> str:
    """Return the configured model name for the given provider."""
    if provider == "openai":
        return openai_model()
    if provider == "gemini":
        return gemini_model()
    if provider == "ollama":
        return ollama_model()
    return anthropic_model()


# ── LLM call helpers ──────────────────────────────────────────────────────────

async def call_anthropic_once(resume: str, job_description: str) -> dict:
    cfg = get_mode_config()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "analysis failed: Anthropic API key is not set — add it in the launcher or config.json"
        )
    mode = os.getenv("ANALYSIS_MODE", "standard")
    model = anthropic_model()
    system_prompt = build_system_prompt(cfg)
    logger.info(
        f"→ anthropic request: model={model} mode={mode} "
        f"max_tokens={cfg['num_predict']} resume={len(resume)} chars jd={len(job_description)} chars"
    )
    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=cfg["num_predict"],
        system=system_prompt,
        messages=[{"role": "user", "content": build_user_prompt(resume, job_description)}],
    )
    raw          = message.content[0].text
    input_tokens = getattr(message.usage, "input_tokens", "?")
    output_tokens = getattr(message.usage, "output_tokens", "?")
    logger.info(
        f"→ anthropic response ({len(raw)} chars) "
        f"input_tokens={input_tokens} output_tokens={output_tokens}"
    )
    _log_chunk(f"→ anthropic raw body:\n{raw}")
    result = parse_response(raw, job_description, cfg)
    result["llm_provider"]  = "anthropic"
    result["llm_model"]     = model
    result["analysis_mode"] = mode
    return result


async def call_openai_once(resume: str, job_description: str) -> dict:
    """Single-shot OpenAI Chat Completions call — same shape as call_anthropic_once."""
    cfg = get_mode_config()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "analysis failed: OpenAI API key is not set — add OPENAI_API_KEY in the launcher"
        )
    model = openai_model()
    system_prompt = build_system_prompt(cfg)
    user_prompt   = build_user_prompt(resume, job_description)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ValueError("openai package is not installed — run: pip install openai")

    mode = os.getenv("ANALYSIS_MODE", "standard")
    logger.info(
        f"→ openai request: model={model} mode={mode} "
        f"max_tokens={cfg['num_predict']} resume={len(resume)} chars jd={len(job_description)} chars"
    )
    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=cfg["num_predict"],
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content or ""
    usage = response.usage
    input_tokens  = getattr(usage, "prompt_tokens", "?")
    output_tokens = getattr(usage, "completion_tokens", "?")
    logger.info(
        f"→ openai response ({len(raw)} chars) "
        f"prompt_tokens={input_tokens} completion_tokens={output_tokens}"
    )
    _log_chunk(f"→ openai raw body:\n{raw}")
    result = parse_response(raw, job_description, cfg)
    result["llm_provider"]  = "openai"
    result["llm_model"]     = model
    result["analysis_mode"] = mode
    return result


async def call_gemini_once(resume: str, job_description: str) -> dict:
    """Single-shot Google Gemini call — same shape as call_anthropic_once."""
    cfg = get_mode_config()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "analysis failed: Gemini API key is not set — add GEMINI_API_KEY in the launcher"
        )
    model = gemini_model()
    system_prompt = build_system_prompt(cfg)
    user_prompt   = build_user_prompt(resume, job_description)

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise ValueError("google-genai package is not installed — run: pip install google-genai")

    mode = os.getenv("ANALYSIS_MODE", "standard")
    logger.info(
        f"→ gemini request: model={model} mode={mode} "
        f"resume={len(resume)} chars jd={len(job_description)} chars"
    )
    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            # Do not cap max_output_tokens for Gemini — thinking models
            # (2.5 Flash etc.) consume tokens internally before visible output,
            # so a hard cap starves the response. Let the model decide.
            temperature=0.2,
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        ),
    )
    raw = response.text or ""
    # Log token usage if available in response metadata
    usage = getattr(response, "usage_metadata", None)
    input_tokens  = getattr(usage, "prompt_token_count", "?") if usage else "?"
    output_tokens = getattr(usage, "candidates_token_count", "?") if usage else "?"
    logger.info(
        f"→ gemini response ({len(raw)} chars) "
        f"prompt_tokens={input_tokens} output_tokens={output_tokens}"
    )
    _log_chunk(f"→ gemini raw body:\n{raw}")
    result = parse_response(raw, job_description, cfg)
    result["llm_provider"]  = "gemini"
    result["llm_model"]     = model
    result["analysis_mode"] = mode
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

    _log_chunk(
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

    from analyzer.config import is_thinking_model
    if is_thinking_model(model):
        # Thinking models need a larger output budget — the detailed JSON schema
        # with 15 matched skills, snippets, missing skills, and suggestions
        # easily exceeds 4096 tokens.
        desired_output = min(desired_output * 2, 8192)
        MIN_OUTPUT = int(desired_output * 0.80)
        # Also reduce snippet lengths and counts to keep output manageable.
        # Even with 8192 tokens, long snippets across 15 skills exhaust the budget.
        cfg = dict(cfg)
        cfg["snippet_len"] = min(cfg.get("snippet_len", 100), 60)
        cfg["max_matched"] = min(cfg.get("max_matched", 15), 10)
        cfg["max_missing"] = min(cfg.get("max_missing", 10), 7)

        # Thinking models tend to produce prose despite JSON instructions.
        # Prepend a hard constraint that overrides the conversational tendency.
        system_prompt = (
            "CRITICAL INSTRUCTION: Your entire response must be a single valid JSON object. "
            "Do NOT write any prose, markdown, headers, bullet points, or explanations. "
            "Do NOT start with 'Based on' or any other sentence. "
            "Start your response with '{' and end with '}'. Nothing else.\n\n"
            + system_prompt
        )

    user_prompt = build_user_prompt(resume, job_description)
    if is_thinking_model(model):
        user_prompt += "\n\nRemember: respond with ONLY a JSON object starting with '{'. No prose."

    num_predict = safe_num_predict(full_prompt, model_name=model,
                                   desired_output=desired_output)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream":  False,
        "options": {"temperature": 0.2, "num_predict": num_predict, "think": False},
    }
    if is_thinking_model(model):
        payload["format"] = "json"
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
    msg = resp_json.get("message", {})
    content  = msg.get("content", "")
    thinking = msg.get("thinking", "")
    raw = content or thinking or ""
    raw = _strip_thinking(raw)

    eval_count   = resp_json.get("eval_count", "?")
    prompt_eval  = resp_json.get("prompt_eval_count", "?")
    logger.info(
        f"→ raw Ollama response ({len(raw)} chars) "
        f"content={len(content)} thinking={len(thinking)} "
        f"prompt_tokens={prompt_eval} output_tokens={eval_count}"
    )
    _log_chunk(f"→ raw Ollama response body:\n{raw}")

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
        "options": {"temperature": 0.1, "num_predict": num_predict, "think": False},
    }
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{ollama_base_url()}/api/chat", json=payload)
            resp.raise_for_status()
        resp_json   = resp.json()
        msg         = resp_json.get("message", {})
        raw         = msg.get("content") or msg.get("thinking") or ""
        eval_count  = resp_json.get("eval_count", "?")
        prompt_eval = resp_json.get("prompt_eval_count", "?")
        logger.info(
            f"→ chunk {chunk_name} response ({len(raw)} chars) "
            f"prompt_tokens={prompt_eval} output_tokens={eval_count}"
        )
        _log_chunk(f"→ chunk {chunk_name} raw body:\n{raw}")
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


def _strip_thinking(raw: str | None) -> str:
    """
    Strip Ollama thinking-model preamble from raw response content.
    Gemma 4 and other thinking models prepend 'Thinking Process:...' or
    '<think>...</think>' blocks before the actual JSON output.
    """
    if not raw:
        return raw or ""
    import re as _re
    # Strip <think>...</think> blocks
    raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL)
    # Strip "Thinking Process:" narrative blocks — everything up to the first { or [
    m = _re.search(r"[{\[]", raw)
    if m and m.start() > 0:
        preamble = raw[:m.start()]
        # Only strip if it looks like a thinking preamble, not legitimate text
        if any(marker in preamble for marker in ("Thinking Process", "thinking process", "Let me analyze", "I need to")):
            raw = raw[m.start():]
    return raw.strip()


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

    raw     = _strip_thinking(raw)
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

    # Last resort: if the array was truncated mid-item, try to extract
    # only the complete items by finding the last valid closing brace
    # before the truncation point and closing the array/object there.
    try:
        match = re.search(r"\{[^{]*\"" + key + r"\":\s*\[(.*)\]", cleaned, re.DOTALL)
        if match:
            array_content = match.group(1)
            last_complete = array_content.rfind("}")
            if last_complete > 0:
                truncated_fixed = '{"'  + key + '": [' + array_content[:last_complete + 1] + ']}'
                data = _json.loads(truncated_fixed)
                if key in data:
                    return data[key]
    except Exception:
        pass

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

    raw     = _strip_thinking(raw)
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


async def call_ollama_thinking(resume: str, job_description: str) -> dict:
    """
    Two-call analysis path for thinking models (Gemma 4, DeepSeek-R1, etc.).

    Thinking models stop generating JSON early when given a large schema.
    Splitting into two focused calls keeps each response within the model's
    self-imposed completion limit:
      Call A — score + reasoning + matched_skills
      Call B — missing_skills + suggestions
    Results are merged into the standard analysis dict.
    """
    cfg   = get_mode_config()
    model = ollama_model()

    # Compact config for thinking models — shorter snippets, fewer items
    cfg = dict(cfg)
    cfg["snippet_len"] = min(cfg.get("snippet_len", 100), 60)
    cfg["max_matched"] = min(cfg.get("max_matched", 15), 10)
    cfg["max_missing"] = min(cfg.get("max_missing", 10), 7)

    slen     = cfg["snippet_len"]
    mmatched = cfg["max_matched"]
    mmissing = cfg["max_missing"]
    timeout  = int(os.getenv("OLLAMA_TIMEOUT", "600"))
    user_msg = build_user_prompt(resume, job_description)
    user_msg_b = user_msg + "\n\nRemember: respond with ONLY a JSON object starting with '{'. No prose."

    thinking_prefix = (
        "CRITICAL INSTRUCTION: Your entire response must be a single valid JSON object. "
        "Do NOT write any prose, markdown, headers, or explanations. "
        "Start your response with '{' and end with '}'. Nothing else.\n\n"
    )

    # ── Call A: score + reasoning + matched_skills ────────────────────────────
    sys_a = thinking_prefix + (
        f"You are an expert technical recruiter. Evaluate resume vs job description.\n\n"
        f"Return ONLY this JSON with at most {mmatched} matched skills:\n"
        f'{{"score": <1-5>, "reasoning": "<2 sentences>", '
        f'"matched_skills": [{{"skill": "...", "match_type": "exact|partial|inferred", '
        f'"jd_snippet": "<{slen} chars max>", "resume_snippet": "<{slen} chars max>"}}]}}'
    )
    num_predict_a = safe_num_predict(
        sys_a + user_msg_b, model_name=model, desired_output=2000
    )
    payload_a = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_a},
            {"role": "user",   "content": user_msg_b},
        ],
        "stream":  False,
        "format":  "json",
        "options": {"temperature": 0.2, "num_predict": num_predict_a, "think": False},
    }

    # ── Call B: missing_skills + suggestions ──────────────────────────────────
    sys_b = thinking_prefix + (
        f"You are an expert technical recruiter. Evaluate resume vs job description.\n\n"
        f"Return ONLY this JSON with at most {mmissing} missing skills and 3 suggestions:\n"
        f'{{"missing_skills": [{{"skill": "...", "severity": "blocker|major|minor", '
        f'"requirement_type": "hard|preferred|bonus", "jd_snippet": "<{slen} chars max>"}}], '
        f'"suggestions": [{{"title": "...", "detail": "..."}}]}}'
    )
    num_predict_b = safe_num_predict(
        sys_b + user_msg_b, model_name=model, desired_output=2000
    )
    payload_b = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_b},
            {"role": "user",   "content": user_msg_b},
        ],
        "stream":  False,
        "format":  "json",
        "options": {"temperature": 0.2, "num_predict": num_predict_b, "think": False},
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Run both calls sequentially — Ollama processes one at a time anyway
        resp_a = await client.post(f"{ollama_base_url()}/api/chat", json=payload_a)
        resp_a.raise_for_status()
        resp_a_json = resp_a.json()
        raw_a     = resp_a_json.get("message", {}).get("content", "")
        think_a   = resp_a_json.get("message", {}).get("thinking", "")
        raw_a     = _strip_thinking(raw_a)
        logger.info(f"→ thinking call A: {len(raw_a)} chars  thinking={len(think_a)} chars")
        _log_chunk(f"→ thinking call A body:\n{raw_a}")
        _log_chunk(f"→ thinking call A reasoning:\n{think_a}")

        resp_b = await client.post(f"{ollama_base_url()}/api/chat", json=payload_b)
        resp_b.raise_for_status()
        resp_b_json = resp_b.json()
        raw_b     = resp_b_json.get("message", {}).get("content", "")
        think_b   = resp_b_json.get("message", {}).get("thinking", "")
        raw_b     = _strip_thinking(raw_b)
        logger.info(f"→ thinking call B: {len(raw_b)} chars  thinking={len(think_b)} chars")
        _log_chunk(f"→ thinking call B body:\n{raw_b}")
        _log_chunk(f"→ thinking call B reasoning:\n{think_b}")

    # ── Parse and merge ───────────────────────────────────────────────────────
    import json as _json
    from analyzer.skills_helpers import parse_matched_skills, parse_missing_skills, parse_suggestions, keyword_boost
    from analyzer.penalties import compute_adjusted_score

    result_a: dict = {}
    result_b: dict = {}

    for raw, target in [(raw_a, "A"), (raw_b, "B")]:
        try:
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            parsed  = _json.loads(cleaned)
            if target == "A":
                result_a = parsed
            else:
                result_b = parsed
        except Exception:
            try:
                repaired = repair_truncated_json(raw)
                parsed   = _json.loads(repaired)
                if target == "A":
                    result_a = parsed
                else:
                    result_b = parsed
            except Exception as e:
                logger.warning(f"✗ thinking call {target} parse failed: {e}")

    score     = result_a.get("score")
    reasoning = result_a.get("reasoning", "")
    if score is None:
        raise ValueError("Thinking analysis failed: could not get score from model")

    try:
        score = int(float(str(score)))
        score = max(1, min(5, score))
    except Exception:
        raise ValueError(f"Thinking analysis failed: invalid score {score!r}")

    if not reasoning.strip():
        reasoning = "Analysis completed. Review matched and missing skills above."

    matched     = result_a.get("matched_skills", [])
    missing     = result_b.get("missing_skills", [])
    suggestions = result_b.get("suggestions", [])

    # ── Apply penalty pipeline (same as call_ollama_chunked) ──────────────────
    parsed_matched     = parse_matched_skills(matched)[:cfg["max_matched"]]
    parsed_missing     = parse_missing_skills(missing)[:cfg["max_missing"]]
    parsed_missing     = keyword_boost(parsed_missing, job_description)
    parsed_suggestions = parse_suggestions(suggestions) if suggestions else []

    adjusted_score, penalty_breakdown = compute_adjusted_score(score, parsed_missing)

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
        "analysis_mode":     cfg.get("mode", "detailed"),
        "retry_count":       0,
        "used_fallback":     False,
        "validation_errors": "",
    }

    corrections = auto_correct_llm_output(result)
    if corrections:
        logger.info(f"→ auto-corrected thinking result: {corrections}")

    return result


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
    # Use a larger budget for chunk 1 — thinking models consume tokens on
    # reasoning before producing JSON, so 350 is too tight.
    chunk1_predict = 2000
    raw1 = await _call_chunk(sys1, user_prompt, model, num_predict=chunk1_predict, chunk_name="1/score+reasoning")
    score, reasoning = _parse_score_chunk(raw1)

    if score is None:
        logger.error("✗ chunk 1 failed — cannot produce analysis without score")
        raise ValueError("Chunked analysis failed: could not get score from model")

    logger.info(f"→ chunk 1 OK: score={score}")

    # ── Chunk 2: matched_skills ───────────────────────────────────────────────
    # detailed mode requests up to 15 skills with full snippets — needs more tokens
    chunk2_predict = 1400 if actual_mode == "detailed" else 800
    sys2    = build_chunk2_prompt(cfg, actual_mode)
    raw2    = await _call_chunk(sys2, user_prompt, model, num_predict=chunk2_predict, chunk_name="2/matched")
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
    Entry point. provider = 'anthropic' | 'openai' | 'gemini' | 'ollama'
    Cloud providers (anthropic, openai, gemini) use single-shot calls.
    Ollama uses chunked calls.
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
            _log_chunk(
                f"→ attempt {attempt + 1}/{MAX_RETRIES} provider={provider} "
                f"resume={len(resume)} chars jd={len(job_description)} chars"
            )
            import analyzer.llm as _self
            if provider == "ollama":
                from analyzer.config import is_thinking_model
                current_model = ollama_model()
                if is_thinking_model(current_model):
                    logger.info(f"→ Routing thinking model {current_model!r} to thinking path")
                    result = await _self.call_ollama_thinking(resume, job_description)
                else:
                    result = await _self.call_ollama_chunked(resume, job_description)
            elif provider == "openai":
                result = await _self.call_openai_once(resume, job_description)
            elif provider == "gemini":
                result = await _self.call_gemini_once(resume, job_description)
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
    fallback["llm_model"]    = _get_model_for_provider(provider)
    return fallback


# ── Backward compat aliases ───────────────────────────────────────────────────
_call_anthropic_once   = call_anthropic_once
_call_ollama_once      = call_ollama_once
_call_ollama_chunked   = call_ollama_chunked
analyze_with_anthropic = call_anthropic_once
analyze_with_ollama    = call_ollama_chunked
analyze_with_openai    = call_openai_once
analyze_with_gemini    = call_gemini_once
