"""
analyzer/parsers.py — LLM response parsing, JSON repair and sanitization.
"""

import json
import logging
import re

from analyzer.config import get_mode_config
from analyzer.penalties import compute_adjusted_score
from analyzer.skills_helpers import (
    parse_matched_skills,
    parse_missing_skills,
    parse_suggestions,
    keyword_boost,
)

logger = logging.getLogger("analyzer.parsers")


# ── JSON repair and sanitization ─────────────────────────────────────────────

def sanitize_json(raw: str) -> str:
    """
    Fix unescaped double quotes and control characters inside JSON string
    values produced by LLMs. Uses a JSON-aware state machine.
    """
    # Normalize curly/smart quotes first
    raw = raw.replace("\u201c", '\\"').replace("\u201d", '\\"')
    raw = raw.replace("\u2018", "'").replace("\u2019", "'")



    out = []
    i = 0
    n = len(raw)

    while i < n:
        c = raw[i]

        if c == '"':
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

            j = i
            while j < n and raw[j] in ' \t\r\n':
                j += 1

            if j < n and raw[j] == ':':
                out.append(raw[i:j + 1])
                i = j + 1
                while i < n and raw[i] in ' \t\r\n':
                    out.append(raw[i])
                    i += 1

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
                            k = i + 1
                            while k < n and raw[k] in ' \t\r\n':
                                k += 1
                            next_ch = raw[k] if k < n else ''
                            if next_ch in (',', '}', ']', '', '"'):
                                break
                            else:
                                value_chars.append('\\"')
                                i += 1
                                continue
                        value_chars.append(c)
                        i += 1
                    out.extend(value_chars)
                    out.append('"')
                    i += 1
                    continue
        else:
            out.append(c)
            i += 1

    return ''.join(out)


def repair_truncated_json(raw: str) -> str:
    """
    Attempt to close a truncated JSON object produced by a model that hit
    its token limit mid-response.
    """
    open_braces   = raw.count('{') - raw.count('}')
    open_brackets = raw.count('[') - raw.count(']')

    if open_braces <= 0 and open_brackets <= 0:
        return raw

    stripped = raw.rstrip()
    while stripped and stripped[-1] in (',', ':', '{', '['):
        stripped = stripped[:-1].rstrip()

    closing  = (']' * open_brackets) + ('}' * open_braces)
    repaired = stripped + closing
    logger.info(f"→ repaired truncated JSON: appended {repr(closing)}")
    return repaired


# ── Response parser ───────────────────────────────────────────────────────────

def _escape_control_chars(raw: str) -> str:
    """
    Escape literal control characters inside JSON string values.
    LLMs often copy raw tabs and newlines from resume/JD text into
    snippet fields, which makes the JSON invalid.
    Strategy: use a simple state machine to only replace control chars
    that appear inside quoted string values.
    """
    out   = []
    in_str = False
    i = 0
    n = len(raw)
    while i < n:
        c = raw[i]
        if in_str:
            if c == '\\':
                out.append(c)
                i += 1
                if i < n:
                    out.append(raw[i])
                    i += 1
                continue
            if c == '"':
                in_str = False
                out.append(c)
            elif c == '\t':
                out.append('\\t')
            elif c == '\r':
                out.append('\\r')
            elif c == '\n':
                out.append('\\n')
            else:
                out.append(c)
        else:
            if c == '"':
                in_str = True
            out.append(c)
        i += 1
    return ''.join(out)


def parse_response(raw: str, job_description: str = "", cfg: dict | None = None) -> dict:
    """Extract and validate JSON from LLM response, apply full penalty pipeline."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Strategy: try the full greedy match first (normal case).
    # If all parse passes fail, split on top-level object boundaries and
    # try each chunk — handles models that concatenate two JSON objects
    # (e.g. llama3.2:3b outputting a failed attempt followed by a second one).
    greedy = re.search(r"\{.*\}", raw, re.DOTALL)
    if not greedy:
        raise ValueError("No JSON object found in LLM response")

    # Five-pass parse loop per candidate
    parse_attempts = [
        lambda s: json.loads(s),
        lambda s: json.loads(_escape_control_chars(s)),
        lambda s: json.loads(repair_truncated_json(s)),
        lambda s: json.loads(sanitize_json(s)),
        lambda s: json.loads(sanitize_json(repair_truncated_json(s))),
    ]

    def _try_parse(json_str: str) -> dict | None:
        """Try all parse passes on a single candidate string."""
        for attempt_fn in parse_attempts:
            try:
                result = attempt_fn(json_str)
                if isinstance(result, dict) and "score" in result:
                    return result
            except json.JSONDecodeError:
                continue
        return None

    # Pass 1: full greedy match (covers 99% of cases)
    json_str = greedy.group()
    data = _try_parse(json_str)

    # Pass 2: if full match failed, try splitting on top-level object boundaries.
    # Find positions of top-level '{' by tracking brace depth.
    if data is None:
        top_level_starts = []
        depth = 0
        for idx, ch in enumerate(json_str):
            if ch == '{':
                if depth == 0:
                    top_level_starts.append(idx)
                depth += 1
            elif ch == '}':
                depth -= 1

        if len(top_level_starts) > 1:
            logger.info(f"→ greedy parse failed, trying {len(top_level_starts)} top-level objects")
            for start in top_level_starts:
                chunk = json_str[start:]
                # Find the end of this top-level object
                d = 0
                for i, ch in enumerate(chunk):
                    if ch == '{': d += 1
                    elif ch == '}':
                        d -= 1
                        if d == 0:
                            chunk = chunk[:i+1]
                            break
                candidate_data = _try_parse(chunk)
                if candidate_data is not None:
                    data = candidate_data
                    break

    if data is None:
        logger.error(
            f"✗ failed to parse LLM JSON after {len(parse_attempts)} passes. "
            f"Raw response (first 1000 chars): {json_str[:1000]!r}"
        )
        raise ValueError(f"failed to parse LLM JSON after {len(parse_attempts)} attempts")

    # Score: accept float and round to nearest int (some models return 4.5 etc.)
    raw_score = data.get("score", 0)
    try:
        score = round(float(raw_score))
    except (TypeError, ValueError):
        score = 0
    if not 1 <= score <= 5:
        raise ValueError(f"Score out of range: {raw_score}")

    if cfg is None:
        cfg = get_mode_config()

    matched = parse_matched_skills(data.get("matched_skills", []))
    missing = parse_missing_skills(data.get("missing_skills", []))
    matched = matched[:cfg["max_matched"]]
    missing = missing[:cfg["max_missing"]]

    suggestions = []
    if cfg["suggestions"]:
        suggestions = parse_suggestions(data.get("suggestions", []))

    if job_description:
        missing = keyword_boost(missing, job_description)

    adjusted_score, penalty_breakdown = compute_adjusted_score(score, missing)

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


# ── Backward compat aliases ───────────────────────────────────────────────────
_sanitize_json         = sanitize_json
_repair_truncated_json = repair_truncated_json
_parse_response        = parse_response
