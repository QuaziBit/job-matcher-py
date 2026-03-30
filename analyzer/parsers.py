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
    Fix unescaped double quotes inside JSON string values produced by LLMs.
    Uses a JSON-aware state machine to identify and escape inner quotes.
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

def parse_response(raw: str, job_description: str = "", cfg: dict | None = None) -> dict:
    """Extract and validate JSON from LLM response, apply full penalty pipeline."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    json_str = match.group()

    # Four-pass parse loop
    data = None
    for attempt_fn in [
        lambda s: json.loads(s),
        lambda s: json.loads(repair_truncated_json(s)),
        lambda s: json.loads(sanitize_json(s)),
        lambda s: json.loads(sanitize_json(repair_truncated_json(s))),
    ]:
        try:
            data = attempt_fn(json_str)
            break
        except json.JSONDecodeError:
            continue

    if data is None:
        raise ValueError(f"failed to parse LLM JSON after 4 attempts")

    score = int(data.get("score", 0))
    if not 1 <= score <= 5:
        raise ValueError(f"Score out of range: {score}")

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
