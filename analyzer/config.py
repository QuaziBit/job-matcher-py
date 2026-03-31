"""
analyzer/config.py — Analysis mode configuration and environment helpers.
"""

import logging
import os

logger = logging.getLogger("analyzer.config")

ANTHROPIC_MODEL = "claude-opus-4-5"
MAX_RETRIES     = 3

# ── Analysis mode configuration ───────────────────────────────────────────────

MODE_CONFIG = {
    "fast": {
        "snippet_len":  40,
        "max_matched":   5,
        "max_missing":   4,
        "suggestions": False,
        "num_predict": 800,
    },
    "standard": {
        "snippet_len":  70,
        "max_matched":   8,
        "max_missing":   6,
        "suggestions": False,   # disabled for smaller models — too complex
        "num_predict": 1800,
    },
    "detailed": {
        "snippet_len":  100,
        "max_matched":  15,
        "max_missing":  10,
        "suggestions": True,
        "num_predict": 4096,
    },
}

# Estimated response time in seconds per mode (used by frontend progress bar)
MODE_ESTIMATES = {
    "fast":     30,
    "standard": 90,
    "detailed": 240,
}

# Hard-blocker keyword patterns for the keyword-based detector
BLOCKER_KEYWORDS = [
    "clearance", "ts/sci", "top secret", "secret", "polygraph",
    "citizenship", "citizen only", "usc only",
]

# ── Model capability map ──────────────────────────────────────────────────────
#
# Defines the maximum analysis mode each local model can reliably handle.
# Models not listed here default to "standard".
#
# Tiers:
#   fast     — model struggles with standard JSON schema (unreliable)
#   standard — model handles basic schema but not suggestions or large counts
#   detailed — model handles full schema reliably
#
MODEL_MAX_MODE = {
    # Llama 3.1 — reliable, handles detailed
    "llama3.1:8b":   "detailed",
    "llama3.1:70b":  "detailed",
    "llama3.1:405b": "detailed",
    # Llama 3.2 — smaller, standard is safe
    "llama3.2:1b":   "fast",
    "llama3.2:3b":   "standard",
    # Llama 3.3
    "llama3.3:70b":  "detailed",
    # Gemma 3 — 27b handles detailed, smaller ones standard
    "gemma3:1b":     "fast",
    "gemma3:4b":     "standard",
    "gemma3:12b":    "standard",
    "gemma3:27b":    "detailed",
    # Mistral — standard is safe
    "mistral:7b":    "standard",
    "mistral:latest": "standard",
    "mixtral:8x7b":  "detailed",
    # Phi — allow all modes, behaviour observed via logs
    "phi3.5:3.8b":   "detailed",
    "phi4:14b":      "standard",
    # DeepSeek
    "deepseek-r1:7b":  "standard",
    "deepseek-r1:14b": "detailed",
    "deepseek-r1:32b": "detailed",
    "deepseek-r1:70b": "detailed",
    # Nemotron
    "nemotron-3-nano:latest": "fast",
    # Qwen
    "qwen2.5:7b":   "standard",
    "qwen2.5:14b":  "detailed",
    "qwen2.5:32b":  "detailed",
    "qwen2.5:72b":  "detailed",
}

# Mode ordering for comparisons
_MODE_ORDER = {"fast": 0, "standard": 1, "detailed": 2}


def get_model_max_mode(model_name: str) -> str:
    """
    Return the maximum analysis mode a model can reliably handle.
    Match strategy:
      1. Exact match (e.g. "llama3.1:8b" == "llama3.1:8b")
      2. Prefix match on the base name before ":" 
         (e.g. "llama3.1:8b-instruct" matches "llama3.1:8b")
    Defaults to 'standard' for unknown models.
    """
    # Exact match first
    if model_name in MODEL_MAX_MODE:
        return MODEL_MAX_MODE[model_name]

    # Prefix match: check if model_name starts with known_model (handles tag variants)
    # Sort by length descending so more specific entries match first
    for known_model in sorted(MODEL_MAX_MODE, key=len, reverse=True):
        if model_name.startswith(known_model.split(":")[0] + ":") or model_name == known_model.split(":")[0]:
            return MODEL_MAX_MODE[known_model]

    return "standard"


def cap_mode_for_model(requested_mode: str, model_name: str) -> str:
    """
    Return the effective mode after capping to the model's capability.
    If the requested mode exceeds what the model can handle, returns the
    model's max mode instead.
    """
    max_mode = get_model_max_mode(model_name)
    if _MODE_ORDER.get(requested_mode, 1) > _MODE_ORDER.get(max_mode, 1):
        return max_mode
    return requested_mode


def get_mode_config() -> dict:
    """Read ANALYSIS_MODE from env and return the corresponding config dict."""
    mode = os.getenv("ANALYSIS_MODE", "standard").lower().strip()
    return MODE_CONFIG.get(mode, MODE_CONFIG["standard"])


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b")
