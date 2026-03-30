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
        "suggestions": True,
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


def get_mode_config() -> dict:
    """Read ANALYSIS_MODE from env and return the corresponding config dict."""
    mode = os.getenv("ANALYSIS_MODE", "standard").lower().strip()
    return MODE_CONFIG.get(mode, MODE_CONFIG["standard"])


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b")
