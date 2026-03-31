# ollama_utils.py
import logging
import math
import os
import httpx

logger = logging.getLogger("ollama_utils")

# Known context windows as a cache seed — extended with common models.
# safe_num_predict() will query Ollama for any model not listed here
# and cache the result for the session.
MODEL_CONTEXT_WINDOWS = {
    # Llama 3 family
    "llama3.1:8b":        8192,
    "llama3.1:70b":      32768,
    "llama3.1:405b":    131072,
    "llama3.2:1b":       32768,
    "llama3.2:3b":       32768,
    "llama3.3:70b":     131072,
    # Gemma 3 family
    "gemma3:1b":         32768,
    "gemma3:4b":         32768,
    "gemma3:12b":        32768,
    "gemma3:27b":        32768,
    # Mistral family
    "mistral:7b":         8192,
    "mistral:latest":     8192,
    "mixtral:8x7b":      32768,
    # DeepSeek
    "deepseek-r1:7b":    32768,
    "deepseek-r1:14b":   32768,
    "deepseek-r1:32b":   32768,
    "deepseek-r1:70b":   32768,
    # Phi
    "phi3.5:3.8b":       32768,
    "phi4:14b":          16384,
    # Nemotron
    "nemotron-3-nano:latest": 16384,
    # Qwen
    "qwen2.5:7b":        32768,
    "qwen2.5:14b":       32768,
    "qwen2.5:32b":       32768,
    "qwen2.5:72b":       32768,
    # CodeLlama
    "codellama:7b":      16384,
    "codellama:13b":     16384,
    "codellama:34b":     16384,
}

# Session cache for dynamically queried models
_CONTEXT_CACHE: dict[str, int] = {}

# Conservative fallback if Ollama query also fails
_FALLBACK_CONTEXT = 4096


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _query_model_context(model_name: str) -> int | None:
    """
    Query Ollama API for a model's context window size.
    Returns None if the query fails or the field is missing.
    """
    try:
        resp = httpx.post(
            f"{_ollama_base_url()}/api/show",
            json={"name": model_name},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns model_info.llama.context_length or similar
        model_info = data.get("model_info", {})
        for key in model_info:
            if "context_length" in key:
                val = model_info[key]
                if isinstance(val, int) and val > 0:
                    return val

        # Fallback: check modelfile parameters
        params = data.get("parameters", "")
        for line in params.splitlines():
            if line.lower().startswith("num_ctx"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])

        return None
    except httpx.ConnectError:
        logger.warning(f"⚠ cannot connect to Ollama at {_ollama_base_url()} — is it running?")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"⚠ Ollama API error querying '{model_name}': {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"⚠ unexpected error querying context window for '{model_name}': {e}")
        return None


def get_context_window(model_name: str) -> int:
    """
    Return the context window size for a model.
    Priority: static table → session cache → Ollama API query → fallback.
    """
    # 1. Check static table
    if model_name in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model_name]

    # 2. Check session cache (already queried this session)
    if model_name in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[model_name]

    # 3. Query Ollama API
    queried = _query_model_context(model_name)
    if queried:
        _CONTEXT_CACHE[model_name] = queried
        logger.info(f"✓ queried context window for '{model_name}': {queried}")
        return queried

    # 4. Conservative fallback
    logger.warning(f"⚠ unknown model '{model_name}' — using fallback context {_FALLBACK_CONTEXT}")
    _CONTEXT_CACHE[model_name] = _FALLBACK_CONTEXT
    return _FALLBACK_CONTEXT


def estimate_tokens(text: str) -> int:
    """Rough token estimator: 1 token ≈ 3.5 characters (more accurate than 4)."""
    return math.ceil(len(text) / 3.5)


def safe_num_predict(prompt: str, model_name: str, desired_output: int = 4096) -> int:
    """
    Calculate a safe num_predict for Ollama.

    Parameters:
    - prompt:          combined system + user messages as a single string
    - model_name:      the model being used
    - desired_output:  desired number of output tokens (from MODE_CONFIG)

    Returns:
    - num_predict: safe value that won't overflow the model's context window
    """
    context_window = get_context_window(model_name)
    prompt_tokens = estimate_tokens(prompt)

    # Add a 20% safety buffer on top of the estimated prompt tokens to
    # account for tokenizer differences between models — the 1 token ≈ 4 chars
    # estimator consistently underestimates, especially for code and JSON.
    # Increased from 15% after observing consistent underestimation in practice.
    buffered_prompt_tokens = int(prompt_tokens * 1.20)
    logger.info(
        f"→ prompt chars={len(prompt)} estimated_tokens={prompt_tokens} "
        f"buffered={buffered_prompt_tokens} context={context_window}"
    )
    max_safe_output = context_window - buffered_prompt_tokens

    if max_safe_output <= 0:
        logger.warning(
            f"⚠ prompt ({prompt_tokens} tokens, buffered={buffered_prompt_tokens}) "
            f"exceeds context window ({context_window}) for '{model_name}'"
        )
        return min(desired_output, context_window // 4)  # emergency fallback

    # Reserve at least 512 tokens as breathing room for model overhead
    MIN_HEADROOM = 512
    result = min(desired_output, max_safe_output - MIN_HEADROOM)
    result = max(result, 256)  # never go below 256 — useless otherwise
    logger.info(
        f"→ num_predict={result} "
        f"(context={context_window} prompt~{buffered_prompt_tokens} "
        f"headroom={MIN_HEADROOM} desired={desired_output})"
    )
    return result
