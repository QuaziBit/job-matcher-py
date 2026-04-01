"""
analyzer/known_models.py — Known models per cloud provider with cost indicators.

Each entry has:
  id    — model string sent to the API
  label — human-readable name shown in the UI dropdown

Update this file when providers release new models.
Ollama models are fetched dynamically from the local Ollama API.
"""

KNOWN_MODELS = {
    "anthropic": [
        {"id": "claude-haiku-4-5",  "label": "claude-haiku-4-5  · cheapest"},
        {"id": "claude-sonnet-4-6", "label": "claude-sonnet-4-6 · balanced"},
        {"id": "claude-opus-4-6",   "label": "claude-opus-4-6   · best · expensive"},
    ],
    "openai": [
        {"id": "gpt-4o-mini",  "label": "gpt-4o-mini  · cheapest"},
        {"id": "gpt-4o",       "label": "gpt-4o       · balanced"},
        {"id": "gpt-4-turbo",  "label": "gpt-4-turbo  · powerful"},
        {"id": "o1-mini",      "label": "o1-mini      · reasoning · cheap"},
        {"id": "o1",           "label": "o1           · reasoning · expensive"},
    ],
    "gemini": [
        {"id": "gemini-2.5-flash",      "label": "gemini-2.5-flash      · cheapest"},
        {"id": "gemini-2.5-flash-lite", "label": "gemini-2.5-flash-lite · cheapest · fastest"},
        {"id": "gemini-2.5-pro",        "label": "gemini-2.5-pro        · best · expensive"},
        {"id": "gemini-2.0-flash",      "label": "gemini-2.0-flash      · fast · cheap"},
    ],
}
