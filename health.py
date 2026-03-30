"""
health.py — Startup health checks for Job Matcher.
Extracted from main.py to keep route handlers separate from infra concerns.
"""

import json
import os
import urllib.request


def check_sqlite(db_path: str) -> tuple[bool, str]:
    """Verify SQLite DB is accessible and all 4 tables exist."""
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur  = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN "
            "('jobs','resumes','analyses','applications')"
        )
        count = cur.fetchone()[0]
        conn.close()
        if count == 4:
            size_kb = os.path.getsize(db_path) / 1024 if os.path.exists(db_path) else 0
            return True, f"{db_path} ({size_kb:.0f} KB, 4 tables)"
        return False, f"{db_path} (only {count}/4 tables found — run the app once to init)"
    except Exception as e:
        return False, f"Error: {e}"


def check_ollama(base_url: str, model: str) -> tuple[bool, str]:
    """Ping Ollama and verify the configured model is available."""
    try:
        req  = urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        data = json.loads(req.read())
        available  = [m["name"] for m in data.get("models", [])]
        model_base = model.split(":")[0]
        matched    = [m for m in available if m.startswith(model_base)]
        if matched:
            return True, f"{model} ready ({len(available)} model(s) installed)"
        return False, (
            f"Ollama running but '{model}' not found. "
            f"Run: ollama pull {model}"
        )
    except Exception:
        return False, "Ollama not reachable — run: ollama serve"


def check_anthropic(api_key: str) -> tuple[bool, str]:
    """Check if the Anthropic API key looks valid (format check only, no API call)."""
    if not api_key or api_key == "sk-ant-...":
        return False, "No API key set — Anthropic provider unavailable"
    if not api_key.startswith("sk-ant-"):
        return False, "Key format looks wrong (should start with sk-ant-)"
    masked = api_key[:12] + "..." + api_key[-4:]
    return True, f"Key present ({masked})"


def run_health_checks() -> bool:
    """
    Run all startup checks and print a formatted report.
    Returns True if critical checks pass (DB accessible).
    """
    from analyzer import _ollama_base_url, _ollama_model
    from database import _db_path

    db_path      = _db_path()
    ollama_url   = _ollama_base_url()
    ollama_model = _ollama_model()
    api_key      = os.getenv("ANTHROPIC_API_KEY", "")

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    def status_line(label, ok, detail, warn_only=False):
        if ok:
            icon = f"{GREEN}✓{RESET}"
        elif warn_only:
            icon = f"{YELLOW}⚠{RESET}"
        else:
            icon = f"{RED}✗{RESET}"
        print(f"  {icon}  {BOLD}{label:<18}{RESET} {detail}")

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))

    print(f"\n{CYAN}{BOLD}{'═' * 54}{RESET}")
    print(f"{CYAN}{BOLD}   Job Matcher{RESET}")
    print(f"{CYAN}{'═' * 54}{RESET}\n")

    db_ok,     db_msg     = check_sqlite(db_path)
    ollama_ok, ollama_msg = check_ollama(ollama_url, ollama_model)
    ant_ok,    ant_msg    = check_anthropic(api_key)

    status_line("SQLite",        db_ok,     db_msg)
    status_line("Ollama",        ollama_ok, ollama_msg, warn_only=True)
    status_line("Anthropic API", ant_ok,    ant_msg,    warn_only=True)

    llm_ok = ollama_ok or ant_ok
    if not llm_ok:
        print(f"\n  {RED}No LLM provider available. Configure Ollama or add ANTHROPIC_API_KEY.{RESET}")

    print(f"\n  {'─' * 50}")
    print(f"  {BOLD}URL{RESET}    http://{host}:{port}")
    print(f"  {BOLD}Model{RESET}  {ollama_model}")
    print(f"  {'─' * 50}\n")

    return db_ok  # Only DB is truly critical


# ── Backward compat aliases ───────────────────────────────────────────────────
_check_sqlite    = check_sqlite
_check_ollama    = check_ollama
_check_anthropic = check_anthropic
_run_health_checks = run_health_checks
