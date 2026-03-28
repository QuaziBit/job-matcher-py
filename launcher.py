"""
launcher.py — Browser-based launcher UI for Job Matcher.

Starts a small HTTP server on a random port, opens the browser to a
config/health-check page, then signals the main app when the user clicks Start.
Also handles Stop and Restart via /api/stop and /api/restart.

Usage:
    from launcher import Launcher
    launcher = Launcher()
    launcher.start()          # opens browser, blocks until user clicks Start
    cfg = launcher.get_config()
    # ... start main app ...
    launcher.stop()
"""

import json
import logging
import os
import platform
import socket
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("launcher")

# ── Health checks ─────────────────────────────────────────────────────────────

def check_sqlite(db_path: str) -> dict:
    """Check if the SQLite DB exists and has the expected tables."""
    try:
        import sqlite3
        if not os.path.exists(db_path):
            return {"status": "warn", "message": f"Will be created at: {db_path}"}
        conn = sqlite3.connect(db_path)
        cur = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN "
            "('jobs','resumes','analyses','applications')"
        )
        count = cur.fetchone()[0]
        conn.close()
        size_kb = os.path.getsize(db_path) / 1024
        if count == 4:
            return {"status": "ok", "message": f"{db_path} ({size_kb:.0f} KB, 4 tables)"}
        return {"status": "warn", "message": f"{db_path} ({count}/4 tables — will init on start)"}
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}"}


def check_ollama(base_url: str) -> tuple[dict, list]:
    """Ping Ollama and return available models."""
    try:
        req = urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        data = json.loads(req.read())
        models = [m["name"] for m in data.get("models", [])]
        msg = f"Running — {len(models)} model(s) available"
        return {"status": "ok", "message": msg}, models
    except Exception:
        return {"status": "warn", "message": "Not reachable — run: ollama serve"}, []


def check_anthropic(api_key: str) -> dict:
    """Validate Anthropic API key format."""
    if not api_key or api_key == "sk-ant-...":
        return {"status": "warn", "message": "No key set — Anthropic provider unavailable"}
    if not api_key.startswith("sk-ant-"):
        return {"status": "error", "message": "Key format invalid (must start with sk-ant-)"}
    masked = api_key[:12] + "..." + api_key[-4:]
    return {"status": "ok", "message": f"Key present ({masked})"}


def run_health_checks(db_path: str, ollama_url: str, api_key: str) -> dict:
    """Run all health checks and return a combined report."""
    ollama_result, models = check_ollama(ollama_url)
    return {
        "sqlite":    check_sqlite(db_path),
        "ollama":    ollama_result,
        "anthropic": check_anthropic(api_key),
        "models":    models,
    }


# ── Launcher page HTML ────────────────────────────────────────────────────────

_LAUNCHER_UI_DIR = os.path.join(os.path.dirname(__file__), "launcher_ui")

_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css",
    ".js":   "application/javascript",
}

def render_launcher_page(cfg: dict) -> str:
    """Read launcher.html and substitute config placeholders."""
    port          = cfg.get("port", 8000)
    host          = cfg.get("host", "127.0.0.1")
    db_path       = cfg.get("db_path", "job_matcher.db")
    api_key       = cfg.get("anthropic_api_key", "")
    ollama_url    = cfg.get("ollama_base_url", "http://localhost:11434")
    ollama_model  = cfg.get("ollama_model", "llama3.1:8b")
    ollama_timeout = cfg.get("ollama_timeout", 600)
    analysis_mode = cfg.get("analysis_mode", "standard")

    html_path = os.path.join(_LAUNCHER_UI_DIR, "launcher.html")
    with open(html_path, encoding="utf-8") as f:
        html = f.read()

    return html.format(
        port           = port,
        host           = host,
        db_path        = db_path,
        api_key        = api_key,
        ollama_url     = ollama_url,
        ollama_model   = ollama_model,
        ollama_timeout = ollama_timeout,
        checked_fast     = "checked" if analysis_mode == "fast"     else "",
        checked_standard = "checked" if analysis_mode == "standard" else "",
        checked_detailed = "checked" if analysis_mode == "detailed" else "",
    )
# ── HTTP request handler ──────────────────────────────────────────────────────

class LauncherHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for the launcher server."""

    launcher: "Launcher" = None  # set by Launcher.start()

    def log_message(self, format, *args):
        # Route to Python logger instead of stderr
        logger.debug("launcher http: " + format % args)

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_form(self) -> dict:
        """Read and parse multipart or urlencoded POST body."""
        from urllib.parse import parse_qs
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")

        if "multipart/form-data" in content_type:
            # Parse multipart manually via cgi module
            import cgi, io
            environ = {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE":   content_type,
                "CONTENT_LENGTH": str(length),
            }
            form = cgi.FieldStorage(
                fp=io.BytesIO(body),
                environ=environ,
                keep_blank_values=True,
            )
            return {k: form.getvalue(k, "") for k in form.keys()}
        else:
            parsed = parse_qs(body.decode(), keep_blank_values=True)
            return {k: v[0] for k, v in parsed.items()}

    def _serve_static(self, file_path: str):
        """Serve a static file from launcher_ui/."""
        import mimetypes
        ext = os.path.splitext(file_path)[1]
        mime = _MIME_TYPES.get(ext, "application/octet-stream")
        try:
            with open(file_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        query  = parse_qs(parsed.query, keep_blank_values=True)

        if path.startswith("/launcher_ui/"):
            filename  = os.path.basename(path)
            file_path = os.path.join(_LAUNCHER_UI_DIR, filename)
            self._serve_static(file_path)
            return

        if path == "/":
            cfg = self.launcher.get_config()
            logger.info("→ Launcher GET /")
            self.send_html(render_launcher_page(cfg))

        elif path == "/health":
            db_path    = query.get("db_path",    [""])[0] or self.launcher.cfg.get("db_path", "job_matcher.db")
            ollama_url = query.get("ollama_url",  [""])[0] or self.launcher.cfg.get("ollama_base_url", "http://localhost:11434")
            api_key    = query.get("api_key",     [""])[0] or self.launcher.cfg.get("anthropic_api_key", "")
            logger.info(f"→ Launcher health: db={db_path} ollama={ollama_url} key_set={bool(api_key)}")
            report = run_health_checks(db_path, ollama_url, api_key)
            logger.info(f"✓ Health: sqlite={report['sqlite']['status']} ollama={report['ollama']['status']} anthropic={report['anthropic']['status']} models={len(report['models'])}")
            self.send_json(report)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/launcher/start":
            self._handle_start()
        elif path == "/api/launcher/stop":
            self._handle_stop()
        elif path == "/api/launcher/restart":
            self._handle_restart()
        else:
            self.send_response(404)
            self.end_headers()

    def _parse_config_from_form(self, form: dict) -> dict:
        """Build config dict from form values, with validation and logging."""
        cfg = dict(self.launcher.cfg)  # start from current

        if v := form.get("port", ""):
            try:
                p = int(v)
                if 1024 <= p <= 65535:
                    cfg["port"] = p
                else:
                    logger.warning(f"✗ Invalid port {v!r}")
            except ValueError:
                logger.warning(f"✗ Non-integer port {v!r}")

        if v := form.get("host", ""):            cfg["host"] = v
        if v := form.get("db_path", ""):         cfg["db_path"] = v
        if v := form.get("anthropic_api_key",""):cfg["anthropic_api_key"] = v
        if v := form.get("ollama_base_url", ""):  cfg["ollama_base_url"] = v
        if v := form.get("ollama_model", ""):    cfg["ollama_model"] = v
        if v := form.get("ollama_timeout", ""):
            try:
                t = int(v)
                if t > 0:
                    cfg["ollama_timeout"] = t
            except ValueError:
                logger.warning(f"✗ Invalid timeout {v!r}")
        if v := form.get("analysis_mode", ""):
            if v in ("fast", "standard", "detailed"):
                cfg["analysis_mode"] = v
            else:
                logger.warning(f"✗ Invalid analysis_mode {v!r}")

        key = cfg.get("anthropic_api_key", "")
        masked = (key[:12] + "...") if key else "not set"
        logger.info(f"✓ Config: port={cfg.get('port')} model={cfg.get('ollama_model')} key={masked} db={cfg.get('db_path')}")
        return cfg

    def _handle_start(self):
        logger.info("→ Launcher POST /api/launcher/start")
        form = self.read_form()
        cfg  = self._parse_config_from_form(form)
        self.launcher.update_config(cfg)
        url  = f"http://{cfg['host']}:{cfg['port']}"
        self.send_json({"ok": True, "url": url})
        threading.Thread(target=self.launcher._signal_start, daemon=True).start()

    def _handle_stop(self):
        logger.info("→ Launcher POST /api/launcher/stop")
        self.send_json({"ok": True})
        threading.Thread(target=self.launcher._signal_stop, daemon=True).start()

    def _handle_restart(self):
        logger.info("→ Launcher POST /api/launcher/restart")
        form = self.read_form()
        cfg  = self._parse_config_from_form(form)
        self.launcher.update_config(cfg)
        url  = f"http://{cfg['host']}:{cfg['port']}"
        self.send_json({"ok": True, "url": url})
        threading.Thread(target=self.launcher._signal_restart, daemon=True).start()


# ── Launcher class ────────────────────────────────────────────────────────────

class Launcher:
    """
    Manages the launcher HTTP server and communicates with the main app
    via threading.Event and callback functions.
    """

    def __init__(self, cfg: dict):
        self.cfg   = dict(cfg)
        self._lock = threading.Lock()
        self._start_event   = threading.Event()
        self._stop_event    = threading.Event()
        self._restart_event = threading.Event()
        self._server: Optional[HTTPServer] = None
        self._port: int = 0

        # Callbacks set by main app
        self.on_start:   Optional[callable] = None
        self.on_stop:    Optional[callable] = None
        self.on_restart: Optional[callable] = None

    def get_config(self) -> dict:
        with self._lock:
            return dict(self.cfg)

    def update_config(self, cfg: dict):
        with self._lock:
            self.cfg = dict(cfg)
        # Persist to .env
        self._save_to_env(cfg)

    def _save_to_env(self, cfg: dict):
        """Update .env file with new config values."""
        env_path = ".env"
        try:
            # Read existing lines
            lines = []
            if os.path.exists(env_path):
                with open(env_path) as f:
                    lines = f.readlines()

            updates = {
                "ANTHROPIC_API_KEY": cfg.get("anthropic_api_key", ""),
                "OLLAMA_BASE_URL":   cfg.get("ollama_base_url", "http://localhost:11434"),
                "OLLAMA_MODEL":      cfg.get("ollama_model", "llama3.1:8b"),
                "OLLAMA_TIMEOUT":    str(cfg.get("ollama_timeout", 600)),
                "APP_PORT":          str(cfg.get("port", 8000)),
                "APP_HOST":          cfg.get("host", "127.0.0.1"),
                "ANALYSIS_MODE":     cfg.get("analysis_mode", "standard"),
            }

            # Update existing lines
            updated_keys = set()
            new_lines = []
            for line in lines:
                key = line.split("=")[0].strip()
                if key in updates:
                    new_lines.append(f"{key}={updates[key]}\n")
                    updated_keys.add(key)
                else:
                    new_lines.append(line)

            # Append any keys not already in the file
            for key, val in updates.items():
                if key not in updated_keys:
                    new_lines.append(f"{key}={val}\n")

            with open(env_path, "w") as f:
                f.writelines(new_lines)

            logger.info(f"✓ .env updated: {env_path}")
        except Exception as e:
            logger.warning(f"✗ Could not update .env: {e}")

    def start(self) -> str:
        """Start launcher server on a random free port. Returns the URL."""
        # Find a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self._port = s.getsockname()[1]

        # Attach self to handler class
        handler_class = type("BoundHandler", (LauncherHandler,), {"launcher": self})

        self._server = HTTPServer(("127.0.0.1", self._port), handler_class)
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()

        url = f"http://127.0.0.1:{self._port}"
        logger.info(f"→ Launcher listening on port {self._port}")
        return url

    def stop(self):
        """Stop the launcher HTTP server."""
        if self._server:
            logger.info("→ Stopping launcher server")
            self._server.shutdown()
            logger.info("✓ Launcher stopped")

    def _signal_start(self):
        logger.info("→ Launcher signaling start")
        if self.on_start:
            self.on_start(self.get_config())

    def _signal_stop(self):
        logger.info("→ Launcher signaling stop")
        if self.on_stop:
            self.on_stop()

    def _signal_restart(self):
        logger.info("→ Launcher signaling restart")
        if self.on_restart:
            self.on_restart(self.get_config())


# ── Browser opener ────────────────────────────────────────────────────────────

def open_browser(url: str):
    """Open a URL in the default system browser."""
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(url)
        elif system == "Darwin":
            subprocess.Popen(["open", url])
        else:
            subprocess.Popen(["xdg-open", url])
        logger.info(f"→ Opened browser: {url}")
    except Exception as e:
        logger.warning(f"✗ Could not open browser: {e} — open manually: {url}")
