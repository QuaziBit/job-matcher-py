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

def render_launcher_page(cfg: dict) -> str:
    port          = cfg.get("port", 8000)
    host          = cfg.get("host", "127.0.0.1")
    db_path       = cfg.get("db_path", "job_matcher.db")
    api_key       = cfg.get("anthropic_api_key", "")
    ollama_url    = cfg.get("ollama_base_url", "http://localhost:11434")
    ollama_model  = cfg.get("ollama_model", "llama3.1:8b")
    ollama_timeout = cfg.get("ollama_timeout", 600)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Job Matcher — Launcher</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root {{
  --bg:#0d0e11; --bg2:#13151a; --bg3:#1c1f27;
  --border:#2a2d38; --text:#d4d8e8; --text-dim:#6b7080; --text-mute:#3d4050;
  --amber:#f59e0b; --green:#22c55e; --red:#ef4444; --yellow:#eab308;
  --radius:4px; --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',sans-serif;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);
  min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px;}}
.card{{background:var(--bg2);border:1px solid var(--border);border-radius:6px;
  width:100%;max-width:540px;overflow:hidden;}}
.card-header{{padding:20px 24px 16px;border-bottom:1px solid var(--border);}}
.card-header h1{{font-size:18px;font-weight:600;}}
.logo-mark{{font-family:var(--mono);font-size:10px;color:var(--amber);
  letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;}}
.card-body{{padding:24px;}}
.section{{margin-bottom:24px;}}
.section-title{{font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:0.1em;color:var(--text-dim);margin-bottom:12px;
  padding-bottom:6px;border-bottom:1px solid var(--border);}}
.form-row{{margin-bottom:14px;}}
.form-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;}}
label{{display:block;font-size:11px;font-weight:500;color:var(--text-dim);
  letter-spacing:0.06em;text-transform:uppercase;margin-bottom:5px;}}
input,select{{background:var(--bg3);border:1px solid var(--border);color:var(--text);
  border-radius:var(--radius);padding:8px 12px;font-family:var(--sans);
  font-size:13px;width:100%;outline:none;transition:border-color .15s;}}
input:focus,select:focus{{border-color:var(--amber);}}
select option{{background:var(--bg3);}}
.health-row{{display:flex;align-items:flex-start;gap:10px;
  padding:10px 12px;background:var(--bg3);border:1px solid var(--border);
  border-radius:var(--radius);margin-bottom:8px;}}
.health-icon{{font-size:14px;flex-shrink:0;margin-top:1px;}}
.health-label{{font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:0.08em;color:var(--text-dim);font-family:var(--mono);}}
.health-msg{{font-size:12px;color:var(--text-dim);margin-top:2px;font-family:var(--mono);}}
.ok .health-icon::before{{content:'✓';color:var(--green);}}
.warn .health-icon::before{{content:'⚠';color:var(--yellow);}}
.error .health-icon::before{{content:'✗';color:var(--red);}}
.loading .health-icon::before{{content:'○';color:var(--text-mute);}}
.btn{{display:inline-flex;align-items:center;justify-content:center;gap:8px;
  padding:10px 20px;border-radius:var(--radius);border:1px solid transparent;
  font-family:var(--sans);font-size:14px;font-weight:500;cursor:pointer;
  transition:all .15s;}}
.btn-primary{{background:var(--amber);color:#0d0e11;border-color:var(--amber);width:100%;}}
.btn-primary:hover:not(:disabled){{background:#fbbf24;}}
.btn-primary:disabled{{opacity:.45;cursor:not-allowed;}}
.btn-warning{{background:transparent;color:var(--amber);border-color:var(--amber);}}
.btn-warning:hover:not(:disabled){{background:rgba(245,158,11,.12);}}
.btn-danger{{background:transparent;color:var(--red);border-color:var(--red);}}
.btn-danger:hover:not(:disabled){{background:rgba(239,68,68,.12);}}
.btn-ghost{{background:transparent;color:var(--text-dim);border-color:var(--border);}}
.btn-ghost:hover:not(:disabled){{border-color:var(--amber);color:var(--amber);}}
.btn-row{{display:flex;gap:8px;margin-top:10px;}}
.btn-row .btn{{flex:1;}}
.spinner{{width:14px;height:14px;border:2px solid rgba(0,0,0,.2);
  border-top-color:#0d0e11;border-radius:50%;animation:spin .6s linear infinite;}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
.url-display{{display:flex;align-items:center;gap:10px;padding:10px 12px;
  background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);
  margin-top:10px;}}
.url-text{{font-family:var(--mono);font-size:12px;color:var(--amber);flex:1;}}
.url-link{{font-size:11px;color:var(--text-dim);text-decoration:none;
  padding:3px 8px;border:1px solid var(--border);border-radius:var(--radius);}}
.url-link:hover{{border-color:var(--amber);color:var(--amber);}}
.restart-note{{margin-top:10px;font-size:11px;color:var(--text-dim);font-family:var(--mono);}}
.hidden{{display:none!important;}}
</style>
</head>
<body>
<div class="card">
  <div class="card-header">
    <div class="logo-mark">// job-matcher</div>
    <h1>Launcher</h1>
  </div>
  <div class="card-body">

    <!-- Health Status -->
    <div class="section">
      <div class="section-title">System Status</div>
      <div id="health-sqlite"    class="health-row loading"><div class="health-icon"></div>
        <div><div class="health-label">SQLite</div><div class="health-msg">Checking…</div></div></div>
      <div id="health-ollama"    class="health-row loading"><div class="health-icon"></div>
        <div><div class="health-label">Ollama</div><div class="health-msg">Checking…</div></div></div>
      <div id="health-anthropic" class="health-row loading"><div class="health-icon"></div>
        <div><div class="health-label">Anthropic API</div><div class="health-msg">Checking…</div></div></div>
    </div>

    <!-- Server -->
    <div class="section">
      <div class="section-title">Server</div>
      <div class="form-grid">
        <div class="form-row" style="margin:0;">
          <label>Port</label>
          <input type="number" id="port" value="{port}" min="1024" max="65535"/>
        </div>
        <div class="form-row" style="margin:0;">
          <label>Host</label>
          <input type="text" id="host" value="{host}"/>
        </div>
      </div>
    </div>

    <!-- Database -->
    <div class="section">
      <div class="section-title">Database</div>
      <div class="form-row">
        <label>SQLite Path</label>
        <input type="text" id="db_path" value="{db_path}"
               placeholder="./job_matcher.db" oninput="scheduleHealthCheck()"/>
      </div>
    </div>

    <!-- LLM Providers -->
    <div class="section">
      <div class="section-title">LLM Providers</div>
      <div class="form-row">
        <label>Anthropic API Key</label>
        <input type="password" id="anthropic_api_key" value="{api_key}"
               placeholder="sk-ant-..." oninput="scheduleHealthCheck()"/>
      </div>
      <div class="form-row">
        <label>Ollama URL</label>
        <input type="text" id="ollama_base_url" value="{ollama_url}"
               oninput="scheduleHealthCheck()"/>
      </div>
      <div class="form-row">
        <label>Ollama Model</label>
        <select id="ollama_model">
          <option value="{ollama_model}">{ollama_model}</option>
        </select>
      </div>
      <div class="form-row">
        <label>Ollama Timeout (seconds)</label>
        <input type="number" id="ollama_timeout" value="{ollama_timeout}" min="30"/>
      </div>
    </div>

    <!-- Start button -->
    <button class="btn btn-primary" id="start-btn" onclick="startApp()">
      ▶ &nbsp;Start Job Matcher
    </button>

    <!-- Running panel (shown after Start) -->
    <div id="running-panel" class="hidden">
      <div class="url-display">
        <span class="url-text" id="url-text"></span>
        <a class="url-link" id="url-link" href="#" target="_blank">Open ↗</a>
      </div>
      <div class="btn-row">
        <button class="btn btn-ghost" onclick="openApp()">🌐 &nbsp;Open App</button>
        <button class="btn btn-warning" id="restart-btn" onclick="restartApp()">↺ &nbsp;Restart</button>
        <button class="btn btn-danger"  id="stop-btn"    onclick="stopApp()">■ &nbsp;Stop</button>
      </div>
      <div class="restart-note">Change model or port above, then click Restart to apply.</div>
    </div>

  </div>
</div>

<script>
console.log('[launcher] Script loaded');

// ── Helpers ───────────────────────────────────────────────────────────────────
const $ = id => {{ const el = document.getElementById(id); if (!el) console.error('[launcher] #' + id + ' not found'); return el; }};
function log(fn, msg, ...a)    {{ console.log('[' + fn + ']', msg, ...a); }}
function logErr(fn, msg, ...a) {{ console.error('[' + fn + '] ERROR:', msg, ...a); }}

let currentAppUrl = '';

// ── Health checks ─────────────────────────────────────────────────────────────
function getFormValues() {{
  return {{
    db_path:    ($('db_path')           || {{value:''}}).value,
    ollama_url: ($('ollama_base_url')   || {{value:''}}).value,
    api_key:    ($('anthropic_api_key') || {{value:''}}).value,
  }};
}}

function updateHealthRow(id, result) {{
  const el = $(id);
  if (!el) return;
  el.className = 'health-row ' + result.status;
  el.querySelector('.health-msg').textContent = result.message;
  log('health', id + ' → ' + result.status + ': ' + result.message);
}}

async function runHealthChecks() {{
  log('runHealthChecks', 'running...');
  const v = getFormValues();
  const params = new URLSearchParams({{ db_path: v.db_path, ollama_url: v.ollama_url, api_key: v.api_key }});
  try {{
    const res  = await fetch('/health?' + params);
    if (!res.ok) {{ logErr('runHealthChecks', 'HTTP ' + res.status); return; }}
    const data = await res.json();
    updateHealthRow('health-sqlite',    data.sqlite);
    updateHealthRow('health-ollama',    data.ollama);
    updateHealthRow('health-anthropic', data.anthropic);
    const sel = $('ollama_model');
    if (sel && data.models && data.models.length > 0) {{
      const current = sel.value;
      sel.innerHTML = '';
      data.models.forEach(m => {{
        const opt = document.createElement('option');
        opt.value = m; opt.textContent = m;
        if (m === current) opt.selected = true;
        sel.appendChild(opt);
      }});
      if (!data.models.includes(current) && current) {{
        const opt = document.createElement('option');
        opt.value = current; opt.textContent = current + ' (not found)';
        opt.selected = true;
        sel.insertBefore(opt, sel.firstChild);
      }}
    }}
  }} catch(e) {{ logErr('runHealthChecks', 'fetch threw:', e); }}
}}

let healthTimer;
function scheduleHealthCheck() {{ clearTimeout(healthTimer); healthTimer = setTimeout(runHealthChecks, 600); }}

// ── Build FormData from current form values ───────────────────────────────────
function buildFormData() {{
  const fd = new FormData();
  fd.append('port',              ($('port')             || {{value:''}}).value);
  fd.append('host',              ($('host')             || {{value:''}}).value);
  fd.append('db_path',           ($('db_path')          || {{value:''}}).value);
  fd.append('anthropic_api_key', ($('anthropic_api_key')|| {{value:''}}).value);
  fd.append('ollama_base_url',   ($('ollama_base_url')  || {{value:''}}).value);
  fd.append('ollama_model',      ($('ollama_model')     || {{value:''}}).value);
  fd.append('ollama_timeout',    ($('ollama_timeout')   || {{value:''}}).value);
  return fd;
}}

// ── Start ─────────────────────────────────────────────────────────────────────
async function startApp() {{
  log('startApp', 'clicked');
  const btn = $('start-btn');
  if (!btn) return;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Starting…';
  try {{
    log('startApp', 'POST /api/launcher/start');
    const res  = await fetch('/api/launcher/start', {{ method: 'POST', body: buildFormData() }});
    const data = await res.json();
    log('startApp', 'response status=' + res.status, data);
    if (!res.ok) {{ logErr('startApp', data.error || res.status); btn.disabled=false; btn.innerHTML='▶ &nbsp;Start Job Matcher'; alert('Error: ' + (data.error || res.status)); return; }}
    if (data.ok) {{ currentAppUrl = data.url; setRunningState(data.url); setTimeout(() => window.open(data.url, '_blank'), 800); }}
  }} catch(e) {{ logErr('startApp', 'fetch threw:', e); btn.disabled=false; btn.innerHTML='▶ &nbsp;Start Job Matcher'; alert('Failed: ' + e); }}
}}

// ── Stop ──────────────────────────────────────────────────────────────────────
async function stopApp() {{
  log('stopApp', 'clicked');
  if (!confirm('Stop the Job Matcher server?')) return;
  const btn = $('stop-btn');
  if (btn) {{ btn.disabled=true; btn.innerHTML='<span class="spinner"></span>'; }}
  try {{
    const res = await fetch('/api/launcher/stop', {{ method: 'POST' }});
    log('stopApp', 'response status=' + res.status);
    if (res.ok) {{ setStoppedState(); }} else {{ logErr('stopApp', 'HTTP ' + res.status); if (btn) {{ btn.disabled=false; btn.innerHTML='■ &nbsp;Stop'; }} }}
  }} catch(e) {{ logErr('stopApp', 'fetch threw:', e); if (btn) {{ btn.disabled=false; btn.innerHTML='■ &nbsp;Stop'; }} }}
}}

// ── Restart ───────────────────────────────────────────────────────────────────
async function restartApp() {{
  log('restartApp', 'clicked');
  const model = ($('ollama_model') || {{value:''}}).value;
  const port  = ($('port')         || {{value:''}}).value;
  if (!confirm('Restart Job Matcher?\\n\\nNew model: ' + model + '\\nNew port: ' + port)) return;
  const btn = $('restart-btn');
  if (btn) {{ btn.disabled=true; btn.innerHTML='<span class="spinner"></span> Restarting…'; }}
  try {{
    const res  = await fetch('/api/launcher/restart', {{ method: 'POST', body: buildFormData() }});
    const data = await res.json();
    log('restartApp', 'response status=' + res.status, data);
    if (res.ok && data.ok) {{ currentAppUrl = data.url; setRunningState(data.url); setTimeout(() => window.open(data.url, '_blank'), 1000); }}
    else {{ logErr('restartApp', data.error || res.status); if (btn) {{ btn.disabled=false; btn.innerHTML='↺ &nbsp;Restart'; }} }}
  }} catch(e) {{ logErr('restartApp', 'fetch threw:', e); if (btn) {{ btn.disabled=false; btn.innerHTML='↺ &nbsp;Restart'; }} }}
}}

// ── Open ──────────────────────────────────────────────────────────────────────
function openApp() {{ if (currentAppUrl) window.open(currentAppUrl, '_blank'); else logErr('openApp', 'no URL set'); }}

// ── UI state ──────────────────────────────────────────────────────────────────
function setRunningState(url) {{
  log('setRunningState', url);
  const btn = $('start-btn');
  if (btn) {{ btn.innerHTML='✓ &nbsp;Running'; btn.style.background='var(--green)'; btn.style.borderColor='var(--green)'; btn.style.color='#fff'; btn.disabled=true; }}
  const urlText = $('url-text'); if (urlText) urlText.textContent = url;
  const urlLink = $('url-link'); if (urlLink) urlLink.href = url;
  const panel = $('running-panel'); if (panel) panel.classList.remove('hidden');
  const rb = $('restart-btn'); if (rb) {{ rb.disabled=false; rb.innerHTML='↺ &nbsp;Restart'; }}
  const sb = $('stop-btn');    if (sb) {{ sb.disabled=false; sb.innerHTML='■ &nbsp;Stop'; }}
}}

function setStoppedState() {{
  log('setStoppedState', 'resetting');
  currentAppUrl = '';
  const btn = $('start-btn');
  if (btn) {{ btn.innerHTML='▶ &nbsp;Start Job Matcher'; btn.style.background=''; btn.style.borderColor=''; btn.style.color=''; btn.disabled=false; }}
  const panel = $('running-panel'); if (panel) panel.classList.add('hidden');
}}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {{
  log('init', 'DOMContentLoaded fired');
  runHealthChecks();
}});
</script>
</body>
</html>"""


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

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        query  = parse_qs(parsed.query, keep_blank_values=True)

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
