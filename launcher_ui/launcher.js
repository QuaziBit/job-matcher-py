console.log('[launcher] Script loaded');

// ── Helpers ───────────────────────────────────────────────────────────────────
const $ = id => {
  const el = document.getElementById(id);
  if (!el) console.error('[launcher] #' + id + ' not found');
  return el;
};
function log(fn, msg, ...a)    { console.log('[' + fn + ']', msg, ...a); }
function logErr(fn, msg, ...a) { console.error('[' + fn + '] ERROR:', msg, ...a); }

let currentAppUrl = '';

// ── Health checks ─────────────────────────────────────────────────────────────
function getFormValues() {
  return {
    db_path:    ($('db_path')           || {value:''}).value,
    ollama_url: ($('ollama_base_url')   || {value:''}).value,
    api_key:    ($('anthropic_api_key') || {value:''}).value,
  };
}

function updateHealthRow(id, result) {
  const el = $(id);
  if (!el) return;
  el.className = 'health-row ' + result.status;
  el.querySelector('.health-msg').textContent = result.message;
  log('health', id + ' → ' + result.status + ': ' + result.message);
}

async function runHealthChecks() {
  log('runHealthChecks', 'running...');
  const v = getFormValues();
  const params = new URLSearchParams({ db_path: v.db_path, ollama_url: v.ollama_url, api_key: v.api_key });
  try {
    const res  = await fetch('/health?' + params);
    if (!res.ok) { logErr('runHealthChecks', 'HTTP ' + res.status); return; }
    const data = await res.json();
    updateHealthRow('health-sqlite',    data.sqlite);
    updateHealthRow('health-ollama',    data.ollama);
    updateHealthRow('health-anthropic', data.anthropic);
    const sel = $('ollama_model');
    if (sel && data.models && data.models.length > 0) {
      const current = sel.value;
      sel.innerHTML = '';
      data.models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m; opt.textContent = m;
        if (m === current) opt.selected = true;
        sel.appendChild(opt);
      });
      if (!data.models.includes(current) && current) {
        const opt = document.createElement('option');
        opt.value = current; opt.textContent = current + ' (not found)';
        opt.selected = true;
        sel.insertBefore(opt, sel.firstChild);
      }
    }
  } catch(e) { logErr('runHealthChecks', 'fetch threw:', e); }
}

let healthTimer;
function scheduleHealthCheck() { clearTimeout(healthTimer); healthTimer = setTimeout(runHealthChecks, 600); }

// ── Build FormData from current form values ───────────────────────────────────
function buildFormData() {
  const fd = new FormData();
  fd.append('port',              ($('port')              || {value:''}).value);
  fd.append('host',              ($('host')              || {value:''}).value);
  fd.append('db_path',           ($('db_path')           || {value:''}).value);
  fd.append('anthropic_api_key', ($('anthropic_api_key') || {value:''}).value);
  fd.append('ollama_base_url',   ($('ollama_base_url')   || {value:''}).value);
  fd.append('ollama_model',      ($('ollama_model')      || {value:''}).value);
  fd.append('ollama_timeout',    ($('ollama_timeout')    || {value:''}).value);
  const modeEl = document.querySelector('input[name="analysis_mode"]:checked');
  fd.append('analysis_mode', modeEl ? modeEl.value : 'standard');
  return fd;
}

// ── Start ─────────────────────────────────────────────────────────────────────
async function startApp() {
  log('startApp', 'clicked');
  const btn = $('start-btn');
  if (!btn) return;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Starting…';
  try {
    log('startApp', 'POST /api/launcher/start');
    const res  = await fetch('/api/launcher/start', { method: 'POST', body: buildFormData() });
    const data = await res.json();
    log('startApp', 'response status=' + res.status, data);
    if (!res.ok) {
      logErr('startApp', data.error || res.status);
      btn.disabled = false;
      btn.innerHTML = '▶ &nbsp;Start Job Matcher';
      alert('Error: ' + (data.error || res.status));
      return;
    }
    if (data.ok) {
      currentAppUrl = data.url;
      setRunningState(data.url);
      setTimeout(() => window.open(data.url, '_blank'), 800);
    }
  } catch(e) {
    logErr('startApp', 'fetch threw:', e);
    btn.disabled = false;
    btn.innerHTML = '▶ &nbsp;Start Job Matcher';
    alert('Failed: ' + e);
  }
}

// ── Stop ──────────────────────────────────────────────────────────────────────
async function stopApp() {
  log('stopApp', 'clicked');
  if (!confirm('Stop the Job Matcher server?')) return;
  const btn = $('stop-btn');
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>'; }
  try {
    const res = await fetch('/api/launcher/stop', { method: 'POST' });
    log('stopApp', 'response status=' + res.status);
    if (res.ok) {
      setStoppedState();
    } else {
      logErr('stopApp', 'HTTP ' + res.status);
      if (btn) { btn.disabled = false; btn.innerHTML = '■ &nbsp;Stop'; }
    }
  } catch(e) {
    logErr('stopApp', 'fetch threw:', e);
    if (btn) { btn.disabled = false; btn.innerHTML = '■ &nbsp;Stop'; }
  }
}

// ── Restart ───────────────────────────────────────────────────────────────────
async function restartApp() {
  log('restartApp', 'clicked');
  const model = ($('ollama_model') || {value:''}).value;
  const port  = ($('port')         || {value:''}).value;
  if (!confirm('Restart Job Matcher?\n\nNew model: ' + model + '\nNew port: ' + port)) return;
  const btn = $('restart-btn');
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Restarting…'; }
  try {
    const res  = await fetch('/api/launcher/restart', { method: 'POST', body: buildFormData() });
    const data = await res.json();
    log('restartApp', 'response status=' + res.status, data);
    if (res.ok && data.ok) {
      currentAppUrl = data.url;
      setRunningState(data.url);
      setTimeout(() => window.open(data.url, '_blank'), 1000);
    } else {
      logErr('restartApp', data.error || res.status);
      if (btn) { btn.disabled = false; btn.innerHTML = '↺ &nbsp;Restart'; }
    }
  } catch(e) {
    logErr('restartApp', 'fetch threw:', e);
    if (btn) { btn.disabled = false; btn.innerHTML = '↺ &nbsp;Restart'; }
  }
}

// ── Open ──────────────────────────────────────────────────────────────────────
function openApp() {
  if (currentAppUrl) window.open(currentAppUrl, '_blank');
  else logErr('openApp', 'no URL set');
}

// ── UI state ──────────────────────────────────────────────────────────────────
function setRunningState(url) {
  log('setRunningState', url);
  const btn = $('start-btn');
  if (btn) {
    btn.innerHTML = '✓ &nbsp;Running';
    btn.style.background   = 'var(--green)';
    btn.style.borderColor  = 'var(--green)';
    btn.style.color        = '#fff';
    btn.disabled = true;
  }
  const urlText = $('url-text'); if (urlText) urlText.textContent = url;
  const urlLink = $('url-link'); if (urlLink) urlLink.href = url;
  const panel   = $('running-panel'); if (panel) panel.classList.remove('hidden');
  const rb = $('restart-btn'); if (rb) { rb.disabled = false; rb.innerHTML = '↺ &nbsp;Restart'; }
  const sb = $('stop-btn');    if (sb) { sb.disabled = false; sb.innerHTML = '■ &nbsp;Stop'; }
}

function setStoppedState() {
  log('setStoppedState', 'resetting');
  currentAppUrl = '';
  const btn = $('start-btn');
  if (btn) {
    btn.innerHTML      = '▶ &nbsp;Start Job Matcher';
    btn.style.background  = '';
    btn.style.borderColor = '';
    btn.style.color       = '';
    btn.disabled = false;
  }
  const panel = $('running-panel'); if (panel) panel.classList.add('hidden');
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  log('init', 'DOMContentLoaded fired');
  runHealthChecks();
});
