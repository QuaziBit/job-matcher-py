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
let isHorizontal  = false;

// ── Layout toggle ─────────────────────────────────────────────────────────────
function toggleLayout() {
  isHorizontal = !isHorizontal;
  applyLayout();
  localStorage.setItem('launcher-layout', isHorizontal ? 'horizontal' : 'vertical');
  log('toggleLayout', isHorizontal ? 'horizontal' : 'vertical');
}

function applyLayout() {
  const card   = $('card');
  const vPanel = $('layout-vertical');
  const hPanel = $('layout-horizontal');
  const btn    = $('layout-toggle');

  if (isHorizontal) {
    card.classList.add('horizontal');
    vPanel.classList.add('hidden');
    hPanel.classList.remove('hidden');
    btn.classList.add('active');
    btn.title = 'Switch to vertical layout';
    syncToHorizontal();
    runHealthChecks();
  } else {
    card.classList.remove('horizontal');
    hPanel.classList.add('hidden');
    vPanel.classList.remove('hidden');
    btn.classList.remove('active');
    btn.title = 'Switch to horizontal layout';
    syncToVertical();
    runHealthChecks();
  }
}

// Sync form values from vertical → horizontal when switching to horizontal
function syncToHorizontal() {
  const fields = ['port','host','db_path','anthropic_api_key','ollama_base_url','ollama_model','ollama_timeout'];
  fields.forEach(id => {
    const src = $(id); const dst = $('h-' + id);
    if (src && dst) dst.value = src.value;
  });
  // :checked may not match on fresh page load — fall back to HTML checked attribute
  let vMode = document.querySelector('input[name="analysis_mode"]:checked');
  if (!vMode) vMode = document.querySelector('input[name="analysis_mode"][checked]');
  if (vMode) {
    const hMode = document.querySelector(`input[name="h-analysis_mode"][value="${vMode.value}"]`);
    if (hMode) hMode.checked = true;
  }
}

// Sync form values from horizontal → vertical when switching to vertical
function syncToVertical() {
  const fields = ['port','host','db_path','anthropic_api_key','ollama_base_url','ollama_model','ollama_timeout'];
  fields.forEach(id => {
    const src = $('h-' + id); const dst = $(id);
    if (src && dst) dst.value = src.value;
  });
  const hMode = document.querySelector('input[name="h-analysis_mode"]:checked');
  if (hMode) {
    const vMode = document.querySelector(`input[name="analysis_mode"][value="${hMode.value}"]`);
    if (vMode) vMode.checked = true;
  }
}

// ── Health checks ─────────────────────────────────────────────────────────────
function getFormValues() {
  if (isHorizontal) {
    return {
      db_path:    ($('h-db_path')           || {value:''}).value,
      ollama_url: ($('h-ollama_base_url')   || {value:''}).value,
      api_key:    ($('h-anthropic_api_key') || {value:''}).value,
    };
  }
  return {
    db_path:    ($('db_path')           || {value:''}).value,
    ollama_url: ($('ollama_base_url')   || {value:''}).value,
    api_key:    ($('anthropic_api_key') || {value:''}).value,
  };
}

function updateHealthRow(id, result) {
  // Update both vertical and horizontal health rows
  const ids = [id, id.replace('health-', 'h-health-')];
  ids.forEach(rid => {
    const el = document.getElementById(rid);
    if (!el) return;
    el.className = 'health-row ' + result.status;
    el.querySelector('.health-msg').textContent = result.message;
  });
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

    // Update both vertical and horizontal model selects
    ['ollama_model', 'h-ollama_model'].forEach(selId => {
      const sel = $(selId);
      if (!sel || !data.models || data.models.length === 0) return;
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
    });
  } catch(e) { logErr('runHealthChecks', 'fetch threw:', e); }
}

let healthTimer;
function scheduleHealthCheck() { clearTimeout(healthTimer); healthTimer = setTimeout(runHealthChecks, 600); }

// ── Build FormData — reads from active layout ─────────────────────────────────
function buildFormData() {
  const fd  = new FormData();
  const pfx = isHorizontal ? 'h-' : '';

  fd.append('port',              ($(pfx + 'port')              || {value:''}).value);
  fd.append('host',              ($(pfx + 'host')              || {value:''}).value);
  fd.append('db_path',           ($(pfx + 'db_path')           || {value:''}).value);
  fd.append('anthropic_api_key', ($(pfx + 'anthropic_api_key') || {value:''}).value);
  fd.append('ollama_base_url',   ($(pfx + 'ollama_base_url')   || {value:''}).value);
  fd.append('ollama_model',      ($(pfx + 'ollama_model')      || {value:''}).value);
  fd.append('ollama_timeout',    ($(pfx + 'ollama_timeout')    || {value:''}).value);

  const modeName = isHorizontal ? 'h-analysis_mode' : 'analysis_mode';
  const modeEl   = document.querySelector(`input[name="${modeName}"]:checked`);
  fd.append('analysis_mode', modeEl ? modeEl.value : 'standard');
  return fd;
}

// ── Start ─────────────────────────────────────────────────────────────────────
async function startApp() {
  log('startApp', 'clicked');
  const pfx = isHorizontal ? 'h-' : '';
  const btn = $(pfx + 'start-btn');
  if (!btn) return;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Starting…';
  try {
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
  const pfx = isHorizontal ? 'h-' : '';
  const btn = $(pfx + 'stop-btn');
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
  const pfx   = isHorizontal ? 'h-' : '';
  const model = ($(pfx + 'ollama_model') || {value:''}).value;
  const port  = ($(pfx + 'port')         || {value:''}).value;
  if (!confirm('Restart Job Matcher?\n\nNew model: ' + model + '\nNew port: ' + port)) return;
  const btn = $(pfx + 'restart-btn');
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Restarting…'; }
  try {
    const res  = await fetch('/api/launcher/restart', { method: 'POST', body: buildFormData() });
    const data = await res.json();
    log('restartApp', 'response status=' + res.status, data);
    if (res.ok && data.ok) {
      currentAppUrl = data.url;
      if (data.analysis_mode) applyAnalysisMode(data.analysis_mode);
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

// ── UI state — updates both layouts ──────────────────────────────────────────
function setRunningState(url) {
  log('setRunningState', url);

  ['start-btn', 'h-start-btn'].forEach(id => {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.innerHTML = '✓ &nbsp;Running';
    btn.style.background  = 'var(--green)';
    btn.style.borderColor = 'var(--green)';
    btn.style.color       = '#fff';
    btn.disabled = true;
  });

  ['url-text', 'h-url-text'].forEach(id => { const el = document.getElementById(id); if (el) el.textContent = url; });
  ['url-link', 'h-url-link'].forEach(id => { const el = document.getElementById(id); if (el) el.href = url; });
  ['running-panel', 'h-running-panel'].forEach(id => { const el = document.getElementById(id); if (el) el.classList.remove('hidden'); });

  ['restart-btn', 'h-restart-btn'].forEach(id => { const el = document.getElementById(id); if (el) { el.disabled = false; el.innerHTML = '↺ &nbsp;Restart'; } });
  ['stop-btn',    'h-stop-btn'   ].forEach(id => { const el = document.getElementById(id); if (el) { el.disabled = false; el.innerHTML = '■ &nbsp;Stop'; } });
}

function setStoppedState() {
  log('setStoppedState', 'resetting');
  currentAppUrl = '';

  ['start-btn', 'h-start-btn'].forEach(id => {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.innerHTML     = '▶ &nbsp;Start Job Matcher';
    btn.style.background  = '';
    btn.style.borderColor = '';
    btn.style.color       = '';
    btn.disabled = false;
  });

  ['running-panel', 'h-running-panel'].forEach(id => { const el = document.getElementById(id); if (el) el.classList.add('hidden'); });
}

// ── Apply analysis mode to both layouts ──────────────────────────────────────
function applyAnalysisMode(mode) {
  const v = document.querySelector(`input[name="analysis_mode"][value="${mode}"]`);
  const h = document.querySelector(`input[name="h-analysis_mode"][value="${mode}"]`);
  if (v) v.checked = true;
  if (h) h.checked = true;
  log('applyAnalysisMode', mode);
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  log('init', 'DOMContentLoaded fired');

  // Restore saved layout preference — default is horizontal
  const saved = localStorage.getItem('launcher-layout');
  const defaultHorizontal = true;
  if (saved !== null ? saved === 'horizontal' : defaultHorizontal) {
    isHorizontal = true;
    applyLayout();
  }

  runHealthChecks();
});
