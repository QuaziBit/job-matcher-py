// ── Location flag map ────────────────────────────────────────────────────────
const LOCATION_FLAGS = [
  { keywords: ['remote'],                                              code: 'REMOTE' },
  { keywords: ['canada', 'ontario', 'quebec', 'british columbia',
               'toronto', 'vancouver', 'montreal', 'ottawa', 'calgary'], code: 'CA' },
  { keywords: ['united kingdom', 'england', 'scotland', 'wales',
               'london', 'manchester', 'birmingham', 'edinburgh', ', uk'],code: 'UK' },
  { keywords: ['germany', 'deutschland', 'berlin', 'munich', 'frankfurt',
               'hamburg', 'cologne', 'stuttgart', 'bavaria'],           code: 'DE' },
  { keywords: ['france', 'paris', 'lyon', 'marseille', 'toulouse'],    code: 'FR' },
  { keywords: ['australia', 'sydney', 'melbourne', 'brisbane',
               'perth', 'adelaide'],                                    code: 'AU' },
  { keywords: ['india', 'bangalore', 'mumbai', 'delhi', 'hyderabad',
               'chennai', 'pune', 'bengaluru'],                         code: 'IN' },
  { keywords: ['netherlands', 'amsterdam', 'rotterdam', 'the hague'],  code: 'NL' },
  { keywords: ['poland', 'warsaw', 'krakow', 'wroclaw', 'gdansk'],     code: 'PL' },
  { keywords: ['ukraine', 'kyiv', 'kharkiv', 'lviv', 'odessa'],        code: 'UA' },
  { keywords: ['singapore'],                                            code: 'SG' },
  { keywords: ['japan', 'tokyo', 'osaka', 'kyoto', 'yokohama'],        code: 'JP' },
  { keywords: ['brazil', 'sao paulo', 'rio de janeiro',
               'brasilia', 'curitiba'],                                 code: 'BR' },
  { keywords: ['ireland', 'dublin', 'cork', 'galway'],                 code: 'IE' },
  { keywords: ['sweden', 'stockholm', 'gothenburg', 'malmo'],          code: 'SE' },
  { keywords: ['spain', 'madrid', 'barcelona', 'valencia', 'seville'], code: 'ES' },
  { keywords: ['portugal', 'lisbon', 'porto'],                         code: 'PT' },
  { keywords: ['italy', 'rome', 'milan', 'naples', 'turin'],           code: 'IT' },
  { keywords: ['switzerland', 'zurich', 'geneva', 'bern', 'basel'],    code: 'CH' },
  { keywords: ['united states', 'usa', 'u.s.'],                        code: 'US' },
];

function getLocationFlag(location) {
  if (!location || /^\d+$/.test(location.trim())) return 'N/A';
  const loc = location.toLowerCase();
  for (const entry of LOCATION_FLAGS) {
    if (entry.keywords.some(kw => loc.includes(kw))) return entry.code;
  }
  return 'US';
}


// ── Date formatting ───────────────────────────────────────────────────────────
function formatJobDate(dateStr) {
  if (!dateStr) return '';
  const date     = new Date(dateStr.replace(' ', 'T') + 'Z');
  const now      = new Date();
  const sameYear = date.getFullYear() === now.getFullYear();
  const opts     = sameYear
    ? { month: 'short', day: 'numeric' }
    : { month: 'short', day: 'numeric', year: 'numeric' };
  return date.toLocaleDateString('en-US', opts);
}


// ── Logging helpers ───────────────────────────────────────────────────────────
function log(fn, msg, ...args)    { console.log(`[${fn}]`, msg, ...args); }
function logErr(fn, msg, ...args) { console.error(`[${fn}] ERROR:`, msg, ...args); }


// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function toast(msg, type = 'info') {
  log('toast', type, msg);
  const el = document.getElementById('toast');
  if (!el) { logErr('toast', '#toast not found'); return; }
  el.textContent = msg;
  el.className = `show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = ''; }, 3500);
}


// ── Tabs ──────────────────────────────────────────────────────────────────────
function activateTab(tabName) {
  const tab = document.querySelector(`.tab[data-tab="${tabName}"]`);
  if (!tab) return;
  const tabBar = tab.closest('.tabs');
  const parent = tabBar ? tabBar.closest('.tab-container') || document : document;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  tab.classList.add('active');
  const content = parent.querySelector(`[data-tab-content="${tabName}"]`);
  if (content) content.classList.add('active');
}

function initTabs() {
  log('initTabs', `found ${document.querySelectorAll('.tabs').length} tab bar(s)`);
  document.querySelectorAll('.tabs').forEach(tabBar => {
    tabBar.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.tab;
        const parent = tabBar.closest('.tab-container') || document;
        tabBar.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        const content = parent.querySelector(`[data-tab-content="${target}"]`);
        if (content) content.classList.add('active');
      });
    });
  });
}


// ── Score meter ───────────────────────────────────────────────────────────────
function renderMeter(score, container) {
  container.innerHTML = '';
  for (let i = 1; i <= 5; i++) {
    const pip = document.createElement('div');
    pip.className = 'score-pip' + (i <= score ? ` filled-${score}` : '');
    container.appendChild(pip);
  }
}


// ── Add mode toggle ───────────────────────────────────────────────────────────
function initAddModeToggle() {
  log('initAddModeToggle', 'init');
  const radios = document.querySelectorAll('input[name="add-mode"]');
  if (!radios.length) return;
  radios.forEach(radio => {
    radio.addEventListener('change', () => {
      const urlForm   = document.getElementById('add-job-form');
      const pasteForm = document.getElementById('paste-job-form-wrap');
      if (radio.value === 'paste' && radio.checked) {
        urlForm.classList.add('hidden');
        pasteForm.classList.remove('hidden');
      } else {
        urlForm.classList.remove('hidden');
        pasteForm.classList.add('hidden');
      }
    });
  });
}


// ── Add job by URL ────────────────────────────────────────────────────────────
async function addJob(e) {
  e.preventDefault();
  const form  = e.target;
  const btn   = form.querySelector('[type=submit]');
  const input = form.querySelector('input[name=url]');
  const url   = input.value.trim();
  log('addJob', `url="${url}"`);
  if (!url) { toast('Please enter a URL', 'error'); return; }

  btn.disabled  = true;
  btn.innerHTML = TMPL.spinner('Scraping\u2026');

  const fd = new FormData();
  fd.append('url', url);

  try {
    log('addJob', 'POST /api/jobs/scrape');
    const res  = await fetch('/api/jobs/scrape', { method: 'POST', body: fd });
    const data = await res.json();
    log('addJob', `response status=${res.status}`, data);
    if (res.status === 409) {
      toast('Job already added', 'info');
      if (data.job_id) setTimeout(() => window.location.href = '/job/' + data.job_id, 600);
      btn.disabled = false; btn.textContent = 'Add Job';
      return;
    }
    if (!res.ok) {
      logErr('addJob', `server error ${res.status}:`, data.error);
      toast(data.error || 'Failed to scrape URL', 'error');
      btn.disabled = false; btn.textContent = 'Add Job';
      return;
    }
    sessionStorage.setItem('job_preview', JSON.stringify(data));
    window.location.href = '/jobs/preview';
  } catch(err) {
    logErr('addJob', 'fetch threw:', err);
    toast('Network error', 'error');
    btn.disabled = false; btn.textContent = 'Add Job';
  }
}


// ── Add job by paste ──────────────────────────────────────────────────────────
async function addJobManual() {
  log('addJobManual', 'called');
  const title       = document.getElementById('paste-title').value.trim();
  const company     = document.getElementById('paste-company').value.trim();
  const location    = (document.getElementById('paste-location') || {value:''}).value.trim();
  const sourceUrl   = (document.getElementById('paste-url')      || {value:''}).value.trim();
  const description = document.getElementById('paste-description').value.trim();
  const btn         = document.getElementById('paste-submit-btn');

  log('addJobManual', `title="${title}" company="${company}" desc_len=${description.length}`);
  if (!description) { toast('Please paste a job description', 'error'); return; }
  if (description.length < 50) { toast('Description too short (min 50 chars)', 'error'); return; }

  btn.disabled  = true;
  btn.innerHTML = TMPL.spinner('Saving\u2026');

  const fd = new FormData();
  fd.append('title',       title);
  fd.append('company',     company);
  fd.append('location',    location);
  fd.append('source_url',  sourceUrl);
  fd.append('description', description);

  try {
    log('addJobManual', 'POST /api/jobs/add-manual');
    const res  = await fetch('/api/jobs/add-manual', { method: 'POST', body: fd });
    const data = await res.json();
    log('addJobManual', `response status=${res.status}`, data);
    if (res.status === 409) {
      toast('This description was already added', 'info');
      btn.disabled = false; btn.textContent = 'Add Job';
      return;
    }
    if (!res.ok) {
      logErr('addJobManual', `server error ${res.status}:`, data.error);
      toast(data.error || 'Failed to save', 'error');
      btn.disabled = false; btn.textContent = 'Add Job';
      return;
    }
    log('addJobManual', `success id=${data.job_id}`);
    toast(`\u2713 Added: ${data.title}`, 'success');
    setTimeout(() => window.location.href = '/job/' + data.job_id, 600);
  } catch(err) {
    logErr('addJobManual', 'fetch threw:', err);
    toast('Network error', 'error');
    btn.disabled = false; btn.textContent = 'Add Job';
  }
}


// ── Analysis progress bar ─────────────────────────────────────────────────────
const MODE_ESTIMATES = { fast: 30, standard: 90, detailed: 240 };
let _progressTimer = null;
let _progressStart = null;

function startProgress(provider, model, mode) {
  const el    = document.getElementById('analysis-progress');
  const label = document.getElementById('progress-label');
  const fill  = document.getElementById('progress-fill');
  const meta  = document.getElementById('progress-meta');
  const est   = MODE_ESTIMATES[mode] || 90;
  if (el) el.classList.remove('hidden');
  _progressStart = Date.now();
  if (label) label.textContent = `Analyzing with ${model} \u00b7 ${mode} mode`;
  if (fill)  fill.style.width = '0%';
  _progressTimer = setInterval(() => {
    const elapsed = (Date.now() - _progressStart) / 1000;
    const pct     = Math.min(elapsed / est * 100, 95);
    if (fill) fill.style.width = pct + '%';
    if (meta) meta.textContent = `${formatElapsed(elapsed)} elapsed \u00b7 ~${formatElapsed(est)} estimated`;
  }, 500);
  log('startProgress', `provider=${provider} model=${model} mode=${mode} est=${est}s`);
}

function stopProgress() {
  clearInterval(_progressTimer);
  _progressTimer = null;
  const fill = document.getElementById('progress-fill');
  if (fill) fill.style.width = '100%';
  setTimeout(() => {
    const el = document.getElementById('analysis-progress');
    if (el) el.classList.add('hidden');
  }, 400);
  log('stopProgress', 'done');
}

function formatElapsed(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}:${s.toString().padStart(2, '0')}` : `${s}s`;
}


// ── Provider model selectors ──────────────────────────────────────────────────
async function populateOllamaModels() {
  const sel = document.getElementById('ollama-model-select');
  if (!sel) return;
  try {
    const res  = await fetch('/api/ollama/models');
    if (!res.ok) return;
    const data   = await res.json();
    const models = (data.models || []).slice().sort();
    if (!models.length) return;
    const current = sel.value;
    sel.innerHTML = '';
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      if (m === current) opt.selected = true;
      sel.appendChild(opt);
    });
    if (!models.includes(current) && current) {
      const opt = document.createElement('option');
      opt.value = current; opt.textContent = current; opt.selected = true;
      sel.insertBefore(opt, sel.firstChild);
    }
  } catch(e) {
    log('populateOllamaModels', 'could not reach Ollama:', e.message);
  }
}

async function populateCloudModels(provider) {
  const sel = document.getElementById('cloud-model-select');
  if (!sel) return;
  try {
    const res  = await fetch(`/api/providers/models?provider=${provider}`);
    if (!res.ok) return;
    const data   = await res.json();
    const models = data.models || [];
    if (!models.length) return;
    const current = sel.dataset[provider] || '';
    sel.innerHTML = '';
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id; opt.textContent = m.label;
      if (m.id === current) opt.selected = true;
      sel.appendChild(opt);
    });
    const ids = models.map(m => m.id);
    if (current && !ids.includes(current)) {
      const opt = document.createElement('option');
      opt.value = current; opt.textContent = current + ' \u00b7 (configured)'; opt.selected = true;
      sel.insertBefore(opt, sel.firstChild);
    }
  } catch(e) {
    log('populateCloudModels', 'error:', e.message);
  }
}

function updateProviderModelRow() {
  const providerInput = document.querySelector('input[name="provider"]:checked');
  const provider      = providerInput ? providerInput.value : '';
  const ollamaRow     = document.getElementById('ollama-model-row');
  const cloudRow      = document.getElementById('cloud-model-row');
  if (!ollamaRow || !cloudRow) return;
  const isOllama = provider === 'ollama';
  const isCloud  = ['anthropic', 'openai', 'gemini'].includes(provider);
  ollamaRow.style.display = isOllama ? '' : 'none';
  cloudRow.style.display  = isCloud  ? '' : 'none';
  if (isOllama) populateOllamaModels();
  if (isCloud)  populateCloudModels(provider);
}


// ── Analyze job ───────────────────────────────────────────────────────────────
async function analyzeJob(jobId) {
  log('analyzeJob', `jobId=${jobId}`);
  const resumeSelect  = document.getElementById('analyze-resume');
  const providerInput = document.querySelector('input[name="provider"]:checked');
  const btn           = document.getElementById('analyze-btn');

  if (!resumeSelect || !resumeSelect.value) {
    toast('Please select a resume first', 'error'); return;
  }

  const provider     = providerInput ? providerInput.value : 'anthropic';
  const modeSelect   = document.getElementById('analysis-mode-select');
  const ollamaSelect = document.getElementById('ollama-model-select');
  const cloudSelect  = document.getElementById('cloud-model-select');
  const mode         = modeSelect ? modeSelect.value : (btn ? (btn.dataset.mode || 'standard') : 'standard');
  const ollamaModel  = (provider === 'ollama' && ollamaSelect) ? ollamaSelect.value : '';
  const cloudModel   = (['anthropic','openai','gemini'].includes(provider) && cloudSelect) ? cloudSelect.value : '';
  const displayModel = ollamaModel || cloudModel || provider;

  btn.disabled    = true;
  btn.textContent = 'Analyzing\u2026';
  startProgress(provider, displayModel, mode);

  const fd = new FormData();
  fd.append('resume_id',     resumeSelect.value);
  fd.append('provider',      provider);
  fd.append('analysis_mode', mode);
  if (provider === 'ollama' && ollamaModel) fd.append('ollama_model', ollamaModel);
  if (['anthropic','openai','gemini'].includes(provider) && cloudModel) fd.append('cloud_model', cloudModel);

  try {
    log('analyzeJob', `POST /api/jobs/${jobId}/analyze`);
    const res  = await fetch(`/api/jobs/${jobId}/analyze`, { method: 'POST', body: fd });
    const data = await res.json();
    log('analyzeJob', `response status=${res.status}`, data);
    stopProgress();
    if (!res.ok) {
      logErr('analyzeJob', `server error ${res.status}:`, data.error);
      toast(data.error || 'Analysis failed', 'error');
      btn.disabled = false; btn.textContent = 'Run Analysis';
      return;
    }
    log('analyzeJob', `success score=${data.score} adjusted=${data.adjusted_score}`);
    toast(`\u2713 Score: ${data.adjusted_score}/5 (raw ${data.score}/5)`, 'success');
    btn.disabled = false; btn.textContent = 'Run Analysis';
    setTimeout(() => refreshJobDetailPage(), 800);
  } catch(err) {
    stopProgress();
    logErr('analyzeJob', 'fetch threw:', err);
    toast('Network error', 'error');
    btn.disabled = false; btn.textContent = 'Run Analysis';
  }
}


// ── Save application ──────────────────────────────────────────────────────────
async function saveApplication(jobId) {
  log('saveApplication', `jobId=${jobId}`);
  const status = document.getElementById('app-status').value;
  const name   = document.getElementById('recruiter-name').value;
  const email  = document.getElementById('recruiter-email').value;
  const phone  = document.getElementById('recruiter-phone').value;
  const notes  = document.getElementById('app-notes').value;
  const btn    = document.getElementById('save-app-btn');

  btn.disabled  = true;
  btn.innerHTML = TMPL.spinner();

  const fd = new FormData();
  fd.append('status',          status);
  fd.append('recruiter_name',  name);
  fd.append('recruiter_email', email);
  fd.append('recruiter_phone', phone);
  fd.append('notes',           notes);

  try {
    log('saveApplication', `POST /api/jobs/${jobId}/application`);
    const res = await fetch(`/api/jobs/${jobId}/application`, { method: 'POST', body: fd });
    log('saveApplication', `response status=${res.status}`);
    if (res.ok) {
      toast('Application info saved', 'success');
      const badge = document.getElementById('status-badge');
      if (badge) {
        badge.className   = `status-badge status-${status}`;
        badge.textContent = status.replace('_', ' ');
      }
    } else {
      logErr('saveApplication', `server error ${res.status} for job=${jobId}`);
      toast('Save failed', 'error');
    }
  } catch(err) {
    logErr('saveApplication', 'fetch threw:', err);
    toast('Network error', 'error');
  }
  btn.disabled  = false;
  btn.innerHTML = TMPL.saveBtn();
}


// ── Delete analysis / job ─────────────────────────────────────────────────────
async function deleteAnalysis(analysisId) {
  if (!confirm('Remove this analysis?')) return;
  log('deleteAnalysis', `id=${analysisId}`);
  try {
    const res = await fetch(`/api/analyses/${analysisId}`, { method: 'DELETE' });
    log('deleteAnalysis', `response status=${res.status}`);
    if (res.ok) {
      const block = document.getElementById(`analysis-${analysisId}`);
      if (block) block.remove();
      toast('Analysis removed', 'info');
    } else {
      logErr('deleteAnalysis', `server error ${res.status} for id=${analysisId}`);
      toast('Delete failed', 'error');
    }
  } catch(err) {
    logErr('deleteAnalysis', 'fetch threw:', err);
    toast('Network error', 'error');
  }
}

async function deleteJob(jobId) {
  if (!confirm('Delete this job and all its analyses?')) return;
  log('deleteJob', `id=${jobId}`);
  try {
    const res = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
    log('deleteJob', `response status=${res.status}`);
    if (res.ok) {
      toast('Job deleted', 'info');
      setTimeout(() => location.href = '/', 600);
    } else {
      logErr('deleteJob', `server error ${res.status} for id=${jobId}`);
      toast('Delete failed', 'error');
    }
  } catch(err) {
    logErr('deleteJob', 'fetch threw:', err);
    toast('Network error', 'error');
  }
}


// ── Edit / save job URL ───────────────────────────────────────────────────────

function toggleUrlEdit() {
  const row = document.getElementById('url-edit-row');
  if (!row) return;
  const visible = row.style.display === 'block';
  row.style.display = visible ? 'none' : 'block';
  if (!visible) {
    const input = document.getElementById('url-edit-input');
    if (input) { input.focus(); input.select(); }
  }
}

async function saveJobUrl(jobId, url) {
  log('saveJobUrl', `jobId=${jobId} url="${url}"`);
  const fd = new FormData();
  fd.append('url', url);
  try {
    const res  = await fetch(`/api/jobs/${jobId}/url`, { method: 'PATCH', body: fd });
    const data = await res.json();
    log('saveJobUrl', `response status=${res.status}`, data);
    if (!res.ok) {
      logErr('saveJobUrl', `server error ${res.status}:`, data.error);
      toast(data.error || 'Failed to update URL', 'error');
      return;
    }
    toast('✓ URL updated', 'success');
    refreshJobDetailPage();
  } catch(err) {
    logErr('saveJobUrl', 'fetch threw:', err);
    toast('Network error', 'error');
  }
}


// ── Delete resume ─────────────────────────────────────────────────────────────
async function deleteResume(resumeId) {
  if (!confirm('Delete this resume version?')) return;
  log('deleteResume', `id=${resumeId}`);
  try {
    const res = await fetch(`/api/resumes/${resumeId}`, { method: 'DELETE' });
    log('deleteResume', `response status=${res.status}`);
    if (res.ok) {
      toast('Resume deleted', 'info');
      setTimeout(() => initResumesPage(), 600);
    } else {
      logErr('deleteResume', `server error ${res.status} for id=${resumeId}`);
      toast('Delete failed', 'error');
    }
  } catch(err) {
    logErr('deleteResume', 'fetch threw:', err);
    toast('Network error', 'error');
  }
}


// ── Add resume ────────────────────────────────────────────────────────────────
async function addResume(e) {
  if (e) e.preventDefault();
  log('addResume', 'called');
  const form = document.getElementById('add-resume-form');
  if (!form) { logErr('addResume', 'form#add-resume-form not found'); return; }

  const fd      = new FormData(form);
  const label   = (fd.get('label')   || '').trim();
  const content = (fd.get('content') || '').trim();
  log('addResume', `label="${label}" content_len=${content.length}`);
  if (!label || !content) {
    logErr('addResume', `validation failed — label="${label}" content_len=${content.length}`);
    toast('Label and content are required', 'error');
    return;
  }

  const btn = form.querySelector('[type=submit]');
  if (btn) { btn.disabled = true; btn.innerHTML = TMPL.spinner(); }

  try {
    log('addResume', 'POST /api/resumes/add');
    const res  = await fetch('/api/resumes/add', { method: 'POST', body: fd });
    const data = await res.json();
    log('addResume', `response status=${res.status}`, data);
    if (res.ok) {
      log('addResume', `success id=${data.resume_id}`);
      toast(`\u2713 Resume "${data.label}" saved`, 'success');
      form.reset();
      if (btn) { btn.disabled = false; btn.textContent = 'Save Resume'; }
      await initResumesPage();
    } else {
      logErr('addResume', `server error ${res.status}:`, data.error);
      toast(data.error || 'Failed', 'error');
      if (btn) { btn.disabled = false; btn.textContent = 'Save Resume'; }
    }
  } catch(err) {
    logErr('addResume', 'fetch threw:', err);
    toast('Network error', 'error');
    if (btn) { btn.disabled = false; btn.textContent = 'Save Resume'; }
  }
}


// ── Snippet toggle ────────────────────────────────────────────────────────────
function toggleSnippets(btn) {
  log('toggleSnippets', 'clicked');
  const container = btn.nextElementSibling;
  if (!container) { logErr('toggleSnippets', 'no sibling found'); return; }
  const isHidden = container.style.display === 'none' || !container.style.display;
  container.style.display = isHidden ? 'block' : 'none';
  btn.innerHTML = isHidden ? TMPL.snippetToggle(true) : TMPL.snippetToggle(false);
}


// ── Toggle description ────────────────────────────────────────────────────────
async function toggleDesc(jobId) {
  log('toggleDesc', `jobId=${jobId}`);
  const box = document.getElementById('desc-box');
  if (!box) { logErr('toggleDesc', '#desc-box not found'); return; }
  if (box.classList.contains('hidden')) {
    if (!box.dataset.loaded) {
      box.textContent = 'Loading\u2026';
      log('toggleDesc', `fetching description for job=${jobId}`);
      try {
        const res  = await fetch(`/api/jobs/${jobId}/description`);
        const data = await res.json();
        log('toggleDesc', `loaded ${data.description?.length || 0} chars`);
        box.textContent    = data.description || '(no description)';
        box.dataset.loaded = '1';
      } catch(err) {
        logErr('toggleDesc', `fetch threw for job=${jobId}:`, err);
        box.textContent = 'Failed to load description.';
      }
    }
    box.classList.remove('hidden');
  } else {
    box.classList.add('hidden');
  }
}


// ── Init (job detail + resumes pages) ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  log('init', 'DOMContentLoaded fired');
  initTabs();
  initAddModeToggle();

  if (window.location.hash) activateTab(window.location.hash.replace('#', ''));

  document.querySelectorAll('input[name="provider"]').forEach(radio => {
    radio.addEventListener('change', updateProviderModelRow);
  });
  updateProviderModelRow();

  ['filter-date-from', 'filter-date-to'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', applyFilters);
  });

  const addJobForm = document.getElementById('add-job-form');
  if (addJobForm) {
    log('init', 'binding #add-job-form submit');
    addJobForm.addEventListener('submit', addJob);
  } else {
    log('init', '#add-job-form not on this page');
  }
});


// ── Jobs list state ───────────────────────────────────────────────────────────
let _currentPage = 1;
let _perPage     = 25;
let _searchTimer = null;


// ── Date filter mode toggle ───────────────────────────────────────────────────
let _dateMode = 'simple';

function toggleDateMode() {
  _dateMode = _dateMode === 'simple' ? 'advanced' : 'simple';
  const simple = document.getElementById('filter-date');
  const fromEl = document.getElementById('filter-date-from');
  const toEl   = document.getElementById('filter-date-to');
  const sep    = document.getElementById('date-range-sep');
  const btn    = document.getElementById('date-mode-btn');
  if (_dateMode === 'advanced') {
    if (simple) simple.style.display = 'none';
    if (fromEl) fromEl.style.display = '';
    if (toEl)   toEl.style.display   = '';
    if (sep)    sep.style.display    = '';
    if (btn)  { btn.title = 'Switch to simple mode'; btn.textContent = '\u2715 range'; }
    if (simple) simple.value = '';
  } else {
    if (simple) simple.style.display = '';
    if (fromEl) fromEl.style.display = 'none';
    if (toEl)   toEl.style.display   = 'none';
    if (sep)    sep.style.display    = 'none';
    if (btn)  { btn.title = 'Switch to date range mode'; btn.textContent = '\u22ef'; }
    if (fromEl) fromEl.value = '';
    if (toEl)   toEl.value   = '';
  }
  applyFilters();
}


// ── Fetch jobs ────────────────────────────────────────────────────────────────
async function fetchJobs() {
  const search    = (document.getElementById('filter-search')    || {value:''}).value.trim();
  const addedDays = (document.getElementById('filter-date')      || {value:''}).value;
  const dateFrom  = (document.getElementById('filter-date-from') || {value:''}).value;
  const dateTo    = (document.getElementById('filter-date-to')   || {value:''}).value;
  const status    = (document.getElementById('filter-status')    || {value:''}).value;
  const score     = (document.getElementById('filter-score')     || {value:''}).value;
  const provider  = (document.getElementById('filter-provider')  || {value:''}).value;

  const params = new URLSearchParams({ page: _currentPage, per_page: _perPage });
  if (search)    params.set('search',     search);
  if (addedDays) params.set('added_days', addedDays);
  if (dateFrom)  params.set('date_from',  dateFrom);
  if (dateTo)    params.set('date_to',    dateTo);
  if (status)    params.set('status',     status);
  if (score)     params.set('score',      score);
  if (provider)  params.set('provider',   provider);

  const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
  history.pushState({}, '', newURL);

  const clearBtn = document.getElementById('clear-btn');
  if (clearBtn) clearBtn.style.display =
    (search || status || score || provider || addedDays || dateFrom || dateTo) ? 'inline-flex' : 'none';

  log('fetchJobs', `page=${_currentPage} per_page=${_perPage} search=${search} status=${status} score=${score} provider=${provider}`);
  showLoading(true);

  try {
    const res = await fetch('/api/jobs/list?' + params);
    if (!res.ok) {
      let errMsg = `Server error ${res.status}`;
      try { const d = await res.json(); errMsg = d.error || errMsg; } catch(_) {}
      logErr('fetchJobs', `HTTP ${res.status}: ${errMsg}`);
      showError(`Failed to load jobs: ${errMsg}`);
      return;
    }
    let data;
    try { data = await res.json(); } catch(e) {
      logErr('fetchJobs', 'failed to parse JSON response:', e);
      showError('Failed to load jobs: server returned invalid data.');
      return;
    }
    if (typeof data.total === 'undefined' || !Array.isArray(data.jobs)) {
      logErr('fetchJobs', 'unexpected response shape:', data);
      showError('Failed to load jobs: unexpected response from server.');
      return;
    }
    log('fetchJobs', `total=${data.total} pages=${data.total_pages}`);
    renderJobs(data);
  } catch(err) {
    logErr('fetchJobs', 'fetch threw:', err);
    showError('Could not reach the server. Is Job Matcher still running?');
  }
}


// ── Render jobs list ──────────────────────────────────────────────────────────
function renderJobs(data) {
  log('renderJobs', `total=${data.total} jobs=${data.jobs?.length}`);
  showLoading(false);

  const list       = document.getElementById('jobs-list');
  const noResults  = document.getElementById('no-results');
  const emptyState = document.getElementById('empty-state');
  const pagBar     = document.getElementById('pagination-bar');

  if (!list) return;

  if (data.total === 0 && !hasActiveFilter()) {
    list.innerHTML = '';
    if (noResults)  noResults.classList.add('hidden');
    if (emptyState) emptyState.classList.remove('hidden');
    if (pagBar)     pagBar.style.display = 'none';
    return;
  }
  if (data.jobs.length === 0) {
    list.innerHTML = '';
    if (noResults)  noResults.classList.remove('hidden');
    if (emptyState) emptyState.classList.add('hidden');
    if (pagBar)     pagBar.style.display = 'none';
    return;
  }
  if (noResults)  noResults.classList.add('hidden');
  if (emptyState) emptyState.classList.add('hidden');

  list.innerHTML = data.jobs.map(job => {
    if (!job || typeof job.id === 'undefined') { logErr('renderJobs', 'malformed job:', job); return ''; }
    return TMPL.jobListItem(job);
  }).join('');

  renderPagination(data);
  list.querySelectorAll('a.job-item').forEach(link => {
    link.addEventListener('click', () => saveFilterState());
  });
}


// ── Render pagination ─────────────────────────────────────────────────────────
function renderPagination(data) {
  log('renderPagination', `page=${data.page}/${data.total_pages} total=${data.total}`);
  const pagBar    = document.getElementById('pagination-bar');
  const info      = document.getElementById('pagination-info');
  const indicator = document.getElementById('page-indicator');
  const prevBtn   = document.getElementById('prev-btn');
  const nextBtn   = document.getElementById('next-btn');
  if (!pagBar) return;

  const perPage    = data.per_page === 0 ? data.total : data.per_page;
  const start      = ((data.page - 1) * perPage) + 1;
  const end        = Math.min(data.page * perPage, data.total);
  const totalPages = data.total_pages;

  pagBar.style.display = data.total > 0 ? 'flex' : 'none';
  if (info)      info.textContent      = `Showing ${start}\u2013${end} of ${data.total} job${data.total !== 1 ? 's' : ''}`;
  if (indicator) indicator.textContent = totalPages > 1 ? `Page ${data.page} of ${totalPages}` : '';
  if (prevBtn)   prevBtn.disabled      = data.page <= 1;
  if (nextBtn)   nextBtn.disabled      = data.page >= totalPages;
  if (totalPages <= 1 && _perPage !== 0) pagBar.style.display = 'none';
}


// ── Loading / error states ────────────────────────────────────────────────────
function showLoading(show) {
  log('showLoading', show);
  const loading = document.getElementById('jobs-loading');
  const list    = document.getElementById('jobs-list');
  const errBox  = document.getElementById('jobs-error');
  if (loading) loading.style.display = show ? 'block' : 'none';
  if (errBox)  errBox.classList.add('hidden');
  if (list && show) list.innerHTML = '';
}

function showError(msg) {
  showLoading(false);
  const errBox  = document.getElementById('jobs-error');
  const errText = document.getElementById('jobs-error-text');
  const list    = document.getElementById('jobs-list');
  const pagBar  = document.getElementById('pagination-bar');
  const noRes   = document.getElementById('no-results');
  const empty   = document.getElementById('empty-state');
  if (list)   list.innerHTML        = '';
  if (pagBar) pagBar.style.display  = 'none';
  if (noRes)  noRes.classList.add('hidden');
  if (empty)  empty.classList.add('hidden');
  if (errBox && errText) {
    errText.textContent = msg;
    errBox.classList.remove('hidden');
  } else {
    toast(msg, 'error');
  }
  logErr('showError', msg);
}

function hasActiveFilter() {
  return ['filter-search','filter-status','filter-score','filter-provider',
          'filter-date','filter-date-from','filter-date-to']
    .some(id => { const el = document.getElementById(id); return el && el.value !== ''; });
}


// ── Filter actions ────────────────────────────────────────────────────────────
function applyFilters()          { _currentPage = 1; fetchJobs(); }
function applyFiltersDebounced() { clearTimeout(_searchTimer); _searchTimer = setTimeout(applyFilters, 300); }

function clearSearch() {
  const el  = document.getElementById('filter-search');
  const btn = document.getElementById('search-clear-btn');
  if (el) { el.value = ''; el.focus(); }
  if (btn) btn.style.display = 'none';
  applyFilters();
}

function updateSearchClearBtn() {
  const el  = document.getElementById('filter-search');
  const btn = document.getElementById('search-clear-btn');
  if (el && btn) btn.style.display = el.value.trim() ? '' : 'none';
}

function changePage(dir) { _currentPage += dir; fetchJobs(); }

function changePerPage() {
  const sel = document.getElementById('per-page');
  _perPage  = sel ? parseInt(sel.value) : 25;
  _currentPage = 1;
  fetchJobs();
  log('changePerPage', `perPage=${_perPage}`);
}

function clearFilters() {
  ['filter-search','filter-status','filter-score','filter-provider',
   'filter-date','filter-date-from','filter-date-to'].forEach(id => {
    const el = document.getElementById(id); if (el) el.value = '';
  });
  applyFilters();
}


// ── Filter state persistence ──────────────────────────────────────────────────
function saveFilterState() {
  const state = {
    search:   (document.getElementById('filter-search')   || {value:''}).value,
    status:   (document.getElementById('filter-status')   || {value:''}).value,
    score:    (document.getElementById('filter-score')    || {value:''}).value,
    provider: (document.getElementById('filter-provider') || {value:''}).value,
    page:     _currentPage,
    per_page: _perPage,
  };
  try {
    sessionStorage.setItem('jobFilterState', JSON.stringify(state));
    log('saveFilterState', 'saved:', JSON.stringify(state));
  } catch(e) {
    logErr('saveFilterState', 'sessionStorage write failed:', e);
  }
}

function restoreFromURL() {
  const rawSearch = window.location.search;
  log('restoreFromURL', 'URL search string:', rawSearch);
  const params        = new URLSearchParams(rawSearch);
  const hasFilters    = params.get('search') || params.get('status') || params.get('score') || params.get('provider');
  const hasPageParams = params.get('page') || params.get('per_page');
  log('restoreFromURL', 'hasFilters=' + hasFilters + ' hasPageParams=' + hasPageParams);

  let saved = null;
  if (!hasFilters) {
    try {
      const raw = sessionStorage.getItem('jobFilterState');
      if (raw) { saved = JSON.parse(raw); log('restoreFromURL', 'found sessionStorage state:', JSON.stringify(saved)); }
    } catch(e) { logErr('restoreFromURL', 'sessionStorage read failed:', e); }
  }

  const getValue = key => { const v = params.get(key); return v !== null ? v : (saved ? (saved[key] || '') : ''); };
  const setEl    = (id, val) => {
    const el = document.getElementById(id);
    if (!el) { logErr('restoreFromURL', 'element not found: #' + id); return; }
    el.value = val || '';
    log('restoreFromURL', id + ' = ' + JSON.stringify(el.value));
  };

  setEl('filter-search',    getValue('search'));
  updateSearchClearBtn();
  setEl('filter-status',    getValue('status'));
  setEl('filter-score',     getValue('score'));
  setEl('filter-provider',  getValue('provider'));
  setEl('filter-date',      getValue('added_days'));
  setEl('filter-date-from', getValue('date_from'));
  setEl('filter-date-to',   getValue('date_to'));

  if (getValue('date_from') || getValue('date_to')) { _dateMode = 'simple'; toggleDateMode(); }

  if (hasPageParams) {
    _currentPage = parseInt(params.get('page')) || 1;
    const pp = params.get('per_page');
    _perPage = (pp !== null && pp !== '') ? parseInt(pp) : 25;
  } else if (saved) {
    _currentPage = saved.page || 1;
    _perPage     = (saved.per_page !== undefined && saved.per_page !== null) ? saved.per_page : 25;
  }

  const perPageSel = document.getElementById('per-page');
  if (perPageSel) perPageSel.value = String(_perPage);
  log('restoreFromURL', 'final: page=' + _currentPage + ' perPage=' + _perPage);
}


// ── Init (jobs list page) ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const jobsList = document.getElementById('jobs-list');
  if (!jobsList) return;

  const searchEl = document.getElementById('filter-search');
  if (searchEl) searchEl.addEventListener('input', () => { applyFiltersDebounced(); updateSearchClearBtn(); });

  ['filter-status','filter-score','filter-provider','filter-date'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', applyFilters);
  });

  window.addEventListener('popstate', () => { restoreFromURL(); fetchJobs(); });
  restoreFromURL();
  fetchJobs();
});


// ── Salary estimation ─────────────────────────────────────────────────────────
async function estimateSalary(jobId) {
  log('estimateSalary', 'job ' + jobId);
  const btn = document.getElementById('salary-btn');
  if (btn) { btn.disabled = true; btn.innerHTML = TMPL.spinner('Estimating\u2026'); }

  const provider = document.querySelector('input[name="provider"]:checked')?.value || 'anthropic';
  const fd = new FormData();
  fd.append('provider', provider);

  try {
    const res  = await fetch(`/api/jobs/${jobId}/estimate-salary`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) {
      toast(data.error || 'Salary estimation failed', 'error');
      if (btn) { btn.disabled = false; btn.innerHTML = TMPL.salaryBtn(); }
      return;
    }
    toast('Salary estimate saved', 'success');
    setTimeout(() => refreshJobDetailPage(), 800);
  } catch(e) {
    logErr('estimateSalary', 'fetch threw:', e);
    toast('Salary estimation failed', 'error');
    if (btn) { btn.disabled = false; btn.innerHTML = TMPL.salaryBtn(); }
  }
}

async function clearSalaryEstimate(jobId) {
  log('clearSalaryEstimate', 'job ' + jobId);
  try {
    await fetch(`/api/jobs/${jobId}/salary-estimate`, { method: 'DELETE' });
    refreshJobDetailPage();
  } catch(e) {
    logErr('clearSalaryEstimate', 'fetch threw:', e);
    refreshJobDetailPage();
  }
}

async function rerunSalaryEstimate(jobId) {
  log('rerunSalaryEstimate', 'job ' + jobId);
  const btn = document.querySelector(`button[onclick="rerunSalaryEstimate(${jobId})"]`);
  if (btn) { btn.disabled = true; btn.innerHTML = TMPL.spinnerSm('re-running\u2026'); }

  try { await fetch(`/api/jobs/${jobId}/salary-estimate`, { method: 'DELETE' }); }
  catch(e) { logErr('rerunSalaryEstimate', 'DELETE threw:', e); }

  const provider = document.querySelector('input[name="provider"]:checked')?.value || 'anthropic';
  const fd = new FormData();
  fd.append('provider', provider);

  try {
    const res  = await fetch(`/api/jobs/${jobId}/estimate-salary`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) {
      toast(data.error || 'Salary estimation failed', 'error');
      if (btn) { btn.disabled = false; btn.innerHTML = TMPL.rerunBtn(); }
    } else {
      toast('Salary updated', 'success');
      setTimeout(() => refreshJobDetailPage(), 600);
    }
  } catch(e) {
    logErr('rerunSalaryEstimate', 'POST threw:', e);
    refreshJobDetailPage();
  }
}


// ══════════════════════════════════════════════════════════════════════════════
// ── Shared frontend — v5 additions ───────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════


// ── HTML escape helper ────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}


// ── Duration formatter ────────────────────────────────────────────────────────
function formatDuration(seconds) {
  if (!seconds) return '';
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r > 0 ? `${m}m ${r}s` : `${m}m`;
}


// ══════════════════════════════════════════════════════════════════════════════
// ── HTML Templates ────────────────────────────────────────────────────────────
// Every HTML string in the app lives here. Controllers call TMPL.*() and
// assign the result to innerHTML — never scatter markup into logic functions.
// ══════════════════════════════════════════════════════════════════════════════

const TMPL = {

  // ── Primitives ──────────────────────────────────────────────────────────────

  // Loading spinner, optionally with a label
  spinner(label = '') {
    return label
      ? `<span class="spinner"></span> ${label}`
      : `<span class="spinner"></span>`;
  },

  // Small inline spinner (used in re-run buttons)
  spinnerSm(label = '') {
    const s = `<span class="spinner" style="width:10px;height:10px;border-width:1.5px;vertical-align:middle;"></span>`;
    return label ? `${s} ${label}` : s;
  },

  // Salary button label (used when resetting after failed estimate)
  salaryBtn() {
    return '\uD83D\uDCB0 Estimate Salary';
  },

  // Generic empty / error state panel
  emptyPanel(msg) {
    return `<div class="empty"><p>${escHtml(msg)}</p></div>`;
  },

  // Evidence snippet toggle button label
  snippetToggle(open) {
    return open ? '&#9660; evidence' : '&#9658; evidence';
  },

  // Save application button label (reset after save)
  saveBtn() { return 'Save'; },

  // Re-run salary button label (reset after failed re-run)
  rerunBtn() { return 're-run'; },

  // Text quality warning block (reused in analysis tab + description tab)
  tqWarning(tq, style) {
    if (!tq || !tq.level || tq.level === 'ok') return '';
    return `
      <div class="text-quality-warning level-${escHtml(tq.level)}"${style ? ` style="${style}"` : ''}>
        <span class="text-quality-title">\u26a0 Job description quality warning</span>
        <ul class="text-quality-issues">
          ${(tq.issues || []).map(i => `<li>${escHtml(i)}</li>`).join('')}
        </ul>
        <small class="text-dim">${tq.char_count || 0} chars \u00b7 ${tq.tech_keywords || 0} tech keywords detected</small>
      </div>`;
  },

  // All provider radios for the available providers
  providerRadios(providers, lastProvider) {
    return [
      ['anthropic', 'Anthropic', providers.has_anthropic],
      ['openai',    'OpenAI',    providers.has_openai],
      ['gemini',    'Gemini',    providers.has_gemini],
      ['ollama',    'Ollama',    providers.has_ollama],
    ].filter(([,, has]) => has).map(([val, label]) => {
      const checked = lastProvider === val || (!lastProvider && val === 'anthropic');
      return TMPL.providerRadio(val, label, checked);
    }).join('');
  },

  // One provider radio + label pair
  providerRadio(val, label, checked) {
    return `<input type="radio" name="provider" id="p-${val}" value="${val}" ${checked ? 'checked' : ''} /><label for="p-${val}">${label}</label>`;
  },

  // Resume <select> or no-resumes message
  resumeSelect(resumes, lastResumeId) {
    if (!resumes.length) {
      return `<div class="text-dim text-sm" style="padding: 8px 0;">No resumes yet \u2014 <a href="/resumes" style="color:var(--amber);">add one first</a></div>`;
    }
    return `<select id="analyze-resume">${
      resumes.map(r => `<option value="${r.id}" ${r.id === lastResumeId ? 'selected' : ''}>${escHtml(r.label)}</option>`).join('')
    }</select>`;
  },

  // Analysis mode <select> options
  modeOptions(selected) {
    return [
      ['fast',     'Fast \u2014 ~30s \u00b7 no suggestions'],
      ['standard', 'Standard \u2014 ~90s \u00b7 suggestions on'],
      ['detailed', 'Detailed \u2014 ~4min \u00b7 all skills'],
    ].map(([v, l]) => `<option value="${v}" ${selected === v ? 'selected' : ''}>${l}</option>`).join('');
  },

  // Application status <select> options
  appStatusOptions(current) {
    return [
      ['not_applied',  'Not Applied'],
      ['applied',      'Applied'],
      ['interviewing', 'Interviewing'],
      ['offered',      'Offered'],
      ['rejected',     'Rejected'],
    ].map(([v, l]) => `<option value="${v}" ${(current || 'not_applied') === v ? 'selected' : ''}>${l}</option>`).join('');
  },

  // Salary action: estimate/extract button or cached + re-run label
  salaryAction(sal, jobId, hasInJd) {
    if (sal) {
      return `<span class="text-xs text-dim" style="font-family: var(--font-mono);">
        Salary ${escHtml(sal.source || 'estimate')} cached \u2014
        <button class="btn-inline-mute" onclick="rerunSalaryEstimate(${jobId})">re-run</button>
      </span>`;
    }
    return `<button id="salary-btn" class="btn btn-ghost btn-sm" onclick="estimateSalary(${jobId})">\uD83D\uDCB0 ${hasInJd ? 'Extract' : 'Estimate'} Salary</button>`;
  },

  // Salary estimate badge (shown in page header when estimate exists)
  salaryBadge(sal, jobId) {
    if (!sal) return '';
    const icon   = sal.source === 'posted' ? '\uD83D\uDCCC' : '~';
    const minFmt = (sal.min || 0).toLocaleString();
    const maxFmt = (sal.max || 0).toLocaleString();
    const model  = sal.llm_model ? ` (${escHtml(sal.llm_model)})` : '';
    return `
      <div class="salary-badge-wrap" style="margin-top: 8px;">
        <span class="salary-badge salary-${escHtml(sal.confidence || '')}">
          ${icon} $${minFmt} \u2013 $${maxFmt} / yr
          <span class="salary-confidence">
            ${escHtml(sal.source || 'estimated')} \u00b7 ${escHtml(sal.confidence || '')} confidence
            \u00b7 ${escHtml(sal.llm_provider || '')}${model}
          </span>
        </span>
        <button class="btn-inline-mute" onclick="clearSalaryEstimate(${jobId})" title="Clear">\u2715</button>
      </div>`;
  },

  // Job URL link or "pasted description" label
  jobUrl(url, jobId) {
    const editBtn = jobId
      ? `<button class="btn btn-ghost btn-xs" style="margin-left:6px;padding:1px 6px;font-size:11px;" onclick="toggleUrlEdit()" title="Edit source URL">&#9998;</button>`
      : '';
    if (!url || url.startsWith('manual://')) {
      return `<span style="color: var(--text-mute);">\u00b7 pasted description${editBtn}</span>`;
    }
    return `<span>\u00b7 <a href="${escHtml(url)}" target="_blank" style="color: var(--amber);">View Original \u2197</a>${editBtn}</span>`;
  },

  // Score explainer <details> block (static content, no dynamic data)
  scoreExplainer() {
    return `
      <details class="score-explainer">
        <summary>How are scores calculated?</summary>
        <div class="score-explainer-body">
          <div class="score-explainer-section">
            <div class="score-explainer-label">Raw Score</div>
            <div class="score-explainer-desc">LLM evaluates overall resume-to-job fit on a 1\u20135 scale.</div>
          </div>
          <div class="score-explainer-section">
            <div class="score-explainer-label">Adjusted Score</div>
            <div class="score-explainer-desc">Raw score minus penalties based on detected skill gaps.</div>
          </div>
          <table class="score-explainer-table">
            <thead><tr><th>Gap Type</th><th>Penalty</th><th>Cap</th><th>Examples</th></tr></thead>
            <tbody>
              <tr><td><span class="penalty-tag penalty-blocker">Blocker</span></td><td>\u22122 each</td><td>max \u22123</td><td>Clearance, citizenship, mandatory certs</td></tr>
              <tr><td><span class="penalty-tag penalty-major">Major</span></td>    <td>\u22121 each</td><td>max \u22122</td><td>Significant missing skills</td></tr>
              <tr><td><span class="penalty-tag penalty-minor">Minor</span></td>    <td>\u22121 per 2</td><td>max \u22121</td><td>Nice-to-have or learnable gaps</td></tr>
              <tr><td><span class="penalty-tag penalty-count">Count</span></td>    <td>\u22121 flat</td><td>max \u22121</td><td>More than 6 total gaps</td></tr>
            </tbody>
          </table>
          <div class="score-explainer-example">
            Example: Raw 4 &middot; 1 blocker (\u22122) &middot; 3 minors (\u22121) = Adjusted <strong>1</strong>
          </div>
        </div>
      </details>`;
  },



  // Past analyses section header + blocks
  analysesSection(analyses, mode) {
    if (!analyses.length) return TMPL.emptyPanel('No analyses yet. Select a resume and run your first analysis above.');
    return `<div class="section-title" style="margin-bottom: 12px;">Past Analyses</div>
      ${analyses.map((a, idx) => TMPL.analysisBlock(a, idx)).join('')}`;
  },

  // Job company + location meta spans
  jobMeta(company, location) {
    const co  = company  ? `<span>${escHtml(company)}</span>`          : '';
    const loc = location ? `<span>\u00b7 ${escHtml(location)}</span>` : '';
    return co + loc;
  },

  // Compare tab button (shown only when comparison data exists)
  compareTabBtn(comp) {
    return comp ? `<button class="tab" data-tab="compare">Compare</button>` : '';
  },

  // ── Job detail sub-sections ─────────────────────────────────────────────────

  // Page header: title, status badge, company/location, URL, salary
  pageHeader(job, statusClass, statusLabel, urlHtml, salaryHtml) {
    const currentUrl = (!job.url || job.url.startsWith('manual://')) ? '' : job.url;
    return `
      <div style="margin-bottom: 20px;">
        <a href="/" class="btn btn-ghost btn-sm">\u2190 Back</a>
      </div>
      <div class="page-header">
        <div class="flex items-center gap-10" style="margin-bottom: 8px;">
          <h2 style="flex:1; min-width:0;">${escHtml(job.title || 'Untitled Job')}</h2>
          <span id="status-badge" class="status-badge status-${escHtml(statusClass)}">${escHtml(statusLabel)}</span>
          <button class="btn btn-danger btn-sm" onclick="deleteJob(${job.id})">Delete</button>
        </div>
        <div class="flex gap-10 text-dim text-mono text-xs">
          ${TMPL.jobMeta(job.company, job.location)}
          ${urlHtml}
        </div>
        <div id="url-edit-row" style="display:none; margin-top:8px; display:none;">
          <div class="flex gap-10 items-center" style="max-width:600px;">
            <input id="url-edit-input" type="text" value="${escHtml(currentUrl)}"
                   placeholder="https://jobs.example.com/posting/12345"
                   style="flex:1; font-size:12px;" />
            <button class="btn btn-primary btn-sm" onclick="saveJobUrl(${job.id}, document.getElementById('url-edit-input').value)">Save</button>
            <button class="btn btn-ghost btn-sm" onclick="toggleUrlEdit()">Cancel</button>
          </div>
        </div>
        ${salaryHtml}
      </div>`;
  },

  // Progress bar block (static structure, shown/hidden via JS)
  progressBar() {
    return `
      <div id="analysis-progress" class="analysis-progress hidden">
        <div class="progress-header">
          <span class="progress-spinner"></span>
          <span id="progress-label">Analyzing...</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" id="progress-fill"></div>
        </div>
        <div class="progress-meta" id="progress-meta"></div>
      </div>`;
  },

  // Cloud model select row
  cloudModelRow(d) {
    return `
      <div class="form-row" id="cloud-model-row" style="display:none; margin:0;">
        <label>Model</label>
        <select id="cloud-model-select"
          data-anthropic="${escHtml(d.anthropic_model || '')}"
          data-openai="${escHtml(d.openai_model || '')}"
          data-gemini="${escHtml(d.gemini_model || '')}">
          <option value="">Loading\u2026</option>
        </select>
      </div>`;
  },

  // Ollama model select row
  ollamaModelRow(d) {
    return `
      <div class="form-row" id="ollama-model-row" style="display:none; margin:0;">
        <label>Ollama Model</label>
        <select id="ollama-model-select">
          <option value="${escHtml(d.ollama_model || '')}">${escHtml(d.ollama_model || '')}</option>
        </select>
      </div>`;
  },

  // "Run New Analysis" card
  analysisCard(job, d, resumes, providerRadios, resumeSelectHtml, modeOpts, salaryBtnHtml, tqWarning) {
    const mode = d.analysis_mode || 'standard';
    return `
      <div class="card mb-16">
        <div class="section-header">
          <span class="section-title">Run New Analysis</span>
        </div>
        <div class="form-grid" style="margin-bottom: 14px;">
          <div class="form-row" style="margin:0;">
            <label>Resume Version</label>
            ${resumeSelectHtml}
          </div>
          <div style="display:flex; flex-direction:column; gap:10px;">
            <div class="form-row" style="margin:0;">
              <label>LLM Provider</label>
              <div class="llm-toggle">${providerRadios}</div>
            </div>
            ${TMPL.cloudModelRow(d)}
            ${TMPL.ollamaModelRow(d)}
            <div class="form-row" style="margin:0;">
              <label>Analysis Mode</label>
              <select id="analysis-mode-select">${modeOpts}</select>
            </div>
          </div>
        </div>
        <button id="analyze-btn" class="btn btn-primary"
                onclick="analyzeJob(${job.id})"
                data-mode="${escHtml(mode)}"
                ${!resumes.length ? 'disabled' : ''}>Run Analysis</button>
        <div style="margin-top: 10px; display: flex; align-items: center; gap: 10px;">
          ${salaryBtnHtml}
        </div>
        ${TMPL.progressBar()}
        ${TMPL.scoreExplainer()}
        ${tqWarning}
      </div>`;
  },

  // Application tab content
  applicationTab(job, app) {
    return `
      <div class="tab-content" data-tab-content="application">
        <div class="card">
          <div class="form-row">
            <label>Application Status</label>
            <select id="app-status">
              ${TMPL.appStatusOptions(app.status)}
            </select>
          </div>
          <div class="section-title" style="margin: 18px 0 12px;">Recruiter Info</div>
          <div class="form-grid">
            <div class="form-row" style="margin:0 0 14px 0;">
              <label>Name</label>
              <input type="text" id="recruiter-name" value="${escHtml(app.recruiter_name || '')}" placeholder="Jane Smith" />
            </div>
            <div class="form-row" style="margin:0 0 14px 0;">
              <label>Email</label>
              <input type="email" id="recruiter-email" value="${escHtml(app.recruiter_email || '')}" placeholder="jane@company.com" />
            </div>
            <div class="form-row" style="margin:0 0 14px 0;">
              <label>Phone</label>
              <input type="text" id="recruiter-phone" value="${escHtml(app.recruiter_phone || '')}" placeholder="+1 (555) 000-0000" />
            </div>
          </div>
          <div class="form-row">
            <label>Notes</label>
            <textarea id="app-notes" placeholder="Follow-up dates, referrals, compensation details, interview notes\u2026">${escHtml(app.notes || '')}</textarea>
          </div>
          <button id="save-app-btn" class="btn btn-primary" onclick="saveApplication(${job.id})">Save</button>
        </div>
      </div>`;
  },

  // Description tab content
  descriptionTab(job, tq) {
    return `
      <div class="tab-content" data-tab-content="description">
        <div class="card">
          ${TMPL.tqWarning(tq, 'margin-bottom:14px;')}
          <div class="flex justify-between items-center mb-16">
            <span class="section-title">Raw Scraped Description</span>
            <button class="btn btn-ghost btn-sm" onclick="toggleDesc(${job.id})">Toggle</button>
          </div>
          <div id="desc-box" class="desc-box hidden"></div>
        </div>
      </div>`;
  },



  // Tab container: wraps all tabs and their content panels
  tabContainer(comp, analysisTabContent, applicationTab, descriptionTab, compareTab) {
    return `
      <div class="tab-container">
        <div class="tabs">
          <button class="tab active" data-tab="analysis">Analysis</button>
          <button class="tab" data-tab="application">Application</button>
          <button class="tab" data-tab="description">Job Description</button>
          ${TMPL.compareTabBtn(comp)}
        </div>
        <div class="tab-content active" data-tab-content="analysis">
          ${analysisTabContent}
        </div>
        ${applicationTab}
        ${descriptionTab}
        ${compareTab}
      </div>`;
  },



  // ── Jobs list ───────────────────────────────────────────────────────────────

  // One row in the jobs list
  jobListItem(job) {
    const score        = job.adjusted_score || job.best_score;
    const scoreBadge   = score
      ? `<div class="score-badge score-${score}">${score}</div>`
      : `<div class="score-badge score-none">\u2014</div>`;

    const isManual     = job.is_manual || (job.url || '').startsWith('manual://');
    const sourceTag    = `<span class="provider-tag">${isManual ? 'manual' : 'scraped'}</span>`;
    const modelTag     = (job.provider && job.last_model)
      ? `<span class="provider-tag" title="${job.provider} \u00b7 ${job.last_model}">${job.provider} \u00b7 ${job.last_model}</span>`
      : job.provider
        ? `<span class="provider-tag">${job.provider}</span>`
        : '';
    const recruiterBadge = job.has_recruiter
      ? `<span class="provider-tag" style="cursor:pointer;"
             onclick="event.preventDefault();event.stopPropagation();window.location='/job/${job.id}#application';"
             title="Recruiter contact saved">\uD83D\uDC64</span>`
      : '';

    const status       = job.status || 'not_applied';
    const statusLabel  = status.replace(/_/g, ' ');
    const company      = job.company  || '';
    const location     = job.location || '';
    const locationFlag = getLocationFlag(location);
    const locationBadge = locationFlag === 'N/A'
      ? `<span class="location-tag"><span class="location-code">N/A</span></span>`
      : locationFlag
        ? `<span class="location-tag" title="${location}"><span class="location-code">${locationFlag}</span> ${location}</span>`
        : '';
    const metaBase  = company ? escHtml(company) : (isManual ? 'pasted description' : (job.url || '').substring(0, 60) + '\u2026');
    const dateLabel = job.scraped_at ? `<span class="date-tag">added ${formatJobDate(job.scraped_at)}</span>` : '';
    const meta      = [metaBase, locationBadge, dateLabel].filter(Boolean).join(' \u00b7 ');
    const rawTitle  = job.title || (job.url
      ? (() => { try { return new URL(job.url).hostname; } catch(e) { return 'Untitled Job'; } })()
      : 'Untitled Job');
    const title     = escHtml(rawTitle);

    return `<a href="/job/${job.id}" class="job-item" style="text-decoration:none;">
      <div>${scoreBadge}</div>
      <div class="job-item-info">
        <div class="job-title">${title}</div>
        <div class="job-meta">${meta}</div>
      </div>
      <div class="job-item-right">
        ${recruiterBadge}
        ${sourceTag}
        ${modelTag}
        <span class="status-badge status-${status}">${statusLabel}</span>
      </div>
    </a>`;
  },


  // ── Resumes page ────────────────────────────────────────────────────────────

  resumeCard(r) {
    const date  = r.created_at ? r.created_at.slice(0, 16) : '';
    const chars = r.char_count != null ? r.char_count : (r.content ? r.content.length : 0);
    return `
      <div class="card card-sm" style="margin-bottom: 10px;">
        <div class="flex justify-between items-center">
          <div>
            <div style="font-weight: 500; font-size: 14px;">${escHtml(r.label)}</div>
            <div class="text-xs text-dim text-mono mt-4">${escHtml(date)} \u00b7 ${chars} chars</div>
          </div>
          <button class="btn btn-danger btn-sm" onclick="deleteResumeAndRefresh(${r.id})">Delete</button>
        </div>
      </div>`;
  },

  resumesEmpty() {
    return `
      <div class="empty">
        <div class="empty-icon">\u25c8</div>
        <p>No resumes saved yet.<br>Add your first resume version above.</p>
      </div>`;
  },


  // ── Preview page ────────────────────────────────────────────────────────────

  previewBlockerList(keywords) {
    return keywords.map(kw => `<li>${escHtml(kw)}</li>`).join('');
  },

  previewTqIssues(issues) {
    return issues.map(i => `<li>${escHtml(i)}</li>`).join('');
  },


  // ── Analysis block ──────────────────────────────────────────────────────────

  analysisBlock(a, idx) {
    const pb = a.penalty_breakdown || {};

    const scoreDiff = a.adjusted_score !== a.score
      ? `<span class="score-arrow">\u2192</span>
         <div class="score-badge score-${a.adjusted_score}" title="Penalty-adjusted score">${a.adjusted_score}</div>
         <span class="score-diff score-diff-down">-${a.score - a.adjusted_score}</span>`
      : `<span class="score-diff score-diff-same">no penalty</span>`;

    const model    = a.llm_model ? ` (${escHtml(a.llm_model)})` : '';
    const dur      = a.duration_seconds ? ` \u00b7 ${formatDuration(a.duration_seconds)}` : '';
    const modeStr  = a.analysis_mode ? ` \u00b7 ${escHtml(a.analysis_mode)}` : '';
    const metaLine = `${escHtml(a.resume_label||'')} \u00b7 ${escHtml(a.llm_provider||'')}${model} \u00b7 ${escHtml((a.created_at||'').slice(0,16))}${dur}${modeStr}`;

    const fallbackHtml = a.used_fallback ? `
      <div class="analysis-warning" style="margin-top:8px;">
        \u26a0 This analysis used a fallback due to repeated model output errors.
        Results may be incomplete. Try re-running the analysis.
        ${a.validation_errors ? `<small class="text-dim"> (${escHtml(a.validation_errors)})</small>` : ''}
      </div>` : '';

    let breakdownHtml = '';
    if (pb && pb.total_penalty) {
      const rows = (severity, label, penalty) =>
        (a.missing_skills||[])
          .filter(s => s && s.severity === severity && s.requirement_type !== 'bonus')
          .map(s => `<div class="breakdown-penalty-row">
                       <span class="penalty-tag penalty-${severity}">${label}</span>
                       <span>${escHtml(s.skill)}</span><span>${penalty}</span>
                     </div>`).join('');
      breakdownHtml = `
        <details class="score-breakdown" style="margin-top:8px;">
          <summary>Score Breakdown</summary>
          <div class="score-breakdown-body">
            <div class="breakdown-row"><span>Base score (LLM)</span><span>${a.score} / 5</span></div>
            <div class="breakdown-penalties">
              ${rows('blocker', 'BLOCKER', '\u22122')}
              ${rows('major',   'MAJOR',   '\u22121')}
            </div>
            <div class="breakdown-row breakdown-total-row"><span>Adjusted score</span><span>${a.adjusted_score} / 5</span></div>
            <div class="breakdown-footer">${escHtml(a.llm_provider||'')}${model}</div>
          </div>
        </details>`;
    }

    const skillPill = (s, side) => {
      if (!s || typeof s !== 'object') {
        const cls = side === 'matched' ? 'pill-green' : 'pill-red';
        return `<span class="pill ${cls}">${escHtml(String(s))}</span>`;
      }
      if (side === 'matched') {
        const mt          = s.match_type || 'exact';
        const snippetHtml = s.jd_snippet ? `
          <button class="snippet-toggle" onclick="toggleSnippets(this)">${TMPL.snippetToggle(false)}</button>
          <div class="evidence-snippets" style="display:none;">
            <div class="snippet jd-snippet"><span class="snippet-label">JD:</span> &#8220;${escHtml(s.jd_snippet)}&#8221;</div>
            ${s.resume_snippet ? `<div class="snippet resume-snippet"><span class="snippet-label">Resume:</span> &#8220;${escHtml(s.resume_snippet)}&#8221;</div>` : ''}
          </div>` : '';
        return `<div class="skill-evidence">
                  <span class="pill pill-green match-${escHtml(mt)}">${escHtml(s.skill)}</span>
                  <span class="match-type-badge match-${escHtml(mt)}">${escHtml(mt)}</span>
                  ${snippetHtml}
                </div>`;
      } else {
        const sev       = s.severity || 'minor';
        const pillClass = sev === 'blocker' ? 'pill-red' : sev === 'major' ? 'pill-orange' : 'pill-dim';
        const reqType   = s.requirement_type
          ? `<span class="req-type req-${escHtml(s.requirement_type)}">${escHtml(s.requirement_type.toUpperCase())}</span>` : '';
        const snippetHtml = s.jd_snippet ? `
          <button class="snippet-toggle" onclick="toggleSnippets(this)">${TMPL.snippetToggle(false)}</button>
          <div class="evidence-snippets" style="display:none;">
            <div class="snippet jd-snippet"><span class="snippet-label">JD:</span> &#8220;${escHtml(s.jd_snippet)}&#8221;</div>
          </div>` : '';
        return `<div class="skill-evidence">
                  <span class="pill ${pillClass}" title="${escHtml(sev)}${s.requirement_type ? ' \u00b7 ' + escHtml(s.requirement_type) : ''}">
                    ${reqType}${escHtml(s.skill)}
                  </span>
                  ${snippetHtml}
                </div>`;
      }
    };

    const matchedHtml = (a.matched_skills||[]).length ? `
      <div style="margin-top: 10px;">
        <div class="text-xs text-dim" style="margin-bottom: 4px;">Matched skills</div>
        <div class="pill-list">${(a.matched_skills).map(s => skillPill(s, 'matched')).join('')}</div>
      </div>` : '';

    const missingHtml = (a.missing_skills||[]).length ? `
      <div style="margin-top: 10px;">
        <div class="text-xs text-dim" style="margin-bottom: 4px;">Missing / gaps</div>
        <div class="pill-list">${(a.missing_skills).map(s => skillPill(s, 'missing')).join('')}</div>
      </div>` : '';

    const reasoningHtml = a.reasoning ? `<div class="reasoning">${escHtml(a.reasoning)}</div>` : '';

    const suggestionsHtml = (a.suggestions||[]).length ? `
      <div class="suggestions-panel">
        <div class="section-title" style="font-size:11px; margin-bottom:6px;">Resume Suggestions</div>
        <div class="suggestions-disclaimer">Based on your existing experience only \u2014 not skills to fabricate.</div>
        ${(a.suggestions).map(s => {
          const title  = typeof s === 'object' ? (s.title  || 'Suggestion') : 'Suggestion';
          const detail = typeof s === 'object' ? (s.detail || s)            : s;
          const req    = typeof s === 'object' && s.job_requirement
            ? `<div class="suggestion-req">Addresses: &#8220;${escHtml(s.job_requirement)}&#8221;</div>` : '';
          return `<div class="suggestion-item">
                    <div class="suggestion-title">${escHtml(title)}</div>
                    <div class="suggestion-detail">${escHtml(detail)}</div>
                    ${req}
                  </div>`;
        }).join('')}
      </div>` : '';

    return `
      <div class="analysis-block" id="analysis-${a.id}">
        <div class="analysis-header">
          <div class="score-adjusted-wrap">
            <div class="score-badge score-${a.score}" title="Raw LLM score">${a.score}</div>
            ${scoreDiff}
          </div>
          <div class="score-meter" id="meter-${idx + 1}"></div>
          <div class="analysis-meta" style="flex:1;">${metaLine}</div>
          <button class="btn btn-danger btn-sm" style="margin-left:10px;flex-shrink:0;"
                  onclick="deleteAnalysis(${a.id})" title="Delete this analysis">\u2715</button>
        </div>
        ${fallbackHtml}
        ${breakdownHtml}
        ${matchedHtml}
        ${missingHtml}
        ${reasoningHtml}
        ${suggestionsHtml}
      </div>`;
  },


  // ── Compare tab ─────────────────────────────────────────────────────────────

  compareTab(comp) {
    if (!comp) return '';
    const a = comp.resume_a, b = comp.resume_b;
    return `
      <div class="tab-content" data-tab-content="compare">
        <div class="card">
          <div class="compare-panel">
            <div class="compare-header">
              <div class="compare-label"></div>
              <div class="compare-col">${escHtml(a.resume_label || '')}</div>
              <div class="compare-col">${escHtml(b.resume_label || '')}</div>
            </div>
            <div class="compare-row">
              <div class="compare-label">Adjusted Score</div>
              <div class="compare-val score-${a.adjusted_score}">${a.adjusted_score} / 5</div>
              <div class="compare-val score-${b.adjusted_score}">${b.adjusted_score} / 5</div>
            </div>
            <div class="compare-row">
              <div class="compare-label">Matched Skills</div>
              <div class="compare-val">${(a.matched_skills || []).length}</div>
              <div class="compare-val">${(b.matched_skills || []).length}</div>
            </div>
            <div class="compare-row">
              <div class="compare-label">Missing Skills</div>
              <div class="compare-val">${(a.missing_skills || []).length}</div>
              <div class="compare-val">${(b.missing_skills || []).length}</div>
            </div>
            <div class="compare-verdict">
              <span class="verdict-label">Better fit:</span>
              <span class="verdict-winner">${escHtml(comp.better_fit || '')}</span>
              <span class="verdict-reason">${escHtml(comp.better_reason || '')}</span>
            </div>
          </div>
        </div>
      </div>`;
  },


  // ── Job detail page ─────────────────────────────────────────────────────────

  jobDetailPage(d, providers) {
    const job  = d.job || {};
    const app  = d.application || {};
    const sal  = d.salary_estimate;
    const tq   = d.text_quality || {};
    const comp = d.comparison;

    const salaryHtml    = TMPL.salaryBadge(sal, job.id);
    const urlHtml       = TMPL.jobUrl(job.url, job.id);
    const tqWarning     = TMPL.tqWarning(tq, 'margin-top:14px;');

    const lastProvider   = d.last_provider || 'anthropic';
    const providerRadios = TMPL.providerRadios(providers, lastProvider);

    const resumes          = d.resumes || [];
    const resumeSelectHtml = TMPL.resumeSelect(resumes, d.last_resume_id);
    const modeOpts         = TMPL.modeOptions(d.analysis_mode || 'standard');
    const salaryBtnHtml    = TMPL.salaryAction(sal, job.id, d.has_salary_in_jd);
    const analyses         = d.analyses || [];
    const statusClass      = app.status || 'not_applied';
    const statusLabel      = (app.status || 'not applied').replace(/_/g, ' ');

    const analysisTabContent = `
      ${TMPL.analysisCard(job, d, resumes, providerRadios, resumeSelectHtml, modeOpts, salaryBtnHtml, tqWarning)}
      ${TMPL.analysesSection(analyses)}`;

    return `
      ${TMPL.pageHeader(job, statusClass, statusLabel, urlHtml, salaryHtml)}
      ${TMPL.tabContainer(comp, analysisTabContent, TMPL.applicationTab(job, app), TMPL.descriptionTab(job, tq), TMPL.compareTab(comp))}`;
  }

};


// ══════════════════════════════════════════════════════════════════════════════
// ── Page controllers ──────────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════


// ── Active nav ────────────────────────────────────────────────────────────────
function initActiveNav() {
  const path      = window.location.pathname;
  const jobsEl    = document.getElementById('nav-jobs');
  const resumesEl = document.getElementById('nav-resumes');
  if (!jobsEl || !resumesEl) return;
  if (path === '/resumes') resumesEl.classList.add('active');
  else                     jobsEl.classList.add('active');
}


// ── Resumes page ──────────────────────────────────────────────────────────────
async function initResumesPage() {
  const container = document.getElementById('resumes-list');
  if (!container) return;
  log('initResumesPage', 'fetching resumes');
  try {
    const res  = await fetch('/api/resumes/');
    const data = await res.json();
    renderResumesList(data.resumes || []);
  } catch(e) {
    logErr('initResumesPage', 'fetch threw:', e);
    container.innerHTML = TMPL.emptyPanel('Failed to load resumes.');
  }
}

function renderResumesList(resumes) {
  const container = document.getElementById('resumes-list');
  if (!container) return;
  if (!resumes.length) { container.innerHTML = TMPL.resumesEmpty(); return; }
  const countEl = document.createElement('div');
  countEl.className = 'section-title';
  countEl.style.marginBottom = '14px';
  countEl.textContent = `Saved Versions (${resumes.length})`;
  container.innerHTML = '';
  container.appendChild(countEl);
  resumes.forEach(r => container.insertAdjacentHTML('beforeend', TMPL.resumeCard(r)));
}

async function deleteResumeAndRefresh(resumeId) {
  if (!confirm('Delete this resume version?')) return;
  log('deleteResumeAndRefresh', `id=${resumeId}`);
  try {
    const res = await fetch(`/api/resumes/${resumeId}`, { method: 'DELETE' });
    if (res.ok) { toast('Resume deleted', 'info'); await initResumesPage(); }
    else        { toast('Delete failed', 'error'); }
  } catch(e) {
    logErr('deleteResumeAndRefresh', 'fetch threw:', e);
    toast('Network error', 'error');
  }
}

async function addResumeAndRefresh(e) {
  if (e) e.preventDefault();
  const form = document.getElementById('add-resume-form');
  if (!form) return;
  const fd      = new FormData(form);
  const label   = (fd.get('label')   || '').trim();
  const content = (fd.get('content') || '').trim();
  if (!label || !content) { toast('Label and content are required', 'error'); return; }
  const btn = form.querySelector('[type=submit]');
  if (btn) { btn.disabled = true; btn.innerHTML = TMPL.spinner(); }
  try {
    const res  = await fetch('/api/resumes/add', { method: 'POST', body: fd });
    const data = await res.json();
    if (res.ok) {
      toast(`\u2713 Resume "${data.label}" saved`, 'success');
      form.reset();
      await initResumesPage();
    } else {
      toast(data.error || 'Failed', 'error');
    }
  } catch(e) {
    logErr('addResumeAndRefresh', 'fetch threw:', e);
    toast('Network error', 'error');
  }
  if (btn) { btn.disabled = false; btn.textContent = 'Save Resume'; }
}


// ── Job preview page ──────────────────────────────────────────────────────────
async function savePreview() {
  const form = document.getElementById('preview-form');
  if (!form) return;
  const btn = form.querySelector('.btn-primary');
  if (btn) { btn.disabled = true; btn.textContent = 'Saving\u2026'; }
  const fd = new FormData(form);
  try {
    log('savePreview', 'POST /api/jobs/save-preview');
    const res  = await fetch('/api/jobs/save-preview', { method: 'POST', body: fd });
    const data = await res.json();
    log('savePreview', `response status=${res.status}`, data);
    if (res.status === 409) {
      toast('Job already added \u2014 ' + (data.title || ''), 'info');
      if (btn) { btn.disabled = false; btn.textContent = 'Save Job'; }
      return;
    }
    if (!res.ok) {
      toast(data.error || 'Failed to save job', 'error');
      if (btn) { btn.disabled = false; btn.textContent = 'Save Job'; }
      return;
    }
    sessionStorage.removeItem('job_preview');
    window.location.href = '/job/' + data.job_id;
  } catch(e) {
    logErr('savePreview', 'fetch threw:', e);
    toast('Network error', 'error');
    if (btn) { btn.disabled = false; btn.textContent = 'Save Job'; }
  }
}

function initJobPreviewPage() {
  const form = document.getElementById('preview-form');
  if (!form) return;
  log('initJobPreviewPage', 'init');

  const descEl = document.getElementById('preview-description');
  function updateCount() {
    const countEl = document.getElementById('preview-char-count');
    if (descEl && countEl) countEl.textContent = descEl.value.length.toLocaleString();
  }
  if (descEl) descEl.addEventListener('input', updateCount);

  const raw = sessionStorage.getItem('job_preview');
  if (!raw) { window.location.href = '/'; return; }
  let data;
  try { data = JSON.parse(raw); } catch(e) { window.location.href = '/'; return; }

  form.querySelector('[name="url"]').value         = data.url         || '';
  form.querySelector('[name="title"]').value       = data.title       || '';
  form.querySelector('[name="company"]').value     = data.company     || '';
  form.querySelector('[name="location"]').value    = data.location    || '';
  form.querySelector('[name="description"]').value = data.description || '';
  updateCount();

  const warnPanel = document.getElementById('warnings-panel');
  if (warnPanel && data.has_warnings) {
    warnPanel.style.display = '';
    if (data.blocker_keywords && data.blocker_keywords.length) {
      const blockerEl = document.getElementById('blocker-list');
      if (blockerEl) {
        blockerEl.innerHTML = TMPL.previewBlockerList(data.blocker_keywords);
        document.getElementById('blocker-block').style.display = '';
      }
    }
    if (data.text_quality && data.text_quality.level !== 'ok') {
      const tqEl = document.getElementById('tq-list');
      if (tqEl) {
        tqEl.innerHTML = TMPL.previewTqIssues(data.text_quality.issues);
        document.getElementById('tq-block').style.display = '';
        document.getElementById('tq-block').className     = `text-quality-warning level-${data.text_quality.level}`;
        document.getElementById('tq-meta').textContent    =
          `${data.text_quality.char_count} chars \u00b7 ${data.text_quality.tech_keywords} tech keywords detected`;
      }
    }
  }
}


// ── Job detail page ───────────────────────────────────────────────────────────
let _currentJobId = null;

async function initJobDetailPage() {
  const main = document.getElementById('job-detail-main');
  if (!main) return;
  const match = window.location.pathname.match(/\/job\/(\d+)/);
  if (!match) { main.innerHTML = TMPL.emptyPanel('Invalid job URL.'); return; }
  _currentJobId = parseInt(match[1], 10);
  log('initJobDetailPage', `jobId=${_currentJobId}`);
  try {
    const [detailRes, statusRes] = await Promise.all([
      fetch(`/api/jobs/${_currentJobId}/detail`),
      fetch('/api/providers/status'),
    ]);
    if (!detailRes.ok) { main.innerHTML = TMPL.emptyPanel('Job not found.'); return; }
    const detail    = await detailRes.json();
    const providers = statusRes.ok ? await statusRes.json() : {};
    renderJobDetailPage(detail, providers);
  } catch(e) {
    logErr('initJobDetailPage', 'fetch threw:', e);
    main.innerHTML = TMPL.emptyPanel('Failed to load job.');
  }
}

async function refreshJobDetailPage() {
  if (!_currentJobId) return;
  log('refreshJobDetailPage', `jobId=${_currentJobId}`);
  try {
    const [detailRes, statusRes] = await Promise.all([
      fetch(`/api/jobs/${_currentJobId}/detail`),
      fetch('/api/providers/status'),
    ]);
    if (!detailRes.ok) return;
    const detail    = await detailRes.json();
    const providers = statusRes.ok ? await statusRes.json() : {};
    renderJobDetailPage(detail, providers);
  } catch(e) {
    logErr('refreshJobDetailPage', 'fetch threw:', e);
  }
}

function renderJobDetailPage(d, providers) {
  const main = document.getElementById('job-detail-main');
  const job  = d.job || {};
  document.title = `${job.title || 'Job Detail'} \u2014 Job Matcher`;
  main.innerHTML = TMPL.jobDetailPage(d, providers);
  initTabs();
  document.querySelectorAll('input[name="provider"]').forEach(r => r.addEventListener('change', updateProviderModelRow));
  updateProviderModelRow();
  if (window.location.hash) activateTab(window.location.hash.replace('#', ''));
  document.querySelectorAll("[id^='meter-']").forEach(el => {
    const block  = el.closest('.analysis-block');
    const badges = block ? block.querySelectorAll('.score-badge') : [];
    const badge  = badges.length > 1 ? badges[1] : badges[0];
    if (badge) {
      const score = parseInt(badge.textContent.trim());
      if (!isNaN(score)) renderMeter(score, el);
    }
  });
}


// ── Shared DOMContentLoaded ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initActiveNav();
  const path = window.location.pathname;
  if (path === '/resumes') {
    const form = document.getElementById('add-resume-form');
    if (form) form.addEventListener('submit', addResumeAndRefresh);
    initResumesPage();
  } else if (path === '/jobs/preview') {
    initJobPreviewPage();
  } else if (/^\/job\/\d+/.test(path)) {
    initJobDetailPage();
  }
});
