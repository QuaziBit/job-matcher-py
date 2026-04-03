// ── Logging helpers ───────────────────────────────────────────────────────────

function log(fn, msg, ...args)    { console.log(`[${fn}]`, msg, ...args); }
function logErr(fn, msg, ...args) { console.error(`[${fn}] ERROR:`, msg, ...args); }


// ── Toast ─────────────────────────────────────────────────────────────────────

let toastTimer;
function toast(msg, type = "info") {
  log("toast", type, msg);
  const el = document.getElementById("toast");
  if (!el) { logErr("toast", "#toast not found"); return; }
  el.textContent = msg;
  el.className = `show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = ""; }, 3500);
}


// ── Tabs ──────────────────────────────────────────────────────────────────────

function activateTab(tabName) {
  const tab = document.querySelector(`.tab[data-tab="${tabName}"]`);
  if (!tab) return;
  const tabBar = tab.closest(".tabs");
  const parent = tabBar ? tabBar.closest(".tab-container") || document : document;
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
  tab.classList.add("active");
  const content = parent.querySelector(`[data-tab-content="${tabName}"]`);
  if (content) content.classList.add("active");
}

function initTabs() {
  log("initTabs", `found ${document.querySelectorAll(".tabs").length} tab bar(s)`);
  document.querySelectorAll(".tabs").forEach(tabBar => {
    tabBar.querySelectorAll(".tab").forEach(tab => {
      tab.addEventListener("click", () => {
        const target = tab.dataset.tab;
        const parent = tabBar.closest(".tab-container") || document;
        tabBar.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        parent.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
        tab.classList.add("active");
        const content = parent.querySelector(`[data-tab-content="${target}"]`);
        if (content) content.classList.add("active");
      });
    });
  });
}


// ── Score meter ───────────────────────────────────────────────────────────────

function renderMeter(score, container) {
  container.innerHTML = "";
  for (let i = 1; i <= 5; i++) {
    const pip = document.createElement("div");
    pip.className = "score-pip" + (i <= score ? ` filled-${score}` : "");
    container.appendChild(pip);
  }
}


// ── Add mode toggle ───────────────────────────────────────────────────────────

function initAddModeToggle() {
  log("initAddModeToggle", "init");
  const radios = document.querySelectorAll('input[name="add-mode"]');
  if (!radios.length) return;
  radios.forEach(radio => {
    radio.addEventListener("change", () => {
      const urlForm   = document.getElementById("add-job-form");
      const pasteForm = document.getElementById("paste-job-form-wrap");
      if (radio.value === "paste" && radio.checked) {
        urlForm.classList.add("hidden");
        pasteForm.classList.remove("hidden");
      } else {
        urlForm.classList.remove("hidden");
        pasteForm.classList.add("hidden");
      }
    });
  });
}


// ── Add job by URL ────────────────────────────────────────────────────────────

async function addJob(e) {
  e.preventDefault();
  const form  = e.target;
  const btn   = form.querySelector("[type=submit]");
  const input = form.querySelector("input[name=url]");
  const url   = input.value.trim();
  log("addJob", `url="${url}"`);
  if (!url) { toast("Please enter a URL", "error"); return; }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Scraping…';

  const fd = new FormData();
  fd.append("url", url);

  try {
    log("addJob", "POST /api/jobs/scrape");
    const res  = await fetch("/api/jobs/scrape", { method: "POST", body: fd });
    const data = await res.json();
    log("addJob", `response status=${res.status}`, data);

    if (res.status === 409) {
      toast("Job already added", "info");
      if (data.job_id) setTimeout(() => window.location.href = "/job/" + data.job_id, 600);
      btn.disabled = false; btn.textContent = "Add Job";
      return;
    }
    if (!res.ok) {
      logErr("addJob", `server error ${res.status}:`, data.error);
      toast(data.error || "Failed to scrape URL", "error");
      btn.disabled = false; btn.textContent = "Add Job";
      return;
    }

    // Store scraped data in sessionStorage and redirect to preview page
    sessionStorage.setItem("job_preview", JSON.stringify(data));
    window.location.href = "/jobs/preview";
  } catch(err) {
    logErr("addJob", "fetch threw:", err);
    toast("Network error", "error");
    btn.disabled = false; btn.textContent = "Add Job";
  }
}


// ── Add job by paste ──────────────────────────────────────────────────────────

async function addJobManual() {
  log("addJobManual", "called");
  const title       = document.getElementById("paste-title").value.trim();
  const company     = document.getElementById("paste-company").value.trim();
  const location    = (document.getElementById("paste-location") || {value:""}).value.trim();
  const description = document.getElementById("paste-description").value.trim();
  const btn         = document.getElementById("paste-submit-btn");

  log("addJobManual", `title="${title}" company="${company}" location="${location}" desc_len=${description.length}`);
  if (!description) { toast("Please paste a job description", "error"); return; }
  if (description.length < 50) { toast("Description too short (min 50 chars)", "error"); return; }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Saving…';

  const fd = new FormData();
  fd.append("title",       title);
  fd.append("company",     company);
  fd.append("location",    location);
  fd.append("description", description);

  try {
    log("addJobManual", "POST /api/jobs/add-manual");
    const res  = await fetch("/api/jobs/add-manual", { method: "POST", body: fd });
    const data = await res.json();
    log("addJobManual", `response status=${res.status}`, data);

    if (res.status === 409) {
      toast("This description was already added", "info");
      btn.disabled = false; btn.textContent = "Add Job";
      return;
    }
    if (!res.ok) {
      logErr("addJobManual", `server error ${res.status}:`, data.error);
      toast(data.error || "Failed to save", "error");
      btn.disabled = false; btn.textContent = "Add Job";
      return;
    }

    log("addJobManual", `success id=${data.job_id}`);
    toast(`✓ Added: ${data.title}`, "success");
    setTimeout(() => window.location.href = "/job/" + data.job_id, 600);
  } catch(err) {
    logErr("addJobManual", "fetch threw:", err);
    toast("Network error", "error");
    btn.disabled = false; btn.textContent = "Add Job";
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
  if (label) label.textContent = `Analyzing with ${model} · ${mode} mode`;
  if (fill)  fill.style.width = '0%';
  _progressTimer = setInterval(() => {
    const elapsed = (Date.now() - _progressStart) / 1000;
    const pct     = Math.min(elapsed / est * 100, 95);
    if (fill) fill.style.width = pct + '%';
    if (meta) meta.textContent = `${formatElapsed(elapsed)} elapsed · ~${formatElapsed(est)} estimated`;
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
    const data = await res.json();
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
      opt.value = current; opt.textContent = current;
      opt.selected = true;
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
    const data = await res.json();
    const models = data.models || [];
    if (!models.length) return;
    // Current configured model from env — stored per-provider in data attributes
    const current = sel.dataset[provider] || '';
    sel.innerHTML = '';
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.label;
      if (m.id === current) opt.selected = true;
      sel.appendChild(opt);
    });
    // If configured model not in list, prepend it
    const ids = models.map(m => m.id);
    if (current && !ids.includes(current)) {
      const opt = document.createElement('option');
      opt.value = current; opt.textContent = current + ' · (configured)';
      opt.selected = true;
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

  const provider      = providerInput ? providerInput.value : 'anthropic';
  const modeSelect    = document.getElementById('analysis-mode-select');
  const ollamaSelect  = document.getElementById('ollama-model-select');
  const cloudSelect   = document.getElementById('cloud-model-select');
  const mode          = modeSelect ? modeSelect.value : (btn ? (btn.dataset.mode || 'standard') : 'standard');
  const ollamaModel   = (provider === 'ollama' && ollamaSelect) ? ollamaSelect.value : '';
  const cloudModel    = (['anthropic','openai','gemini'].includes(provider) && cloudSelect) ? cloudSelect.value : '';
  const displayModel  = ollamaModel || cloudModel || provider;

  btn.disabled    = true;
  btn.textContent = 'Analyzing…';
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
    toast(`✓ Score: ${data.adjusted_score}/5 (raw ${data.score}/5)`, 'success');
    btn.disabled = false; btn.textContent = 'Run Analysis';
    setTimeout(() => location.reload(), 800);
  } catch(err) {
    stopProgress();
    logErr('analyzeJob', 'fetch threw:', err);
    toast('Network error', 'error');
    btn.disabled = false; btn.textContent = 'Run Analysis';
  }
}


// ── Save application ──────────────────────────────────────────────────────────

async function saveApplication(jobId) {
  log("saveApplication", `jobId=${jobId}`);
  const status = document.getElementById("app-status").value;
  const name   = document.getElementById("recruiter-name").value;
  const email  = document.getElementById("recruiter-email").value;
  const phone  = document.getElementById("recruiter-phone").value;
  const notes  = document.getElementById("app-notes").value;
  const btn    = document.getElementById("save-app-btn");

  btn.disabled  = true;
  btn.innerHTML = '<span class="spinner"></span>';

  const fd = new FormData();
  fd.append("status",          status);
  fd.append("recruiter_name",  name);
  fd.append("recruiter_email", email);
  fd.append("recruiter_phone", phone);
  fd.append("notes",           notes);

  try {
    log("saveApplication", `POST /api/jobs/${jobId}/application`);
    const res = await fetch(`/api/jobs/${jobId}/application`, { method: "POST", body: fd });
    log("saveApplication", `response status=${res.status}`);
    if (res.ok) {
      toast("Application info saved", "success");
      const badge = document.getElementById("status-badge");
      if (badge) {
        badge.className   = `status-badge status-${status}`;
        badge.textContent = status.replace("_", " ");
      }
    } else {
      logErr("saveApplication", `server error ${res.status} for job=${jobId}`);
      toast("Save failed", "error");
    }
  } catch(err) {
    logErr("saveApplication", "fetch threw:", err);
    toast("Network error", "error");
  }
  btn.disabled  = false;
  btn.innerHTML = "Save";
}


// ── Delete analysis / job ─────────────────────────────────────────────────────

async function deleteAnalysis(analysisId) {
  if (!confirm("Remove this analysis?")) return;
  log("deleteAnalysis", `id=${analysisId}`);
  try {
    const res = await fetch(`/api/analyses/${analysisId}`, { method: "DELETE" });
    log("deleteAnalysis", `response status=${res.status}`);
    if (res.ok) {
      const block = document.getElementById(`analysis-${analysisId}`);
      if (block) block.remove();
      toast("Analysis removed", "info");
    } else {
      logErr("deleteAnalysis", `server error ${res.status} for id=${analysisId}`);
      toast("Delete failed", "error");
    }
  } catch(err) {
    logErr("deleteAnalysis", "fetch threw:", err);
    toast("Network error", "error");
  }
}

async function deleteJob(jobId) {
  if (!confirm("Delete this job and all its analyses?")) return;
  log("deleteJob", `id=${jobId}`);
  try {
    const res = await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
    log("deleteJob", `response status=${res.status}`);
    if (res.ok) {
      toast("Job deleted", "info");
      setTimeout(() => location.href = "/", 600);
    } else {
      logErr("deleteJob", `server error ${res.status} for id=${jobId}`);
      toast("Delete failed", "error");
    }
  } catch(err) {
    logErr("deleteJob", "fetch threw:", err);
    toast("Network error", "error");
  }
}


// ── Delete resume ─────────────────────────────────────────────────────────────

async function deleteResume(resumeId) {
  if (!confirm("Delete this resume version?")) return;
  log("deleteResume", `id=${resumeId}`);
  try {
    const res = await fetch(`/api/resumes/${resumeId}`, { method: "DELETE" });
    log("deleteResume", `response status=${res.status}`);
    if (res.ok) {
      toast("Resume deleted", "info");
      setTimeout(() => location.reload(), 600);
    } else {
      logErr("deleteResume", `server error ${res.status} for id=${resumeId}`);
      toast("Delete failed", "error");
    }
  } catch(err) {
    logErr("deleteResume", "fetch threw:", err);
    toast("Network error", "error");
  }
}


// ── Add resume ────────────────────────────────────────────────────────────────

async function addResume(e) {
  if (e) e.preventDefault();
  log("addResume", "called");

  const form = document.getElementById("add-resume-form");
  if (!form) { logErr("addResume", "form#add-resume-form not found"); return; }

  const fd      = new FormData(form);
  const label   = (fd.get("label")   || "").trim();
  const content = (fd.get("content") || "").trim();

  log("addResume", `label="${label}" content_len=${content.length}`);

  if (!label || !content) {
    logErr("addResume", `validation failed — label="${label}" content_len=${content.length}`);
    toast("Label and content are required", "error");
    return;
  }

  const btn = form.querySelector("[type=submit]");
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>'; }

  try {
    log("addResume", "POST /api/resumes/add");
    const res  = await fetch("/api/resumes/add", { method: "POST", body: fd });
    const data = await res.json();
    log("addResume", `response status=${res.status}`, data);
    if (res.ok) {
      log("addResume", `success id=${data.resume_id}`);
      toast(`✓ Resume "${data.label}" saved`, "success");
      form.reset();
      if (btn) { btn.disabled = false; btn.textContent = "Save Resume"; }
      setTimeout(() => location.reload(), 800);
    } else {
      logErr("addResume", `server error ${res.status}:`, data.error);
      toast(data.error || "Failed", "error");
      if (btn) { btn.disabled = false; btn.textContent = "Save Resume"; }
    }
  } catch(err) {
    logErr("addResume", "fetch threw:", err);
    toast("Network error", "error");
    if (btn) { btn.disabled = false; btn.textContent = "Save Resume"; }
  }
}


// ── Snippet toggle ────────────────────────────────────────────────────────────

function toggleSnippets(btn) {
  log('toggleSnippets', 'clicked');
  const container = btn.nextElementSibling;
  if (!container) { logErr('toggleSnippets', 'no sibling found'); return; }
  const isHidden = container.style.display === 'none' || !container.style.display;
  container.style.display = isHidden ? 'block' : 'none';
  btn.innerHTML = isHidden ? '&#9660; evidence' : '&#9658; evidence';
}


// ── Toggle description ────────────────────────────────────────────────────────

async function toggleDesc(jobId) {
  log("toggleDesc", `jobId=${jobId}`);
  const box = document.getElementById("desc-box");
  if (!box) { logErr("toggleDesc", "#desc-box not found"); return; }
  if (box.classList.contains("hidden")) {
    if (!box.dataset.loaded) {
      box.textContent = "Loading…";
      log("toggleDesc", `fetching description for job=${jobId}`);
      try {
        const res  = await fetch(`/api/jobs/${jobId}/description`);
        const data = await res.json();
        log("toggleDesc", `loaded ${data.description?.length || 0} chars`);
        box.textContent    = data.description || "(no description)";
        box.dataset.loaded = "1";
      } catch(err) {
        logErr("toggleDesc", `fetch threw for job=${jobId}:`, err);
        box.textContent = "Failed to load description.";
      }
    }
    box.classList.remove("hidden");
  } else {
    box.classList.add("hidden");
  }
}


// ── Init (job detail + resumes pages) ────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  log("init", "DOMContentLoaded fired");
  initTabs();
  initAddModeToggle();

  // Activate tab from URL hash (e.g. /job/1#application)
  if (window.location.hash) {
    const hashTab = window.location.hash.replace('#', '');
    activateTab(hashTab);
  }

  // Wire provider radio buttons to show/hide model row
  document.querySelectorAll('input[name="provider"]').forEach(radio => {
    radio.addEventListener('change', updateProviderModelRow);
  });
  updateProviderModelRow();

  const addJobForm = document.getElementById("add-job-form");
  if (addJobForm) {
    log("init", "binding #add-job-form submit");
    addJobForm.addEventListener("submit", addJob);
  } else {
    log("init", "#add-job-form not on this page");
  }

  const addResumeForm = document.getElementById("add-resume-form");
  if (addResumeForm) {
    log("init", "binding #add-resume-form submit");
    addResumeForm.addEventListener("submit", addResume);
  } else {
    log("init", "#add-resume-form not on this page");
  }
});


// ── Jobs list state ───────────────────────────────────────────────────────────

let _currentPage = 1;
let _perPage     = 25;
let _searchTimer = null;


// ── Fetch jobs ────────────────────────────────────────────────────────────────

async function fetchJobs() {
  const search   = (document.getElementById('filter-search')   || {value:''}).value.trim();
  const status   = (document.getElementById('filter-status')   || {value:''}).value;
  const score    = (document.getElementById('filter-score')    || {value:''}).value;
  const provider = (document.getElementById('filter-provider') || {value:''}).value;

  const params = new URLSearchParams({ page: _currentPage, per_page: _perPage });
  if (search)   params.set('search',   search);
  if (status)   params.set('status',   status);
  if (score)    params.set('score',    score);
  if (provider) params.set('provider', provider);

  const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
  history.pushState({}, '', newURL);

  const clearBtn = document.getElementById('clear-btn');
  if (clearBtn) clearBtn.style.display = (search || status || score || provider) ? 'inline-flex' : 'none';

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

    const score      = job.adjusted_score || job.best_score;
    const scoreBadge = score
      ? `<div class="score-badge score-${score}">${score}</div>`
      : `<div class="score-badge score-none">—</div>`;

    const isManual     = job.is_manual || (job.url || '').startsWith('manual://');
    const providerTag  = isManual
      ? `<span class="provider-tag">manual</span>`
      : (job.provider ? `<span class="provider-tag">${job.provider}</span>` : '');
    const recruiterBadge = job.has_recruiter
      ? `<span class="recruiter-tag" title="Open recruiter info" onclick="event.preventDefault();event.stopPropagation();window.location='/job/${job.id}#application';">👤 recruiter</span>`
      : '';

    const status      = job.status || 'not_applied';
    const statusLabel = status.replace(/_/g, ' ');
    const company     = job.company  || '';
    const location    = job.location || '';
    const sep         = company && location ? ' · ' : '';
    const metaBase    = (company + sep + location) ||
                        (isManual ? 'pasted description' : (job.url || '').substring(0, 60) + '…');
    const meta        = metaBase
                      ? (recruiterBadge ? `${metaBase} · ${recruiterBadge}` : metaBase)
                      : recruiterBadge;

    return `<a href="/job/${job.id}" class="job-item" style="text-decoration:none;">
      <div>${scoreBadge}</div>
      <div class="job-item-info">
        <div class="job-title">${job.title || (job.url ? (() => { try { return new URL(job.url).hostname; } catch(e) { return 'Untitled Job'; } })() : 'Untitled Job')}</div>
        <div class="job-meta">${meta}</div>
      </div>
      <div class="job-item-right">
        ${providerTag}
        <span class="status-badge status-${status}">${statusLabel}</span>
      </div>
    </a>`;
  }).join('');

  renderPagination(data);

  list.querySelectorAll('a.job-item').forEach(link => {
    link.addEventListener('click', () => { saveFilterState(); });
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

  if (info)      info.textContent = `Showing ${start}–${end} of ${data.total} job${data.total !== 1 ? 's' : ''}`;
  if (indicator) indicator.textContent = totalPages > 1 ? `Page ${data.page} of ${totalPages}` : '';
  if (prevBtn)   prevBtn.disabled = data.page <= 1;
  if (nextBtn)   nextBtn.disabled = data.page >= totalPages;

  if (totalPages <= 1 && _perPage !== 0) {
    pagBar.style.display = 'none';
  }
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

  if (list)   list.innerHTML = '';
  if (pagBar) pagBar.style.display = 'none';
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
  return ['filter-search', 'filter-status', 'filter-score', 'filter-provider']
    .some(id => { const el = document.getElementById(id); return el && el.value !== ''; });
}


// ── Filter actions ────────────────────────────────────────────────────────────

function applyFilters()         { _currentPage = 1; fetchJobs(); }
function applyFiltersDebounced() { clearTimeout(_searchTimer); _searchTimer = setTimeout(applyFilters, 300); }
function changePage(dir)        { _currentPage += dir; fetchJobs(); }

function changePerPage() {
  const sel = document.getElementById('per-page');
  _perPage  = sel ? parseInt(sel.value) : 25;
  _currentPage = 1;
  fetchJobs();
  log('changePerPage', `perPage=${_perPage}`);
}

function clearFilters() {
  ['filter-search', 'filter-status', 'filter-score', 'filter-provider'].forEach(id => {
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

  const params     = new URLSearchParams(rawSearch);
  const hasFilters = params.get('search') || params.get('status') ||
                     params.get('score')  || params.get('provider');
  const hasPageParams = params.get('page') || params.get('per_page');

  log('restoreFromURL', 'hasFilters=' + hasFilters + ' hasPageParams=' + hasPageParams);

  let saved = null;
  if (!hasFilters) {
    try {
      const raw = sessionStorage.getItem('jobFilterState');
      if (raw) {
        saved = JSON.parse(raw);
        log('restoreFromURL', 'found sessionStorage state:', JSON.stringify(saved));
      }
    } catch(e) {
      logErr('restoreFromURL', 'sessionStorage read failed:', e);
    }
  }

  const getValue = (key) => {
    const fromURL = params.get(key);
    if (fromURL !== null) return fromURL;
    return saved ? (saved[key] || '') : '';
  };

  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (!el) { logErr('restoreFromURL', 'element not found: #' + id); return; }
    el.value = val || '';
    log('restoreFromURL', id + ' = ' + JSON.stringify(el.value));
  };

  setEl('filter-search',   getValue('search'));
  setEl('filter-status',   getValue('status'));
  setEl('filter-score',    getValue('score'));
  setEl('filter-provider', getValue('provider'));

  if (hasPageParams) {
    _currentPage = parseInt(params.get('page')) || 1;
    const pp = params.get('per_page');
    _perPage = (pp !== null && pp !== '') ? parseInt(pp) : 25;
  } else if (saved) {
    _currentPage = saved.page     || 1;
    _perPage     = (saved.per_page !== undefined && saved.per_page !== null) ? saved.per_page : 25;
  }

  const perPageSel = document.getElementById('per-page');
  if (perPageSel) perPageSel.value = String(_perPage);

  log('restoreFromURL', 'final: page=' + _currentPage + ' perPage=' + _perPage);
}


// ── Init (jobs list page) ─────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  const jobsList = document.getElementById('jobs-list');
  if (!jobsList) return; // not on jobs page

  const searchEl = document.getElementById('filter-search');
  if (searchEl) searchEl.addEventListener('input', applyFiltersDebounced);

  ['filter-status', 'filter-score', 'filter-provider'].forEach(id => {
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
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Estimating…'; }

  const provider = document.querySelector('input[name="provider"]:checked')?.value || 'anthropic';
  const fd = new FormData();
  fd.append('provider', provider);

  try {
    const res  = await fetch(`/api/jobs/${jobId}/estimate-salary`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) {
      toast(data.error || 'Salary estimation failed', 'error');
      if (btn) { btn.disabled = false; btn.innerHTML = '💰 Estimate Salary'; }
      return;
    }
    toast('Salary estimate saved', 'success');
    setTimeout(() => location.reload(), 800);
  } catch(e) {
    logErr('estimateSalary', 'fetch threw:', e);
    toast('Salary estimation failed', 'error');
    if (btn) { btn.disabled = false; btn.innerHTML = '💰 Estimate Salary'; }
  }
}

async function clearSalaryEstimate(jobId) {
  // X button — just remove the estimate and reload, do not re-run
  log('clearSalaryEstimate', 'job ' + jobId);
  try {
    await fetch(`/api/jobs/${jobId}/salary-estimate`, { method: 'DELETE' });
    location.reload();
  } catch(e) {
    logErr('clearSalaryEstimate', 'fetch threw:', e);
    location.reload();
  }
}

async function rerunSalaryEstimate(jobId) {
  // re-run link — delete then immediately re-estimate
  log('rerunSalaryEstimate', 'job ' + jobId);

  const btn = document.querySelector(`button[onclick="rerunSalaryEstimate(${jobId})"]`);
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner" style="width:10px;height:10px;border-width:1.5px;vertical-align:middle;"></span> re-running…'; }

  try {
    await fetch(`/api/jobs/${jobId}/salary-estimate`, { method: 'DELETE' });
  } catch(e) {
    logErr('rerunSalaryEstimate', 'DELETE salary-estimate threw:', e);
  }

  const provider = document.querySelector('input[name="provider"]:checked')?.value || 'anthropic';
  const fd = new FormData();
  fd.append('provider', provider);

  try {
    const res  = await fetch(`/api/jobs/${jobId}/estimate-salary`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) {
      toast(data.error || 'Salary estimation failed', 'error');
      if (btn) { btn.disabled = false; btn.innerHTML = 're-run'; }
    } else {
      toast('Salary updated', 'success');
      setTimeout(() => location.reload(), 600);
    }
  } catch(e) {
    logErr('rerunSalaryEstimate', 'POST estimate-salary threw:', e);
    location.reload();
  }
}
