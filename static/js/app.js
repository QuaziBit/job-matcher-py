// =============================================== //
// == Script loaded ============================== //
// =============================================== //
console.log("[app.js] Script loaded v1");

// =============================================== //
// == Logging helpers ============================ //
// =============================================== //
function log(fn, msg, ...args)    { console.log(`[${fn}]`, msg, ...args); }
function logErr(fn, msg, ...args) { console.error(`[${fn}] ERROR:`, msg, ...args); }

// =============================================== //
// == Toast ====================================== //
// =============================================== //
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

// =============================================== //
// == Tabs ======================================= //
// =============================================== //
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

// =============================================== //
// == Score meter ================================ //
// =============================================== //
function renderMeter(score, container) {
  container.innerHTML = "";
  for (let i = 1; i <= 5; i++) {
    const pip = document.createElement("div");
    pip.className = "score-pip" + (i <= score ? ` filled-${score}` : "");
    container.appendChild(pip);
  }
}

// =============================================== //
// == Add Mode Toggle ============================ //
// =============================================== //
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

// =============================================== //
// == Add Job by URL ============================= //
// =============================================== //
async function addJob(e) {
  e.preventDefault();
  const form = e.target;
  const btn = form.querySelector("[type=submit]");
  const input = form.querySelector("input[name=url]");
  const url = input.value.trim();
  log("addJob", `url="${url}"`);
  if (!url) { toast("Please enter a URL", "error"); return; }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Scraping…';

  const fd = new FormData();
  fd.append("url", url);

  try {
    log("addJob", "POST /api/jobs/add");
    const res = await fetch("/api/jobs/add", { method: "POST", body: fd });
    const data = await res.json();
    log("addJob", `response status=${res.status}`, data);

    if (res.status === 409) {
      toast("Job already added — " + (data.title || url), "info");
      btn.disabled = false; btn.textContent = "Add Job";
      return;
    }
    if (!res.ok) {
      logErr("addJob", `server error ${res.status}:`, data.error);
      toast(data.error || "Failed to scrape URL", "error");
      btn.disabled = false; btn.textContent = "Add Job";
      return;
    }

    log("addJob", `success id=${data.job_id} title="${data.title}"`);
    toast(`✓ Added: ${data.title || url}`, "success");
    input.value = "";
    btn.disabled = false; btn.textContent = "Add Job";
    setTimeout(() => location.reload(), 800);
  } catch(err) {
    logErr("addJob", "fetch threw:", err);
    toast("Network error", "error");
    btn.disabled = false; btn.textContent = "Add Job";
  }
}

// =============================================== //
// == Add Job by Paste =========================== //
// =============================================== //
async function addJobManual() {
  log("addJobManual", "called");
  const title       = document.getElementById("paste-title").value.trim();
  const company     = document.getElementById("paste-company").value.trim();
  const description = document.getElementById("paste-description").value.trim();
  const btn         = document.getElementById("paste-submit-btn");

  log("addJobManual", `title="${title}" company="${company}" desc_len=${description.length}`);
  if (!description) { toast("Please paste a job description", "error"); return; }
  if (description.length < 50) { toast("Description too short (min 50 chars)", "error"); return; }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Saving…';

  const fd = new FormData();
  fd.append("title",       title);
  fd.append("company",     company);
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
    document.getElementById("paste-title").value = "";
    document.getElementById("paste-company").value = "";
    document.getElementById("paste-description").value = "";
    btn.disabled = false; btn.textContent = "Add Job";
    setTimeout(() => location.reload(), 800);
  } catch(err) {
    logErr("addJobManual", "fetch threw:", err);
    toast("Network error", "error");
    btn.disabled = false; btn.textContent = "Add Job";
  }
}

// =============================================== //
// == Analyze Job ================================ //
// =============================================== //
async function analyzeJob(jobId) {
  log("analyzeJob", `jobId=${jobId}`);
  const resumeSelect = document.getElementById("analyze-resume");
  const providerInput = document.querySelector('input[name="provider"]:checked');
  const btn = document.getElementById("analyze-btn");

  if (!resumeSelect || !resumeSelect.value) {
    toast("Please select a resume first", "error"); return;
  }
  const provider = providerInput ? providerInput.value : "anthropic";

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analyzing…';

  const fd = new FormData();
  fd.append("resume_id", resumeSelect.value);
  fd.append("provider", provider);

  try {
    log("analyzeJob", `POST /api/jobs/${jobId}/analyze`);
    const res = await fetch(`/api/jobs/${jobId}/analyze`, { method: "POST", body: fd });
    const data = await res.json();
    log("analyzeJob", `response status=${res.status}`, data);
    if (!res.ok) {
      logErr("analyzeJob", `server error ${res.status}:`, data.error);
      toast(data.error || "Analysis failed", "error");
      btn.disabled = false; btn.textContent = "Run Analysis";
      return;
    }
    log("analyzeJob", `success score=${data.score} adjusted=${data.adjusted_score}`);
    toast(`✓ Score: ${data.adjusted_score}/5 (raw ${data.score}/5)`, "success");
    btn.disabled = false; btn.textContent = "Run Analysis";
    setTimeout(() => location.reload(), 800);
  } catch(err) {
    logErr("analyzeJob", "fetch threw:", err);
    toast("Network error", "error");
    btn.disabled = false; btn.textContent = "Run Analysis";
  }
}

// =============================================== //
// == Save Application =========================== //
// =============================================== //
async function saveApplication(jobId) {
  log("saveApplication", `jobId=${jobId}`);
  const status   = document.getElementById("app-status").value;
  const name     = document.getElementById("recruiter-name").value;
  const email    = document.getElementById("recruiter-email").value;
  const phone    = document.getElementById("recruiter-phone").value;
  const notes    = document.getElementById("app-notes").value;
  const btn      = document.getElementById("save-app-btn");

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';

  const fd = new FormData();
  fd.append("status", status);
  fd.append("recruiter_name", name);
  fd.append("recruiter_email", email);
  fd.append("recruiter_phone", phone);
  fd.append("notes", notes);

  try {
    log("saveApplication", `POST /api/jobs/${jobId}/application`);
    const res = await fetch(`/api/jobs/${jobId}/application`, { method: "POST", body: fd });
    log("saveApplication", `response status=${res.status}`);
    if (res.ok) {
      toast("Application info saved", "success");
      // Update status badge inline
      const badge = document.getElementById("status-badge");
      if (badge) {
        badge.className = `status-badge status-${status}`;
        badge.textContent = status.replace("_", " ");
      }
    } else {
      logErr("saveApplication", `server error ${res.status}`);
      toast("Save failed", "error");
    }
  } catch(err) {
    logErr("saveApplication", "fetch threw:", err);
    toast("Network error", "error");
  }
  btn.disabled = false; btn.innerHTML = "Save";
}

// =============================================== //
// == Delete Analysis ============================ //
// =============================================== //
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
      logErr("deleteAnalysis", `server error ${res.status}`);
      toast("Delete failed", "error");
    }
  } catch(err) { logErr("deleteAnalysis", "fetch threw:", err); toast("Network error", "error"); }
}
async function deleteJob(jobId) {
  if (!confirm("Delete this job and all its analyses?")) return;
  log("deleteJob", `id=${jobId}`);
  try {
    const res = await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
    log("deleteJob", `response status=${res.status}`);
    if (res.ok) { toast("Job deleted", "info"); setTimeout(() => location.href = "/", 600); }
  else { logErr("deleteJob", `server error ${res.status}`); toast("Delete failed", "error"); }
  } catch(err) { logErr("deleteJob", "fetch threw:", err); toast("Network error", "error"); }
}

// =============================================== //
// == Delete Resume ============================== //
// =============================================== //
async function deleteResume(resumeId) {
  if (!confirm("Delete this resume version?")) return;
  log("deleteResume", `id=${resumeId}`);
  try {
    const res = await fetch(`/api/resumes/${resumeId}`, { method: "DELETE" });
    log("deleteResume", `response status=${res.status}`);
    if (res.ok) { toast("Resume deleted", "info"); setTimeout(() => location.reload(), 600); }
  else { logErr("deleteResume", `server error ${res.status}`); toast("Delete failed", "error"); }
  } catch(err) { logErr("deleteResume", "fetch threw:", err); toast("Network error", "error"); }
}

// =============================================== //
// == Add Resume ================================= //
// =============================================== //
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

// =============================================== //
// == Toggle description ========================= //
// =============================================== //
async function toggleDesc(jobId) {
  log("toggleDesc", `jobId=${jobId}`);
  const box = document.getElementById("desc-box");
  if (!box) { logErr("toggleDesc", "#desc-box not found"); return; }
  if (box.classList.contains("hidden")) {
    if (!box.dataset.loaded) {
      box.textContent = "Loading…";
      log("toggleDesc", `fetching description for job=${jobId}`);
      try {
        const res = await fetch(`/api/jobs/${jobId}/description`);
        const data = await res.json();
        log("toggleDesc", `loaded ${data.description?.length || 0} chars`);
        box.textContent = data.description || "(no description)";
        box.dataset.loaded = "1";
      } catch(err) { logErr("toggleDesc", "fetch threw:", err); box.textContent = "Failed to load description."; }
    }
    box.classList.remove("hidden");
  } else {
    box.classList.add("hidden");
  }
}

// =============================================== //
// == Init ======================================= //
// =============================================== //
document.addEventListener("DOMContentLoaded", () => {
  log("init", "DOMContentLoaded fired");
  initTabs();
  initAddModeToggle();

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
