# Job Matcher — Python Version

> Local AI-powered resume-to-job matching and application tracker.
> Compare job postings against your resume using Claude or a local Ollama model.
> No cloud storage. No subscriptions. Runs entirely on your machine.

---

## Screenshots

<div align="center">
  <img src="screenshots/01-launcher.jpg" width="350"/>
  <p><em>Launcher — live health checks, model selector, Start / Restart / Stop</em></p>
</div>

<div align="center">
  <img src="screenshots/05-job-tracker.jpg" width="800"/>
  <p><em>Job Tracker — jobs list with score badges and status tags</em></p>
</div>

<div align="center">
  <img src="screenshots/03-analysis-ollama-2.jpg" width="800"/>
  <p><em>Ollama (gemma3:27b) — analysis tab with resume selector and LLM provider toggle</em></p>
</div>

<div align="center">
  <img src="screenshots/03-analysis-ollama-1.jpg" width="800"/>
  <p><em>Ollama (llama3.1:8b) — same job, lighter penalty, different model perspective</em></p>
</div>

<div align="center">
  <img src="screenshots/02-analysis-anthropic.jpg" width="800"/>
  <p><em>Anthropic (claude-opus-4-5) — penalty pipeline with severity-weighted gaps</em></p>
</div>

---

## What It Does

- Scrapes job postings from a URL (Indeed, LinkedIn, Greenhouse, Workday, etc.)
- Compares each job description against your resume using an LLM
- Scores the match from **1** (poor) to **5** (excellent) with a skill breakdown
- Applies a **penalty pipeline** — detects hard blockers like clearance requirements and experience minimums and adjusts the raw score automatically
- Tracks application status, recruiter contact info, and personal notes
- Supports **Anthropic API** (Claude) or **Ollama** (local models, fully offline)

**Stack:** FastAPI · SQLite · Jinja2 · Plain JS · BeautifulSoup · Python 3.10+

---

## Requirements

- Python 3.10 or higher
- pip
- An Anthropic API key → https://console.anthropic.com
  *(or Ollama installed locally for the free/offline option)*

---

## Installation

### Step 1 — Clone or download

```bash
git clone https://github.com/alexm/job-matcher-py.git
cd job-matcher-py
```

---

### Step 2 — Create a virtual environment

```bash
# Create the environment
py -3 -m venv .venv

# Or target a specific Python version
py -3.11 -m venv .venv
```

**Activate it:**

| Platform | Command |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows CMD | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |

Your prompt should now show `(.venv)` on the left — that means it's active.

---

### PowerShell Permission Fix (Windows only)

If you get an execution policy error in PowerShell, run this first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate normally:

```powershell
.venv\Scripts\Activate.ps1
```

> This only applies to the current terminal session — it does not change any system settings permanently.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---|---|---|
| fastapi | 0.115.0 | Web framework |
| uvicorn | 0.30.6 | ASGI server |
| jinja2 | 3.1.4 | HTML templating |
| python-multipart | 0.0.9 | Form parsing |
| httpx | 0.27.2 | Async HTTP client |
| beautifulsoup4 | 4.12.3 | HTML parsing |
| anthropic | 0.34.2 | Claude API client |
| python-dotenv | 1.0.1 | `.env` file loader |
| aiosqlite | 0.20.0 | Async SQLite driver |

---

### Step 4 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in any editor and fill in your key:

```env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
```

Optional Ollama settings (only needed if using local models):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT=600
```

Optional server settings:

```env
APP_PORT=8000
APP_HOST=127.0.0.1
```

---

## Running the App

```bash
python main.py
```

A browser window opens automatically with the **Launcher** — a config page where you can:

- Review health status of SQLite, Ollama, and Anthropic API
- Change the port, model, or API key before starting
- Click **▶ Start Job Matcher** to launch the main app
- Use **↺ Restart** to switch models without restarting manually
- Use **■ Stop** to shut down the server

Alternatively, run directly with uvicorn (skips the launcher):

```bash
uvicorn main:app --reload --port 8000
```

**Common port choices:**

| Port | Use case |
|---|---|
| `8000` | Default |
| `8080` | If 8000 is taken |
| `8888` | Avoids conflict with Jupyter |
| `9000` | Multiple local tools running |

To allow access from other devices on your local network:

```bash
APP_HOST=0.0.0.0 python main.py
# Then open http://<your-machine-ip>:8000 from another device
```

Press `Ctrl+C` to stop.

---

## Running the Tests

```bash
python tests.py
```

No API key or network connection required — all external calls are mocked.

```
════════════════════════════════════════════════════
  Job Matcher — Unit Test Suite  (Python 3.11.x)
════════════════════════════════════════════════════

TestAPIEndpoints
  ✓ test_add_duplicate_job_returns_409
  ✓ test_add_job_success
  ... (72 tests)

------------------------------------------------------------
  72/72 passed  |  0 failed | 0 errors
```

**Test coverage:**

| Class | What it covers |
|---|---|
| `TestParseResponse` | LLM JSON parsing, score validation, fence stripping |
| `TestPenaltyPipeline` | Blocker/major/minor scoring, caps, count penalty |
| `TestBuildUserPrompt` | Prompt construction |
| `TestAnalyzeWithAnthropic` | API client, mocked responses |
| `TestAnalyzeWithOllama` | Ollama client, connection errors |
| `TestCleanText` | Whitespace normalization |
| `TestExtractMeta` | Meta tag fallback chain (og → title → h1) |
| `TestScrapeJob` | Selector priority, noise stripping, truncation, HTTP errors |
| `TestDatabase` | CRUD, duplicate constraint, cascade delete, upsert |
| `TestAPIEndpoints` | All routes — 200/404/409/422 responses |
| `TestEndToEndFlow` | Full flow: resume → job → analyze → track |

---

## Usage

### 1. Add your resume

Go to the **Resumes** page → paste your full resume text → give it a label.

```
Example labels: "DevSecOps v2", "Full-Stack General", "Federal Contractor"
```

You can store multiple versions and pick which one to compare per job.

### 2. Add a job

**From a URL** — paste any job posting URL and click Add Job. The app scrapes and parses it automatically.

**By pasting** — switch to the Paste tab, paste the description directly. Useful for Workday or login-gated pages that block scraping.

### 3. Run an analysis

Open a job → **Analysis tab** → pick a resume + LLM provider → **Run Analysis**.

You get:

- A **1–5 raw score** from the LLM
- A **penalty-adjusted score** accounting for hard blockers (clearance, years of experience, mandatory certs)
- **Matched skills** — what aligns
- **Missing skills** — each rated `blocker / major / minor`
- A short **reasoning** summary

Run multiple analyses (different resumes or providers) and compare them side by side.

### 4. Track your application

**Application tab** → set status, add recruiter info, and write notes.

| Status | Meaning |
|---|---|
| `not_applied` | Saved for later |
| `applied` | Submitted |
| `interviewing` | In process |
| `offered` | Offer received |
| `rejected` | Closed |

---

## LLM Providers

### Anthropic (default)

Requires `ANTHROPIC_API_KEY` in `.env`. Uses `claude-opus-4-5`.
Costs roughly $0.01–0.05 per analysis depending on job description length.
Get a key at: https://console.anthropic.com

### Ollama (local, free, offline)

Runs entirely on your machine — no API key, no cost, no data leaves your box.

```bash
# 1. Install Ollama
#    https://ollama.com/download

# 2. Pull a model
ollama pull llama3.1:8b

# 3. Start the server
ollama serve

# 4. Toggle to "Ollama" in the analysis UI
```

**Recommended models:**

| Model | RAM needed | Quality |
|---|---|---|
| `llama3.1:8b` | ~8 GB | Best balance for everyday use |
| `gemma3:27b` | ~32 GB | Near-Anthropic quality |
| `mistral:7b` | ~5 GB | Decent, lightweight |
| `phi3.5:3.8b` | ~4 GB | Fast triage only |

**Recommended workflow:**

- Quick triage → `llama3.1:8b`
- Serious roles → `gemma3:27b`
- Final decision → Anthropic toggle

---

## Scraping Notes

Works out of the box with most public job boards: Indeed, LinkedIn, Greenhouse, Lever, SmartRecruiters, BambooHR.

May not work if the page:

- Requires a login to view the full description
- Renders content entirely via JavaScript (some Workday / iCIMS pages)
- Has aggressive bot detection

**Workaround:** Use the **Paste** tab — copy the job description text from your browser and paste it directly.

---

## Project Structure

```
job-matcher-py/
├── main.py              FastAPI app — all routes and request handling
├── launcher.py          Browser-based launcher (health checks, Start/Stop/Restart)
├── database.py          SQLite schema init and async connection helper
├── scraper.py           URL fetching, HTML parsing, metadata extraction
├── ollama_utils.py      Ollama API client with error handling
├── analyzer.py          LLM prompt construction, API calls, penalty pipeline
├── tests.py             Full unit + integration test suite
├── screenshots/            Example screenshots for the README
├── templates/
│   ├── base.html        Sidebar layout shell
│   ├── index.html       Jobs list page
│   ├── job_detail.html  Analysis, application tracking, description viewer
│   └── resumes.html     Resume version manager
├── static/
│   ├── css/style.css    Dark theme, IBM Plex fonts
│   └── js/app.js        Form submissions, toasts, tabs, score meters
├── .env                 Your secrets — never commit this file
├── .env.example         Template for .env
├── requirements.txt     Python dependencies with pinned versions
├── job_matcher.db       SQLite database — auto-created on first run
├── LICENSE
├── README.md
└── .gitignore
```

---

## Related

- **Go version** → [github.com/alexm/job-matcher-go](https://github.com/alexm/job-matcher-go)
  Single binary, no Python required. Includes a browser launcher, embeds all templates and static files, builds to `.exe` (Windows) and a Linux binary via GitHub Actions.

---

## License

MIT
