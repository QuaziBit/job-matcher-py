import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

import aiosqlite
from database import get_db, init_db
from scraper import scrape_job, assess_job_text_quality
from analyzer import analyze_match, _ollama_model

load_dotenv()

# ── Startup health checks ──────────────────────────────────────────────────────

def _check_sqlite(db_path: str) -> tuple[bool, str]:
    """Verify SQLite DB is accessible and tables exist."""
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN "
            "('jobs','resumes','analyses','applications')"
        )
        count = cur.fetchone()[0]
        conn.close()
        if count == 4:
            size_kb = os.path.getsize(db_path) / 1024 if os.path.exists(db_path) else 0
            return True, f"{db_path} ({size_kb:.0f} KB, 4 tables)"
        else:
            return False, f"{db_path} (only {count}/4 tables found — run the app once to init)"
    except Exception as e:
        return False, f"Error: {e}"


def _check_ollama(base_url: str, model: str) -> tuple[bool, str]:
    """Ping Ollama and verify the configured model is available."""
    try:
        import urllib.request
        import json as _json
        # Check if Ollama is running
        req = urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        data = _json.loads(req.read())
        available = [m["name"] for m in data.get("models", [])]
        # Check if our model is in the list (match on prefix to handle tag variants)
        model_base = model.split(":")[0]
        matched = [m for m in available if m.startswith(model_base)]
        if matched:
            return True, f"{model} ready ({len(available)} model(s) installed)"
        else:
            return False, (
                f"Ollama running but '{model}' not found. "
                f"Run: ollama pull {model}"
            )
    except Exception:
        return False, "Ollama not reachable — run: ollama serve"


def _check_anthropic(api_key: str) -> tuple[bool, str]:
    """Check if the Anthropic API key looks valid (format check, no actual API call)."""
    if not api_key or api_key == "sk-ant-...":
        return False, "No API key set — Anthropic provider unavailable"
    if not api_key.startswith("sk-ant-"):
        return False, "Key format looks wrong (should start with sk-ant-)"
    # Mask the key for display
    masked = api_key[:12] + "..." + api_key[-4:]
    return True, f"Key present ({masked})"


def _run_health_checks() -> bool:
    """Run all startup checks and print a formatted report. Returns True if critical checks pass."""
    from analyzer import _ollama_base_url, _ollama_model
    from database import _db_path

    db_path   = _db_path()
    ollama_url = _ollama_base_url()
    ollama_model = _ollama_model()
    api_key   = os.getenv("ANTHROPIC_API_KEY", "")

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

    # DB check (critical)
    db_ok, db_msg = _check_sqlite(db_path)
    status_line("SQLite", db_ok, db_msg)

    # Ollama check (warning only — user may want Anthropic only)
    ollama_ok, ollama_msg = _check_ollama(ollama_url, ollama_model)
    status_line("Ollama", ollama_ok, ollama_msg, warn_only=True)

    # Anthropic check (warning only — user may want Ollama only)
    ant_ok, ant_msg = _check_anthropic(api_key)
    status_line("Anthropic API", ant_ok, ant_msg, warn_only=True)

    # At least one LLM provider must be available
    llm_ok = ollama_ok or ant_ok
    if not llm_ok:
        print(f"\n  {RED}No LLM provider available. Configure Ollama or add ANTHROPIC_API_KEY.{RESET}")

    print(f"\n  {'─' * 50}")
    print(f"  {BOLD}URL{RESET}    http://{host}:{port}")
    print(f"  {BOLD}Model{RESET}  {ollama_model}")
    print(f"  {'─' * 50}\n")

    return db_ok  # Only DB is truly critical




@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    _run_health_checks()
    yield


app = FastAPI(title="Job Matcher", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def _format_duration(seconds: int) -> str:
    """Format seconds as '1:23' or '45s'. Returns '' for zero/None."""
    if not seconds:
        return ""
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}" if m > 0 else f"{s}s"

templates.env.filters["format_duration"] = _format_duration


# ─── Pages ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: aiosqlite.Connection = Depends(get_db)):
    async with db.execute("SELECT id, label FROM resumes ORDER BY created_at DESC") as cur:
        resumes = [dict(r) for r in await cur.fetchall()]
    # Jobs are loaded client-side via /api/jobs/list
    return templates.TemplateResponse("index.html", {
        "request": request,
        "resumes": resumes,
    })


@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_detail(job_id: int, request: Request, db: aiosqlite.Connection = Depends(get_db)):
    async with db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cur:
        job = await cur.fetchone()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job = dict(job)

    async with db.execute("SELECT * FROM applications WHERE job_id = ?", (job_id,)) as cur:
        app_row = await cur.fetchone()
    application = dict(app_row) if app_row else {}

    async with db.execute("""
        SELECT a.*, r.label as resume_label
        FROM analyses a JOIN resumes r ON r.id = a.resume_id
        WHERE a.job_id = ?
        ORDER BY a.created_at DESC
    """, (job_id,)) as cur:
        analyses = [dict(r) for r in await cur.fetchall()]

    # Parse JSON fields — use v2 columns if present, fall back to v1
    for analysis in analyses:
        # matched_skills: try v2 (rich objects) first, fall back to v1 (plain strings)
        v2_matched = analysis.get("matched_skills_v2", "[]") or "[]"
        try:
            parsed_v2 = json.loads(v2_matched)
            if parsed_v2:
                analysis["matched_skills"] = parsed_v2
            else:
                v1 = json.loads(analysis.get("matched_skills", "[]") or "[]")
                analysis["matched_skills"] = [
                    {"skill": s, "match_type": "exact",
                     "jd_snippet": "", "resume_snippet": "", "category": "other"}
                    for s in v1 if isinstance(s, str)
                ]
        except Exception:
            analysis["matched_skills"] = []

        # missing_skills: try v2 first, fall back to v1
        v2_missing = analysis.get("missing_skills_v2", "[]") or "[]"
        try:
            parsed_v2 = json.loads(v2_missing)
            if parsed_v2:
                analysis["missing_skills"] = parsed_v2
            else:
                v1 = json.loads(analysis.get("missing_skills", "[]") or "[]")
                analysis["missing_skills"] = [
                    {"skill": s["skill"] if isinstance(s, dict) else s,
                     "severity": s.get("severity", "minor") if isinstance(s, dict) else "minor",
                     "requirement_type": s.get("requirement_type", "preferred") if isinstance(s, dict) else "preferred",
                     "jd_snippet": "", "cluster_group": "other"}
                    for s in v1
                ]
        except Exception:
            analysis["missing_skills"] = []

        # penalty_breakdown
        pb = analysis.get("penalty_breakdown")
        if isinstance(pb, str):
            try:
                analysis["penalty_breakdown"] = json.loads(pb)
            except Exception:
                analysis["penalty_breakdown"] = {}

        # suggestions
        sugg = analysis.get("suggestions", "[]") or "[]"
        try:
            analysis["suggestions"] = json.loads(sugg) if isinstance(sugg, str) else sugg
        except Exception:
            analysis["suggestions"] = []

        # adjusted_score fallback
        if not analysis.get("adjusted_score"):
            analysis["adjusted_score"] = analysis["score"]

        # used_fallback as bool
        analysis["used_fallback"] = bool(analysis.get("used_fallback", 0))

    async with db.execute("SELECT id, label FROM resumes ORDER BY created_at DESC") as cur:
        resumes = [dict(r) for r in await cur.fetchall()]

    # Job text quality assessment
    text_quality = assess_job_text_quality(job.get("raw_description") or "")

    # Resume comparison (shown when 2+ different resumes analyzed)
    comparison = _build_comparison(analyses)

    return templates.TemplateResponse("job_detail.html", {
        "request":       request,
        "job":           job,
        "application":   application,
        "analyses":      analyses,
        "resumes":       resumes,
        "ollama_model":  _ollama_model(),
        "text_quality":  text_quality,
        "comparison":    comparison,
        "analysis_mode": os.getenv("ANALYSIS_MODE", "standard"),
    })



@app.get("/resumes", response_class=HTMLResponse)
async def resumes_page(request: Request, db: aiosqlite.Connection = Depends(get_db)):
    async with db.execute("SELECT * FROM resumes ORDER BY created_at DESC") as cur:
        resumes = [dict(r) for r in await cur.fetchall()]
    return templates.TemplateResponse("resumes.html", {"request": request, "resumes": resumes})


# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/jobs/list")
async def jobs_list(
    request: Request,
    db: aiosqlite.Connection = Depends(get_db),
    page:     int = 1,
    per_page: int = 25,
    search:   str = "",
    status:   str = "",
    score:    str = "",
    provider: str = "",
):
    import logging
    logger = logging.getLogger("jobs_list")

    # ── Validate inputs ───────────────────────────────────────────────────────
    if page < 1:
        return JSONResponse({"error": f"invalid page {page!r} — must be a positive integer"}, status_code=400)
    if per_page < 0:
        return JSONResponse({"error": f"invalid per_page {per_page!r} — must be 0 (all) or a positive integer"}, status_code=400)

    valid_statuses = {"", "not_applied", "applied", "interviewing", "offered", "rejected"}
    if status not in valid_statuses:
        return JSONResponse({"error": f"invalid status {status!r} — must be one of: not_applied, applied, interviewing, offered, rejected"}, status_code=400)

    valid_scores = {"", "0", "1", "2", "3", "4", "5"}
    if score not in valid_scores:
        return JSONResponse({"error": f"invalid score {score!r} — must be one of: 0, 1, 2, 3, 4, 5"}, status_code=400)

    valid_providers = {"", "anthropic", "ollama", "manual"}
    if provider not in valid_providers:
        return JSONResponse({"error": f"invalid provider {provider!r} — must be one of: anthropic, ollama, manual"}, status_code=400)

    logger.info(f"→ /api/jobs/list page={page} per_page={per_page} search={search!r} status={status!r} score={score!r} provider={provider!r}")

    # ── Build WHERE clause ────────────────────────────────────────────────────
    where = []
    args  = []

    if search:
        where.append("(LOWER(j.title) LIKE ? OR LOWER(j.company) LIKE ?)")
        like = f"%{search.lower()}%"
        args += [like, like]

    if status:
        where.append("COALESCE(a.status, 'not_applied') = ?")
        args.append(status)

    if provider:
        if provider == "manual":
            where.append("j.url LIKE 'manual://%'")
        else:
            where.append(
                "(SELECT llm_provider FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1) = ?"
            )
            args.append(provider)

    if score:
        if score == "0":
            logger.info("→ score filter = not scored")
            where.append("(SELECT COUNT(*) FROM analyses WHERE job_id = j.id) = 0")
        elif score == "5":
            logger.info("→ score filter = exactly 5")
            where.append(
                "COALESCE((SELECT adjusted_score FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1),"
                "(SELECT score FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1), 0) = 5"
            )
        else:
            min_score = int(score)
            logger.info(f"→ score filter >= {min_score}")
            where.append(
                "COALESCE((SELECT adjusted_score FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1),"
                "(SELECT score FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1), 0) >= ?"
            )
            args.append(min_score)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    base_query = f"""
        SELECT j.id, j.url, j.title, j.company, j.location, j.scraped_at,
               COALESCE(a.status, 'not_applied') as status,
               (SELECT score          FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1) as best_score,
               (SELECT adjusted_score FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1) as adjusted_score,
               (SELECT llm_provider   FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1) as provider
        FROM jobs j
        LEFT JOIN applications a ON a.job_id = j.id
        {where_sql}
        ORDER BY j.scraped_at DESC
    """

    # ── Count total ───────────────────────────────────────────────────────────
    count_query = f"SELECT COUNT(*) FROM jobs j LEFT JOIN applications a ON a.job_id = j.id {where_sql}"
    try:
        async with db.execute(count_query, args) as cur:
            row = await cur.fetchone()
            total = row[0] if row else 0
    except Exception as e:
        logger.error(f"✗ count query failed: {e}")
        return JSONResponse({"error": "Failed to load jobs from database. Check the terminal for details."}, status_code=500)

    # ── Pagination ────────────────────────────────────────────────────────────
    total_pages = 1
    if per_page > 0 and total > 0:
        total_pages = (total + per_page - 1) // per_page

    # Clamp page to valid range
    if page > total_pages and total_pages > 0:
        logger.info(f"→ page {page} out of range (max {total_pages}) — clamping")
        page = total_pages

    paginated_query = base_query
    paginated_args  = list(args)
    if per_page > 0:
        offset = (page - 1) * per_page
        paginated_query += " LIMIT ? OFFSET ?"
        paginated_args  += [per_page, offset]

    # ── Fetch jobs ────────────────────────────────────────────────────────────
    try:
        async with db.execute(paginated_query, paginated_args) as cur:
            rows = await cur.fetchall()
            jobs = [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"✗ jobs query failed: {e}")
        return JSONResponse({"error": "Failed to load jobs from database. Check the terminal for details."}, status_code=500)

    # Mark manual jobs
    for job in jobs:
        job["is_manual"] = (job.get("url") or "").startswith("manual://")

    logger.info(f"✓ /api/jobs/list total={total} page={page}/{total_pages} returned={len(jobs)}")

    return JSONResponse({
        "jobs":        jobs,
        "total":       total,
        "page":        page,
        "per_page":    per_page,
        "total_pages": total_pages,
    })


@app.post("/api/jobs/add")
async def add_job(url: str = Form(...), db: aiosqlite.Connection = Depends(get_db)):
    """Scrape a job URL and store it."""
    url = url.strip()
    # Check duplicate
    async with db.execute("SELECT id FROM jobs WHERE url = ?", (url,)) as cur:
        existing = await cur.fetchone()
    if existing:
        return JSONResponse({"error": "This URL has already been added.", "job_id": existing[0]}, status_code=409)

    try:
        data = await scrape_job(url)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    async with db.execute(
        "INSERT INTO jobs (url, title, company, location, raw_description) VALUES (?, ?, ?, ?, ?)",
        (url, data["title"], data["company"], data["location"], data["raw_description"]),
    ) as cur:
        job_id = cur.lastrowid
    await db.commit()

    return JSONResponse({"job_id": job_id, "title": data["title"], "company": data["company"]})


@app.post("/api/jobs/add-manual")
async def add_job_manual(
    title: str = Form(""),
    company: str = Form(""),
    description: str = Form(...),
    db: aiosqlite.Connection = Depends(get_db),
):
    """Store a manually pasted job description (no URL scraping)."""
    import hashlib

    description = description.strip()
    if len(description) < 50:
        return JSONResponse(
            {"error": "Description is too short (minimum 50 characters)."},
            status_code=422,
        )

    title   = title.strip()   or "Untitled Job"
    company = company.strip() or ""

    # Synthetic unique key so the UNIQUE constraint on url still works
    slug = hashlib.md5(description[:200].encode()).hexdigest()[:12]
    synthetic_url = f"manual://{slug}"

    async with db.execute("SELECT id FROM jobs WHERE url = ?", (synthetic_url,)) as cur:
        existing = await cur.fetchone()
    if existing:
        return JSONResponse(
            {"error": "This description has already been added.", "job_id": existing[0]},
            status_code=409,
        )

    if len(description) > 8000:
        description = description[:8000] + "\n\n[...truncated for analysis]"

    async with db.execute(
        "INSERT INTO jobs (url, title, company, location, raw_description) VALUES (?, ?, ?, ?, ?)",
        (synthetic_url, title, company, "", description),
    ) as cur:
        job_id = cur.lastrowid
    await db.commit()

    return JSONResponse({"job_id": job_id, "title": title, "company": company})


# ── Resume comparison helpers ─────────────────────────────────────────────────

def _has_blocker(missing_skills: list) -> bool:
    return any(
        (s.get("severity") if isinstance(s, dict) else "") == "blocker"
        for s in missing_skills
    )


def _determine_better_fit(a: dict, b: dict) -> tuple:
    a_has_blocker = _has_blocker(a.get("missing_skills", []))
    b_has_blocker = _has_blocker(b.get("missing_skills", []))

    if a_has_blocker and not b_has_blocker:
        return b["resume_label"], f"No hard blockers vs {a['resume_label']} which has blockers"
    if b_has_blocker and not a_has_blocker:
        return a["resume_label"], f"No hard blockers vs {b['resume_label']} which has blockers"

    a_score = a.get("adjusted_score", 0)
    b_score = b.get("adjusted_score", 0)
    if a_score > b_score:
        return a["resume_label"], f"Higher adjusted score ({a_score} vs {b_score})"
    if b_score > a_score:
        return b["resume_label"], f"Higher adjusted score ({b_score} vs {a_score})"
    return "Tie", "Both resumes score equally for this role"


def _build_comparison(analyses: list) -> dict | None:
    seen: dict = {}
    for a in analyses:
        rid = a.get("resume_id")
        if rid not in seen:
            seen[rid] = a
        if len(seen) == 2:
            break
    if len(seen) < 2:
        return None

    ids = list(seen.keys())
    ra, rb = seen[ids[0]], seen[ids[1]]
    better, reason = _determine_better_fit(ra, rb)
    return {
        "resume_a":     ra,
        "resume_b":     rb,
        "better_fit":   better,
        "better_reason": reason,
    }


@app.post("/api/jobs/{job_id}/analyze")
async def analyze_job(
    job_id: int,
    resume_id: int = Form(...),
    provider: str = Form("anthropic"),
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute("SELECT raw_description FROM jobs WHERE id = ?", (job_id,)) as cur:
        job = await cur.fetchone()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async with db.execute("SELECT content FROM resumes WHERE id = ?", (resume_id,)) as cur:
        resume = await cur.fetchone()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    try:
        import time as _time
        _start = _time.monotonic()
        result = await analyze_match(resume["content"], job["raw_description"], provider)
        duration_seconds = int(_time.monotonic() - _start)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    # v1 backward-compat: plain skill name lists
    matched_v1 = json.dumps([
        s["skill"] if isinstance(s, dict) else s
        for s in result.get("matched_skills", [])
    ])
    missing_v1 = json.dumps([
        {"skill": s["skill"], "severity": s.get("severity", "minor")}
        if isinstance(s, dict) else {"skill": s, "severity": "minor"}
        for s in result.get("missing_skills", [])
    ])

    await db.execute(
        """INSERT INTO analyses
           (job_id, resume_id, score, adjusted_score, penalty_breakdown,
            matched_skills, missing_skills, reasoning, llm_provider, llm_model,
            matched_skills_v2, missing_skills_v2, suggestions,
            validation_errors, retry_count, used_fallback, duration_seconds,
            analysis_mode)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            job_id,
            resume_id,
            result["score"],
            result.get("adjusted_score", result["score"]),
            json.dumps(result.get("penalty_breakdown", {})),
            matched_v1,
            missing_v1,
            result["reasoning"],
            result["llm_provider"],
            result.get("llm_model", ""),
            json.dumps(result.get("matched_skills", [])),
            json.dumps(result.get("missing_skills", [])),
            json.dumps(result.get("suggestions", [])),
            result.get("validation_errors", ""),
            result.get("retry_count", 0),
            1 if result.get("used_fallback") else 0,
            duration_seconds,
            os.getenv("ANALYSIS_MODE", "standard"),
        ),
    )
    await db.commit()
    return JSONResponse(result)


@app.post("/api/jobs/{job_id}/application")
async def upsert_application(
    job_id: int,
    status: str = Form("not_applied"),
    recruiter_name: str = Form(""),
    recruiter_email: str = Form(""),
    recruiter_phone: str = Form(""),
    notes: str = Form(""),
    db: aiosqlite.Connection = Depends(get_db),
):
    await db.execute(
        """INSERT INTO applications (job_id, status, recruiter_name, recruiter_email, recruiter_phone, notes)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(job_id) DO UPDATE SET
               status=excluded.status,
               recruiter_name=excluded.recruiter_name,
               recruiter_email=excluded.recruiter_email,
               recruiter_phone=excluded.recruiter_phone,
               notes=excluded.notes,
               updated_at=CURRENT_TIMESTAMP""",
        (job_id, status, recruiter_name, recruiter_email, recruiter_phone, notes),
    )
    await db.commit()
    return JSONResponse({"ok": True})


@app.delete("/api/analyses/{analysis_id}")
async def delete_analysis(analysis_id: int, db: aiosqlite.Connection = Depends(get_db)):
    async with db.execute("SELECT id FROM analyses WHERE id = ?", (analysis_id,)) as cur:
        if not await cur.fetchone():
            raise HTTPException(status_code=404, detail="Analysis not found")
    await db.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    await db.commit()
    return JSONResponse({"ok": True})


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: int, db: aiosqlite.Connection = Depends(get_db)):
    await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    await db.commit()
    return JSONResponse({"ok": True})


@app.post("/api/resumes/add")
async def add_resume(
    label: str = Form(...),
    content: str = Form(...),
    db: aiosqlite.Connection = Depends(get_db),
):
    async with db.execute(
        "INSERT INTO resumes (label, content) VALUES (?, ?)", (label.strip(), content.strip())
    ) as cur:
        resume_id = cur.lastrowid
    await db.commit()
    return JSONResponse({"resume_id": resume_id, "label": label.strip()})


@app.delete("/api/resumes/{resume_id}")
async def delete_resume(resume_id: int, db: aiosqlite.Connection = Depends(get_db)):
    await db.execute("DELETE FROM resumes WHERE id = ?", (resume_id,))
    await db.commit()
    return JSONResponse({"ok": True})


@app.get("/api/jobs/{job_id}/description")
async def get_description(job_id: int, db: aiosqlite.Connection = Depends(get_db)):
    async with db.execute("SELECT raw_description FROM jobs WHERE id = ?", (job_id,)) as cur:
        row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404)
    return JSONResponse({"description": row["raw_description"]})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import signal
    import time
    import uvicorn
    import logging
    from launcher import Launcher, open_browser

    # Configure logging — same → / ✓ / ✗ pattern as Go version
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    logger = logging.getLogger("main")

    # ── Read initial config from .env ────────────────────────────────────────
    load_dotenv()
    initial_cfg = {
        "port":              int(os.getenv("APP_PORT", "8000")),
        "host":              os.getenv("APP_HOST", "127.0.0.1"),
        "db_path":           os.getenv("DB_PATH", "job_matcher.db"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "ollama_base_url":   os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_model":      os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "ollama_timeout":    int(os.getenv("OLLAMA_TIMEOUT", "600")),
    }

    # ── Start launcher ───────────────────────────────────────────────────────
    launcher_instance = Launcher(initial_cfg)

    uvicorn_server: list = []  # mutable container for current server reference

    def start_app(cfg: dict):
        """Start (or restart) the uvicorn server with new config."""
        host = cfg.get("host", "127.0.0.1")
        port = cfg.get("port", 8000)
        key  = cfg.get("anthropic_api_key", "")
        masked = (key[:12] + "...") if key else "not set"

        logger.info(f"\n  Config loaded:")
        logger.info(f"    Anthropic key : {masked}")
        logger.info(f"    Ollama model  : {cfg.get('ollama_model')}")
        logger.info(f"    DB path       : {cfg.get('db_path')}")
        logger.info(f"\n  Starting app on http://{host}:{port} …")

        # Reload .env so FastAPI picks up new values
        load_dotenv(override=True)
        os.environ["APP_HOST"]          = host
        os.environ["APP_PORT"]          = str(port)
        os.environ["ANTHROPIC_API_KEY"] = key
        os.environ["OLLAMA_MODEL"]      = cfg.get("ollama_model", "llama3.1:8b")
        os.environ["OLLAMA_BASE_URL"]   = cfg.get("ollama_base_url", "http://localhost:11434")
        os.environ["OLLAMA_TIMEOUT"]    = str(cfg.get("ollama_timeout", 600))

        config = uvicorn.Config("main:app", host=host, port=port, reload=False)
        server = uvicorn.Server(config)
        uvicorn_server.clear()
        uvicorn_server.append(server)

        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        logger.info(f"✓  Job Matcher running at http://{host}:{port}")

    def stop_app():
        """Gracefully stop the uvicorn server."""
        if uvicorn_server:
            logger.info("→ Stopping app server...")
            uvicorn_server[0].should_exit = True
            time.sleep(1)
            uvicorn_server.clear()
            logger.info("✓ App server stopped")
        else:
            logger.info("→ Stop requested but no server running")

    def restart_app(cfg: dict):
        """Stop current server and start with new config."""
        logger.info(f"↺  Restarting app with model={cfg.get('ollama_model')} port={cfg.get('port')}")
        stop_app()
        time.sleep(0.5)
        start_app(cfg)

    launcher_instance.on_start   = start_app
    launcher_instance.on_stop    = stop_app
    launcher_instance.on_restart = restart_app

    launcher_url = launcher_instance.start()

    # ── Print banner ─────────────────────────────────────────────────────────
    CYAN  = "\033[96m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"
    print(f"\n{CYAN}{BOLD}{'═' * 54}{RESET}")
    print(f"{CYAN}{BOLD}   Job Matcher{RESET}")
    print(f"{CYAN}{'═' * 54}{RESET}\n")
    print(f"  Launcher   {launcher_url}\n")
    print(f"  Opening launcher in your browser…")
    print(f"  Configure settings and click Start.\n")

    open_browser(launcher_url)

    # ── Keep main thread alive ────────────────────────────────────────────────
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down…")
        stop_app()
        launcher_instance.stop()
        print("  Stopped. Goodbye!")

