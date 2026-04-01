import json
import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

import aiosqlite
from database import get_db, init_db
from scraper import scrape_job, assess_job_text_quality
from analyzer import analyze_match, estimate_salary, extract_salary, _job_has_salary, _ollama_model
from analyzer.config import anthropic_model, openai_model, gemini_model
from analyzer.known_models import KNOWN_MODELS
from health import run_health_checks
from utils import build_comparison, format_duration, clean_text

load_dotenv()

logger = logging.getLogger("main")

# ── App setup ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    run_health_checks()
    yield


app = FastAPI(title="Job Matcher", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.filters["format_duration"] = format_duration


# ── Pages ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: aiosqlite.Connection = Depends(get_db)):
    try:
        async with db.execute("SELECT id, label FROM resumes ORDER BY created_at DESC") as cur:
            resumes = [dict(r) for r in await cur.fetchall()]
    except Exception as e:
        logger.error(f"✗ index page DB error: {e}")
        resumes = []
    return templates.TemplateResponse("index.html", {"request": request, "resumes": resumes})


@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_detail(job_id: int, request: Request, db: aiosqlite.Connection = Depends(get_db)):
    try:
        async with db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cur:
            job = await cur.fetchone()
    except Exception as e:
        logger.error(f"✗ job_detail DB error fetching job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job = dict(job)

    try:
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

        async with db.execute("SELECT id, label FROM resumes ORDER BY created_at DESC") as cur:
            resumes = [dict(r) for r in await cur.fetchall()]
    except Exception as e:
        logger.error(f"✗ job_detail DB error fetching related data for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    # Parse JSON fields — v2 columns preferred, v1 as fallback
    for analysis in analyses:
        try:
            v2_matched = analysis.get("matched_skills_v2", "[]") or "[]"
            parsed_v2  = json.loads(v2_matched)
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

        try:
            v2_missing = analysis.get("missing_skills_v2", "[]") or "[]"
            parsed_v2  = json.loads(v2_missing)
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

        try:
            pb = analysis.get("penalty_breakdown")
            if isinstance(pb, str):
                analysis["penalty_breakdown"] = json.loads(pb)
        except Exception:
            analysis["penalty_breakdown"] = {}

        try:
            sugg = analysis.get("suggestions", "[]") or "[]"
            analysis["suggestions"] = json.loads(sugg) if isinstance(sugg, str) else sugg
        except Exception:
            analysis["suggestions"] = []

        if not analysis.get("adjusted_score"):
            analysis["adjusted_score"] = analysis["score"]

        analysis["used_fallback"] = bool(analysis.get("used_fallback", 0))

    text_quality   = assess_job_text_quality(job.get("raw_description") or "")
    comparison     = build_comparison(analyses)
    last_resume_id = analyses[0]["resume_id"] if analyses else None

    salary_data = json.loads(job["salary_estimate"]) if job.get("salary_estimate") else None
    if analyses:
        last_provider = analyses[0]["llm_provider"]
    elif salary_data and salary_data.get("llm_provider"):
        last_provider = salary_data["llm_provider"]
    else:
        last_provider = "anthropic"

    return templates.TemplateResponse("job_detail.html", {
        "request":          request,
        "job":              job,
        "application":      application,
        "analyses":         analyses,
        "resumes":          resumes,
        "ollama_model":     _ollama_model(),
        "anthropic_model":  anthropic_model(),
        "openai_model":     openai_model(),
        "gemini_model":     gemini_model(),
        "text_quality":     text_quality,
        "comparison":       comparison,
        "analysis_mode":    os.getenv("ANALYSIS_MODE", "standard"),
        "salary_estimate":  json.loads(job["salary_estimate"]) if job.get("salary_estimate") else None,
        "has_salary_in_jd": _job_has_salary(job.get("raw_description", "")),
        "last_resume_id":   last_resume_id,
        "last_provider":    last_provider,
    })


@app.get("/resumes", response_class=HTMLResponse)
async def resumes_page(request: Request, db: aiosqlite.Connection = Depends(get_db)):
    try:
        async with db.execute("SELECT * FROM resumes ORDER BY created_at DESC") as cur:
            resumes = [dict(r) for r in await cur.fetchall()]
    except Exception as e:
        logger.error(f"✗ resumes page DB error: {e}")
        resumes = []
    return templates.TemplateResponse("resumes.html", {"request": request, "resumes": resumes})


# ── API Endpoints ──────────────────────────────────────────────────────────────

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
    logger_jl = logging.getLogger("jobs_list")

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

    valid_providers = {"", "anthropic", "openai", "gemini", "ollama", "manual"}
    if provider not in valid_providers:
        return JSONResponse({"error": f"invalid provider {provider!r} — must be one of: anthropic, openai, gemini, ollama, manual"}, status_code=400)

    logger_jl.info(f"→ /api/jobs/list page={page} per_page={per_page} search={search!r} status={status!r} score={score!r} provider={provider!r}")

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
            where.append("(SELECT COUNT(*) FROM analyses WHERE job_id = j.id) = 0")
        elif score == "5":
            where.append(
                "COALESCE((SELECT adjusted_score FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1),"
                "(SELECT score FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1), 0) = 5"
            )
        else:
            min_score = int(score)
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

    count_query = f"SELECT COUNT(*) FROM jobs j LEFT JOIN applications a ON a.job_id = j.id {where_sql}"
    try:
        async with db.execute(count_query, args) as cur:
            row   = await cur.fetchone()
            total = row[0] if row else 0
    except Exception as e:
        logger_jl.error(f"✗ count query failed: {e}")
        return JSONResponse({"error": "Failed to load jobs from database. Check the terminal for details."}, status_code=500)

    total_pages = 1
    if per_page > 0 and total > 0:
        total_pages = (total + per_page - 1) // per_page

    if page > total_pages and total_pages > 0:
        page = total_pages

    paginated_query = base_query
    paginated_args  = list(args)
    if per_page > 0:
        offset = (page - 1) * per_page
        paginated_query += " LIMIT ? OFFSET ?"
        paginated_args  += [per_page, offset]

    try:
        async with db.execute(paginated_query, paginated_args) as cur:
            rows = await cur.fetchall()
            jobs = [dict(r) for r in rows]
    except Exception as e:
        logger_jl.error(f"✗ jobs query failed: {e}")
        return JSONResponse({"error": "Failed to load jobs from database. Check the terminal for details."}, status_code=500)

    for job in jobs:
        job["is_manual"] = (job.get("url") or "").startswith("manual://")

    logger_jl.info(f"✓ /api/jobs/list total={total} page={page}/{total_pages} returned={len(jobs)}")

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

    try:
        async with db.execute("SELECT id FROM jobs WHERE url = ?", (url,)) as cur:
            existing = await cur.fetchone()
    except Exception as e:
        logger.error(f"✗ add_job DB error checking duplicate: {e}")
        return JSONResponse({"error": "Database error. Check the terminal for details."}, status_code=500)

    if existing:
        return JSONResponse({"error": "This URL has already been added.", "job_id": existing[0]}, status_code=409)

    try:
        data = await scrape_job(url)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        logger.error(f"✗ add_job unexpected scrape error: {e}")
        return JSONResponse({"error": "Unexpected error while scraping. Check the terminal for details."}, status_code=500)

    try:
        async with db.execute(
            "INSERT INTO jobs (url, title, company, location, raw_description) VALUES (?, ?, ?, ?, ?)",
            (url, data["title"], data["company"], data["location"], clean_text(data["raw_description"])),
        ) as cur:
            job_id = cur.lastrowid
        await db.commit()
    except Exception as e:
        logger.error(f"✗ add_job DB insert error: {e}")
        return JSONResponse({"error": "Failed to save job. Check the terminal for details."}, status_code=500)

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

    description = clean_text(description.strip())
    if len(description) < 50:
        return JSONResponse(
            {"error": "Description is too short (minimum 50 characters)."},
            status_code=422,
        )

    title         = title.strip()   or "Untitled Job"
    company       = company.strip() or ""
    slug          = hashlib.md5(description[:200].encode()).hexdigest()[:12]
    synthetic_url = f"manual://{slug}"

    try:
        async with db.execute("SELECT id FROM jobs WHERE url = ?", (synthetic_url,)) as cur:
            existing = await cur.fetchone()
    except Exception as e:
        logger.error(f"✗ add_job_manual DB error checking duplicate: {e}")
        return JSONResponse({"error": "Database error. Check the terminal for details."}, status_code=500)

    if existing:
        return JSONResponse(
            {"error": "This description has already been added.", "job_id": existing[0]},
            status_code=409,
        )

    if len(description) > 8000:
        description = description[:8000] + "\n\n[...truncated for analysis]"

    try:
        async with db.execute(
            "INSERT INTO jobs (url, title, company, location, raw_description) VALUES (?, ?, ?, ?, ?)",
            (synthetic_url, title, company, "", description),
        ) as cur:
            job_id = cur.lastrowid
        await db.commit()
    except Exception as e:
        logger.error(f"✗ add_job_manual DB insert error: {e}")
        return JSONResponse({"error": "Failed to save job. Check the terminal for details."}, status_code=500)

    return JSONResponse({"job_id": job_id, "title": title, "company": company})


@app.post("/api/jobs/{job_id}/analyze")
async def analyze_job(
    job_id: int,
    resume_id: int = Form(...),
    provider: str = Form("anthropic"),
    analysis_mode: str = Form(""),
    ollama_model: str = Form(""),
    cloud_model: str = Form(""),
    db: aiosqlite.Connection = Depends(get_db),
):
    try:
        async with db.execute("SELECT raw_description FROM jobs WHERE id = ?", (job_id,)) as cur:
            job = await cur.fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        async with db.execute("SELECT content FROM resumes WHERE id = ?", (resume_id,)) as cur:
            resume = await cur.fetchone()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ analyze_job DB error fetching job/resume: {e}")
        return JSONResponse({"error": "Database error. Check the terminal for details."}, status_code=500)

    try:
        import time as _time
        # Apply analysis_mode, ollama_model, and cloud_model from form if provided
        _valid_modes = {"fast", "standard", "detailed"}
        if analysis_mode and analysis_mode in _valid_modes:
            os.environ["ANALYSIS_MODE"] = analysis_mode
        if ollama_model and provider == "ollama":
            os.environ["OLLAMA_MODEL"] = ollama_model
        if cloud_model:
            _model_env = {
                "anthropic": "ANTHROPIC_MODEL",
                "openai":    "OPENAI_MODEL",
                "gemini":    "GEMINI_MODEL",
            }
            if env_key := _model_env.get(provider):
                os.environ[env_key] = cloud_model
        _start           = _time.monotonic()
        result           = await analyze_match(resume["content"], job["raw_description"], provider)
        duration_seconds = int(_time.monotonic() - _start)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        logger.error(f"✗ analyze_job LLM error for job {job_id}: {e}")
        return JSONResponse({"error": "Analysis failed unexpectedly. Check the terminal for details."}, status_code=500)

    matched_v1 = json.dumps([
        s["skill"] if isinstance(s, dict) else s
        for s in result.get("matched_skills", [])
    ])
    missing_v1 = json.dumps([
        {"skill": s["skill"], "severity": s.get("severity", "minor")}
        if isinstance(s, dict) else {"skill": s, "severity": "minor"}
        for s in result.get("missing_skills", [])
    ])

    try:
        await db.execute(
            """INSERT INTO analyses
               (job_id, resume_id, score, adjusted_score, penalty_breakdown,
                matched_skills, missing_skills, reasoning, llm_provider, llm_model,
                matched_skills_v2, missing_skills_v2, suggestions,
                validation_errors, retry_count, used_fallback, duration_seconds,
                analysis_mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id, resume_id,
                result["score"],
                result.get("adjusted_score", result["score"]),
                json.dumps(result.get("penalty_breakdown", {})),
                matched_v1, missing_v1,
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
                result.get("analysis_mode", os.getenv("ANALYSIS_MODE", "standard")),
            ),
        )
        await db.commit()
    except Exception as e:
        logger.error(f"✗ analyze_job DB insert error for job {job_id}: {e}")
        return JSONResponse({"error": "Failed to save analysis. Check the terminal for details."}, status_code=500)

    return JSONResponse(result)


@app.post("/api/jobs/{job_id}/estimate-salary")
async def estimate_job_salary(
    job_id: int,
    provider: str = Form("anthropic"),
    db: aiosqlite.Connection = Depends(get_db),
):
    """Estimate or extract salary for a job using the configured LLM."""
    try:
        async with db.execute(
            "SELECT title, company, location, raw_description, salary_estimate FROM jobs WHERE id = ?",
            (job_id,)
        ) as cur:
            job = await cur.fetchone()
    except Exception as e:
        logger.error(f"✗ estimate_job_salary DB error fetching job {job_id}: {e}")
        return JSONResponse({"error": "Database error. Check the terminal for details."}, status_code=500)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["salary_estimate"]:
        return JSONResponse(json.loads(job["salary_estimate"]))

    SALARY_INCOMPATIBLE_MODELS = ["gemma3"]
    if provider == "ollama":
        current_model = _ollama_model()
        if any(m in current_model for m in SALARY_INCOMPATIBLE_MODELS):
            return JSONResponse(
                {"error": f"{current_model} is not supported for salary estimation. Please switch to Anthropic or llama3.1:8b."},
                status_code=422,
            )

    has_salary = _job_has_salary(job["raw_description"] or "")

    try:
        if has_salary:
            try:
                result = await extract_salary(
                    title=job["title"] or "",
                    company=job["company"] or "",
                    location=job["location"] or "",
                    job_description=job["raw_description"] or "",
                    provider=provider,
                )
            except ValueError:
                logger.warning(f"→ extract_salary failed for job {job_id}, falling back to estimate_salary")
                result = await estimate_salary(
                    title=job["title"] or "",
                    company=job["company"] or "",
                    location=job["location"] or "",
                    job_description=job["raw_description"] or "",
                    provider=provider,
                    _skip_salary_check=True,
                )
                result["source"] = "estimated"
        else:
            result = await estimate_salary(
                title=job["title"] or "",
                company=job["company"] or "",
                location=job["location"] or "",
                job_description=job["raw_description"] or "",
                provider=provider,
            )
            result["source"] = "estimated"
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        logger.error(f"✗ estimate_job_salary LLM error for job {job_id}: {e}")
        return JSONResponse({"error": "Salary estimation failed unexpectedly. Check the terminal for details."}, status_code=500)

    try:
        await db.execute(
            "UPDATE jobs SET salary_estimate = ? WHERE id = ?",
            (json.dumps(result), job_id)
        )
        await db.commit()
    except Exception as e:
        logger.error(f"✗ estimate_job_salary DB update error for job {job_id}: {e}")

    return JSONResponse(result)


@app.delete("/api/jobs/{job_id}/salary-estimate")
async def clear_salary_estimate(job_id: int, db: aiosqlite.Connection = Depends(get_db)):
    """Clear a cached salary estimate so it can be re-run."""
    try:
        await db.execute("UPDATE jobs SET salary_estimate = '' WHERE id = ?", (job_id,))
        await db.commit()
    except Exception as e:
        logger.error(f"✗ clear_salary_estimate DB error for job {job_id}: {e}")
        return JSONResponse({"error": "Failed to clear salary estimate."}, status_code=500)
    return JSONResponse({"ok": True})


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
    try:
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
    except Exception as e:
        logger.error(f"✗ upsert_application DB error for job {job_id}: {e}")
        return JSONResponse({"error": "Failed to save application. Check the terminal for details."}, status_code=500)
    return JSONResponse({"ok": True})


@app.delete("/api/analyses/{analysis_id}")
async def delete_analysis(analysis_id: int, db: aiosqlite.Connection = Depends(get_db)):
    try:
        async with db.execute("SELECT id FROM analyses WHERE id = ?", (analysis_id,)) as cur:
            if not await cur.fetchone():
                raise HTTPException(status_code=404, detail="Analysis not found")
        await db.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        await db.commit()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ delete_analysis DB error for analysis {analysis_id}: {e}")
        return JSONResponse({"error": "Failed to delete analysis."}, status_code=500)
    return JSONResponse({"ok": True})


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: int, db: aiosqlite.Connection = Depends(get_db)):
    try:
        await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        await db.commit()
    except Exception as e:
        logger.error(f"✗ delete_job DB error for job {job_id}: {e}")
        return JSONResponse({"error": "Failed to delete job."}, status_code=500)
    return JSONResponse({"ok": True})


@app.post("/api/resumes/add")
async def add_resume(
    label: str = Form(...),
    content: str = Form(...),
    db: aiosqlite.Connection = Depends(get_db),
):
    try:
        async with db.execute(
            "INSERT INTO resumes (label, content) VALUES (?, ?)", (label.strip(), clean_text(content.strip()))
        ) as cur:
            resume_id = cur.lastrowid
        await db.commit()
    except Exception as e:
        logger.error(f"✗ add_resume DB error: {e}")
        return JSONResponse({"error": "Failed to save resume. Check the terminal for details."}, status_code=500)
    return JSONResponse({"resume_id": resume_id, "label": label.strip()})


@app.delete("/api/resumes/{resume_id}")
async def delete_resume(resume_id: int, db: aiosqlite.Connection = Depends(get_db)):
    try:
        await db.execute("DELETE FROM resumes WHERE id = ?", (resume_id,))
        await db.commit()
    except Exception as e:
        logger.error(f"✗ delete_resume DB error for resume {resume_id}: {e}")
        return JSONResponse({"error": "Failed to delete resume."}, status_code=500)
    return JSONResponse({"ok": True})


@app.get("/api/jobs/{job_id}/description")
async def get_description(job_id: int, db: aiosqlite.Connection = Depends(get_db)):
    try:
        async with db.execute("SELECT raw_description FROM jobs WHERE id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
    except Exception as e:
        logger.error(f"✗ get_description DB error for job {job_id}: {e}")
        return JSONResponse({"error": "Database error."}, status_code=500)
    if not row:
        raise HTTPException(status_code=404)
    return JSONResponse({"description": row["raw_description"]})


# ── Known models endpoint ────────────────────────────────────────────────────

@app.get("/api/providers/models")
async def get_provider_models(provider: str):
    """Return known models for a cloud provider with cost labels."""
    models = KNOWN_MODELS.get(provider, [])
    return JSONResponse({"provider": provider, "models": models})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    import uvicorn
    from launcher import Launcher, open_browser

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    load_dotenv()
    initial_cfg = {
        "port":              int(os.getenv("APP_PORT", "8000")),
        "host":              os.getenv("APP_HOST", "127.0.0.1"),
        "db_path":           os.getenv("DB_PATH", "job_matcher.db"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "anthropic_model":   os.getenv("ANTHROPIC_MODEL", ""),
        "openai_api_key":    os.getenv("OPENAI_API_KEY", ""),
        "gemini_api_key":    os.getenv("GEMINI_API_KEY", ""),
        "ollama_base_url":   os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_model":      os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "ollama_timeout":    int(os.getenv("OLLAMA_TIMEOUT", "600")),
        "analysis_mode":     os.getenv("ANALYSIS_MODE", "standard"),
        "show_more_logs":    os.getenv("SHOW_MORE_LOGS", "").lower() in ("1", "true", "yes"),
    }

    launcher_instance = Launcher(initial_cfg)
    uvicorn_server: list = []

    def start_app(cfg: dict):
        host   = cfg.get("host", "127.0.0.1")
        port   = cfg.get("port", 8000)
        key    = cfg.get("anthropic_api_key", "")
        masked = (key[:12] + "...") if key else "not set"

        logger.info(f"\n  Config loaded:")
        logger.info(f"    Anthropic key : {masked}")
        logger.info(f"    Ollama model  : {cfg.get('ollama_model')}")
        logger.info(f"    DB path       : {cfg.get('db_path')}")
        logger.info(f"\n  Starting app on http://{host}:{port} …")

        load_dotenv(override=True)
        os.environ["APP_HOST"]          = host
        os.environ["APP_PORT"]          = str(port)
        os.environ["ANTHROPIC_API_KEY"] = key
        if m := cfg.get("anthropic_model", ""): os.environ["ANTHROPIC_MODEL"] = m
        os.environ["OPENAI_API_KEY"]    = cfg.get("openai_api_key", "")
        os.environ["GEMINI_API_KEY"]    = cfg.get("gemini_api_key", "")
        os.environ["OLLAMA_MODEL"]      = cfg.get("ollama_model", "llama3.1:8b")
        os.environ["OLLAMA_BASE_URL"]   = cfg.get("ollama_base_url", "http://localhost:11434")
        os.environ["OLLAMA_TIMEOUT"]    = str(cfg.get("ollama_timeout", 600))

        config = uvicorn.Config("main:app", host=host, port=port, reload=False)
        server = uvicorn.Server(config)
        uvicorn_server.clear()
        uvicorn_server.append(server)
        threading.Thread(target=server.run, daemon=True).start()
        logger.info(f"✓  Job Matcher running at http://{host}:{port}")

    def stop_app():
        if uvicorn_server:
            logger.info("→ Stopping app server...")
            uvicorn_server[0].should_exit = True
            time.sleep(1)
            uvicorn_server.clear()
            logger.info("✓ App server stopped")
        else:
            logger.info("→ Stop requested but no server running")

    def restart_app(cfg: dict):
        logger.info(f"↺  Restarting app with model={cfg.get('ollama_model')} port={cfg.get('port')}")
        stop_app()
        time.sleep(0.5)
        start_app(cfg)

    launcher_instance.on_start   = start_app
    launcher_instance.on_stop    = stop_app
    launcher_instance.on_restart = restart_app

    launcher_url = launcher_instance.start()

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

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down…")
        stop_app()
        launcher_instance.stop()
        print("  Stopped. Goodbye!")
