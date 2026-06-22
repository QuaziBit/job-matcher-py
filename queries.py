"""
queries.py — SQL text used by main.py / database.py.

This module holds raw SQL strings, and named string constants for every
fixed query. Callers in main.py / database.py import the constant they
need and call db.execute(...) / db.executescript(...) themselves — no
data-access layer, no connection handling here.

A small number of queries are genuinely dynamic (the column list or
placeholder count depends on runtime input, not just `?` parameter
values) — those get a builder function instead of a fixed string, since
a plain constant can't represent them. The function still only returns a
SQL string; it doesn't take a `db` connection or execute anything itself.

Organized by table/domain with a `# ── Section ──` header per group:
  - Jobs
  - Company meta
  - Analyses
  - Applications
  - Job emails
  - Resumes
  - MX cache
  - Schema / migrations (used by database.init_db)
"""


# ── Jobs ───────────────────────────────────────────────────────────────────

GET_JOB_ID_BY_URL = "SELECT id FROM jobs WHERE url = ?"

GET_MANUAL_JOB_ID_BY_URL_OR_SLUG = "SELECT id FROM jobs WHERE url = ? OR (url = ? AND ? = '')"

INSERT_JOB_WITH_COMPANY_URL = (
    "INSERT INTO jobs (url, title, company, location, company_url, raw_description) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)

INSERT_JOB_NO_COMPANY_URL = (
    "INSERT INTO jobs (url, title, company, location, raw_description) VALUES (?, ?, ?, ?, ?)"
)

GET_JOB_RAW_DESCRIPTION = "SELECT raw_description FROM jobs WHERE id = ?"

GET_JOB_FOR_SALARY = (
    "SELECT title, company, location, raw_description, salary_estimate FROM jobs WHERE id = ?"
)

UPDATE_JOB_SALARY_ESTIMATE = "UPDATE jobs SET salary_estimate = ? WHERE id = ?"

CLEAR_JOB_SALARY_ESTIMATE = "UPDATE jobs SET salary_estimate = '' WHERE id = ?"

DELETE_JOB = "DELETE FROM jobs WHERE id = ?"

GET_JOB_ID_AND_DESCRIPTION = "SELECT id, raw_description FROM jobs WHERE id = ?"

GET_JOB_ID_ONLY = "SELECT id FROM jobs WHERE id = ?"

GET_JOB_ID_AND_COMPANY = "SELECT id, company FROM jobs WHERE id = ?"

UPDATE_JOB_URL = "UPDATE jobs SET url = ? WHERE id = ?"

UPDATE_JOB_TITLE = "UPDATE jobs SET title = ? WHERE id = ?"

UPDATE_JOB_COMPANY = "UPDATE jobs SET company = ? WHERE id = ?"

UPDATE_JOB_LOCATION = "UPDATE jobs SET location = ? WHERE id = ?"

UPDATE_JOB_COMPANY_URL = "UPDATE jobs SET company_url = ? WHERE id = ?"

GET_JOB_FULL_ROW = "SELECT * FROM jobs WHERE id = ?"

# {where_sql} is filled in by the caller (built from request filter params).
COUNT_JOBS = "SELECT COUNT(*) FROM jobs j LEFT JOIN applications a ON a.job_id = j.id {where_sql}"

# {where_sql} is filled in by the caller (built from request filter params).
# Caller appends " LIMIT ? OFFSET ?" itself when paginating.
LIST_JOBS_PAGINATED = """
    SELECT j.id, j.url, j.title, j.company, j.location, j.scraped_at,
           COALESCE(a.status, 'not_applied') as status,
           (SELECT score          FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1) as best_score,
           (SELECT adjusted_score FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1) as adjusted_score,
           (SELECT llm_provider   FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1) as provider,
           (SELECT llm_model     FROM analyses WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1) as last_model,
           CASE WHEN (a.recruiter_name IS NOT NULL AND a.recruiter_name != '')
                  OR (a.recruiter_email IS NOT NULL AND a.recruiter_email != '')
                  OR (a.recruiter_phone IS NOT NULL AND a.recruiter_phone != '')
                THEN 1 ELSE 0 END as has_recruiter
    FROM jobs j
    LEFT JOIN applications a ON a.job_id = j.id
    {where_sql}
    ORDER BY j.scraped_at DESC
"""

GET_VETTING_ROWS = """
    SELECT
        j.id, j.title, j.company, j.url, j.scraped_at,
        a.status, a.recruiter_name, a.recruiter_email, a.recruiter_phone
    FROM jobs j
    LEFT JOIN applications a ON a.job_id = j.id
    ORDER BY j.company COLLATE NOCASE, j.scraped_at DESC
"""


# ── Company meta ────────────────────────────────────────────────────────────

GET_COMPANY_META = "SELECT * FROM company_meta WHERE company_name = ?"

INSERT_OR_IGNORE_COMPANY_META = "INSERT OR IGNORE INTO company_meta (company_name) VALUES (?)"

# {cols}/{placeholders}/{set_clause} are filled in by upsert_company_meta()
# at call time, since the column list depends on which fields were passed.
UPSERT_COMPANY_META = """INSERT INTO company_meta ({cols}) VALUES ({placeholders})
               ON CONFLICT(company_name) DO UPDATE SET {set_clause},
               crawled_at = CURRENT_TIMESTAMP"""

UPSERT_COMPANY_VETTING = """INSERT INTO company_meta (company_name, llm_risk_level, llm_assessment,
               llm_signals, llm_provider, llm_model, llm_assessed_at)
               VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(company_name) DO UPDATE SET
                   llm_risk_level  = excluded.llm_risk_level,
                   llm_assessment  = excluded.llm_assessment,
                   llm_signals     = excluded.llm_signals,
                   llm_provider    = excluded.llm_provider,
                   llm_model       = excluded.llm_model,
                   llm_assessed_at = excluded.llm_assessed_at"""

# Used by get_vetting_data, save_job_preview, add_job_manual, update_job_company_url,
# and get_job_detail to sync a job's company_url onto its company_meta row.
SYNC_COMPANY_URL_TO_META = """INSERT INTO company_meta (company_name, company_url)
                   VALUES (?, ?)
                   ON CONFLICT(company_name) DO UPDATE SET company_url = excluded.company_url"""

DELETE_COMPANY_META = "DELETE FROM company_meta WHERE company_name = ?"

# Used by update_job_company when a job's company name is edited — keeps
# the company_meta row (ratings, vetting data) associated with the new name.
RENAME_COMPANY_META = "UPDATE company_meta SET company_name = ? WHERE company_name = ?"

# {placeholders} depends on how many company names are being looked up at once
# (built from a dynamic IN (...) list) — genuinely variable, so this needs a
# small builder function rather than a fixed string with one format() slot.
_GET_COMPANY_META_BATCH_TEMPLATE = """SELECT company_name,
                   company_url,
                   glassdoor_url, glassdoor_rating, glassdoor_review_count,
                   linkedin_url, linkedin_employee_count, linkedin_founded,
                   bbb_url, bbb_rating,
                   indeed_url, indeed_rating, indeed_review_count,
                   llm_risk_level, llm_assessment,
                   llm_signals, llm_provider, llm_model, llm_assessed_at
            FROM company_meta WHERE company_name IN ({placeholders})"""


def get_company_meta_batch_query(company_count: int) -> str:
    """Build the company_meta batch-lookup query for `company_count` names.
    Used by get_vetting_data, which looks up vetting/rating data for every
    distinct company on the vetting page in one query rather than one
    query per company."""
    placeholders = ",".join(["?"] * company_count)
    return _GET_COMPANY_META_BATCH_TEMPLATE.format(placeholders=placeholders)


# ── Analyses ──────────────────────────────────────────────────────────────────

INSERT_ANALYSIS = """INSERT INTO analyses
               (job_id, resume_id, score, adjusted_score, penalty_breakdown,
                matched_skills, missing_skills, reasoning, llm_provider, llm_model,
                matched_skills_v2, missing_skills_v2, suggestions,
                validation_errors, retry_count, used_fallback, duration_seconds,
                analysis_mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

GET_ANALYSIS_ID = "SELECT id FROM analyses WHERE id = ?"

DELETE_ANALYSIS = "DELETE FROM analyses WHERE id = ?"

GET_ANALYSES_WITH_RESUME_LABEL = """
    SELECT a.*, r.label as resume_label
    FROM analyses a JOIN resumes r ON r.id = a.resume_id
    WHERE a.job_id = ?
    ORDER BY a.created_at DESC
"""


# ── Applications ──────────────────────────────────────────────────────────────

UPSERT_APPLICATION = """INSERT INTO applications (job_id, status, recruiter_name, recruiter_email, recruiter_phone, notes)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(job_id) DO UPDATE SET
                   status=excluded.status,
                   recruiter_name=excluded.recruiter_name,
                   recruiter_email=excluded.recruiter_email,
                   recruiter_phone=excluded.recruiter_phone,
                   notes=excluded.notes,
                   updated_at=CURRENT_TIMESTAMP"""

GET_APPLICATION_BY_JOB = "SELECT * FROM applications WHERE job_id = ?"


# ── Job emails ────────────────────────────────────────────────────────────────

GET_JOB_EMAIL = "SELECT id, raw_html, created_at FROM job_emails WHERE job_id = ?"

UPSERT_JOB_EMAIL = """INSERT INTO job_emails (job_id, raw_html)
               VALUES (?, ?)
               ON CONFLICT(job_id) DO UPDATE SET raw_html=excluded.raw_html,
               created_at=CURRENT_TIMESTAMP"""

DELETE_JOB_EMAIL = "DELETE FROM job_emails WHERE job_id = ?"


# ── Resumes ───────────────────────────────────────────────────────────────────

INSERT_RESUME = "INSERT INTO resumes (label, content) VALUES (?, ?)"

GET_RESUME = "SELECT id, label, content, created_at, LENGTH(content) as char_count FROM resumes WHERE id = ?"

GET_RESUME_CONTENT = "SELECT content FROM resumes WHERE id = ?"

DELETE_RESUME = "DELETE FROM resumes WHERE id = ?"

LIST_RESUME_IDS_AND_LABELS = "SELECT id, label FROM resumes ORDER BY created_at DESC"

LIST_RESUMES_FULL = "SELECT id, label, created_at, LENGTH(content) as char_count FROM resumes ORDER BY created_at DESC"


# ── MX cache ──────────────────────────────────────────────────────────────────

GET_MX_CACHE = "SELECT domain, has_mx, mx_records FROM domain_mx_cache"


# ── Schema (used by database.init_db) ────────────────────────────────────────

SCHEMA = """
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL UNIQUE,
        title TEXT,
        company TEXT,
        location TEXT,
        company_url TEXT DEFAULT '',
        raw_description TEXT,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        resume_id INTEGER NOT NULL,
        score INTEGER NOT NULL,
        adjusted_score INTEGER NOT NULL DEFAULT 0,
        penalty_breakdown TEXT DEFAULT '{}',
        matched_skills TEXT,
        missing_skills TEXT,
        reasoning TEXT,
        llm_provider TEXT DEFAULT 'anthropic',
        llm_model TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
        FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL UNIQUE,
        status TEXT DEFAULT 'not_applied',
        recruiter_name TEXT,
        recruiter_email TEXT,
        recruiter_phone TEXT,
        notes TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS job_emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL UNIQUE,
        raw_html TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS company_meta (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_name TEXT NOT NULL UNIQUE,
        glassdoor_url TEXT DEFAULT '',
        glassdoor_rating REAL DEFAULT NULL,
        glassdoor_review_count INTEGER DEFAULT NULL,
        linkedin_url TEXT DEFAULT '',
        linkedin_employee_count TEXT DEFAULT '',
        linkedin_founded TEXT DEFAULT '',
        bbb_url TEXT DEFAULT '',
        bbb_rating TEXT DEFAULT '',
        indeed_url TEXT DEFAULT '',
        indeed_rating REAL DEFAULT NULL,
        company_url TEXT DEFAULT '',
        indeed_review_count INTEGER DEFAULT NULL,
        crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        llm_assessment TEXT DEFAULT NULL,
        llm_risk_level TEXT DEFAULT NULL,
        llm_signals TEXT DEFAULT NULL,
        llm_provider TEXT DEFAULT NULL,
        llm_model TEXT DEFAULT NULL,
        llm_assessed_at TIMESTAMP DEFAULT NULL
    );

    CREATE TABLE IF NOT EXISTS domain_mx_cache (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        domain      TEXT NOT NULL UNIQUE,
        has_mx      INTEGER NOT NULL DEFAULT 0,
        mx_records  TEXT DEFAULT '',
        checked_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
"""

# Migrations: add columns to existing databases that predate these fields.
# database.init_db() runs each of these in a try/except, ignoring failures
# (column already exists), so order doesn't matter for correctness — kept
# in the original chronological order for readability/history only.
MIGRATIONS = [
    "ALTER TABLE analyses ADD COLUMN llm_model TEXT DEFAULT ''",
    "ALTER TABLE analyses ADD COLUMN adjusted_score INTEGER DEFAULT 0",
    "ALTER TABLE analyses ADD COLUMN penalty_breakdown TEXT DEFAULT '{}'",
    "ALTER TABLE analyses ADD COLUMN matched_skills_v2 TEXT DEFAULT '[]'",
    "ALTER TABLE analyses ADD COLUMN missing_skills_v2 TEXT DEFAULT '[]'",
    "ALTER TABLE analyses ADD COLUMN suggestions TEXT DEFAULT '[]'",
    "ALTER TABLE analyses ADD COLUMN validation_errors TEXT DEFAULT ''",
    "ALTER TABLE analyses ADD COLUMN retry_count INTEGER DEFAULT 0",
    "ALTER TABLE analyses ADD COLUMN used_fallback INTEGER DEFAULT 0",
    "ALTER TABLE analyses ADD COLUMN duration_seconds INTEGER DEFAULT 0",
    "ALTER TABLE analyses ADD COLUMN analysis_mode TEXT DEFAULT 'standard'",
    "ALTER TABLE jobs ADD COLUMN salary_estimate TEXT DEFAULT ''",
    "ALTER TABLE company_meta ADD COLUMN llm_assessment TEXT DEFAULT NULL",
    "ALTER TABLE company_meta ADD COLUMN llm_risk_level TEXT DEFAULT NULL",
    "ALTER TABLE company_meta ADD COLUMN llm_signals TEXT DEFAULT NULL",
    "ALTER TABLE company_meta ADD COLUMN llm_provider TEXT DEFAULT NULL",
    "ALTER TABLE company_meta ADD COLUMN llm_model TEXT DEFAULT NULL",
    "ALTER TABLE company_meta ADD COLUMN llm_assessed_at TIMESTAMP DEFAULT NULL",
    "ALTER TABLE company_meta ADD COLUMN indeed_url TEXT DEFAULT ''",
    "ALTER TABLE company_meta ADD COLUMN indeed_rating REAL DEFAULT NULL",
    "ALTER TABLE company_meta ADD COLUMN indeed_review_count INTEGER DEFAULT NULL",
    "ALTER TABLE jobs ADD COLUMN company_url TEXT DEFAULT ''",
    "ALTER TABLE company_meta ADD COLUMN company_url TEXT DEFAULT ''",
]
