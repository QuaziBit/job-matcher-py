import os
import sys

# ── SQLite driver selection ────────────────────────────────────────────────────
# Python's built-in sqlite3 depends on libsqlite3 being present on the OS.
# On minimal Linux systems (stripped Docker images, some cloud environments)
# it may be missing. pysqlite3-binary bundles its own SQLite for those cases.
# On Windows and macOS, sqlite3 is always available — no fallback needed.
#
# To install the fallback on Linux only:
#   pip install pysqlite3-binary
#
_sqlite3_ok = False
if sys.platform != "win32" and sys.platform != "darwin":
    # Only attempt fallback on Linux where sqlite3 may be absent
    try:
        import sqlite3 as _test
        _test.connect(":memory:").execute("SELECT sqlite_version()").fetchone()
        _sqlite3_ok = True
    except Exception:
        pass

    if not _sqlite3_ok:
        try:
            import pysqlite3
            sys.modules["sqlite3"] = pysqlite3
            print("  ✓  SQLite  using bundled pysqlite3 (system sqlite3 unavailable)")
        except ImportError:
            print("  ⚠  SQLite  pysqlite3-binary not installed — run: pip install pysqlite3-binary")
            print("             Falling back to system sqlite3 (may fail on minimal systems)")

import aiosqlite


def _db_path() -> str:
    """Read DB_PATH at call time so tests can override it via os.environ."""
    return os.getenv("DB_PATH", "job_matcher.db")


async def get_db():
    db = await aiosqlite.connect(_db_path())
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys = ON")
    try:
        yield db
    finally:
        await db.close()


async def init_db():
    async with aiosqlite.connect(_db_path()) as db:
        await db.executescript("""
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
        """)
        await db.commit()

        # Migrations: add columns to existing databases that predate these fields
        for migration in [
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
        ]:
            try:
                await db.execute(migration)
                await db.commit()
            except Exception:
                pass  # Column already exists — safe to ignore


# ── company_meta helpers ──────────────────────────────────────────────────────

COMPANY_META_FIELDS = [
    "glassdoor_url", "glassdoor_rating", "glassdoor_review_count",
    "linkedin_url", "linkedin_employee_count", "linkedin_founded",
    "bbb_url", "bbb_rating",
    "indeed_url", "indeed_rating", "indeed_review_count",
    "company_url",
    "crawled_at",
    "llm_assessment", "llm_risk_level", "llm_signals",
    "llm_provider", "llm_model", "llm_assessed_at",
]


async def get_company_meta(company_name: str) -> dict | None:
    """Return cached company_meta row or None if not found."""
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM company_meta WHERE company_name = ?",
            (company_name,)
        ) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    return dict(row)


async def upsert_company_meta(company_name: str, data: dict) -> None:
    """Insert or update only the provided fields in company_meta.
    Uses ON CONFLICT DO UPDATE so existing columns are preserved."""
    fields = [f for f in COMPANY_META_FIELDS if f != "crawled_at" and f in data]
    if not fields:
        # Nothing to write — just ensure the row exists
        async with aiosqlite.connect(_db_path()) as db:
            await db.execute(
                "INSERT OR IGNORE INTO company_meta (company_name) VALUES (?)",
                (company_name,),
            )
            await db.commit()
        return
    set_clause = ", ".join(f"{f} = excluded.{f}" for f in fields)
    cols = ", ".join(["company_name"] + fields + ["crawled_at"])
    placeholders = ", ".join(["?"] * (len(fields) + 1)) + ", CURRENT_TIMESTAMP"
    values = [company_name] + [data.get(f) for f in fields]
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(
            f"""INSERT INTO company_meta ({cols}) VALUES ({placeholders})
               ON CONFLICT(company_name) DO UPDATE SET {set_clause},
               crawled_at = CURRENT_TIMESTAMP""",
            values,
        )
        await db.commit()


async def upsert_company_vetting(
    company_name: str,
    risk_level: str,
    assessment: str,
    signals: list,
    provider: str,
    model: str,
) -> None:
    """Update the LLM vetting columns for an existing company_meta row.
    Creates the row if it doesn't exist yet."""
    import json as _json
    signals_json = _json.dumps(signals) if signals else "[]"
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(
            """INSERT INTO company_meta (company_name, llm_risk_level, llm_assessment,
               llm_signals, llm_provider, llm_model, llm_assessed_at)
               VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(company_name) DO UPDATE SET
                   llm_risk_level  = excluded.llm_risk_level,
                   llm_assessment  = excluded.llm_assessment,
                   llm_signals     = excluded.llm_signals,
                   llm_provider    = excluded.llm_provider,
                   llm_model       = excluded.llm_model,
                   llm_assessed_at = excluded.llm_assessed_at""",
            (company_name, risk_level, assessment, signals_json, provider, model),
        )
        await db.commit()
