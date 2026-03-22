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
        """)
        await db.commit()

        # Migrations: add columns to existing databases that predate these fields
        for migration in [
            "ALTER TABLE analyses ADD COLUMN llm_model TEXT DEFAULT ''",
            "ALTER TABLE analyses ADD COLUMN adjusted_score INTEGER DEFAULT 0",
            "ALTER TABLE analyses ADD COLUMN penalty_breakdown TEXT DEFAULT '{}'",
        ]:
            try:
                await db.execute(migration)
                await db.commit()
            except Exception:
                pass  # Column already exists — safe to ignore
