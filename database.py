import aiosqlite
import os


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
