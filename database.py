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

import queries


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
        await db.executescript(queries.SCHEMA)
        await db.commit()

        # Migrations: add columns to existing databases that predate these fields
        for migration in queries.MIGRATIONS:
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
            queries.GET_COMPANY_META,
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
                queries.INSERT_OR_IGNORE_COMPANY_META,
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
            queries.UPSERT_COMPANY_META.format(cols=cols, placeholders=placeholders, set_clause=set_clause),
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
            queries.UPSERT_COMPANY_VETTING,
            (company_name, risk_level, assessment, signals_json, provider, model),
        )
        await db.commit()
