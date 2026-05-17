"""
tests/test_database.py — Integration tests for database module.
Uses a real temporary SQLite DB — no mocking.
"""

import os
import tempfile
import unittest

from tests.mock_data import run, MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS


class TestDatabase(unittest.TestCase):
    """
    Integration tests against a real temporary SQLite DB.

    Each test method wraps all async operations into a single coroutine and
    calls run() exactly once — this avoids any cross-loop connection issues.
    """

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        os.environ["DB_PATH"] = self.tmp.name

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_init_creates_all_tables(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ) as cur:
                    return {row[0] for row in await cur.fetchall()}

        tables = run(ops())
        self.assertIn("resumes", tables)
        self.assertIn("jobs", tables)
        self.assertIn("analyses", tables)
        self.assertIn("applications", tables)
        self.assertIn("company_meta", tables)
        self.assertIn("domain_mx_cache", tables)

    def test_company_meta_has_llm_columns(self):
        from database import init_db
        import aiosqlite

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                async with db.execute("PRAGMA table_info(company_meta)") as cur:
                    return {row[1] for row in await cur.fetchall()}

        columns = run(ops())
        for col in ("llm_assessment", "llm_risk_level", "llm_signals",
                    "llm_provider", "llm_model", "llm_assessed_at",
                    "indeed_url", "indeed_rating", "indeed_review_count"):
            self.assertIn(col, columns, f"Missing column: {col}")

    def test_llm_columns_added_via_migration_to_existing_db(self):
        """Simulates an existing DB without LLM columns — migration must add them."""
        import aiosqlite
        from database import init_db

        async def ops():
            # Create DB with old schema (no LLM columns)
            async with aiosqlite.connect(self.tmp.name) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS company_meta (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_name TEXT NOT NULL UNIQUE,
                        bbb_rating TEXT DEFAULT '',
                        crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await db.commit()
            # Run init_db — migrations should add the missing columns
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                async with db.execute("PRAGMA table_info(company_meta)") as cur:
                    return {row[1] for row in await cur.fetchall()}

        columns = run(ops())
        for col in ("llm_assessment", "llm_risk_level", "llm_signals",
                    "llm_provider", "llm_model", "llm_assessed_at"):
            self.assertIn(col, columns, f"Migration did not add column: {col}")

    def test_init_is_idempotent(self):
        from database import init_db

        async def ops():
            await init_db()
            await init_db()  # Second call must not raise

        run(ops())

    def test_resume_insert_and_fetch(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                db.row_factory = aiosqlite.Row
                await db.execute(
                    "INSERT INTO resumes (label, content) VALUES (?, ?)",
                    ("DevSecOps v1", MOCK_RESUME_DEVSECOPS)
                )
                await db.commit()
                async with db.execute("SELECT * FROM resumes") as cur:
                    return [dict(r) for r in await cur.fetchall()]

        rows = run(ops())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "DevSecOps v1")
        self.assertIn("Security+", rows[0]["content"])


    def test_jobs_table_has_company_url_column(self):
        """jobs table should have company_url column after migration."""
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                async with db.execute("PRAGMA table_info(jobs)") as cur:
                    cols = {row[1] for row in await cur.fetchall()}
                return "company_url" in cols

        self.assertTrue(run(ops()))

    def test_job_insert_and_fetch(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                db.row_factory = aiosqlite.Row
                await db.execute(
                    "INSERT INTO jobs (url, title, company, location, raw_description) "
                    "VALUES (?,?,?,?,?)",
                    ("https://example.com/job/1", "DevSecOps Engineer",
                     "Acme Federal", "Arlington, VA", MOCK_JOB_DEVSECOPS)
                )
                await db.commit()
                async with db.execute("SELECT * FROM jobs") as cur:
                    return [dict(r) for r in await cur.fetchall()]

        rows = run(ops())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "DevSecOps Engineer")
        self.assertEqual(rows[0]["company"], "Acme Federal")

    def test_duplicate_job_url_raises(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                await db.execute(
                    "INSERT INTO jobs (url, title, raw_description) VALUES (?,?,?)",
                    ("https://example.com/job/1", "Job A", "description")
                )
                await db.commit()
                await db.execute(
                    "INSERT INTO jobs (url, title, raw_description) VALUES (?,?,?)",
                    ("https://example.com/job/1", "Job B", "description")
                )
                await db.commit()

        with self.assertRaises(Exception):
            run(ops())

    def test_analysis_insert_and_cascade_delete(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA foreign_keys = ON")

                async with db.execute(
                    "INSERT INTO resumes (label, content) VALUES (?,?)",
                    ("v1", "resume text")
                ) as cur:
                    resume_id = cur.lastrowid

                async with db.execute(
                    "INSERT INTO jobs (url, raw_description) VALUES (?,?)",
                    ("https://example.com/job/1", "job desc")
                ) as cur:
                    job_id = cur.lastrowid

                await db.execute(
                    "INSERT INTO analyses "
                    "(job_id, resume_id, score, matched_skills, missing_skills, reasoning) "
                    "VALUES (?,?,?,?,?,?)",
                    (job_id, resume_id, 4, '["Python"]', '["K8s"]', "Good match")
                )
                await db.commit()

                await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                await db.commit()

                async with db.execute(
                    "SELECT * FROM analyses WHERE job_id = ?", (job_id,)
                ) as cur:
                    return await cur.fetchall()

        remaining = run(ops())
        self.assertEqual(len(remaining), 0)

    def test_application_upsert(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                db.row_factory = aiosqlite.Row

                async with db.execute(
                    "INSERT INTO jobs (url, raw_description) VALUES (?,?)",
                    ("https://example.com/job/1", "desc")
                ) as cur:
                    job_id = cur.lastrowid

                await db.execute(
                    "INSERT INTO applications (job_id, status, recruiter_name) VALUES (?,?,?)",
                    (job_id, "applied", "Jane Smith")
                )
                await db.commit()

                await db.execute(
                    """INSERT INTO applications (job_id, status, recruiter_name)
                       VALUES (?,?,?)
                       ON CONFLICT(job_id) DO UPDATE SET status=excluded.status""",
                    (job_id, "interviewing", "Jane Smith")
                )
                await db.commit()

                async with db.execute(
                    "SELECT * FROM applications WHERE job_id = ?", (job_id,)
                ) as cur:
                    return [dict(r) for r in await cur.fetchall()]

        rows = run(ops())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "interviewing")
