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
