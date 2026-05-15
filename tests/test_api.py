"""
tests/test_api.py — FastAPI route tests and end-to-end smoke test.
Uses httpx.AsyncClient with ASGITransport against a real temp DB.
"""

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from analyzer.config import anthropic_model
from tests.mock_data import (
    MOCK_RESUME_DEVSECOPS,
    MOCK_JOB_DEVSECOPS,
    MOCK_JOB_MARKETING,
)


class TestAPIEndpoints(unittest.IsolatedAsyncioTestCase):
    """
    FastAPI route tests using httpx.AsyncClient as a test client.

    IsolatedAsyncioTestCase spins up its own event loop per test method.
    DB_PATH is read lazily in database.py (not at import time), so setting
    os.environ["DB_PATH"] in asyncSetUp reliably points the app at the
    test's temp database for every route call.
    """

    async def asyncSetUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        os.environ["DB_PATH"] = self.tmp.name

        from database import init_db
        await init_db()

        from httpx import AsyncClient, ASGITransport
        from main import app
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        os.unlink(self.tmp.name)

    async def test_delete_analysis(self):
        """Delete a specific analysis without affecting the job."""
        import aiosqlite

        r = await self.client.post("/api/resumes/add", data={
            "label": "v1", "content": MOCK_RESUME_DEVSECOPS
        })
        rid = r.json()["resume_id"]

        with patch("main.scrape_job") as mock_scrape:
            mock_scrape.return_value = {
                "title": "Job", "company": "Co", "location": "VA",
                "raw_description": MOCK_JOB_DEVSECOPS
            }
            j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/del-analysis"})
        jid = j.json()["job_id"]

        with patch("main.analyze_match") as mock_analyze:
            mock_analyze.return_value = {
                "score": 3, "adjusted_score": 3,
                "penalty_breakdown": {"blockers": 0, "majors": 0, "minors": 0, "blocker_penalty": 0, "major_penalty": 0, "minor_penalty": 0, "count_penalty": 0, "total_penalty": 0},
                "matched_skills": ["Python"], "missing_skills": [],
                "reasoning": "Ok.", "llm_provider": "anthropic", "llm_model": anthropic_model(),
            }
            await self.client.post(f"/api/jobs/{jid}/analyze", data={
                "resume_id": rid, "provider": "anthropic"
            })

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT id FROM analyses WHERE job_id=?", (jid,)) as cur:
                analysis_id = (await cur.fetchone())["id"]

        resp = await self.client.delete(f"/api/analyses/{analysis_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

        job_resp = await self.client.get(f"/job/{jid}")
        self.assertEqual(job_resp.status_code, 200)

        async with aiosqlite.connect(self.tmp.name) as db:
            async with db.execute("SELECT id FROM analyses WHERE id=?", (analysis_id,)) as cur:
                self.assertIsNone(await cur.fetchone())

    async def test_delete_analysis_404_for_missing(self):
        resp = await self.client.delete("/api/analyses/9999")
        self.assertEqual(resp.status_code, 404)

    async def test_add_resume(self):
        resp = await self.client.post("/api/resumes/add", data={
            "label": "DevSecOps v1",
            "content": MOCK_RESUME_DEVSECOPS
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("resume_id", data)
        self.assertEqual(data["label"], "DevSecOps v1")

    async def test_delete_resume(self):
        add = await self.client.post("/api/resumes/add", data={
            "label": "Temp Resume",
            "content": "some content here"
        })
        rid = add.json()["resume_id"]
        resp = await self.client.delete(f"/api/resumes/{rid}")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

    async def test_add_manual_job_success(self):
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title":       "DevSecOps Engineer",
            "company":     "Acme Federal",
            "description": MOCK_JOB_DEVSECOPS,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["title"], "DevSecOps Engineer")
        self.assertEqual(data["company"], "Acme Federal")

    async def test_add_manual_job_with_location(self):
        """Location field should be saved and appear on the job detail page."""
        import aiosqlite
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title":       "Cloud Engineer",
            "company":     "Acme",
            "location":    "Washington, DC",
            "description": MOCK_JOB_DEVSECOPS,
        })
        self.assertEqual(resp.status_code, 200)
        jid = resp.json()["job_id"]

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT location FROM jobs WHERE id = ?", (jid,)) as cur:
                row = await cur.fetchone()
        self.assertEqual(row["location"], "Washington, DC")

    async def test_add_manual_job_location_defaults_to_empty(self):
        """Omitting location should store empty string, not error."""
        import aiosqlite
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title":       "Dev",
            "company":     "Co",
            "description": MOCK_JOB_DEVSECOPS,
        })
        self.assertEqual(resp.status_code, 200)
        jid = resp.json()["job_id"]

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT location FROM jobs WHERE id = ?", (jid,)) as cur:
                row = await cur.fetchone()
        self.assertEqual(row["location"], "")

    async def test_add_manual_job_title_defaults_to_untitled(self):
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title":       "",
            "company":     "",
            "description": MOCK_JOB_DEVSECOPS,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["title"], "Untitled Job")

    async def test_add_manual_job_too_short_returns_422(self):
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title":       "Some Job",
            "company":     "Co",
            "description": "Too short",
        })
        self.assertEqual(resp.status_code, 422)
        self.assertIn("too short", resp.json()["error"])

    async def test_add_manual_job_duplicate_returns_409(self):
        payload = {
            "title":       "DevSecOps Engineer",
            "company":     "Acme",
            "description": MOCK_JOB_DEVSECOPS,
        }
        await self.client.post("/api/jobs/add-manual", data=payload)
        resp = await self.client.post("/api/jobs/add-manual", data=payload)
        self.assertEqual(resp.status_code, 409)
        self.assertIn("already been added", resp.json()["error"])

    async def test_add_manual_job_can_be_analyzed(self):
        r = await self.client.post("/api/resumes/add", data={
            "label": "v1", "content": MOCK_RESUME_DEVSECOPS
        })
        rid = r.json()["resume_id"]

        j = await self.client.post("/api/jobs/add-manual", data={
            "title": "DevSecOps Engineer", "company": "Acme",
            "description": MOCK_JOB_DEVSECOPS,
        })
        jid = j.json()["job_id"]

        with patch("main.analyze_match") as mock_analyze:
            mock_analyze.return_value = {
                "score": 4, "adjusted_score": 2,
                "penalty_breakdown": {"blockers": 1, "majors": 0, "minors": 0, "blocker_penalty": 2, "major_penalty": 0, "minor_penalty": 0, "count_penalty": 0, "total_penalty": 2},
                "matched_skills": ["Python", "Docker"],
                "missing_skills": [{"skill": "Clearance", "severity": "blocker"}],
                "reasoning": "Good match.",
                "llm_provider": "anthropic", "llm_model": anthropic_model(),
            }
            resp = await self.client.post(f"/api/jobs/{jid}/analyze", data={
                "resume_id": rid, "provider": "anthropic"
            })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["score"], 4)

    async def test_analyze_stores_llm_model(self):
        import aiosqlite

        r = await self.client.post("/api/resumes/add", data={
            "label": "v1", "content": MOCK_RESUME_DEVSECOPS
        })
        rid = r.json()["resume_id"]

        with patch("main.scrape_job") as mock_scrape:
            mock_scrape.return_value = {
                "title": "DevSecOps Engineer", "company": "Acme",
                "location": "VA", "raw_description": MOCK_JOB_DEVSECOPS
            }
            j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/model-test"})
        jid = j.json()["job_id"]

        with patch("main.analyze_match") as mock_analyze:
            mock_analyze.return_value = {
                "score": 4, "adjusted_score": 4,
                "penalty_breakdown": {"blockers": 0, "majors": 0, "minors": 0, "blocker_penalty": 0, "major_penalty": 0, "minor_penalty": 0, "count_penalty": 0, "total_penalty": 0},
                "matched_skills": ["Python"], "missing_skills": [],
                "reasoning": "Good.", "llm_provider": "ollama", "llm_model": "llama3.1:8b",
            }
            await self.client.post(f"/api/jobs/{jid}/analyze", data={
                "resume_id": rid, "provider": "ollama"
            })

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT llm_provider, llm_model FROM analyses WHERE job_id=?", (jid,)
            ) as cur:
                row = dict(await cur.fetchone())

        self.assertEqual(row["llm_provider"], "ollama")
        self.assertEqual(row["llm_model"], "llama3.1:8b")

    @patch("main.scrape_job")
    async def test_add_job_success(self, mock_scrape):
        mock_scrape.return_value = {
            "title": "DevSecOps Engineer", "company": "Acme Federal",
            "location": "Arlington, VA", "raw_description": MOCK_JOB_DEVSECOPS
        }
        resp = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/1"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["title"], "DevSecOps Engineer")
        self.assertIn("job_id", data)

    @patch("main.scrape_job")
    async def test_add_duplicate_job_returns_409(self, mock_scrape):
        mock_scrape.return_value = {
            "title": "Job", "company": "", "location": "",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/dupe"})
        resp = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/dupe"})
        self.assertEqual(resp.status_code, 409)
        self.assertIn("already been added", resp.json()["error"])

    @patch("main.scrape_job")
    async def test_add_job_scrape_failure_returns_422(self, mock_scrape):
        mock_scrape.side_effect = ValueError("Could not extract job description.")
        resp = await self.client.post("/api/jobs/add", data={"url": "https://bad.example.com"})
        self.assertEqual(resp.status_code, 422)
        self.assertIn("error", resp.json())

    # ── /api/jobs/scrape ──────────────────────────────────────────────────────

    @patch("main.scrape_job")
    async def test_scrape_returns_data_without_saving(self, mock_scrape):
        """POST /api/jobs/scrape should return scraped data but NOT save to DB."""
        mock_scrape.return_value = {
            "title": "Python Dev", "company": "Acme", "location": "Remote",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        resp = await self.client.post("/api/jobs/scrape", data={"url": "https://example.com/scrape-only"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["title"], "Python Dev")
        self.assertIn("description", data)
        self.assertIn("has_warnings", data)
        self.assertIn("blocker_keywords", data)
        self.assertIn("text_quality", data)
        self.assertNotIn("job_id", data)

    @patch("main.scrape_job")
    async def test_scrape_duplicate_returns_409(self, mock_scrape):
        """Scraping an already-saved URL should return 409."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "", "location": "",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        await self.client.post("/api/jobs/add", data={"url": "https://example.com/scrape-dupe"})
        resp = await self.client.post("/api/jobs/scrape", data={"url": "https://example.com/scrape-dupe"})
        self.assertEqual(resp.status_code, 409)

    @patch("main.scrape_job")
    async def test_scrape_failure_returns_422(self, mock_scrape):
        mock_scrape.side_effect = ValueError("Could not extract content.")
        resp = await self.client.post("/api/jobs/scrape", data={"url": "https://bad.example.com/scrape"})
        self.assertEqual(resp.status_code, 422)

    @patch("main.scrape_job")
    async def test_scrape_detects_blocker_keywords(self, mock_scrape):
        """Scrape endpoint should flag clearance keywords in warnings."""
        mock_scrape.return_value = {
            "title": "Gov Dev", "company": "Agency", "location": "DC",
            "raw_description": "This role requires active Secret clearance and US citizenship.",
        }
        resp = await self.client.post("/api/jobs/scrape", data={"url": "https://example.com/clearance-job"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["has_warnings"])
        self.assertGreater(len(data["blocker_keywords"]), 0)

    # ── /api/jobs/save-preview ─────────────────────────────────────────────────

    async def test_save_preview_creates_job(self):
        """POST /api/jobs/save-preview should save job and return job_id."""
        resp = await self.client.post("/api/jobs/save-preview", data={
            "url":         "https://example.com/preview-save",
            "title":       "Cleaned Up Dev",
            "company":     "Acme",
            "location":    "Remote",
            "description": MOCK_JOB_DEVSECOPS,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["title"], "Cleaned Up Dev")

    async def test_save_preview_duplicate_returns_409(self):
        url = "https://example.com/preview-dupe"
        await self.client.post("/api/jobs/save-preview", data={
            "url": url, "title": "Job", "company": "", "location": "",
            "description": MOCK_JOB_DEVSECOPS,
        })
        resp = await self.client.post("/api/jobs/save-preview", data={
            "url": url, "title": "Job", "company": "", "location": "",
            "description": MOCK_JOB_DEVSECOPS,
        })
        self.assertEqual(resp.status_code, 409)

    async def test_save_preview_short_description_returns_422(self):
        resp = await self.client.post("/api/jobs/save-preview", data={
            "url": "https://example.com/preview-short",
            "title": "Dev", "company": "", "location": "",
            "description": "Too short.",
        })
        self.assertEqual(resp.status_code, 422)
        self.assertIn("too short", resp.json()["error"])

    async def test_save_preview_job_accessible_after_save(self):
        """Job saved via save-preview should be accessible on its detail page."""
        resp = await self.client.post("/api/jobs/save-preview", data={
            "url":         "https://example.com/preview-accessible",
            "title":       "Accessible Job",
            "company":     "Co",
            "location":    "NY",
            "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]
        page = await self.client.get(f"/job/{jid}")
        self.assertEqual(page.status_code, 200)
        # Page is now a static shell — job data served via /api/jobs/{id}/detail
        self.assertIn(b"job-detail-main", page.content)
        detail = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertIn("Accessible Job", detail.json()["job"]["title"])

    # ── GET /jobs/preview ──────────────────────────────────────────────────────

    async def test_preview_page_renders(self):
        """GET /jobs/preview should return 200 with the preview form."""
        resp = await self.client.get("/jobs/preview")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Preview Job", resp.content)
        self.assertIn(b"preview-form", resp.content)

    @patch("main.scrape_job")
    async def test_delete_job(self, mock_scrape):
        mock_scrape.return_value = {
            "title": "Temp Job", "company": "", "location": "",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        add = await self.client.post("/api/jobs/add", data={"url": "https://example.com/delete-me"})
        jid = add.json()["job_id"]
        resp = await self.client.delete(f"/api/jobs/{jid}")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

    @patch("main.scrape_job")
    @patch("main.analyze_match")
    async def test_analyze_job(self, mock_analyze, mock_scrape):
        mock_scrape.return_value = {
            "title": "DevSecOps Engineer", "company": "Acme", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        mock_analyze.return_value = {
            "score": 5, "adjusted_score": 5,
            "penalty_breakdown": {"blockers": 0, "majors": 0, "minors": 0, "blocker_penalty": 0, "major_penalty": 0, "minor_penalty": 0, "count_penalty": 0, "total_penalty": 0},
            "matched_skills": ["Python", "Docker", "AWS"],
            "missing_skills": [{"skill": "Secret Clearance", "severity": "blocker"}],
            "reasoning": "Great match.", "llm_provider": "anthropic", "llm_model": anthropic_model(),
        }

        r = await self.client.post("/api/resumes/add", data={
            "label": "DevSecOps v1", "content": MOCK_RESUME_DEVSECOPS
        })
        rid = r.json()["resume_id"]

        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/analyze"})
        jid = j.json()["job_id"]

        resp = await self.client.post(f"/api/jobs/{jid}/analyze", data={
            "resume_id": rid, "provider": "anthropic"
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["score"], 5)
        self.assertIn("Python", resp.json()["matched_skills"])

    async def test_analyze_missing_job_returns_404(self):
        r = await self.client.post("/api/resumes/add", data={
            "label": "v1", "content": "resume text"
        })
        rid = r.json()["resume_id"]
        resp = await self.client.post("/api/jobs/9999/analyze", data={
            "resume_id": rid, "provider": "anthropic"
        })
        self.assertEqual(resp.status_code, 404)

    async def test_analyze_missing_resume_returns_404(self):
        import aiosqlite
        async with aiosqlite.connect(self.tmp.name) as db:
            async with db.execute(
                "INSERT INTO jobs (url, raw_description) VALUES (?,?)",
                ("https://example.com/job/noresume", "job desc")
            ) as cur:
                jid = cur.lastrowid
            await db.commit()

        resp = await self.client.post(f"/api/jobs/{jid}/analyze", data={
            "resume_id": 9999, "provider": "anthropic"
        })
        self.assertEqual(resp.status_code, 404)

    async def test_analyze_passes_analysis_mode(self):
        """analysis_mode form param should be applied before LLM call."""
        import os
        r = await self.client.post("/api/resumes/add", data={
            "label": "v1", "content": MOCK_RESUME_DEVSECOPS
        })
        rid = r.json()["resume_id"]

        with patch("main.scrape_job") as mock_scrape:
            mock_scrape.return_value = {
                "title": "Dev", "company": "Co", "location": "VA",
                "raw_description": MOCK_JOB_DEVSECOPS
            }
            j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/mode-test"})
        jid = j.json()["job_id"]

        captured_mode = {}
        with patch("main.analyze_match") as mock_analyze:
            def capture(*args, **kwargs):
                captured_mode["mode"] = os.environ.get("ANALYSIS_MODE")
                return {
                    "score": 3, "adjusted_score": 3,
                    "penalty_breakdown": {"blockers":0,"majors":0,"minors":0,"blocker_penalty":0,"major_penalty":0,"minor_penalty":0,"count_penalty":0,"total_penalty":0},
                    "matched_skills": [], "missing_skills": [],
                    "reasoning": "ok.", "llm_provider": "anthropic", "llm_model": anthropic_model(),
                }
            mock_analyze.side_effect = capture
            await self.client.post(f"/api/jobs/{jid}/analyze", data={
                "resume_id": rid, "provider": "anthropic", "analysis_mode": "detailed"
            })
        self.assertEqual(captured_mode.get("mode"), "detailed")

    async def test_analyze_passes_ollama_model(self):
        """ollama_model form param should be applied to OLLAMA_MODEL env before call."""
        import os
        r = await self.client.post("/api/resumes/add", data={
            "label": "v1", "content": MOCK_RESUME_DEVSECOPS
        })
        rid = r.json()["resume_id"]

        with patch("main.scrape_job") as mock_scrape:
            mock_scrape.return_value = {
                "title": "Dev", "company": "Co", "location": "VA",
                "raw_description": MOCK_JOB_DEVSECOPS
            }
            j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/ollama-model-test"})
        jid = j.json()["job_id"]

        captured = {}
        with patch("main.analyze_match") as mock_analyze:
            def capture(*args, **kwargs):
                captured["model"] = os.environ.get("OLLAMA_MODEL")
                return {
                    "score": 3, "adjusted_score": 3,
                    "penalty_breakdown": {"blockers":0,"majors":0,"minors":0,"blocker_penalty":0,"major_penalty":0,"minor_penalty":0,"count_penalty":0,"total_penalty":0},
                    "matched_skills": [], "missing_skills": [],
                    "reasoning": "ok.", "llm_provider": "ollama", "llm_model": "gemma3:27b",
                }
            mock_analyze.side_effect = capture
            await self.client.post(f"/api/jobs/{jid}/analyze", data={
                "resume_id": rid, "provider": "ollama", "ollama_model": "gemma3:27b"
            })
        self.assertEqual(captured.get("model"), "gemma3:27b")

    async def test_analyze_passes_cloud_model(self):
        """cloud_model form param should set the correct provider env var."""
        import os
        r = await self.client.post("/api/resumes/add", data={
            "label": "v1", "content": MOCK_RESUME_DEVSECOPS
        })
        rid = r.json()["resume_id"]

        with patch("main.scrape_job") as mock_scrape:
            mock_scrape.return_value = {
                "title": "Dev", "company": "Co", "location": "VA",
                "raw_description": MOCK_JOB_DEVSECOPS
            }
            j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/cloud-model-test"})
        jid = j.json()["job_id"]

        captured = {}
        with patch("main.analyze_match") as mock_analyze:
            def capture(*args, **kwargs):
                captured["model"] = os.environ.get("OPENAI_MODEL")
                return {
                    "score": 3, "adjusted_score": 3,
                    "penalty_breakdown": {"blockers":0,"majors":0,"minors":0,"blocker_penalty":0,"major_penalty":0,"minor_penalty":0,"count_penalty":0,"total_penalty":0},
                    "matched_skills": [], "missing_skills": [],
                    "reasoning": "ok.", "llm_provider": "openai", "llm_model": "gpt-4o",
                }
            mock_analyze.side_effect = capture
            await self.client.post(f"/api/jobs/{jid}/analyze", data={
                "resume_id": rid, "provider": "openai", "cloud_model": "gpt-4o"
            })
        self.assertEqual(captured.get("model"), "gpt-4o")

    async def test_get_provider_models_anthropic(self):
        """GET /api/providers/models?provider=anthropic returns known models."""
        resp = await self.client.get("/api/providers/models?provider=anthropic")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["provider"], "anthropic")
        self.assertIsInstance(data["models"], list)
        self.assertGreater(len(data["models"]), 0)
        ids = [m["id"] for m in data["models"]]
        self.assertIn("claude-sonnet-4-6", ids)

    async def test_get_provider_models_openai(self):
        resp = await self.client.get("/api/providers/models?provider=openai")
        self.assertEqual(resp.status_code, 200)
        ids = [m["id"] for m in resp.json()["models"]]
        self.assertIn("gpt-4o-mini", ids)

    async def test_get_provider_models_gemini(self):
        resp = await self.client.get("/api/providers/models?provider=gemini")
        self.assertEqual(resp.status_code, 200)
        ids = [m["id"] for m in resp.json()["models"]]
        self.assertIn("gemini-2.5-flash", ids)

    async def test_get_provider_models_unknown_returns_empty(self):
        resp = await self.client.get("/api/providers/models?provider=unknown")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["models"], [])

    async def test_get_provider_models_each_has_id_and_label(self):
        """Every model entry must have id and label fields."""
        for provider in ["anthropic", "openai", "gemini"]:
            resp = await self.client.get(f"/api/providers/models?provider={provider}")
            for m in resp.json()["models"]:
                self.assertIn("id", m, f"{provider} model missing 'id'")
                self.assertIn("label", m, f"{provider} model missing 'label'")

    @patch("main.scrape_job")
    async def test_save_and_update_application(self, mock_scrape):
        mock_scrape.return_value = {
            "title": "Job", "company": "Co", "location": "DC",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/app"})
        jid = j.json()["job_id"]

        resp = await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "applied", "recruiter_name": "Jane Smith",
            "recruiter_email": "jane@company.com", "recruiter_phone": "555-1234",
            "notes": "Great conversation on LinkedIn"
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

        resp2 = await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "interviewing", "recruiter_name": "Jane Smith",
            "recruiter_email": "jane@company.com", "recruiter_phone": "555-1234",
            "notes": "Phone screen scheduled"
        })
        self.assertEqual(resp2.status_code, 200)

    # ── has_recruiter flag in job list ────────────────────────────────────────

    @patch("main.scrape_job")
    async def test_job_list_has_recruiter_false_by_default(self, mock_scrape):
        """Job with no application data should have has_recruiter=0."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/no-recruiter"})
        jid = j.json()["job_id"]
        resp = await self.client.get("/api/jobs/list")
        self.assertEqual(resp.status_code, 200)
        jobs = resp.json()["jobs"]
        job = next((j for j in jobs if j["id"] == jid), None)
        self.assertIsNotNone(job)
        self.assertEqual(job["has_recruiter"], 0)

    @patch("main.scrape_job")
    async def test_job_list_has_recruiter_true_when_name_saved(self, mock_scrape):
        """Job with recruiter_name saved should have has_recruiter=1."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/with-recruiter"})
        jid = j.json()["job_id"]
        await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "applied", "recruiter_name": "Jane Smith",
            "recruiter_email": "", "recruiter_phone": "",
        })
        resp = await self.client.get("/api/jobs/list")
        jobs = resp.json()["jobs"]
        job = next((j for j in jobs if j["id"] == jid), None)
        self.assertIsNotNone(job)
        self.assertEqual(job["has_recruiter"], 1)

    @patch("main.scrape_job")
    async def test_job_list_has_recruiter_true_when_email_only(self, mock_scrape):
        """Email alone (no name) should also set has_recruiter=1."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/email-recruiter"})
        jid = j.json()["job_id"]
        await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "not_applied", "recruiter_name": "",
            "recruiter_email": "recruiter@co.com", "recruiter_phone": "",
        })
        resp = await self.client.get("/api/jobs/list")
        jobs = resp.json()["jobs"]
        job = next((j for j in jobs if j["id"] == jid), None)
        self.assertEqual(job["has_recruiter"], 1)

    @patch("main.scrape_job")
    async def test_job_list_has_recruiter_false_when_status_only(self, mock_scrape):
        """Application with only status (no contact info) should have has_recruiter=0."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/status-only"})
        jid = j.json()["job_id"]
        await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "applied", "recruiter_name": "",
            "recruiter_email": "", "recruiter_phone": "",
        })
        resp = await self.client.get("/api/jobs/list")
        jobs = resp.json()["jobs"]
        job = next((j for j in jobs if j["id"] == jid), None)
        self.assertEqual(job["has_recruiter"], 0)

    # ── date filter and scraped_at ───────────────────────────────────────────

    @patch("main.scrape_job")
    async def test_job_list_returns_scraped_at(self, mock_scrape):
        """Job list should include scraped_at timestamp for each job."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/date-test"})
        jid = j.json()["job_id"]

        resp = await self.client.get("/api/jobs/list")
        jobs = resp.json()["jobs"]
        job = next((j for j in jobs if j["id"] == jid), None)
        self.assertIsNotNone(job)
        self.assertIn("scraped_at", job)
        self.assertIsNotNone(job["scraped_at"])

    @patch("main.scrape_job")
    async def test_added_days_filter_includes_recent_job(self, mock_scrape):
        """Jobs added within the filter window should be returned."""
        mock_scrape.return_value = {
            "title": "Recent Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/recent-job"})
        jid = j.json()["job_id"]

        resp = await self.client.get("/api/jobs/list?added_days=7")
        jobs = resp.json()["jobs"]
        ids = [j["id"] for j in jobs]
        self.assertIn(jid, ids)

    @patch("main.scrape_job")
    async def test_date_from_filter_includes_recent_job(self, mock_scrape):
        """Jobs added on or after date_from should be returned."""
        from datetime import date
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/date-from-test"})
        jid = j.json()["job_id"]

        today = date.today().isoformat()
        resp = await self.client.get(f"/api/jobs/list?date_from={today}")
        ids = [j["id"] for j in resp.json()["jobs"]]
        self.assertIn(jid, ids)

    @patch("main.scrape_job")
    async def test_date_to_filter_includes_recent_job(self, mock_scrape):
        """Jobs added on or before date_to should be returned."""
        from datetime import date, timedelta
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/date-to-test"})
        jid = j.json()["job_id"]

        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        resp = await self.client.get(f"/api/jobs/list?date_to={tomorrow}")
        ids = [j["id"] for j in resp.json()["jobs"]]
        self.assertIn(jid, ids)

    @patch("main.scrape_job")
    async def test_date_range_filter(self, mock_scrape):
        """Jobs within date_from/date_to range should be returned."""
        from datetime import date, timedelta
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/date-range-test"})
        jid = j.json()["job_id"]

        today    = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        resp = await self.client.get(f"/api/jobs/list?date_from={today}&date_to={tomorrow}")
        ids = [j["id"] for j in resp.json()["jobs"]]
        self.assertIn(jid, ids)

    @patch("main.scrape_job")
    async def test_date_from_future_returns_empty(self, mock_scrape):
        """date_from in the future should return no jobs."""
        from datetime import date, timedelta
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        await self.client.post("/api/jobs/add", data={"url": "https://example.com/future-date-test"})

        future = (date.today() + timedelta(days=30)).isoformat()
        resp = await self.client.get(f"/api/jobs/list?date_from={future}")
        self.assertEqual(resp.json()["total"], 0)

    @patch("main.scrape_job")
    async def test_added_days_invalid_value_ignored(self, mock_scrape):
        """Invalid added_days value should be ignored, returning all jobs."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        await self.client.post("/api/jobs/add", data={"url": "https://example.com/invalid-date"})

        resp = await self.client.get("/api/jobs/list?added_days=notanumber")
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(resp.json()["total"], 0)

    # ── last_model in job list ────────────────────────────────────────────────

    @patch("main.scrape_job")
    @patch("main.analyze_match")
    async def test_job_list_last_model_none_before_analysis(self, mock_analyze, mock_scrape):
        """Job with no analysis should have last_model=None in list."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/no-analysis-model"})
        jid = j.json()["job_id"]

        resp = await self.client.get("/api/jobs/list")
        jobs = resp.json()["jobs"]
        job = next((j for j in jobs if j["id"] == jid), None)
        self.assertIsNotNone(job)
        self.assertIsNone(job["last_model"])

    @patch("main.scrape_job")
    @patch("main.analyze_match")
    async def test_job_list_last_model_after_analysis(self, mock_analyze, mock_scrape):
        """Job list should return last_model from most recent analysis."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        mock_analyze.return_value = {
            "score": 4, "adjusted_score": 4,
            "penalty_breakdown": {"blockers":0,"majors":0,"minors":0,"blocker_penalty":0,"major_penalty":0,"minor_penalty":0,"count_penalty":0,"total_penalty":0},
            "matched_skills": ["Python"], "missing_skills": [],
            "reasoning": "ok.", "llm_provider": "openai", "llm_model": "gpt-4o-mini",
        }
        r = await self.client.post("/api/resumes/add", data={
            "label": "v1", "content": MOCK_RESUME_DEVSECOPS
        })
        rid = r.json()["resume_id"]
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/with-model"})
        jid = j.json()["job_id"]
        await self.client.post(f"/api/jobs/{jid}/analyze", data={
            "resume_id": rid, "provider": "openai", "cloud_model": "gpt-4o-mini"
        })

        resp = await self.client.get("/api/jobs/list")
        jobs = resp.json()["jobs"]
        job = next((j for j in jobs if j["id"] == jid), None)
        self.assertIsNotNone(job)
        self.assertEqual(job["last_model"], "gpt-4o-mini")
        self.assertEqual(job["provider"], "openai")

    # ── Search — recruiter fields ─────────────────────────────────────────────

    @patch("main.scrape_job")
    async def test_search_by_recruiter_name(self, mock_scrape):
        """Search should match recruiter_name in applications."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/recruiter-search-name"})
        jid = j.json()["job_id"]
        await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "applied", "recruiter_name": "Jane Smith",
            "recruiter_email": "", "recruiter_phone": "",
        })

        resp = await self.client.get("/api/jobs/list?search=jane")
        jobs = resp.json()["jobs"]
        ids = [j["id"] for j in jobs]
        self.assertIn(jid, ids)

    @patch("main.scrape_job")
    async def test_search_by_recruiter_email(self, mock_scrape):
        """Search should match recruiter_email in applications."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/recruiter-search-email"})
        jid = j.json()["job_id"]
        await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "applied", "recruiter_name": "",
            "recruiter_email": "recruiter@uniquecompany.com", "recruiter_phone": "",
        })

        resp = await self.client.get("/api/jobs/list?search=uniquecompany")
        jobs = resp.json()["jobs"]
        ids = [j["id"] for j in jobs]
        self.assertIn(jid, ids)

    @patch("main.scrape_job")
    async def test_search_by_recruiter_phone(self, mock_scrape):
        """Search should match recruiter_phone in applications."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/recruiter-search-phone"})
        jid = j.json()["job_id"]
        await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "applied", "recruiter_name": "",
            "recruiter_email": "", "recruiter_phone": "571-999-0001",
        })

        resp = await self.client.get("/api/jobs/list?search=571-999-0001")
        jobs = resp.json()["jobs"]
        ids = [j["id"] for j in jobs]
        self.assertIn(jid, ids)

    @patch("main.scrape_job")
    async def test_search_no_match_returns_empty(self, mock_scrape):
        """Search with no matching term should return empty list."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        await self.client.post("/api/jobs/add", data={"url": "https://example.com/search-no-match"})

        resp = await self.client.get("/api/jobs/list?search=xyznonexistent999")
        self.assertEqual(resp.json()["total"], 0)

    # ── /api/ollama/models proxy ──────────────────────────────────────────────

    async def test_ollama_models_returns_list(self):
        """Proxy should return sorted model list when Ollama is running."""
        import httpx
        mock_resp = {"models": [{"name": "llama3.1:8b"}, {"name": "gemma3:27b"}]}
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp_obj = MagicMock()
            mock_resp_obj.json.return_value = mock_resp
            mock_resp_obj.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp_obj)
            mock_cls.return_value = mock_client
            resp = await self.client.get("/api/ollama/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("models", data)
        self.assertIn("llama3.1:8b", data["models"])
        self.assertIn("gemma3:27b", data["models"])

    async def test_ollama_models_returns_empty_when_offline(self):
        """Proxy should return empty list when Ollama is not reachable."""
        import httpx
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_cls.return_value = mock_client
            resp = await self.client.get("/api/ollama/models")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["models"], [])

    async def test_index_page_renders(self):
        resp = await self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Job Tracker", resp.content)

    async def test_resumes_page_renders(self):
        resp = await self.client.get("/resumes")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Resume Versions", resp.content)

    @patch("main.scrape_job")
    @patch("main.analyze_match")
    async def test_job_detail_preselects_last_model(self, mock_analyze, mock_scrape):
        """Job detail page should pre-select the model used in the last analysis."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        mock_analyze.return_value = {
            "score": 4, "adjusted_score": 4,
            "penalty_breakdown": {"blockers":0,"majors":0,"minors":0,"blocker_penalty":0,"major_penalty":0,"minor_penalty":0,"count_penalty":0,"total_penalty":0},
            "matched_skills": ["Python"], "missing_skills": [],
            "reasoning": "ok.", "llm_provider": "openai", "llm_model": "gpt-4o",
            "analysis_mode": "detailed",
        }

        r = await self.client.post("/api/resumes/add", data={
            "label": "v1", "content": MOCK_RESUME_DEVSECOPS
        })
        rid = r.json()["resume_id"]
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/preselect-model"})
        jid = j.json()["job_id"]

        await self.client.post(f"/api/jobs/{jid}/analyze", data={
            "resume_id": rid, "provider": "openai", "cloud_model": "gpt-4o", "analysis_mode": "detailed"
        })

        resp = await self.client.get(f"/job/{jid}")
        self.assertEqual(resp.status_code, 200)
        # Model pre-selection is now served via /api/jobs/{id}/detail
        detail = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(detail.json()["openai_model"], "gpt-4o")
        self.assertEqual(detail.json()["last_provider"], "openai")

    @patch("main.scrape_job")
    @patch("main.analyze_match")
    async def test_job_detail_preselects_last_analysis_mode(self, mock_analyze, mock_scrape):
        """Job detail page should pre-select the analysis mode from the last analysis."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        mock_analyze.return_value = {
            "score": 4, "adjusted_score": 4,
            "penalty_breakdown": {"blockers":0,"majors":0,"minors":0,"blocker_penalty":0,"major_penalty":0,"minor_penalty":0,"count_penalty":0,"total_penalty":0},
            "matched_skills": ["Python"], "missing_skills": [],
            "reasoning": "ok.", "llm_provider": "anthropic", "llm_model": anthropic_model(),
            "analysis_mode": "fast",
        }

        r = await self.client.post("/api/resumes/add", data={
            "label": "v1", "content": MOCK_RESUME_DEVSECOPS
        })
        rid = r.json()["resume_id"]
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/preselect-mode"})
        jid = j.json()["job_id"]

        await self.client.post(f"/api/jobs/{jid}/analyze", data={
            "resume_id": rid, "provider": "anthropic", "analysis_mode": "fast"
        })

        resp = await self.client.get(f"/job/{jid}")
        self.assertEqual(resp.status_code, 200)
        # Analysis mode pre-selection is now served via /api/jobs/{id}/detail
        detail = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(detail.json()["analysis_mode"], "fast")

    @patch("main.scrape_job")
    async def test_job_detail_no_analyses_uses_env_defaults(self, mock_scrape):
        """With no analyses, model and mode should fall back to env defaults."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/no-analyses"})
        jid = j.json()["job_id"]

        resp = await self.client.get(f"/job/{jid}")
        self.assertEqual(resp.status_code, 200)
        # Static shell always renders — verify defaults via API
        self.assertIn(b"job-detail-main", resp.content)
        detail = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(detail.json()["analyses"], [])
        self.assertEqual(detail.json()["last_provider"], "anthropic")

    @patch("main.scrape_job")
    async def test_job_detail_hides_anthropic_when_no_key(self, mock_scrape):
        """Anthropic radio button should not appear when API key is not set."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/no-anthropic"})
        jid = j.json()["job_id"]

        import os
        orig = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            resp = await self.client.get(f"/job/{jid}")
            self.assertEqual(resp.status_code, 200)
            # Provider availability now served via /api/providers/status
            status = await self.client.get("/api/providers/status")
            self.assertFalse(status.json()["has_anthropic"])
        finally:
            if orig is not None:
                os.environ["ANTHROPIC_API_KEY"] = orig

    @patch("main.scrape_job")
    async def test_job_detail_shows_anthropic_when_key_set(self, mock_scrape):
        """Anthropic radio button should appear when API key is set."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/with-anthropic"})
        jid = j.json()["job_id"]

        import os
        orig = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-testkey"
        try:
            resp = await self.client.get(f"/job/{jid}")
            self.assertEqual(resp.status_code, 200)
            # Provider availability now served via /api/providers/status
            status = await self.client.get("/api/providers/status")
            self.assertTrue(status.json()["has_anthropic"])
        finally:
            if orig is not None:
                os.environ["ANTHROPIC_API_KEY"] = orig
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)

    @patch("main.scrape_job")
    async def test_job_detail_shows_openai_when_key_set(self, mock_scrape):
        """OpenAI radio button should appear when API key is set."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/with-openai"})
        jid = j.json()["job_id"]

        import os
        orig = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-proj-testkey"
        try:
            resp = await self.client.get(f"/job/{jid}")
            self.assertEqual(resp.status_code, 200)
            # Provider availability now served via /api/providers/status
            status = await self.client.get("/api/providers/status")
            self.assertTrue(status.json()["has_openai"])
        finally:
            if orig is not None:
                os.environ["OPENAI_API_KEY"] = orig
            else:
                os.environ.pop("OPENAI_API_KEY", None)

    @patch("main.scrape_job")
    async def test_job_detail_hides_openai_when_no_key(self, mock_scrape):
        """OpenAI radio button should not appear when API key is not set."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/no-openai"})
        jid = j.json()["job_id"]

        import os
        orig = os.environ.pop("OPENAI_API_KEY", None)
        try:
            resp = await self.client.get(f"/job/{jid}")
            self.assertEqual(resp.status_code, 200)
            # Provider availability now served via /api/providers/status
            status = await self.client.get("/api/providers/status")
            self.assertFalse(status.json()["has_openai"])
        finally:
            if orig is not None:
                os.environ["OPENAI_API_KEY"] = orig

    async def test_job_detail_404_for_missing(self):
        """Page shell always returns 200 — 404 is served by the API endpoint."""
        resp = await self.client.get("/job/9999")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"job-detail-main", resp.content)
        api_resp = await self.client.get("/api/jobs/9999/detail")
        self.assertEqual(api_resp.status_code, 404)

    @patch("main.scrape_job")
    async def test_job_detail_page_renders(self, mock_scrape):
        mock_scrape.return_value = {
            "title": "DevSecOps Engineer", "company": "Acme", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/detail"})
        jid = j.json()["job_id"]
        resp = await self.client.get(f"/job/{jid}")
        self.assertEqual(resp.status_code, 200)
        # Page is now a static shell — job data served via /api/jobs/{id}/detail
        self.assertIn(b"job-detail-main", resp.content)
        self.assertIn(b"app.js", resp.content)


class TestEndToEndFlow(unittest.IsolatedAsyncioTestCase):
    """
    Full happy-path: add resume -> add job -> analyze -> save application -> verify DB.
    """

    async def asyncSetUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        os.environ["DB_PATH"] = self.tmp.name

        from database import init_db
        await init_db()

        from httpx import AsyncClient, ASGITransport
        from main import app
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        os.unlink(self.tmp.name)

    @patch("main.scrape_job")
    @patch("main.analyze_match")
    async def test_full_flow(self, mock_analyze, mock_scrape):
        mock_scrape.return_value = {
            "title": "DevSecOps Engineer", "company": "Acme Federal",
            "location": "Arlington, VA", "raw_description": MOCK_JOB_DEVSECOPS
        }
        mock_analyze.return_value = {
            "score": 5, "adjusted_score": 3,
            "penalty_breakdown": {"blockers": 1, "majors": 0, "minors": 0, "blocker_penalty": 2, "major_penalty": 0, "minor_penalty": 0, "count_penalty": 0, "total_penalty": 2},
            "matched_skills": ["Python", "Docker", "AWS", "Security+", "Splunk"],
            "missing_skills": [{"skill": "Active Secret Clearance", "severity": "blocker"}],
            "reasoning": "Excellent match for this DevSecOps federal role.",
            "llm_provider": "anthropic", "llm_model": anthropic_model(),
        }

        r = await self.client.post("/api/resumes/add", data={
            "label": "DevSecOps v1", "content": MOCK_RESUME_DEVSECOPS
        })
        self.assertEqual(r.status_code, 200)
        resume_id = r.json()["resume_id"]

        j = await self.client.post("/api/jobs/add", data={
            "url": "https://clearancejobs.com/jobs/devsecops-engineer"
        })
        self.assertEqual(j.status_code, 200)
        job_id = j.json()["job_id"]
        self.assertEqual(j.json()["title"], "DevSecOps Engineer")

        a = await self.client.post(f"/api/jobs/{job_id}/analyze", data={
            "resume_id": resume_id, "provider": "anthropic"
        })
        self.assertEqual(a.status_code, 200)
        self.assertEqual(a.json()["score"], 5)
        self.assertIn("Python", a.json()["matched_skills"])

        app_resp = await self.client.post(f"/api/jobs/{job_id}/application", data={
            "status": "applied", "recruiter_name": "Sumpter",
            "recruiter_email": "sumpter@acme.gov", "recruiter_phone": "571-555-0100",
            "notes": "Responded to LinkedIn outreach. Strong interest in the role."
        })
        self.assertEqual(app_resp.status_code, 200)

        page = await self.client.get(f"/job/{job_id}")
        self.assertEqual(page.status_code, 200)
        # Page is now a static shell — job data served via /api/jobs/{id}/detail
        self.assertIn(b"job-detail-main", page.content)
        detail = await self.client.get(f"/api/jobs/{job_id}/detail")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["job"]["title"], "DevSecOps Engineer")
        self.assertEqual(detail.json()["job"]["company"], "Acme Federal")

        import aiosqlite
        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT COUNT(*) as n FROM analyses WHERE job_id=?", (job_id,)
            ) as cur:
                count = (await cur.fetchone())["n"]
            async with db.execute(
                "SELECT * FROM applications WHERE job_id=?", (job_id,)
            ) as cur:
                app_row = dict(await cur.fetchone())

        self.assertEqual(count, 1)
        self.assertEqual(app_row["status"], "applied")
        self.assertEqual(app_row["recruiter_name"], "Sumpter")

    # ── GET /api/jobs/{job_id}/detail ─────────────────────────────────────────

    async def test_job_detail_api_not_found(self):
        """Non-existent job_id should return 404."""
        resp = await self.client.get("/api/jobs/99999/detail")
        self.assertEqual(resp.status_code, 404)

    @patch("main.scrape_job")
    async def test_job_detail_api_returns_job_fields(self, mock_scrape):
        """Response must include job object with expected fields."""
        mock_scrape.return_value = {
            "title": "Backend Engineer", "company": "Acme", "location": "Remote",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-job-fields"})
        jid = j.json()["job_id"]

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("job", data)
        self.assertEqual(data["job"]["id"], jid)
        self.assertEqual(data["job"]["title"], "Backend Engineer")
        self.assertEqual(data["job"]["company"], "Acme")

    @patch("main.scrape_job")
    async def test_job_detail_api_empty_application(self, mock_scrape):
        """Job with no application saved should return empty application dict."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-no-app"})
        jid = j.json()["job_id"]

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["application"], {})

    @patch("main.scrape_job")
    async def test_job_detail_api_application_populated(self, mock_scrape):
        """Saved application data should appear in the response."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-with-app"})
        jid = j.json()["job_id"]
        await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "applied", "recruiter_name": "Jane",
            "recruiter_email": "jane@co.com", "recruiter_phone": "",
        })

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(resp.status_code, 200)
        app = resp.json()["application"]
        self.assertEqual(app["status"], "applied")
        self.assertEqual(app["recruiter_name"], "Jane")

    @patch("main.scrape_job")
    async def test_job_detail_api_empty_analyses(self, mock_scrape):
        """Job with no analyses should return empty list."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-no-analyses"})
        jid = j.json()["job_id"]

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["analyses"], [])

    @patch("main.scrape_job")
    async def test_job_detail_api_analyses_parsed(self, mock_scrape):
        """matched_skills and missing_skills in analyses must be lists, not raw strings."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-analyses-parsed"})
        jid = j.json()["job_id"]

        r = await self.client.post("/api/resumes/add", data={"label": "v1", "content": MOCK_RESUME_DEVSECOPS})
        rid = r.json()["resume_id"]

        with patch("main.analyze_match") as mock_analyze:
            mock_analyze.return_value = {
                "score": 4, "adjusted_score": 4,
                "penalty_breakdown": {"blockers": 0, "majors": 0, "minors": 0,
                                      "blocker_penalty": 0, "major_penalty": 0,
                                      "minor_penalty": 0, "count_penalty": 0, "total_penalty": 0},
                "matched_skills": ["Python", "Docker"],
                "missing_skills": [],
                "reasoning": "Good match.", "llm_provider": "anthropic",
                "llm_model": anthropic_model(),
            }
            await self.client.post(f"/api/jobs/{jid}/analyze", data={
                "resume_id": rid, "provider": "anthropic"
            })

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(resp.status_code, 200)
        analyses = resp.json()["analyses"]
        self.assertEqual(len(analyses), 1)
        self.assertIsInstance(analyses[0]["matched_skills"], list)
        self.assertIsInstance(analyses[0]["missing_skills"], list)

    @patch("main.scrape_job")
    async def test_job_detail_api_used_fallback_is_bool(self, mock_scrape):
        """used_fallback field in analyses must be a bool, not an int."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-fallback-bool"})
        jid = j.json()["job_id"]

        r = await self.client.post("/api/resumes/add", data={"label": "v1", "content": MOCK_RESUME_DEVSECOPS})
        rid = r.json()["resume_id"]

        with patch("main.analyze_match") as mock_analyze:
            mock_analyze.return_value = {
                "score": 3, "adjusted_score": 3,
                "penalty_breakdown": {"blockers": 0, "majors": 0, "minors": 0,
                                      "blocker_penalty": 0, "major_penalty": 0,
                                      "minor_penalty": 0, "count_penalty": 0, "total_penalty": 0},
                "matched_skills": [], "missing_skills": [],
                "reasoning": "Ok.", "llm_provider": "anthropic",
                "llm_model": anthropic_model(),
            }
            await self.client.post(f"/api/jobs/{jid}/analyze", data={
                "resume_id": rid, "provider": "anthropic"
            })

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        analyses = resp.json()["analyses"]
        self.assertIsInstance(analyses[0]["used_fallback"], bool)

    @patch("main.scrape_job")
    async def test_job_detail_api_resumes_list(self, mock_scrape):
        """Resumes list should be present and contain saved resumes."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        await self.client.post("/api/resumes/add", data={"label": "My Resume", "content": MOCK_RESUME_DEVSECOPS})
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-resumes"})
        jid = j.json()["job_id"]

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(resp.status_code, 200)
        resumes = resp.json()["resumes"]
        self.assertIsInstance(resumes, list)
        self.assertTrue(any(r["label"] == "My Resume" for r in resumes))

    @patch("main.scrape_job")
    async def test_job_detail_api_has_required_top_level_keys(self, mock_scrape):
        """Response must include all expected top-level keys."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-keys"})
        jid = j.json()["job_id"]

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for key in ["job", "application", "analyses", "resumes", "text_quality",
                    "comparison", "salary_estimate", "has_salary_in_jd",
                    "last_resume_id", "last_provider", "analysis_mode",
                    "anthropic_model", "openai_model", "gemini_model", "ollama_model"]:
            self.assertIn(key, data, f"Missing key: {key}")

    @patch("main.scrape_job")
    async def test_job_detail_api_text_quality_present(self, mock_scrape):
        """text_quality must be a dict with a 'level' key."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-text-quality"})
        jid = j.json()["job_id"]

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        tq = resp.json()["text_quality"]
        self.assertIsInstance(tq, dict)
        self.assertIn("level", tq)

    @patch("main.scrape_job")
    async def test_job_detail_api_salary_estimate_none_by_default(self, mock_scrape):
        """salary_estimate should be null for a fresh job with no estimate."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-no-salary"})
        jid = j.json()["job_id"]

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertIsNone(resp.json()["salary_estimate"])

    @patch("main.scrape_job")
    async def test_job_detail_api_default_provider_is_anthropic(self, mock_scrape):
        """last_provider should default to 'anthropic' when no analyses exist."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-default-provider"})
        jid = j.json()["job_id"]

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(resp.json()["last_provider"], "anthropic")

    @patch("main.scrape_job")
    async def test_job_detail_api_last_provider_from_analysis(self, mock_scrape):
        """last_provider should reflect the provider used in the most recent analysis."""
        mock_scrape.return_value = {
            "title": "Dev", "company": "Co", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS,
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/detail-last-provider"})
        jid = j.json()["job_id"]

        r = await self.client.post("/api/resumes/add", data={"label": "v1", "content": MOCK_RESUME_DEVSECOPS})
        rid = r.json()["resume_id"]

        with patch("main.analyze_match") as mock_analyze:
            mock_analyze.return_value = {
                "score": 3, "adjusted_score": 3,
                "penalty_breakdown": {"blockers": 0, "majors": 0, "minors": 0,
                                      "blocker_penalty": 0, "major_penalty": 0,
                                      "minor_penalty": 0, "count_penalty": 0, "total_penalty": 0},
                "matched_skills": [], "missing_skills": [],
                "reasoning": "Ok.", "llm_provider": "ollama",
                "llm_model": "llama3.1:8b",
            }
            await self.client.post(f"/api/jobs/{jid}/analyze", data={
                "resume_id": rid, "provider": "ollama", "model": "llama3.1:8b"
            })

        resp = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(resp.json()["last_provider"], "ollama")

    # ── GET /api/providers/status ─────────────────────────────────────────────

    async def test_providers_status_has_required_keys(self):
        """Response must include all expected keys."""
        resp = await self.client.get("/api/providers/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for key in ["has_anthropic", "has_openai", "has_gemini", "has_ollama",
                    "anthropic_model", "openai_model", "gemini_model", "ollama_model",
                    "mx_auto_check"]:
            self.assertIn(key, data, f"Missing key: {key}")

    async def test_providers_status_has_anthropic_false_when_key_unset(self):
        """has_anthropic must be False when ANTHROPIC_API_KEY is not set."""
        orig = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            resp = await self.client.get("/api/providers/status")
            self.assertFalse(resp.json()["has_anthropic"])
        finally:
            if orig:
                os.environ["ANTHROPIC_API_KEY"] = orig

    async def test_providers_status_has_anthropic_true_when_key_set(self):
        """has_anthropic must be True when ANTHROPIC_API_KEY is set."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-test-key"
        try:
            resp = await self.client.get("/api/providers/status")
            self.assertTrue(resp.json()["has_anthropic"])
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    async def test_providers_status_has_openai_false_when_key_unset(self):
        """has_openai must be False when OPENAI_API_KEY is not set."""
        orig = os.environ.pop("OPENAI_API_KEY", None)
        try:
            resp = await self.client.get("/api/providers/status")
            self.assertFalse(resp.json()["has_openai"])
        finally:
            if orig:
                os.environ["OPENAI_API_KEY"] = orig

    async def test_providers_status_has_gemini_false_when_key_unset(self):
        """has_gemini must be False when GEMINI_API_KEY is not set."""
        orig = os.environ.pop("GEMINI_API_KEY", None)
        try:
            resp = await self.client.get("/api/providers/status")
            self.assertFalse(resp.json()["has_gemini"])
        finally:
            if orig:
                os.environ["GEMINI_API_KEY"] = orig

    async def test_providers_status_has_ollama_false_when_unreachable(self):
        """has_ollama must be False when Ollama is unreachable."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=mock_client):
            resp = await self.client.get("/api/providers/status")
        self.assertFalse(resp.json()["has_ollama"])

    async def test_providers_status_model_fields_are_strings(self):
        """All model fields must be non-empty strings."""
        resp = await self.client.get("/api/providers/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for key in ["anthropic_model", "openai_model", "gemini_model", "ollama_model"]:
            self.assertIsInstance(data[key], str, f"{key} should be a string")
            self.assertTrue(len(data[key]) > 0, f"{key} should not be empty")

    async def test_providers_status_mx_auto_check_defaults_true(self):
        """mx_auto_check should default to True when MX_AUTO_CHECK env var is unset."""
        import os
        os.environ.pop("MX_AUTO_CHECK", None)
        resp = await self.client.get("/api/providers/status")
        self.assertEqual(resp.status_code, 200)
        self.assertIs(resp.json()["mx_auto_check"], True)

    # ── PATCH /api/jobs/{id}/url ───────────────────────────────────────────

    async def test_update_job_url_sets_url(self):
        """PATCH /api/jobs/{id}/url should update the job URL."""
        import aiosqlite
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        patch_resp = await self.client.patch(f"/api/jobs/{jid}/url", data={
            "url": "https://example.com/job/123"
        })
        self.assertEqual(patch_resp.status_code, 200)
        self.assertTrue(patch_resp.json()["ok"])
        self.assertEqual(patch_resp.json()["url"], "https://example.com/job/123")

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT url FROM jobs WHERE id = ?", (jid,)) as cur:
                row = await cur.fetchone()
        self.assertEqual(row["url"], "https://example.com/job/123")

    async def test_update_job_url_clear_restores_synthetic(self):
        """Clearing URL should restore a manual:// synthetic URL."""
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        patch_resp = await self.client.patch(f"/api/jobs/{jid}/url", data={"url": ""})
        self.assertEqual(patch_resp.status_code, 200)
        self.assertTrue(patch_resp.json()["url"].startswith("manual://"))

    async def test_update_job_url_invalid_scheme_returns_422(self):
        """Non-http(s) URL should return 422."""
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        patch_resp = await self.client.patch(f"/api/jobs/{jid}/url", data={
            "url": "ftp://bad.example.com"
        })
        self.assertEqual(patch_resp.status_code, 422)

    async def test_update_job_url_not_found_returns_404(self):
        """Patching a non-existent job should return 404."""
        patch_resp = await self.client.patch("/api/jobs/99999/url", data={
            "url": "https://example.com"
        })
        self.assertEqual(patch_resp.status_code, 404)

    # ── PATCH /api/jobs/{id}/title ─────────────────────────────────────────

    async def test_update_job_title_sets_title(self):
        """PATCH /api/jobs/{id}/title should update the job title."""
        import aiosqlite
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Old Title", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        patch_resp = await self.client.patch(f"/api/jobs/{jid}/title", data={
            "title": "New Title"
        })
        self.assertEqual(patch_resp.status_code, 200)
        self.assertTrue(patch_resp.json()["ok"])
        self.assertEqual(patch_resp.json()["title"], "New Title")

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT title FROM jobs WHERE id = ?", (jid,)) as cur:
                row = await cur.fetchone()
        self.assertEqual(row["title"], "New Title")

    async def test_update_job_title_empty_returns_422(self):
        """Empty title should return 422."""
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        patch_resp = await self.client.patch(f"/api/jobs/{jid}/title", data={"title": ""})
        self.assertEqual(patch_resp.status_code, 422)
        self.assertIn("empty", patch_resp.json()["error"].lower())

    async def test_update_job_title_not_found_returns_404(self):
        """Patching title on a non-existent job should return 404."""
        patch_resp = await self.client.patch("/api/jobs/99999/title", data={"title": "X"})
        self.assertEqual(patch_resp.status_code, 404)

    async def test_update_job_title_reflected_in_detail_api(self):
        """After PATCH, /api/jobs/{id}/detail should return the new title."""
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Original", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        await self.client.patch(f"/api/jobs/{jid}/title", data={"title": "Updated Title"})

        detail = await self.client.get(f"/api/jobs/{jid}/detail")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["job"]["title"], "Updated Title")

    # ── PATCH /api/jobs/{id}/company ───────────────────────────────────────

    async def test_update_job_company_sets_company(self):
        """PATCH /api/jobs/{id}/company should update the company name."""
        import aiosqlite
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "OldCo", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        patch_resp = await self.client.patch(f"/api/jobs/{jid}/company", data={
            "company": "NewCo"
        })
        self.assertEqual(patch_resp.status_code, 200)
        self.assertTrue(patch_resp.json()["ok"])
        self.assertEqual(patch_resp.json()["company"], "NewCo")

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT company FROM jobs WHERE id = ?", (jid,)) as cur:
                row = await cur.fetchone()
        self.assertEqual(row["company"], "NewCo")

    async def test_update_job_company_allows_empty(self):
        """Empty company is valid — some postings have no company."""
        import aiosqlite
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Acme", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        patch_resp = await self.client.patch(f"/api/jobs/{jid}/company", data={"company": ""})
        self.assertEqual(patch_resp.status_code, 200)
        self.assertEqual(patch_resp.json()["company"], "")

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT company FROM jobs WHERE id = ?", (jid,)) as cur:
                row = await cur.fetchone()
        self.assertEqual(row["company"], "")

    async def test_update_job_company_not_found_returns_404(self):
        """Patching company on a non-existent job should return 404."""
        patch_resp = await self.client.patch("/api/jobs/99999/company", data={"company": "X"})
        self.assertEqual(patch_resp.status_code, 404)

    # ── PATCH /api/jobs/{id}/location ──────────────────────────────────────

    async def test_update_job_location_sets_location(self):
        """PATCH /api/jobs/{id}/location should update the location."""
        import aiosqlite
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        patch_resp = await self.client.patch(f"/api/jobs/{jid}/location", data={
            "location": "Washington, DC"
        })
        self.assertEqual(patch_resp.status_code, 200)
        self.assertTrue(patch_resp.json()["ok"])
        self.assertEqual(patch_resp.json()["location"], "Washington, DC")

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT location FROM jobs WHERE id = ?", (jid,)) as cur:
                row = await cur.fetchone()
        self.assertEqual(row["location"], "Washington, DC")

    async def test_update_job_location_allows_empty(self):
        """Empty location is valid."""
        import aiosqlite
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        patch_resp = await self.client.patch(f"/api/jobs/{jid}/location", data={"location": ""})
        self.assertEqual(patch_resp.status_code, 200)
        self.assertEqual(patch_resp.json()["location"], "")

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT location FROM jobs WHERE id = ?", (jid,)) as cur:
                row = await cur.fetchone()
        self.assertEqual(row["location"], "")

    async def test_update_job_location_not_found_returns_404(self):
        """Patching location on a non-existent job should return 404."""
        patch_resp = await self.client.patch("/api/jobs/99999/location", data={"location": "X"})
        self.assertEqual(patch_resp.status_code, 404)

    async def test_add_manual_job_with_source_url(self):
        """Providing source_url should store it instead of manual:// URL."""
        import aiosqlite
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title":      "Dev",
            "company":    "Co",
            "source_url": "https://linkedin.com/jobs/view/12345",
            "description": MOCK_JOB_DEVSECOPS,
        })
        self.assertEqual(resp.status_code, 200)
        jid = resp.json()["job_id"]

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT url FROM jobs WHERE id = ?", (jid,)) as cur:
                row = await cur.fetchone()
        self.assertEqual(row["url"], "https://linkedin.com/jobs/view/12345")

    async def test_add_manual_job_without_source_url_is_manual(self):
        """Omitting source_url should still generate a manual:// URL."""
        import aiosqlite
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        self.assertEqual(resp.status_code, 200)
        jid = resp.json()["job_id"]

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT url FROM jobs WHERE id = ?", (jid,)) as cur:
                row = await cur.fetchone()
        self.assertTrue(row["url"].startswith("manual://"))

    # ── POST /api/resumes/extract ──────────────────────────────────────────

    async def test_extract_resume_txt(self):
        """TXT file should be extracted and returned as text."""
        content = b"John Doe\nSoftware Engineer\nPython, Go, Docker, AWS, PostgreSQL\n" * 5
        resp = await self.client.post(
            "/api/resumes/extract",
            files={"file": ("resume.txt", content, "text/plain")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("text", data)
        self.assertIn("char_count", data)
        self.assertGreater(data["char_count"], 0)

    async def test_extract_resume_unsupported_type(self):
        """Unsupported file type should return 422."""
        resp = await self.client.post(
            "/api/resumes/extract",
            files={"file": ("resume.md", b"# Resume", "text/markdown")},
        )
        self.assertEqual(resp.status_code, 422)

    async def test_extract_resume_txt_too_short(self):
        """TXT file with less than 50 chars should return 422."""
        resp = await self.client.post(
            "/api/resumes/extract",
            files={"file": ("resume.txt", b"Too short", "text/plain")},
        )
        self.assertEqual(resp.status_code, 422)

    async def test_extract_resume_txt_content_matches(self):
        """Extracted text should match the uploaded content."""
        text = b"John Doe\nSenior Python Developer\nSkills: Python, FastAPI, PostgreSQL, Docker, AWS, CI/CD\n"
        resp = await self.client.post(
            "/api/resumes/extract",
            files={"file": ("resume.txt", text, "text/plain")},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("John Doe", resp.json()["text"])

    # ── GET /api/resumes/{id} ──────────────────────────────────────────────

    async def test_get_resume_returns_content(self):
        """GET /api/resumes/{id} should return full resume content."""
        add = await self.client.post("/api/resumes/add", data={
            "label": "v1", "content": "John Doe\nSoftware Engineer\nPython Go Docker AWS\n" * 10,
        })
        rid = add.json()["resume_id"]

        resp = await self.client.get(f"/api/resumes/{rid}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("content", data)
        self.assertIn("label", data)
        self.assertIn("char_count", data)
        self.assertEqual(data["label"], "v1")
        self.assertIn("John Doe", data["content"])

    async def test_get_resume_not_found_returns_404(self):
        """GET /api/resumes/99999 should return 404."""
        resp = await self.client.get("/api/resumes/99999")
        self.assertEqual(resp.status_code, 404)

    async def test_get_resume_char_count_matches_content(self):
        """char_count should match actual content length."""
        text = "John Doe\nSoftware Engineer\nPython Go Docker AWS\n" * 10
        add = await self.client.post("/api/resumes/add", data={"label": "v1", "content": text})
        rid = add.json()["resume_id"]

        resp = await self.client.get(f"/api/resumes/{rid}")
        data = resp.json()
        self.assertEqual(data["char_count"], len(data["content"]))

    # ── /api/jobs/{id}/email ───────────────────────────────────────────────

    async def test_get_job_email_returns_null_when_none(self):
        """GET /api/jobs/{id}/email returns null when no email saved."""
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        r = await self.client.get(f"/api/jobs/{jid}/email")
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["email"])

    async def test_save_and_get_job_email(self):
        """POST then GET should return the saved email."""
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        html = "<html><body><p>Interview confirmed for Monday.</p></body></html>"
        save = await self.client.post(f"/api/jobs/{jid}/email", data={"raw_html": html})
        self.assertEqual(save.status_code, 200)
        self.assertTrue(save.json()["ok"])

        r = await self.client.get(f"/api/jobs/{jid}/email")
        self.assertEqual(r.status_code, 200)
        data = r.json()["email"]
        self.assertIsNotNone(data)
        self.assertEqual(data["raw_html"], html)

    async def test_save_job_email_upserts(self):
        """Saving again should replace the previous email."""
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        await self.client.post(f"/api/jobs/{jid}/email", data={"raw_html": "<p>First</p>"})
        await self.client.post(f"/api/jobs/{jid}/email", data={"raw_html": "<p>Second</p>"})

        r = await self.client.get(f"/api/jobs/{jid}/email")
        self.assertIn("Second", r.json()["email"]["raw_html"])

    async def test_save_job_email_empty_returns_422(self):
        """POST with empty raw_html should return 422."""
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        r = await self.client.post(f"/api/jobs/{jid}/email", data={"raw_html": ""})
        self.assertEqual(r.status_code, 422)

    async def test_delete_job_email(self):
        """DELETE should remove the saved email."""
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]

        await self.client.post(f"/api/jobs/{jid}/email", data={"raw_html": "<p>Email</p>"})
        r = await self.client.delete(f"/api/jobs/{jid}/email")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

        r2 = await self.client.get(f"/api/jobs/{jid}/email")
        self.assertIsNone(r2.json()["email"])

    async def test_email_cascades_on_job_delete(self):
        """Deleting a job should also delete its email via FK cascade."""
        import aiosqlite
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Co", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]
        await self.client.post(f"/api/jobs/{jid}/email", data={"raw_html": "<p>Email</p>"})
        await self.client.delete(f"/api/jobs/{jid}")

        async with aiosqlite.connect(self.tmp.name) as db:
            async with db.execute("SELECT COUNT(*) FROM job_emails WHERE job_id = ?", (jid,)) as cur:
                count = (await cur.fetchone())[0]
        self.assertEqual(count, 0)

    # ── GET /api/vetting ───────────────────────────────────────────────────

    async def test_vetting_returns_companies_and_recruiters_keys(self):
        """GET /api/vetting should return companies and recruiters keys."""
        resp = await self.client.get("/api/vetting")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("companies", data)
        self.assertIn("recruiters", data)

    async def test_vetting_groups_by_company(self):
        """Jobs should be grouped by company name."""
        await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev A", "company": "Acme",
            "description": MOCK_JOB_DEVSECOPS,
        })
        await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev B", "company": "Acme",
            "description": MOCK_JOB_MARKETING,  # different description = different hash
        })
        resp = await self.client.get("/api/vetting")
        companies = resp.json()["companies"]
        acme = next((c for c in companies if c["company"] == "Acme"), None)
        self.assertIsNotNone(acme)
        self.assertEqual(len(acme["jobs"]), 2)
        self.assertIn("meta", acme)  # LLM vetting meta field must be present

    async def test_vetting_groups_recruiter_by_email(self):
        """Jobs with same recruiter email should be grouped together."""
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "Acme", "description": MOCK_JOB_DEVSECOPS,
        })
        jid = resp.json()["job_id"]
        await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "applied",
            "recruiter_name": "Jane Doe",
            "recruiter_email": "jane@acme.com",
            "recruiter_phone": "", "notes": "",
        })
        resp2 = await self.client.get("/api/vetting")
        recruiters = resp2.json()["recruiters"]
        jane = next((r for r in recruiters if r["email"] == "jane@acme.com"), None)
        self.assertIsNotNone(jane)
        self.assertEqual(len(jane["jobs"]), 1)
        # scraped_at should be present in recruiter jobs for date display
        self.assertIn("scraped_at", jane["jobs"][0])

    async def test_vetting_empty_db_returns_empty_lists(self):
        """Empty DB should return empty companies and recruiters lists."""
        resp = await self.client.get("/api/vetting")
        data = resp.json()
        self.assertEqual(data["companies"], [])
        self.assertEqual(data["recruiters"], [])

    async def test_vetting_page_renders(self):
        """GET /vetting should return HTML."""
        resp = await self.client.get("/vetting")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("content-type", ""))

    # ── POST /api/companies/crawl ──────────────────────────────────────────

    async def test_crawl_company_empty_name_returns_422(self):
        """Empty company_name should return 422."""
        resp = await self.client.post("/api/companies/crawl", data={"company_name": ""})
        self.assertEqual(resp.status_code, 422)

    async def test_crawl_company_returns_ok_with_mocked_crawler(self):
        """With mocked crawl_company, endpoint returns ok:True and caches result."""
        from unittest.mock import patch, AsyncMock
        mock_data = {
            "glassdoor_url": "https://glassdoor.com/acme",
            "glassdoor_rating": 4.2,
            "bbb_url": "https://bbb.org/acme",
            "bbb_rating": "A+",
        }
        with patch("main.crawl_company", new_callable=AsyncMock, return_value=mock_data):
            resp = await self.client.post("/api/companies/crawl", data={"company_name": "Acme Corp"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertFalse(data["cached"])
        self.assertEqual(data["company_name"], "Acme Corp")

    async def test_crawl_company_returns_cached_on_second_call(self):
        """Second call within 7 days should return cached=True without re-crawling."""
        from unittest.mock import patch, AsyncMock
        mock_data = {"glassdoor_url": "https://glassdoor.com/acme", "glassdoor_rating": 4.2}
        with patch("main.crawl_company", new_callable=AsyncMock, return_value=mock_data) as mock_fn:
            await self.client.post("/api/companies/crawl", data={"company_name": "CacheTest Inc"})
            resp2 = await self.client.post("/api/companies/crawl", data={"company_name": "CacheTest Inc"})
            self.assertEqual(mock_fn.call_count, 1)  # only called once
        self.assertTrue(resp2.json()["cached"])

    async def test_get_company_meta_not_found_returns_cached_false(self):
        """GET /api/companies/meta for unknown company returns cached:False."""
        resp = await self.client.get("/api/companies/meta?company_name=Nobody+Corp")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["cached"])

    async def test_get_company_meta_after_crawl_returns_cached_true(self):
        """GET /api/companies/meta after a crawl returns cached:True with data."""
        from unittest.mock import patch, AsyncMock
        mock_data = {"bbb_rating": "B", "bbb_url": "https://bbb.org/test"}
        with patch("main.crawl_company", new_callable=AsyncMock, return_value=mock_data):
            await self.client.post("/api/companies/crawl", data={"company_name": "MetaTest LLC"})
        resp = await self.client.get("/api/companies/meta?company_name=MetaTest+LLC")
        data = resp.json()
        self.assertTrue(data["cached"])
        self.assertEqual(data["company_name"], "MetaTest LLC")

    # ── GET /api/email/mx-cache ───────────────────────────────────────────────

    async def test_mx_cache_returns_empty_dict_when_no_entries(self):
        """GET /api/email/mx-cache returns empty dict on fresh DB."""
        resp = await self.client.get("/api/email/mx-cache")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {})

    async def test_mx_cache_returns_cached_entry_after_validate(self):
        """GET /api/email/mx-cache returns entry after a domain is checked."""
        import aiosqlite
        async with aiosqlite.connect(os.environ["DB_PATH"]) as db:
            await db.execute(
                """INSERT INTO domain_mx_cache (domain, has_mx, mx_records)
                   VALUES (?, ?, ?)""",
                ("example.com", 1, "mx1.example.com"),
            )
            await db.commit()

        resp = await self.client.get("/api/email/mx-cache")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("example.com", data)
        self.assertTrue(data["example.com"]["has_mx"])
        self.assertTrue(data["example.com"]["checked"])

    async def test_mx_cache_response_has_required_fields(self):
        """Each entry in mx-cache must have has_mx, mx_records, checked."""
        import aiosqlite
        async with aiosqlite.connect(os.environ["DB_PATH"]) as db:
            await db.execute(
                """INSERT INTO domain_mx_cache (domain, has_mx, mx_records)
                   VALUES (?, ?, ?)""",
                ("testcorp.com", 0, ""),
            )
            await db.commit()

        resp = await self.client.get("/api/email/mx-cache")
        data = resp.json()
        self.assertIn("testcorp.com", data)
        entry = data["testcorp.com"]
        for key in ("has_mx", "mx_records", "checked"):
            self.assertIn(key, entry, f"Missing key: {key}")

    # ── POST /api/companies/vet ───────────────────────────────────────────────

    async def test_vet_company_empty_name_returns_422(self):
        """POST /api/companies/vet with no company_name returns 422."""
        resp = await self.client.post("/api/companies/vet", data={})
        self.assertEqual(resp.status_code, 422)

    async def test_vet_company_returns_structured_result(self):
        """POST /api/companies/vet returns risk_level, assessment, signals."""
        from unittest.mock import patch, AsyncMock
        mock_result = {
            "risk_level": "low", "assessment": "Established company.",
            "signals": ["A+ BBB"], "company": "Acme Corp",
            "provider": "anthropic", "model": "claude-sonnet-4-5",
        }
        with patch("main.vet_company", new=AsyncMock(return_value=mock_result)), \
             patch("main.crawl_company", new=AsyncMock(return_value={})):
            resp = await self.client.post(
                "/api/companies/vet",
                data={"company_name": "Acme Corp", "provider": "anthropic"},
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["risk_level"], "low")
        self.assertIn("assessment", data)
        self.assertIn("signals", data)

    async def test_vet_company_returns_cached_result(self):
        """POST /api/companies/vet returns cached:True when within TTL."""
        import aiosqlite, json
        async with aiosqlite.connect(os.environ["DB_PATH"]) as db:
            await db.execute(
                """INSERT INTO company_meta
                   (company_name, llm_risk_level, llm_assessment, llm_signals,
                    llm_provider, llm_model, llm_assessed_at)
                   VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
                ("CachedCo", "medium", "Some concerns.", "[]", "anthropic", "claude-sonnet-4-5"),
            )
            await db.commit()

        resp = await self.client.post(
            "/api/companies/vet",
            data={"company_name": "CachedCo", "provider": "anthropic"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["cached"])
        self.assertEqual(data["risk_level"], "medium")

    async def test_vet_company_force_bypasses_cache(self):
        """POST /api/companies/vet with force=true skips the cache and re-runs LLM."""
        import aiosqlite
        async with aiosqlite.connect(os.environ["DB_PATH"]) as db:
            await db.execute(
                """INSERT INTO company_meta
                   (company_name, llm_risk_level, llm_assessment, llm_signals,
                    llm_provider, llm_model, llm_assessed_at)
                   VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
                ("ForceCo", "low", "Old result.", "[]", "anthropic", "claude-sonnet-4-5"),
            )
            await db.commit()

        from unittest.mock import patch, AsyncMock
        new_result = {
            "risk_level": "high", "assessment": "Fresh result.",
            "signals": ["new signal"], "company": "ForceCo",
            "provider": "anthropic", "model": "claude-sonnet-4-5",
        }
        with patch("main.vet_company", new=AsyncMock(return_value=new_result)), \
             patch("main.crawl_company", new=AsyncMock(return_value={})):
            resp = await self.client.post(
                "/api/companies/vet",
                data={"company_name": "ForceCo", "provider": "anthropic", "force": "true"},
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["cached"])
        self.assertEqual(data["risk_level"], "high")
        self.assertEqual(data["assessment"], "Fresh result.")

    async def test_vet_company_llm_error_returns_422(self):
        """POST /api/companies/vet returns 422 when LLM raises ValueError."""
        from unittest.mock import patch, AsyncMock
        with patch("main.vet_company", new=AsyncMock(side_effect=ValueError("API key not set"))), \
             patch("main.crawl_company", new=AsyncMock(return_value={})):
            resp = await self.client.post(
                "/api/companies/vet",
                data={"company_name": "ErrorCo", "provider": "anthropic"},
            )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("error", resp.json())

    # ── POST /api/companies/parse-snippet ─────────────────────────────────────

    async def test_parse_snippet_empty_company_returns_422(self):
        resp = await self.client.post(
            "/api/companies/parse-snippet",
            data={"text": "Glassdoor 4.2 stars", "provider": "anthropic"},
        )
        self.assertEqual(resp.status_code, 422)

    async def test_parse_snippet_empty_text_returns_422(self):
        resp = await self.client.post(
            "/api/companies/parse-snippet",
            data={"company_name": "Acme", "text": "", "provider": "anthropic"},
        )
        self.assertEqual(resp.status_code, 422)

    async def test_parse_snippet_text_too_long_returns_422(self):
        resp = await self.client.post(
            "/api/companies/parse-snippet",
            data={"company_name": "Acme", "text": "x" * 6000, "provider": "anthropic"},
        )
        self.assertEqual(resp.status_code, 422)

    async def test_parse_snippet_returns_structured_result(self):
        from unittest.mock import patch, AsyncMock
        mock_data = {"glassdoor_rating": 4.2, "glassdoor_review_count": 500,
                     "bbb_rating": "A+"}
        with patch("main.parse_company_snippet", new=AsyncMock(return_value=mock_data)):
            resp = await self.client.post(
                "/api/companies/parse-snippet",
                data={"company_name": "Acme", "text": "some snippet text",
                      "provider": "anthropic"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["found"])
        self.assertEqual(body["data"]["glassdoor_rating"], 4.2)

    async def test_parse_snippet_empty_result_returns_not_found(self):
        from unittest.mock import patch, AsyncMock
        with patch("main.parse_company_snippet", new=AsyncMock(return_value={})):
            resp = await self.client.post(
                "/api/companies/parse-snippet",
                data={"company_name": "Ghost", "text": "blah blah",
                      "provider": "anthropic"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertFalse(body["found"])

    async def test_parse_snippet_llm_error_returns_422(self):
        from unittest.mock import patch, AsyncMock
        with patch("main.parse_company_snippet",
                   new=AsyncMock(side_effect=ValueError("API key not set"))):
            resp = await self.client.post(
                "/api/companies/parse-snippet",
                data={"company_name": "Acme", "text": "some text",
                      "provider": "anthropic"},
            )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("error", resp.json())

    # ── POST /api/companies/meta/update ──────────────────────────────────────

    async def test_update_meta_empty_company_returns_422(self):
        resp = await self.client.post(
            "/api/companies/meta/update",
            data={"glassdoor_rating": "4.2"},
        )
        self.assertEqual(resp.status_code, 422)

    async def test_update_meta_no_fields_returns_422(self):
        resp = await self.client.post(
            "/api/companies/meta/update",
            data={"company_name": "Acme"},
        )
        self.assertEqual(resp.status_code, 422)

    async def test_update_meta_invalid_rating_returns_422(self):
        resp = await self.client.post(
            "/api/companies/meta/update",
            data={"company_name": "Acme", "glassdoor_rating": "99"},
        )
        self.assertEqual(resp.status_code, 422)

    async def test_update_meta_invalid_url_returns_422(self):
        resp = await self.client.post(
            "/api/companies/meta/update",
            data={"company_name": "Acme", "glassdoor_url": "not-a-url"},
        )
        self.assertEqual(resp.status_code, 422)

    async def test_update_meta_saves_rating(self):
        resp = await self.client.post(
            "/api/companies/meta/update",
            data={"company_name": "RatingCo", "glassdoor_rating": "4.2",
                  "glassdoor_review_count": "500"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertIn("glassdoor_rating", body["updated"])
        self.assertEqual(body["meta"]["glassdoor_rating"], 4.2)

    async def test_update_meta_saves_bbb_grade(self):
        resp = await self.client.post(
            "/api/companies/meta/update",
            data={"company_name": "BBBCo", "bbb_rating": "a+"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        # Should be uppercased
        self.assertEqual(body["meta"]["bbb_rating"], "A+")

    async def test_update_meta_saves_urls(self):
        resp = await self.client.post(
            "/api/companies/meta/update",
            data={"company_name": "URLCo",
                  "glassdoor_url": "https://glassdoor.com/acme",
                  "linkedin_url":  "https://linkedin.com/company/acme"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertIn("glassdoor_url", body["updated"])
        self.assertIn("linkedin_url", body["updated"])

    async def test_update_meta_indeed_rating(self):
        resp = await self.client.post(
            "/api/companies/meta/update",
            data={"company_name": "IndeedCo", "indeed_rating": "3.8",
                  "indeed_review_count": "150"},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["meta"]["indeed_rating"], 3.8)

    async def test_vetting_meta_includes_crawl_fields(self):
        """GET /api/vetting meta should include crawl fields on page load."""
        await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "MetaCo",
            "description": MOCK_JOB_DEVSECOPS,
        })
        await self.client.post("/api/companies/meta/update", data={
            "company_name":     "MetaCo",
            "glassdoor_rating": "4.3",
            "indeed_rating":    "3.9",
            "bbb_rating":       "A",
        })
        resp = await self.client.get("/api/vetting")
        companies = resp.json()["companies"]
        metaco = next((c for c in companies if c["company"] == "MetaCo"), None)
        self.assertIsNotNone(metaco)
        meta = metaco["meta"]
        self.assertEqual(meta["glassdoor_rating"], 4.3)
        self.assertEqual(meta["indeed_rating"], 3.9)
        self.assertEqual(meta["bbb_rating"], "A")

    async def test_vetting_meta_empty_when_no_crawl_data(self):
        """GET /api/vetting meta should be empty dict when no company_meta row exists."""
        await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "NoCrawlCo",
            "description": MOCK_JOB_DEVSECOPS,
        })
        resp = await self.client.get("/api/vetting")
        companies = resp.json()["companies"]
        co = next((c for c in companies if c["company"] == "NoCrawlCo"), None)
        self.assertIsNotNone(co)
        self.assertEqual(co["meta"], {})

    # ── DELETE /api/companies/meta ────────────────────────────────────────────

    async def test_delete_company_meta_empty_name_returns_422(self):
        resp = await self.client.delete("/api/companies/meta?company_name=")
        self.assertEqual(resp.status_code, 422)

    async def test_delete_company_meta_removes_row(self):
        await self.client.post("/api/companies/meta/update", data={
            "company_name": "DeleteMe", "glassdoor_rating": "4.2",
        })
        resp = await self.client.get("/api/companies/meta?company_name=DeleteMe")
        self.assertTrue(resp.json()["cached"])
        resp = await self.client.delete("/api/companies/meta?company_name=DeleteMe")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])
        resp = await self.client.get("/api/companies/meta?company_name=DeleteMe")
        self.assertFalse(resp.json()["cached"])

    async def test_delete_company_meta_nonexistent_returns_ok(self):
        resp = await self.client.delete("/api/companies/meta?company_name=GhostCo")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

    # ── PATCH /api/jobs/{id}/company — meta rename ───────────────────────────

    async def test_update_company_renames_meta(self):
        """Renaming a job's company should move company_meta to the new name."""
        # Add a job
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "OldCo",
            "description": MOCK_JOB_DEVSECOPS,
        })
        job_id = resp.json()["job_id"]
        # Seed meta for OldCo
        await self.client.post("/api/companies/meta/update", data={
            "company_name": "OldCo", "glassdoor_rating": "4.5",
        })
        # Rename company
        resp = await self.client.patch(f"/api/jobs/{job_id}/company",
                                       data={"company": "NewCo Inc."})
        self.assertEqual(resp.status_code, 200)
        # OldCo meta should be gone
        resp = await self.client.get("/api/companies/meta?company_name=OldCo")
        self.assertFalse(resp.json()["cached"])
        # NewCo Inc. meta should have the rating
        resp = await self.client.get("/api/companies/meta?company_name=NewCo Inc.")
        self.assertTrue(resp.json()["cached"])
        self.assertEqual(resp.json()["glassdoor_rating"], 4.5)

    async def test_update_company_no_meta_is_fine(self):
        """Renaming a company with no meta row should not error."""
        resp = await self.client.post("/api/jobs/add-manual", data={
            "title": "Dev", "company": "NoMetaCo",
            "description": MOCK_JOB_DEVSECOPS,
        })
        job_id = resp.json()["job_id"]
        resp = await self.client.patch(f"/api/jobs/{job_id}/company",
                                       data={"company": "NoMetaCoNew"})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])
