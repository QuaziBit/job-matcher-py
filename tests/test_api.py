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
        self.assertIn(b"Accessible Job", page.content)

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
        # gpt-4o should appear as the pre-selected model in the page
        self.assertIn(b"gpt-4o", resp.content)

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
        # fast option should be selected in the mode dropdown
        self.assertIn(b'value="fast"', resp.content)

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
        # Page should render without error — defaults applied
        self.assertIn(b"Run New Analysis", resp.content)

    async def test_job_detail_404_for_missing(self):
        resp = await self.client.get("/job/9999")
        self.assertEqual(resp.status_code, 404)

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
        self.assertIn(b"DevSecOps Engineer", resp.content)


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
        content = page.content.decode()
        self.assertIn("DevSecOps Engineer", content)
        self.assertIn("Acme Federal", content)

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
