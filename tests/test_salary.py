"""
tests/test_salary.py — Unit and API tests for salary estimation feature.
Covers: detection, prompt building, response parsing, LLM callers, endpoints.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.mock_data import (
    run,
    MOCK_JOB_NO_SALARY,
    MOCK_JOB_WITH_SALARY,
    MOCK_SALARY_RESPONSE,
)


class TestSalaryDetection(unittest.TestCase):
    """Tests for _job_has_salary salary keyword detection."""

    def test_detects_dollar_amount(self):
        from analyzer import _job_has_salary
        self.assertTrue(_job_has_salary("Salary: $120,000 – $150,000 per year"))

    def test_detects_salary_with_number(self):
        from analyzer import _job_has_salary
        self.assertTrue(_job_has_salary("salary range: $120,000 - $150,000"))

    def test_detects_salary_keyword_vague_no_match(self):
        from analyzer import _job_has_salary
        self.assertFalse(_job_has_salary("The salary for this role is competitive"))

    def test_detects_compensation_keyword(self):
        from analyzer import _job_has_salary
        self.assertTrue(_job_has_salary("Total compensation includes base pay and equity"))

    def test_detects_annual_keyword(self):
        from analyzer import _job_has_salary
        self.assertTrue(_job_has_salary("Annual pay range: $100k-$130k"))

    def test_detects_annual_keyword_no_dollar_no_match(self):
        from analyzer import _job_has_salary
        self.assertFalse(_job_has_salary("Annual pay range: 100k-130k"))

    def test_no_salary_in_description(self):
        from analyzer import _job_has_salary
        self.assertFalse(_job_has_salary(MOCK_JOB_NO_SALARY))

    def test_empty_description(self):
        from analyzer import _job_has_salary
        self.assertFalse(_job_has_salary(""))


class TestSalaryPromptBuilder(unittest.TestCase):
    """Tests for _build_salary_prompt."""

    def test_includes_title(self):
        from analyzer import _build_salary_prompt
        prompt = _build_salary_prompt("Senior Python Dev", "", "", "Some JD")
        self.assertIn("Senior Python Dev", prompt)

    def test_includes_company(self):
        from analyzer import _build_salary_prompt
        prompt = _build_salary_prompt("", "Acme Corp", "", "Some JD")
        self.assertIn("Acme Corp", prompt)

    def test_includes_location(self):
        from analyzer import _build_salary_prompt
        prompt = _build_salary_prompt("", "", "Washington DC", "Some JD")
        self.assertIn("Washington DC", prompt)

    def test_truncates_long_description(self):
        from analyzer import _build_salary_prompt
        long_jd = "x" * 10000
        prompt = _build_salary_prompt("Dev", "Co", "NYC", long_jd)
        self.assertLess(len(prompt), 6000)
        self.assertNotIn("x" * 5000, prompt)

    def test_requests_json_response(self):
        from analyzer import _build_salary_prompt
        prompt = _build_salary_prompt("Dev", "Co", "NYC", "JD text")
        self.assertIn("JSON", prompt)
        self.assertIn("min", prompt)
        self.assertIn("max", prompt)
        self.assertIn("confidence", prompt)


class TestSalaryResponseParser(unittest.TestCase):
    """Tests for _parse_salary_response."""

    def test_parses_valid_json(self):
        from analyzer import _parse_salary_response
        result = _parse_salary_response(MOCK_SALARY_RESPONSE)
        self.assertEqual(result["min"], 120000)
        self.assertEqual(result["max"], 150000)
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["confidence"], "medium")
        self.assertIsInstance(result["signals"], list)

    def test_strips_markdown_fences(self):
        from analyzer import _parse_salary_response
        raw = "```json\n" + MOCK_SALARY_RESPONSE + "\n```"
        result = _parse_salary_response(raw)
        self.assertEqual(result["min"], 120000)

    def test_swaps_reversed_range(self):
        from analyzer import _parse_salary_response
        raw = '{"min": 150000, "max": 120000, "currency": "USD", "period": "annual", "confidence": "high", "signals": []}'
        result = _parse_salary_response(raw)
        self.assertLess(result["min"], result["max"])

    def test_normalizes_unknown_confidence(self):
        from analyzer import _parse_salary_response
        raw = '{"min": 100000, "max": 130000, "currency": "USD", "period": "annual", "confidence": "very_high", "signals": []}'
        result = _parse_salary_response(raw)
        self.assertEqual(result["confidence"], "low")

    def test_raises_on_zero_values(self):
        from analyzer import _parse_salary_response
        raw = '{"min": 0, "max": 0, "currency": "USD", "period": "annual", "confidence": "low", "signals": []}'
        with self.assertRaises(ValueError):
            _parse_salary_response(raw)

    def test_raises_on_invalid_json(self):
        from analyzer import _parse_salary_response
        with self.assertRaises((ValueError, Exception)):
            _parse_salary_response("not json at all ~~~")

    def test_extracts_json_from_surrounding_text(self):
        from analyzer import _parse_salary_response
        raw = 'Here is the estimate: {"min": 100000, "max": 140000, "currency": "USD", "period": "annual", "confidence": "medium", "signals": ["senior"]}'
        result = _parse_salary_response(raw)
        self.assertEqual(result["min"], 100000)


class TestEstimateSalaryAnthropic(unittest.TestCase):
    """Tests for estimate_salary with Anthropic provider."""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test-key"})
    def test_estimate_salary_returns_structured_result(self):
        from analyzer import estimate_salary

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=MOCK_SALARY_RESPONSE)]

        with patch("analyzer.salary.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_message
            mock_cls.return_value = mock_client

            result = run(estimate_salary(
                title="Senior Python Dev",
                company="Acme",
                location="Remote",
                job_description=MOCK_JOB_NO_SALARY,
                provider="anthropic",
            ))

        self.assertEqual(result["min"], 120000)
        self.assertEqual(result["max"], 150000)
        self.assertEqual(result["llm_provider"], "anthropic")

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""})
    def test_raises_without_api_key(self):
        from analyzer import estimate_salary
        with self.assertRaises(ValueError) as ctx:
            run(estimate_salary("Dev", "Co", "NYC", MOCK_JOB_NO_SALARY, "anthropic"))
        self.assertIn("API key", str(ctx.exception))

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test-key"})
    def test_raises_if_salary_already_in_jd(self):
        from analyzer import estimate_salary
        with self.assertRaises(ValueError) as ctx:
            run(estimate_salary("Dev", "Co", "NYC", MOCK_JOB_WITH_SALARY, "anthropic"))
        self.assertIn("already contains", str(ctx.exception))


class TestEstimateSalaryOllama(unittest.TestCase):
    """Tests for estimate_salary with Ollama provider."""

    def test_estimate_salary_ollama_success(self):
        from analyzer import estimate_salary

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": MOCK_SALARY_RESPONSE}}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("analyzer.salary.httpx.AsyncClient", return_value=mock_client):
            result = run(estimate_salary(
                title="Python Dev",
                company="Co",
                location="Remote",
                job_description=MOCK_JOB_NO_SALARY,
                provider="ollama",
            ))

        self.assertEqual(result["min"], 120000)
        self.assertEqual(result["llm_provider"], "ollama")

    def test_estimate_salary_ollama_connect_error(self):
        import httpx
        from analyzer import estimate_salary

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("analyzer.salary.httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(ValueError) as ctx:
                run(estimate_salary("Dev", "Co", "NYC", MOCK_JOB_NO_SALARY, "ollama"))
        self.assertIn("Ollama", str(ctx.exception))


class TestSalaryAPIEndpoints(unittest.IsolatedAsyncioTestCase):
    """API endpoint tests for salary estimation routes."""

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

    async def _add_job(self, description=None, has_salary=False):
        """Helper — add a job directly via mock scrape."""
        if description is None:
            description = MOCK_JOB_NO_SALARY
        with patch("main.scrape_job") as mock_scrape:
            mock_scrape.return_value = {
                "title": "Python Dev", "company": "Acme", "location": "Remote",
                "raw_description": MOCK_JOB_WITH_SALARY if has_salary else description,
            }
            r = await self.client.post("/api/jobs/add", data={"url": "https://example.com/salary-test"})
        return r.json()["job_id"]

    async def test_estimate_salary_endpoint_success(self):
        jid = await self._add_job()

        with patch("main.estimate_salary", new_callable=AsyncMock) as mock_est:
            mock_est.return_value = {
                "min": 120000, "max": 150000, "currency": "USD",
                "period": "annual", "confidence": "medium",
                "signals": ["senior"], "llm_provider": "anthropic",
                "llm_model": "claude-opus-4-5",
            }
            r = await self.client.post(f"/api/jobs/{jid}/estimate-salary",
                                       data={"provider": "anthropic"})

        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["min"], 120000)
        self.assertEqual(data["max"], 150000)
        self.assertEqual(data["confidence"], "medium")

    async def test_estimate_salary_returns_cached(self):
        """Second call should return cached result without calling LLM."""
        jid = await self._add_job()

        cached = {"min": 110000, "max": 140000, "currency": "USD",
                  "period": "annual", "confidence": "high", "signals": []}

        import aiosqlite
        async with aiosqlite.connect(self.tmp.name) as db:
            await db.execute("UPDATE jobs SET salary_estimate = ? WHERE id = ?",
                             (json.dumps(cached), jid))
            await db.commit()

        with patch("main.estimate_salary", new_callable=AsyncMock) as mock_est:
            r = await self.client.post(f"/api/jobs/{jid}/estimate-salary",
                                       data={"provider": "anthropic"})
            mock_est.assert_not_called()

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["min"], 110000)

    async def test_extract_salary_from_jd_with_posted_salary(self):
        """When JD has salary info, endpoint should extract it and return 200 with source=posted."""
        jid = await self._add_job(has_salary=True)

        with patch("main.extract_salary", new_callable=AsyncMock) as mock_ext:
            mock_ext.return_value = {
                "min": 130000, "max": 160000, "currency": "USD",
                "period": "annual", "confidence": "high",
                "signals": ["Salary: $130,000 - $160,000"], "source": "posted",
                "llm_provider": "anthropic", "llm_model": "claude-opus-4-5",
            }
            r = await self.client.post(f"/api/jobs/{jid}/estimate-salary",
                                       data={"provider": "anthropic"})

        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["source"], "posted")
        self.assertEqual(data["min"], 130000)

    async def test_estimate_salary_404_for_missing_job(self):
        r = await self.client.post("/api/jobs/99999/estimate-salary",
                                   data={"provider": "anthropic"})
        self.assertEqual(r.status_code, 404)

    async def test_clear_salary_estimate(self):
        jid = await self._add_job()

        import aiosqlite
        async with aiosqlite.connect(self.tmp.name) as db:
            await db.execute("UPDATE jobs SET salary_estimate = ? WHERE id = ?",
                             (json.dumps({"min": 100000, "max": 130000}), jid))
            await db.commit()

        r = await self.client.delete(f"/api/jobs/{jid}/salary-estimate")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT salary_estimate FROM jobs WHERE id = ?", (jid,)) as cur:
                row = await cur.fetchone()
        self.assertEqual(row["salary_estimate"], "")

    async def test_estimate_salary_llm_error_returns_422(self):
        jid = await self._add_job()
        with patch("main.estimate_salary", new_callable=AsyncMock) as mock_est:
            mock_est.side_effect = ValueError("Cannot connect to Ollama")
            r = await self.client.post(f"/api/jobs/{jid}/estimate-salary",
                                       data={"provider": "ollama"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("error", r.json())


class TestSalaryDatabaseMigration(unittest.TestCase):
    """Verify salary_estimate column is added by migration."""

    def test_salary_estimate_column_exists_after_init(self):
        import aiosqlite
        from database import init_db

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        os.environ["DB_PATH"] = tmp.name

        async def ops():
            await init_db()
            async with aiosqlite.connect(tmp.name) as db:
                async with db.execute("PRAGMA table_info(jobs)") as cur:
                    cols = [row[1] for row in await cur.fetchall()]
            return cols

        try:
            cols = run(ops())
            self.assertIn("salary_estimate", cols)
        finally:
            os.unlink(tmp.name)
