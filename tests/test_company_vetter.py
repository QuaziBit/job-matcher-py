"""
tests/test_company_vetter.py — Tests for LLM company vetting.
"""

import json
import unittest
from unittest.mock import AsyncMock, patch
from asyncio import run

from analyzer.company_vetter import (
    build_company_prompt,
    _parse_vetting_response,
    vet_company,
    RISK_LEVELS,
    CACHE_TTL_DAYS,
)


# ── build_company_prompt ──────────────────────────────────────────────────────

class TestBuildCompanyPrompt(unittest.TestCase):

    def test_includes_company_name(self):
        prompt = build_company_prompt("Acme Corp", {})
        self.assertIn("Acme Corp", prompt)

    def test_includes_bbb_rating(self):
        prompt = build_company_prompt("Co", {"bbb_rating": "A+"})
        self.assertIn("A+", prompt)

    def test_bbb_no_listing_when_empty(self):
        prompt = build_company_prompt("Co", {})
        self.assertIn("no listing", prompt.lower())

    def test_includes_glassdoor_rating(self):
        prompt = build_company_prompt("Co", {"glassdoor_rating": 4.2, "glassdoor_review_count": 150})
        self.assertIn("4.2", prompt)
        self.assertIn("150", prompt)

    def test_includes_linkedin_employees(self):
        prompt = build_company_prompt("Co", {"linkedin_employee_count": "501-1000", "linkedin_founded": "2005"})
        self.assertIn("501-1000", prompt)
        self.assertIn("2005", prompt)

    def test_requests_json_response(self):
        prompt = build_company_prompt("Co", {})
        self.assertIn("risk_level", prompt)
        self.assertIn("assessment", prompt)

    def test_risk_levels_documented_in_prompt(self):
        prompt = build_company_prompt("Co", {})
        for level in ("low", "medium", "high", "unknown"):
            self.assertIn(level, prompt)

    def test_no_pii_in_prompt(self):
        # Prompt should never contain email addresses or personal names
        prompt = build_company_prompt("Acme Corp", {
            "bbb_rating": "A",
            "glassdoor_rating": 4.0,
            "linkedin_employee_count": "51-200",
        })
        self.assertNotIn("@", prompt)


# ── _parse_vetting_response ───────────────────────────────────────────────────

class TestParseVettingResponse(unittest.TestCase):

    def _valid_json(self, risk="low", assessment="Looks good.", signals=None):
        return json.dumps({
            "risk_level": risk,
            "assessment": assessment,
            "signals": signals or ["Has BBB listing", "Good Glassdoor rating"],
        })

    def test_parses_valid_json(self):
        result = _parse_vetting_response(self._valid_json(), "Acme")
        self.assertEqual(result["risk_level"], "low")
        self.assertEqual(result["assessment"], "Looks good.")
        self.assertEqual(len(result["signals"]), 2)

    def test_strips_markdown_fences(self):
        raw = "```json\n" + self._valid_json() + "\n```"
        result = _parse_vetting_response(raw, "Acme")
        self.assertEqual(result["risk_level"], "low")

    def test_invalid_risk_level_becomes_unknown(self):
        raw = json.dumps({"risk_level": "terrible", "assessment": "Bad.", "signals": []})
        result = _parse_vetting_response(raw, "Acme")
        self.assertEqual(result["risk_level"], "unknown")

    def test_all_risk_levels_accepted(self):
        for level in RISK_LEVELS:
            raw = json.dumps({"risk_level": level, "assessment": "ok", "signals": []})
            result = _parse_vetting_response(raw, "Acme")
            self.assertEqual(result["risk_level"], level)

    def test_missing_assessment_gets_default(self):
        raw = json.dumps({"risk_level": "low", "signals": []})
        result = _parse_vetting_response(raw, "Acme")
        self.assertIsInstance(result["assessment"], str)
        self.assertTrue(len(result["assessment"]) > 0)

    def test_signals_defaults_to_empty_list(self):
        raw = json.dumps({"risk_level": "low", "assessment": "Fine."})
        result = _parse_vetting_response(raw, "Acme")
        self.assertEqual(result["signals"], [])

    def test_raises_on_no_json(self):
        with self.assertRaises(Exception):
            _parse_vetting_response("no json here at all", "Acme")

    def test_json_embedded_in_prose(self):
        raw = 'Here is my analysis:\n{"risk_level":"medium","assessment":"Mixed signals.","signals":["a"]}\nEnd.'
        result = _parse_vetting_response(raw, "Acme")
        self.assertEqual(result["risk_level"], "medium")

    def test_company_name_in_result(self):
        result = _parse_vetting_response(self._valid_json(), "Acme Corp")
        self.assertEqual(result["company"], "Acme Corp")


# ── vet_company ───────────────────────────────────────────────────────────────

class TestVetCompany(unittest.IsolatedAsyncioTestCase):

    async def test_returns_structured_result(self):
        mock_response = json.dumps({
            "risk_level": "low",
            "assessment": "Established company with good ratings.",
            "signals": ["A+ BBB rating", "4.5 Glassdoor"],
        })
        with patch("analyzer.company_vetter._call_vetting_llm",
                   new=AsyncMock(return_value=mock_response)):
            result = await vet_company("Acme Corp", {"bbb_rating": "A+"}, "anthropic")
        self.assertEqual(result["risk_level"], "low")
        self.assertEqual(result["provider"], "anthropic")
        self.assertIn("signals", result)

    async def test_provider_stored_in_result(self):
        mock_response = json.dumps({"risk_level": "medium", "assessment": "ok", "signals": []})
        with patch("analyzer.company_vetter._call_vetting_llm",
                   new=AsyncMock(return_value=mock_response)):
            result = await vet_company("Co", {}, "openai", "gpt-4o-mini")
        self.assertEqual(result["provider"], "openai")
        self.assertEqual(result["model"], "gpt-4o-mini")

    async def test_works_with_empty_meta(self):
        mock_response = json.dumps({"risk_level": "unknown", "assessment": "No data.", "signals": []})
        with patch("analyzer.company_vetter._call_vetting_llm",
                   new=AsyncMock(return_value=mock_response)):
            result = await vet_company("Ghost Corp", {}, "anthropic")
        self.assertEqual(result["risk_level"], "unknown")

    async def test_all_providers_route_correctly(self):
        mock_response = json.dumps({"risk_level": "low", "assessment": "ok", "signals": []})
        for provider in ("anthropic", "openai", "gemini", "ollama"):
            with patch("analyzer.company_vetter._call_vetting_llm",
                       new=AsyncMock(return_value=mock_response)) as mock:
                await vet_company("Co", {}, provider)
                mock.assert_called_once()
                call_args = mock.call_args
                self.assertEqual(call_args[0][1], provider)


# ── constants ─────────────────────────────────────────────────────────────────

class TestVettingConstants(unittest.TestCase):

    def test_cache_ttl_is_7_days(self):
        self.assertEqual(CACHE_TTL_DAYS, 7)

    def test_risk_levels_complete(self):
        for level in ("low", "medium", "high", "unknown"):
            self.assertIn(level, RISK_LEVELS)


if __name__ == "__main__":
    unittest.main()
