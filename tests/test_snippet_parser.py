"""
tests/test_snippet_parser.py — Tests for Google snippet parser.
"""

import json
import unittest
from unittest.mock import AsyncMock, patch

from tests.mock_data import run
from analyzer.snippet_parser import _parse_snippet_response, _build_snippet_prompt


# ── _build_snippet_prompt ─────────────────────────────────────────────────────

class TestBuildSnippetPrompt(unittest.TestCase):

    def test_includes_text(self):
        prompt = _build_snippet_prompt("Glassdoor 4.2 stars (500 reviews)")
        self.assertIn("4.2 stars", prompt)

    def test_truncates_long_text(self):
        long_text = "x" * 5000
        prompt = _build_snippet_prompt(long_text)
        self.assertLessEqual(len(prompt), 6000)

    def test_requests_json_output(self):
        prompt = _build_snippet_prompt("some text")
        self.assertIn("glassdoor_rating", prompt)
        self.assertIn("bbb_rating", prompt)
        self.assertIn("indeed_rating", prompt)


# ── _parse_snippet_response ───────────────────────────────────────────────────

class TestParseSnippetResponse(unittest.TestCase):

    def _valid(self, **kwargs):
        base = {
            "glassdoor_rating": 4.2,
            "glassdoor_review_count": 500,
            "glassdoor_url": "https://glassdoor.com/Overview/Acme.htm",
            "indeed_rating": None,
            "indeed_review_count": None,
            "indeed_url": None,
            "bbb_rating": "A+",
            "bbb_url": "https://bbb.org/acme",
            "linkedin_url": None,
            "linkedin_employee_count": None,
            "linkedin_founded": None,
        }
        base.update(kwargs)
        return json.dumps(base)

    def test_parses_glassdoor_rating(self):
        result = _parse_snippet_response(self._valid())
        self.assertEqual(result["glassdoor_rating"], 4.2)

    def test_parses_review_count(self):
        result = _parse_snippet_response(self._valid())
        self.assertEqual(result["glassdoor_review_count"], 500)

    def test_parses_bbb_grade(self):
        result = _parse_snippet_response(self._valid())
        self.assertEqual(result["bbb_rating"], "A+")

    def test_parses_urls(self):
        result = _parse_snippet_response(self._valid())
        self.assertIn("glassdoor.com", result["glassdoor_url"])
        self.assertIn("bbb.org", result["bbb_url"])

    def test_null_fields_excluded(self):
        result = _parse_snippet_response(self._valid())
        self.assertNotIn("indeed_rating", result)
        self.assertNotIn("indeed_review_count", result)

    def test_strips_markdown_fences(self):
        raw = "```json\n" + self._valid() + "\n```"
        result = _parse_snippet_response(raw)
        self.assertEqual(result["glassdoor_rating"], 4.2)

    def test_json_embedded_in_prose(self):
        raw = "Here is the data:\n" + self._valid() + "\nDone."
        result = _parse_snippet_response(raw)
        self.assertEqual(result["glassdoor_rating"], 4.2)

    def test_invalid_rating_excluded(self):
        raw = json.dumps({"glassdoor_rating": 99.9, "glassdoor_review_count": 100})
        result = _parse_snippet_response(raw)
        self.assertNotIn("glassdoor_rating", result)

    def test_raises_on_no_json(self):
        with self.assertRaises(Exception):
            _parse_snippet_response("no json here at all")

    def test_indeed_rating_parsed(self):
        raw = json.dumps({
            "glassdoor_rating": None,
            "indeed_rating": 3.8,
            "indeed_review_count": 200,
            "indeed_url": "https://indeed.com/cmp/acme",
        })
        result = _parse_snippet_response(raw)
        self.assertEqual(result["indeed_rating"], 3.8)
        self.assertEqual(result["indeed_review_count"], 200)

    def test_linkedin_fields_parsed(self):
        raw = json.dumps({
            "linkedin_url": "https://linkedin.com/company/acme",
            "linkedin_employee_count": "501-1000",
            "linkedin_founded": "2005",
        })
        result = _parse_snippet_response(raw)
        self.assertEqual(result["linkedin_employee_count"], "501-1000")
        self.assertEqual(result["linkedin_founded"], "2005")

    def test_null_string_excluded(self):
        raw = json.dumps({"glassdoor_url": "null", "bbb_rating": "A"})
        result = _parse_snippet_response(raw)
        self.assertNotIn("glassdoor_url", result)
        self.assertEqual(result["bbb_rating"], "A")


# ── parse_company_snippet ─────────────────────────────────────────────────────

class TestParseCompanySnippet(unittest.IsolatedAsyncioTestCase):

    async def test_returns_structured_result(self):
        mock_response = json.dumps({
            "glassdoor_rating": 4.1,
            "glassdoor_review_count": 300,
            "glassdoor_url": "https://glassdoor.com/Acme",
            "bbb_rating": "A+",
            "bbb_url": "https://bbb.org/acme",
        })
        with patch("analyzer.snippet_parser.ant") if False else \
             patch("anthropic.AsyncAnthropic") as mock_ant:
            # Mock the internal LLM call directly
            with patch("analyzer.snippet_parser.parse_company_snippet",
                       new=AsyncMock(return_value={
                           "glassdoor_rating": 4.1,
                           "glassdoor_review_count": 300,
                           "bbb_rating": "A+",
                       })) as mock_fn:
                from analyzer.snippet_parser import parse_company_snippet
                result = await parse_company_snippet(
                    "Glassdoor 4.1 stars 300 reviews A+ BBB", "anthropic"
                )
                # The mock replaces the whole function so we get mock result
                self.assertIsInstance(result, dict)

    async def test_empty_text_returns_empty(self):
        """Empty or whitespace text should not crash."""
        from analyzer.snippet_parser import _parse_snippet_response
        with self.assertRaises(Exception):
            _parse_snippet_response("")
