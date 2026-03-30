"""
tests/test_scraper.py — Unit tests for scraper module.
Covers: text cleaning, meta extraction, job scraping.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.mock_data import (
    run,
    MOCK_HTML_INDEED,
    MOCK_HTML_GENERIC,
    MOCK_HTML_NO_CONTENT,
    MOCK_JOB_DEVSECOPS,
)


class TestCleanText(unittest.TestCase):
    def test_collapses_blank_lines(self):
        from scraper import _clean_text
        result = _clean_text("line1\n\n\n\nline2")
        self.assertNotIn("\n\n\n", result)
        self.assertIn("line1", result)
        self.assertIn("line2", result)

    def test_collapses_extra_spaces(self):
        from scraper import _clean_text
        result = _clean_text("word1    word2\t\tword3")
        self.assertNotIn("  ", result)

    def test_strips_leading_trailing(self):
        from scraper import _clean_text
        result = _clean_text("  \n  hello  \n  ")
        self.assertEqual(result, "hello")


class TestExtractMeta(unittest.TestCase):
    def _soup(self, html):
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "html.parser")

    def test_og_title(self):
        from scraper import _extract_meta
        html = '<html><head><meta property="og:title" content="DevSecOps Engineer"/></head></html>'
        meta = _extract_meta(self._soup(html))
        self.assertEqual(meta["title"], "DevSecOps Engineer")

    def test_falls_back_to_title_tag(self):
        from scraper import _extract_meta
        html = "<html><head><title>My Job Title</title></head></html>"
        meta = _extract_meta(self._soup(html))
        self.assertEqual(meta["title"], "My Job Title")

    def test_falls_back_to_h1(self):
        from scraper import _extract_meta
        html = "<html><body><h1>Software Engineer</h1></body></html>"
        meta = _extract_meta(self._soup(html))
        self.assertEqual(meta["title"], "Software Engineer")

    def test_company_from_meta(self):
        from scraper import _extract_meta
        html = '<html><head><meta name="job-company" content="Acme Corp"/></head></html>'
        meta = _extract_meta(self._soup(html))
        self.assertEqual(meta["company"], "Acme Corp")

    def test_location_from_meta(self):
        from scraper import _extract_meta
        html = '<html><head><meta name="job-location" content="Arlington, VA"/></head></html>'
        meta = _extract_meta(self._soup(html))
        self.assertEqual(meta["location"], "Arlington, VA")

    def test_title_truncated_at_200(self):
        from scraper import _extract_meta
        html = f'<html><head><title>{"x" * 300}</title></head></html>'
        meta = _extract_meta(self._soup(html))
        self.assertLessEqual(len(meta["title"]), 200)

    def test_empty_page_returns_empty_strings(self):
        from scraper import _extract_meta
        meta = _extract_meta(self._soup("<html></html>"))
        self.assertEqual(meta["title"], "")
        self.assertEqual(meta["company"], "")
        self.assertEqual(meta["location"], "")


class TestScrapeJob(unittest.TestCase):
    def _mock_response(self, html, status=200):
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.status_code = status
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @patch("scraper.httpx.AsyncClient")
    def test_scrapes_indeed_selector(self, mock_client_cls):
        mock_resp = self._mock_response(MOCK_HTML_INDEED)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from scraper import scrape_job
        result = run(scrape_job("https://indeed.com/viewjob?jk=abc123"))

        self.assertEqual(result["title"], "DevSecOps Engineer")
        self.assertEqual(result["company"], "Acme Corp")
        self.assertIn("Python", result["raw_description"])
        self.assertIn("Docker", result["raw_description"])
        self.assertNotIn("Navigation stuff", result["raw_description"])
        self.assertNotIn("Footer stuff", result["raw_description"])

    @patch("scraper.httpx.AsyncClient")
    def test_scrapes_generic_main_tag(self, mock_client_cls):
        mock_resp = self._mock_response(MOCK_HTML_GENERIC)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from scraper import scrape_job
        result = run(scrape_job("https://techcorp.com/jobs/123"))

        self.assertIn("Python", result["raw_description"])
        self.assertIn("PostgreSQL", result["raw_description"])

    @patch("scraper.httpx.AsyncClient")
    def test_raises_on_too_short_content(self, mock_client_cls):
        mock_resp = self._mock_response(MOCK_HTML_NO_CONTENT)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from scraper import scrape_job
        with self.assertRaises(ValueError) as ctx:
            run(scrape_job("https://example.com/job"))
        self.assertIn("Could not extract job description", str(ctx.exception))

    @patch("scraper.httpx.AsyncClient")
    def test_truncates_long_descriptions(self, mock_client_cls):
        long_html = f"<html><body><main>{'word ' * 3000}</main></body></html>"
        mock_resp = self._mock_response(long_html)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from scraper import scrape_job
        result = run(scrape_job("https://example.com/job"))
        self.assertLessEqual(len(result["raw_description"]), 8100)
        self.assertIn("truncated", result["raw_description"])

    @patch("scraper.httpx.AsyncClient")
    def test_raises_on_http_error(self, mock_client_cls):
        import httpx
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_resp
        )
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from scraper import scrape_job
        with self.assertRaises(ValueError) as ctx:
            run(scrape_job("https://example.com/job"))
        self.assertIn("HTTP 404", str(ctx.exception))

    @patch("scraper.httpx.AsyncClient")
    def test_raises_on_network_error(self, mock_client_cls):
        import httpx
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("timeout"))
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from scraper import scrape_job
        with self.assertRaises(ValueError) as ctx:
            run(scrape_job("https://example.com/job"))
        self.assertIn("Network error", str(ctx.exception))
