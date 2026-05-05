"""
tests/test_company_crawler.py — Unit tests for company_crawler module.
All HTTP calls are mocked — no real network access.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.mock_data import run


def _mock_client(html: str, status: int = 200):
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.status_code = status
    mock_resp.url = "https://example.com/result"
    mock_resp.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ── crawl_glassdoor ───────────────────────────────────────────────────────────

GLASSDOOR_HTML = """
<html><body>
  <a data-test="cell-Employment-Overview-company-name" href="/Overview/Working-at-Acme-EI_IE123.htm">Acme</a>
  <span data-test="rating">4.2</span>
  <span data-test="reviewsCount">1,234 reviews</span>
</body></html>
"""

GLASSDOOR_EMPTY = "<html><body><p>No results found.</p></body></html>"


class TestCrawlGlassdoor(unittest.TestCase):
    @patch("company_crawler._client")
    def test_returns_url_rating_review_count(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client(GLASSDOOR_HTML)
        from company_crawler import crawl_glassdoor
        result = run(crawl_glassdoor("Acme"))
        self.assertIn("glassdoor_url", result)
        self.assertIn("glassdoor.com", result["glassdoor_url"])
        self.assertEqual(result.get("glassdoor_rating"), 4.2)
        self.assertEqual(result.get("glassdoor_review_count"), 1234)

    @patch("company_crawler._client")
    def test_returns_empty_on_no_results(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client(GLASSDOOR_EMPTY)
        from company_crawler import crawl_glassdoor
        result = run(crawl_glassdoor("UnknownCorp"))
        self.assertNotIn("glassdoor_rating", result)

    @patch("company_crawler._client")
    def test_returns_empty_dict_on_exception(self, mock_client_fn):
        mock_client_fn.side_effect = Exception("network down")
        from company_crawler import crawl_glassdoor
        result = run(crawl_glassdoor("Acme"))
        self.assertEqual(result, {})


# ── crawl_bbb ─────────────────────────────────────────────────────────────────

BBB_HTML = """
<html><body>
  <a href="/profile/acme-corp-123">Acme Corp</a>
  <span class="dtm-rating">A+</span>
</body></html>
"""

BBB_EMPTY = "<html><body><p>No results.</p></body></html>"


class TestCrawlBBB(unittest.TestCase):
    @patch("company_crawler._client")
    def test_returns_url_and_grade(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client(BBB_HTML)
        from company_crawler import crawl_bbb
        result = run(crawl_bbb("Acme"))
        self.assertIn("bbb_url", result)
        self.assertIn("bbb.org", result["bbb_url"])
        self.assertEqual(result.get("bbb_rating"), "A+")

    @patch("company_crawler._client")
    def test_returns_empty_on_no_results(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client(BBB_EMPTY)
        from company_crawler import crawl_bbb
        result = run(crawl_bbb("UnknownCorp"))
        self.assertNotIn("bbb_rating", result)

    @patch("company_crawler._client")
    def test_returns_empty_dict_on_exception(self, mock_client_fn):
        mock_client_fn.side_effect = Exception("timeout")
        from company_crawler import crawl_bbb
        result = run(crawl_bbb("Acme"))
        self.assertEqual(result, {})


# ── crawl_linkedin ────────────────────────────────────────────────────────────

LINKEDIN_HTML = """
<html><head>
  <meta name="description" content="Acme Corp | 1,200 employees on LinkedIn">
</head><body>
  <span>Founded 2010</span>
</body></html>
"""


class TestCrawlLinkedIn(unittest.TestCase):
    @patch("company_crawler._client")
    def test_returns_url_employee_count_founded(self, mock_client_fn):
        mock_client_fn.return_value = _mock_client(LINKEDIN_HTML)
        from company_crawler import crawl_linkedin
        result = run(crawl_linkedin("Acme"))
        self.assertIn("linkedin_url", result)
        self.assertEqual(result.get("linkedin_employee_count"), "1200")
        self.assertEqual(result.get("linkedin_founded"), "2010")

    @patch("company_crawler._client")
    def test_returns_empty_dict_on_exception(self, mock_client_fn):
        mock_client_fn.side_effect = Exception("blocked")
        from company_crawler import crawl_linkedin
        result = run(crawl_linkedin("Acme"))
        self.assertEqual(result, {})


# ── crawl_company (orchestrator) ──────────────────────────────────────────────

class TestCrawlCompany(unittest.TestCase):
    @patch("company_crawler.crawl_glassdoor", new_callable=AsyncMock)
    @patch("company_crawler.crawl_linkedin",  new_callable=AsyncMock)
    @patch("company_crawler.crawl_bbb",       new_callable=AsyncMock)
    def test_merges_all_results(self, mock_bbb, mock_li, mock_gd):
        mock_gd.return_value  = {"glassdoor_url": "https://glassdoor.com/acme", "glassdoor_rating": 4.2}
        mock_li.return_value  = {"linkedin_url": "https://linkedin.com/company/acme"}
        mock_bbb.return_value = {"bbb_url": "https://bbb.org/acme", "bbb_rating": "A+"}

        from company_crawler import crawl_company
        result = run(crawl_company("Acme"))

        self.assertIn("glassdoor_url", result)
        self.assertIn("linkedin_url", result)
        self.assertIn("bbb_url", result)
        self.assertEqual(result["glassdoor_rating"], 4.2)
        self.assertEqual(result["bbb_rating"], "A+")

    @patch("company_crawler.crawl_glassdoor", new_callable=AsyncMock)
    @patch("company_crawler.crawl_linkedin",  new_callable=AsyncMock)
    @patch("company_crawler.crawl_bbb",       new_callable=AsyncMock)
    def test_partial_results_on_some_failures(self, mock_bbb, mock_li, mock_gd):
        mock_gd.return_value  = {"glassdoor_url": "https://glassdoor.com/acme"}
        mock_li.return_value  = {}
        mock_bbb.return_value = {"bbb_rating": "B"}

        from company_crawler import crawl_company
        result = run(crawl_company("Acme"))

        self.assertIn("glassdoor_url", result)
        self.assertIn("bbb_rating", result)
        self.assertNotIn("linkedin_employee_count", result)

    @patch("company_crawler.crawl_glassdoor", new_callable=AsyncMock)
    @patch("company_crawler.crawl_linkedin",  new_callable=AsyncMock)
    @patch("company_crawler.crawl_bbb",       new_callable=AsyncMock)
    def test_returns_empty_dict_when_all_fail(self, mock_bbb, mock_li, mock_gd):
        for m in [mock_gd, mock_li, mock_bbb]:
            m.return_value = {}

        from company_crawler import crawl_company
        result = run(crawl_company("Ghost Corp"))
        self.assertEqual(result, {})
