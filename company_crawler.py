"""
company_crawler.py — Best-effort web crawlers for company vetting.

Each crawler is independent: returns a partial dict on success, empty dict
on any failure. Results are merged by the caller.

Sources: Glassdoor, Indeed, LinkedIn, Crunchbase, Trustpilot, G2, BBB.
"""

import logging
import re
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TIMEOUT = 15


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=TIMEOUT)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _first_text(soup: BeautifulSoup, *selectors: str) -> str:
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el.get_text(strip=True)
    return ""


def _first_attr(soup: BeautifulSoup, attr: str, *selectors: str) -> str:
    for sel in selectors:
        el = soup.select_one(sel)
        if el and el.get(attr):
            return el[attr]
    return ""


# ── Glassdoor ─────────────────────────────────────────────────────────────────

async def crawl_glassdoor(company_name: str) -> dict:
    """Scrape Glassdoor search for top company result — URL and rating."""
    url = f"https://www.glassdoor.com/Search/results.htm?keyword={quote_plus(company_name)}"
    try:
        async with _client() as client:
            r = await client.get(url)
            r.raise_for_status()
        s = _soup(r.text)

        # Company card links
        link_el = s.select_one("a[data-test='cell-Employment-Overview-company-name']") or \
                  s.select_one("a[href*='/Overview/Working-at']") or \
                  s.select_one(".company-tile a")
        company_url = ""
        if link_el and link_el.get("href"):
            href = link_el["href"]
            company_url = href if href.startswith("http") else f"https://www.glassdoor.com{href}"

        # Rating
        rating_el = s.select_one("[data-test='rating']") or \
                    s.select_one(".rating") or \
                    s.select_one("[class*='ratingNumber']")
        rating = None
        if rating_el:
            try:
                rating = float(re.search(r"[\d.]+", rating_el.get_text()).group())
            except Exception:
                pass

        # Review count
        review_count = None
        rc_el = s.select_one("[data-test='reviewsCount']") or \
                s.select_one("[class*='reviewCount']")
        if rc_el:
            m = re.search(r"[\d,]+", rc_el.get_text())
            if m:
                try:
                    review_count = int(m.group().replace(",", ""))
                except Exception:
                    pass

        result = {}
        if company_url:
            result["glassdoor_url"] = company_url
        if rating is not None:
            result["glassdoor_rating"] = rating
        if review_count is not None:
            result["glassdoor_review_count"] = review_count
        return result

    except Exception as e:
        logger.warning(f"crawl_glassdoor({company_name!r}) failed: {e}")
        return {}


# ── LinkedIn ──────────────────────────────────────────────────────────────────

async def crawl_linkedin(company_name: str) -> dict:
    """Scrape LinkedIn public company page for employee count and founding year."""
    slug = re.sub(r"[^a-z0-9]+", "-", company_name.lower()).strip("-")
    url = f"https://www.linkedin.com/company/{quote_plus(slug)}"
    try:
        async with _client() as client:
            r = await client.get(url)
            r.raise_for_status()
        s = _soup(r.text)

        # Employee count from meta description or structured data
        employee_count = ""
        for el in s.find_all("meta"):
            content = el.get("content", "")
            m = re.search(r"([\d,]+)\s+employee", content, re.I)
            if m:
                employee_count = m.group(1).replace(",", "")
                break

        # Founding year
        founded = ""
        for el in s.find_all(string=re.compile(r"Founded", re.I)):
            m = re.search(r"\b(19|20)\d{2}\b", el)
            if m:
                founded = m.group()
                break

        result: dict = {"linkedin_url": url}
        if employee_count:
            result["linkedin_employee_count"] = employee_count
        if founded:
            result["linkedin_founded"] = founded
        return result

    except Exception as e:
        logger.warning(f"crawl_linkedin({company_name!r}) failed: {e}")
        return {}


# ── BBB ───────────────────────────────────────────────────────────────────────

async def crawl_bbb(company_name: str) -> dict:
    """Scrape BBB search for top result — URL and letter grade."""
    url = f"https://www.bbb.org/search?find_text={quote_plus(company_name)}"
    try:
        async with _client() as client:
            r = await client.get(url)
            r.raise_for_status()
        s = _soup(r.text)

        link_el = s.select_one("a[href*='/profile/']") or \
                  s.select_one(".result-item a")
        company_url = ""
        if link_el:
            href = link_el["href"]
            company_url = href if href.startswith("http") else f"https://www.bbb.org{href}"

        # Letter grade
        grade = ""
        grade_el = s.select_one(".dtm-rating") or \
                   s.select_one("[class*='grade']") or \
                   s.select_one("[class*='rating-letter']")
        if grade_el:
            text = grade_el.get_text(strip=True)
            m = re.search(r"[A-F][+-]?", text)
            if m:
                grade = m.group()

        result = {}
        if company_url:
            result["bbb_url"] = company_url
        if grade:
            result["bbb_rating"] = grade
        return result

    except Exception as e:
        logger.warning(f"crawl_bbb({company_name!r}) failed: {e}")
        return {}


# ── Orchestrator ──────────────────────────────────────────────────────────────

import asyncio

CRAWLERS = [
    crawl_glassdoor,
    crawl_linkedin,
    crawl_bbb,
]

_CRAWLER_NAMES = [
    "crawl_glassdoor",
    "crawl_linkedin",
    "crawl_bbb",
]


async def crawl_company(company_name: str) -> dict:
    """
    Run all crawlers concurrently and merge results.
    Always returns a dict — partial on partial failure, empty on total failure.
    Looks up crawler functions at call time so test patches are respected.
    """
    import company_crawler as _mod
    fns = [getattr(_mod, name) for name in _CRAWLER_NAMES]
    results = await asyncio.gather(*[fn(company_name) for fn in fns], return_exceptions=True)
    merged: dict = {}
    for r in results:
        if isinstance(r, dict):
            merged.update(r)
    return merged
