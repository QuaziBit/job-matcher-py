import httpx
from bs4 import BeautifulSoup
import re


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Selectors to try for main job content (ordered by specificity)
CONTENT_SELECTORS = [
    "[data-testid='jobDescriptionText']",   # Indeed
    ".job-description",
    ".description__text",                   # LinkedIn
    "#job-details",
    ".jobsearch-jobDescriptionText",
    "[class*='jobDescription']",
    "[class*='job-description']",
    "[class*='description']",
    "main",
    "article",
]

# Tags that are mostly noise
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "iframe", "form"]


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract_meta(soup: BeautifulSoup) -> dict:
    """Best-effort extraction of title, company, location from meta tags."""
    title = ""
    company = ""
    location = ""

    # Open Graph / meta
    og_title = soup.find("meta", property="og:title")
    if og_title:
        title = og_title.get("content", "")

    # LinkedIn structured
    for tag in soup.find_all("meta"):
        name = tag.get("name", "").lower()
        content = tag.get("content", "")
        if "title" in name and not title:
            title = content
        if "company" in name or "employer" in name:
            company = content
        if "location" in name or "city" in name:
            location = content

    # Fallback to <title>
    if not title and soup.title:
        title = soup.title.string or ""

    # Try h1 as job title if still empty
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    return {"title": title[:200], "company": company[:200], "location": location[:200]}


async def scrape_job(url: str) -> dict:
    """
    Fetches a job posting URL and extracts the job description text plus metadata.
    Returns: {title, company, location, raw_description}
    Raises: ValueError if scraping fails or content too short.
    """
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=20) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ValueError(f"HTTP {e.response.status_code} fetching URL: {url}")
        except httpx.RequestError as e:
            raise ValueError(f"Network error fetching URL: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    meta = _extract_meta(soup)

    # Try targeted selectors first
    description = ""
    for selector in CONTENT_SELECTORS:
        el = soup.select_one(selector)
        if el:
            description = _clean_text(el.get_text(separator="\n"))
            if len(description) > 200:
                break

    # Fallback to body
    if len(description) < 200:
        body = soup.find("body")
        if body:
            description = _clean_text(body.get_text(separator="\n"))

    if len(description) < 100:
        raise ValueError(
            "Could not extract job description. "
            "The page may require JavaScript or a login. "
            "Try copying the job description text manually."
        )

    # Cap at ~8000 chars to keep LLM tokens reasonable
    if len(description) > 8000:
        description = description[:8000] + "\n\n[...truncated for analysis]"

    return {
        "title": meta["title"],
        "company": meta["company"],
        "location": meta["location"],
        "raw_description": description,
    }
