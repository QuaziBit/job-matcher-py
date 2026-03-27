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

# ── Job text quality assessment ───────────────────────────────────────────────

TECH_KEYWORDS = [
    "python", "javascript", "typescript", "go", "java", "react", "angular",
    "vue", "node", "django", "flask", "fastapi", "spring", "docker",
    "kubernetes", "aws", "azure", "gcp", "sql", "postgresql", "mongodb",
    "redis", "terraform", "ci/cd", "jenkins", "git", "linux", "api",
    "rest", "graphql", "microservices", "llm", "machine learning",
]

BUZZWORDS = [
    "synergy", "leverage", "paradigm", "holistic", "proactive",
    "dynamic", "innovative", "passionate", "rockstar", "ninja",
    "guru", "wizard", "thought leader", "disruptive", "agile mindset",
]


def assess_job_text_quality(text: str) -> dict:
    """
    Run deterministic quality checks on a job description.
    Returns dict with level ("ok"|"warn"|"poor"), issues, char_count,
    tech_keywords, and buzzword_count.
    """
    issues = []
    char_count = len(text)
    words = text.lower().split()

    # Too short
    if char_count < 300:
        issues.append(f"Description too short ({char_count} chars) — analysis may be unreliable")

    # Non-ASCII noise
    if char_count > 0:
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / char_count > 0.15:
            issues.append("High non-ASCII content — possible scraping noise")

    # Bullet heavy
    bullet_count = text.count("•") + text.count("·") + text.count("-\n")
    if bullet_count > 60:
        issues.append("Bullet-heavy posting — limited context for inference")

    # Buzzword density
    buzzword_count = sum(1 for w in words if w in BUZZWORDS)
    if words and buzzword_count / len(words) > 0.15:
        issues.append("High buzzword density — requirements may be vague")

    # Mixed seniority signals
    text_lower = text.lower()
    junior_signals = "junior" in text_lower or "entry level" in text_lower or "entry-level" in text_lower
    senior_signals = "senior" in text_lower or "5+ years" in text_lower or "7+ years" in text_lower
    if junior_signals and senior_signals:
        issues.append("Mixed seniority signals — job level is unclear")

    # Tech keyword count
    tech_keywords = sum(1 for kw in TECH_KEYWORDS if kw in text_lower)
    if char_count > 500 and tech_keywords == 0:
        issues.append("No recognized tech keywords — job requirements may be too vague")

    level = "ok"
    if len(issues) >= 2 or char_count < 150:
        level = "poor"
    elif len(issues) >= 1:
        level = "warn"

    return {
        "level":          level,
        "issues":         issues,
        "char_count":     char_count,
        "tech_keywords":  tech_keywords,
        "buzzword_count": buzzword_count,
    }
