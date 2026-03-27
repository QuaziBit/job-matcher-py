"""
Job Matcher — Unit Test Suite
Run: python tests.py

No API keys or network required. All external calls are mocked.
Compatible with Python 3.11+.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Shared event loop for sync test classes ───────────────────────────────────
#
# Python 3.10+ removed the implicit "current event loop" on the main thread.
# asyncio.run() creates a fresh loop per call which is fine for isolated calls,
# but breaks when a single test method needs to open a DB connection in one
# awaited call and reuse it in another — each new loop would see a different
# connection object.
#
# Fix: one persistent loop shared across all sync TestCase classes, started at
# module load and closed after the runner finishes. IsolatedAsyncioTestCase
# (used for API + E2E tests) manages its own loop independently per test
# method — that's correct and untouched.

_LOOP = asyncio.new_event_loop()


def run(coro):
    """Run an async coroutine from a sync TestCase. Uses the shared loop."""
    return _LOOP.run_until_complete(coro)


# ── Mock data ─────────────────────────────────────────────────────────────────

MOCK_RESUME_DEVSECOPS = """
Alex Johnson
Senior DevSecOps Engineer | alex@email.com | Washington D.C.

CERTIFICATIONS
- CompTIA Security+
- AWS Cloud Practitioner (CLF-C02)

SKILLS
Python, Go, Docker, Kubernetes, CI/CD, Jenkins, GitHub Actions,
AWS (EC2, S3, IAM, Secrets Manager, CloudWatch), Splunk, Terraform,
Linux, Bash, PostgreSQL, SQLite, REST APIs, OWASP, SAST/DAST tools,
React, Java, Git

EXPERIENCE
DevSecOps Engineer — Acme Federal (2022-present)
- Built CI/CD pipelines with security gates (SAST, dependency scanning)
- Managed AWS infrastructure using Terraform and CloudFormation
- Integrated Splunk dashboards for SOC alerting
- Worked with Montana DOJ and Montana DOL as government clients

Software Developer — Startup Co (2019-2022)
- Full-stack development with Python/Flask and React
- Containerized services with Docker
- Implemented AWS Secrets Manager for credential rotation
""".strip()

MOCK_RESUME_FRONTEND = """
Jordan Lee
Frontend Developer | jordan@email.com | Remote

SKILLS
HTML, CSS, JavaScript, TypeScript, React, Vue.js, Figma, Sass,
Webpack, Node.js, Jest, Cypress

EXPERIENCE
Frontend Developer — WebAgency (2021-present)
- Built responsive UIs in React and Vue
- Wrote component tests with Jest and Cypress
""".strip()

MOCK_JOB_DEVSECOPS = """
DevSecOps Engineer — ClearanceJobs Federal Contractor
Location: Arlington, VA (Hybrid)

We are looking for a DevSecOps Engineer to join our federal contracting team.

Requirements:
- 3+ years DevSecOps or related experience
- Proficiency in Python or Go
- Experience with CI/CD pipelines (Jenkins, GitHub Actions, or similar)
- AWS experience (IAM, EC2, CloudWatch, Secrets Manager)
- Security tools: SAST, DAST, Splunk, or similar SIEM
- Linux systems administration
- CompTIA Security+ or equivalent preferred
- Experience with federal government clients a plus
- Docker and Kubernetes experience
- Terraform or CloudFormation

Nice to have:
- Active Secret Clearance
- Experience with Kubernetes in production
""".strip()

MOCK_JOB_MARKETING = """
Digital Marketing Manager — BrandCo
Location: New York, NY

We need a creative marketing leader to drive brand growth.

Requirements:
- 5+ years in digital marketing
- Experience with Google Ads, Facebook Ads, HubSpot
- SEO/SEM expertise
- Content strategy and copywriting
- Analytics: Google Analytics, Mixpanel
- No technical background required
""".strip()

MOCK_HTML_INDEED = """
<!DOCTYPE html>
<html>
<head>
  <title>DevSecOps Engineer - Acme Corp - Indeed</title>
  <meta property="og:title" content="DevSecOps Engineer" />
  <meta name="job-company" content="Acme Corp" />
  <meta name="job-location" content="Arlington, VA" />
</head>
<body>
  <nav>Navigation stuff</nav>
  <div data-testid="jobDescriptionText">
    We are looking for a DevSecOps Engineer with Python, Docker,
    and AWS experience. Must have CompTIA Security+. Experience with
    CI/CD pipelines and Splunk required. Federal government client
    experience a strong plus. Terraform and Kubernetes knowledge preferred.
    This is a hybrid role based in Arlington, VA supporting federal agencies.
  </div>
  <footer>Footer stuff</footer>
</body>
</html>
""".strip()

MOCK_HTML_GENERIC = """
<!DOCTYPE html>
<html>
<head><title>Software Engineer at TechCorp</title></head>
<body>
  <main>
    <h1>Software Engineer</h1>
    <p>TechCorp is hiring a Software Engineer to join our platform team.
    You will work with Python, PostgreSQL, Docker, and REST APIs.
    Experience with cloud platforms (AWS, GCP, or Azure) is required.
    Strong communication skills and ability to work in an agile team.
    3+ years of professional software development experience needed.
    Competitive salary and full benefits package offered.</p>
  </main>
</body>
</html>
""".strip()

MOCK_HTML_NO_CONTENT = """
<!DOCTYPE html>
<html>
<head><title>Jobs</title></head>
<body><p>Hi</p></body>
</html>
""".strip()

MOCK_LLM_RESPONSE_GOOD = json.dumps({
    "score": 5,
    "matched_skills": [
        {"skill": "Python",    "match_type": "exact", "jd_snippet": "Python required",    "resume_snippet": "Python developer"},
        {"skill": "Docker",    "match_type": "exact", "jd_snippet": "Docker containers",  "resume_snippet": "Docker experience"},
        {"skill": "AWS",       "match_type": "exact", "jd_snippet": "AWS cloud",          "resume_snippet": "AWS Secrets Manager"},
        {"skill": "CI/CD",     "match_type": "exact", "jd_snippet": "CI/CD pipelines",    "resume_snippet": "Jenkins CI/CD"},
        {"skill": "Splunk",    "match_type": "exact", "jd_snippet": "Splunk SIEM",        "resume_snippet": "Splunk dashboards"},
        {"skill": "Security+", "match_type": "exact", "jd_snippet": "Security+ preferred","resume_snippet": "CompTIA Security+"},
        {"skill": "Linux",     "match_type": "exact", "jd_snippet": "Linux systems",      "resume_snippet": "Linux administration"},
    ],
    "missing_skills": [
        {"skill": "Active Secret Clearance", "severity": "blocker",
         "requirement_type": "hard", "jd_snippet": "Active Secret clearance required"},
    ],
    "reasoning": (
        "Excellent match. The candidate's DevSecOps background, Security+ cert, "
        "AWS Secrets Manager experience, and prior federal government client work "
        "align almost perfectly with this role. The only gap is an active clearance."
    ),
    "suggestions": [
        {"title": "Clarify clearance status", "detail": "If pursuing clearance, note it.",
         "job_requirement": "Active Secret clearance required"}
    ]
})

MOCK_LLM_RESPONSE_POOR = json.dumps({
    "score": 1,
    "matched_skills": [],
    "missing_skills": [
        {"skill": "Google Ads",       "severity": "blocker", "requirement_type": "hard",      "jd_snippet": "Google Ads required"},
        {"skill": "Facebook Ads",     "severity": "blocker", "requirement_type": "hard",      "jd_snippet": "Facebook Ads required"},
        {"skill": "HubSpot",          "severity": "major",   "requirement_type": "preferred", "jd_snippet": "HubSpot preferred"},
        {"skill": "SEO",              "severity": "major",   "requirement_type": "preferred", "jd_snippet": "SEO experience"},
        {"skill": "Content Strategy", "severity": "minor",   "requirement_type": "bonus",     "jd_snippet": "content strategy a plus"},
    ],
    "reasoning": (
        "Poor match. The candidate's background is entirely in software engineering "
        "and DevSecOps. This role requires digital marketing expertise with no "
        "technical overlap."
    ),
    "suggestions": []
})

MOCK_LLM_RESPONSE_WITH_FENCES = f"```json\n{MOCK_LLM_RESPONSE_GOOD}\n```"

MOCK_LLM_RESPONSE_INVALID_SCORE = json.dumps({
    "score": 9,
    "matched_skills": [],
    "missing_skills": [],
    "reasoning": "Bad score"
})

MOCK_LLM_RESPONSE_NO_JSON = "I think this is a great match! Definitely apply."


# ── Terminal colors ───────────────────────────────────────────────────────────

class Colors:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


# =============================================================================
# 1. ANALYZER TESTS
# =============================================================================

class TestParseResponse(unittest.TestCase):
    """Tests for analyzer._parse_response — pure function, no mocking needed."""

    def test_valid_json_score_5(self):
        from analyzer import _parse_response
        result = _parse_response(MOCK_LLM_RESPONSE_GOOD)
        self.assertEqual(result["score"], 5)
        skill_names = [s["skill"] if isinstance(s, dict) else s for s in result["matched_skills"]]
        self.assertIn("Python", skill_names)
        # missing_skills is now structured [{skill, severity}]
        skill_names = [s["skill"] for s in result["missing_skills"]]
        self.assertIn("Active Secret Clearance", skill_names)
        self.assertIsInstance(result["reasoning"], str)
        self.assertIn("adjusted_score", result)
        self.assertIn("penalty_breakdown", result)

    def test_valid_json_score_1(self):
        from analyzer import _parse_response
        result = _parse_response(MOCK_LLM_RESPONSE_POOR)
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["matched_skills"], [])
        skill_names = [s["skill"] for s in result["missing_skills"]]
        self.assertIn("Google Ads", skill_names)

    def test_strips_markdown_code_fences(self):
        from analyzer import _parse_response
        result = _parse_response(MOCK_LLM_RESPONSE_WITH_FENCES)
        self.assertEqual(result["score"], 5)

    def test_raises_on_invalid_score(self):
        from analyzer import _parse_response
        with self.assertRaises(ValueError) as ctx:
            _parse_response(MOCK_LLM_RESPONSE_INVALID_SCORE)
        self.assertIn("Score out of range", str(ctx.exception))

    def test_raises_on_no_json(self):
        from analyzer import _parse_response
        with self.assertRaises(ValueError) as ctx:
            _parse_response(MOCK_LLM_RESPONSE_NO_JSON)
        self.assertIn("No JSON object found", str(ctx.exception))

    def test_score_boundaries(self):
        from analyzer import _parse_response
        for valid_score in [1, 2, 3, 4, 5]:
            raw = json.dumps({
                "score": valid_score,
                "matched_skills": [],
                "missing_skills": [],
                "reasoning": "ok"
            })
            result = _parse_response(raw)
            self.assertEqual(result["score"], valid_score)

    def test_score_0_is_invalid(self):
        from analyzer import _parse_response
        raw = json.dumps({"score": 0, "matched_skills": [], "missing_skills": [], "reasoning": "ok"})
        with self.assertRaises(ValueError):
            _parse_response(raw)

    def test_score_6_is_invalid(self):
        from analyzer import _parse_response
        raw = json.dumps({"score": 6, "matched_skills": [], "missing_skills": [], "reasoning": "ok"})
        with self.assertRaises(ValueError):
            _parse_response(raw)

    def test_missing_skills_defaults_to_empty_list(self):
        from analyzer import _parse_response
        raw = json.dumps({"score": 3, "matched_skills": ["Python"], "reasoning": "ok"})
        result = _parse_response(raw)
        self.assertEqual(result["missing_skills"], [])

    def test_json_embedded_in_prose(self):
        from analyzer import _parse_response
        raw = (
            'Here is my evaluation: '
            '{"score": 4, "matched_skills": ["Docker"], "missing_skills": [], "reasoning": "Good fit."}'
            ' Hope that helps!'
        )
        result = _parse_response(raw)
        self.assertEqual(result["score"], 4)

    def test_flat_missing_skills_still_accepted(self):
        """Old flat string format should still parse without error."""
        from analyzer import _parse_response
        raw = json.dumps({
            "score": 3,
            "matched_skills": ["Python"],
            "missing_skills": ["Kubernetes", "Terraform"],
            "reasoning": "ok"
        })
        result = _parse_response(raw)
        skill_names = [s["skill"] for s in result["missing_skills"]]
        self.assertIn("Kubernetes", skill_names)


class TestPenaltyPipeline(unittest.TestCase):
    """Tests for the adjusted score penalty pipeline."""

    def test_blocker_reduces_score(self):
        from analyzer import _compute_adjusted_score
        # TS/SCI Clearance is in "security" cluster — cap is 2
        missing = [{"skill": "TS/SCI Clearance", "severity": "blocker",
                    "requirement_type": "hard", "cluster_group": "security"}]
        adjusted, breakdown = _compute_adjusted_score(4, missing)
        self.assertLess(adjusted, 4)
        self.assertEqual(breakdown["blockers"], 1)
        # In cluster pipeline, security cluster penalty goes into clusters dict
        self.assertEqual(breakdown["clusters"].get("security", 0), 2)

    def test_no_penalty_when_no_gaps(self):
        from analyzer import _compute_adjusted_score
        adjusted, breakdown = _compute_adjusted_score(5, [])
        self.assertEqual(adjusted, 5)
        self.assertEqual(breakdown["total_penalty"], 0)

    def test_adjusted_score_never_below_1(self):
        from analyzer import _compute_adjusted_score
        missing = [
            {"skill": "Clearance",    "severity": "blocker",
             "requirement_type": "hard",      "cluster_group": "security"},
            {"skill": "10 years exp", "severity": "blocker",
             "requirement_type": "hard",      "cluster_group": "other"},
            {"skill": "Kubernetes",   "severity": "major",
             "requirement_type": "preferred", "cluster_group": "devops"},
        ]
        adjusted, _ = _compute_adjusted_score(2, missing)
        self.assertGreaterEqual(adjusted, 1)

    def test_minor_gaps_small_penalty(self):
        from analyzer import _compute_adjusted_score
        # minor severity returns 0 from penalty_for_skill — no penalty
        missing = [{"skill": "Nice-to-have", "severity": "minor",
                    "requirement_type": "preferred", "cluster_group": "other"}]
        adjusted, breakdown = _compute_adjusted_score(4, missing)
        self.assertEqual(breakdown["total_penalty"], 0)
        self.assertEqual(adjusted, 4)

    def test_two_minors_give_penalty(self):
        from analyzer import _compute_adjusted_score
        # Two minors: penalty_for_skill returns 0 each — cluster cap irrelevant
        # New pipeline does not penalize minors individually — count threshold removed
        # Adjusted score stays the same
        missing = [
            {"skill": "A", "severity": "minor",
             "requirement_type": "preferred", "cluster_group": "other"},
            {"skill": "B", "severity": "minor",
             "requirement_type": "preferred", "cluster_group": "frontend"},
        ]
        adjusted, breakdown = _compute_adjusted_score(5, missing)
        self.assertEqual(breakdown["total_penalty"], 0)
        self.assertEqual(adjusted, 5)

    def test_count_penalty_above_6_gaps(self):
        from analyzer import _compute_adjusted_score
        # Count penalty was removed in cluster pipeline — each cluster is capped instead
        # 7 minor skills across different clusters: each cluster capped at 0 (minors = 0 penalty)
        missing = [{"skill": f"skill{i}", "severity": "minor",
                    "requirement_type": "preferred", "cluster_group": f"cat{i}"}
                   for i in range(7)]
        _, breakdown = _compute_adjusted_score(4, missing)
        # No count penalty in new pipeline
        self.assertEqual(breakdown["count_penalty"], 0)

    def test_bonus_requirement_zero_penalty(self):
        from analyzer import penalty_for_skill
        skill = {"skill": "Kubernetes", "severity": "major",
                 "requirement_type": "bonus"}
        self.assertEqual(penalty_for_skill(skill), 0)

    def test_hard_blocker_penalty(self):
        from analyzer import penalty_for_skill
        skill = {"skill": "Secret Clearance", "severity": "blocker",
                 "requirement_type": "hard"}
        self.assertEqual(penalty_for_skill(skill), 2)

    def test_preferred_major_penalty(self):
        from analyzer import penalty_for_skill
        skill = {"skill": "AWS", "severity": "major",
                 "requirement_type": "preferred"}
        self.assertEqual(penalty_for_skill(skill), 1)

    def test_cloud_cluster_capped(self):
        from analyzer import _compute_adjusted_score
        # 3 major cloud skills should be capped at -1 total, not -3
        missing = [
            {"skill": "AWS",    "severity": "major",
             "requirement_type": "preferred", "cluster_group": "cloud"},
            {"skill": "Lambda", "severity": "major",
             "requirement_type": "preferred", "cluster_group": "cloud"},
            {"skill": "S3",     "severity": "major",
             "requirement_type": "preferred", "cluster_group": "cloud"},
        ]
        adjusted, breakdown = _compute_adjusted_score(4, missing)
        self.assertEqual(breakdown["clusters"].get("cloud", 0), 1)
        self.assertEqual(adjusted, 3)

    def test_keyword_detector_upgrades_clearance(self):
        from analyzer import _keyword_boost
        skills = [{"skill": "Active TS/SCI Clearance", "severity": "minor",
                   "requirement_type": "preferred", "cluster_group": "security"}]
        result = _keyword_boost(skills, "Must have clearance to apply")
        self.assertEqual(result[0]["severity"], "blocker")

    def test_keyword_detector_upgrades_years(self):
        from analyzer import _keyword_boost
        skills = [{"skill": "7 years experience", "severity": "major",
                   "requirement_type": "preferred", "cluster_group": "other"}]
        result = _keyword_boost(skills, "Requires 7+ years of experience")
        self.assertEqual(result[0]["severity"], "blocker")


class TestBuildUserPrompt(unittest.TestCase):
    def test_prompt_contains_resume_and_jd(self):
        from analyzer import _build_user_prompt
        prompt = _build_user_prompt("My Resume", "Job Description Here")
        self.assertIn("My Resume", prompt)
        self.assertIn("Job Description Here", prompt)
        self.assertIn("RESUME", prompt)
        self.assertIn("JOB DESCRIPTION", prompt)


class TestAnalyzeWithAnthropic(unittest.TestCase):
    @patch("analyzer.anthropic.Anthropic")
    def test_returns_parsed_result_with_provider(self, mock_cls):
        from analyzer import analyze_with_anthropic, ANTHROPIC_MODEL
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=MOCK_LLM_RESPONSE_GOOD)]
        mock_client.messages.create.return_value = mock_msg

        result = run(analyze_with_anthropic(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))

        self.assertEqual(result["score"], 5)
        self.assertEqual(result["llm_provider"], "anthropic")
        self.assertEqual(result["llm_model"], ANTHROPIC_MODEL)
        skill_names = [s["skill"] if isinstance(s, dict) else s for s in result["matched_skills"]]
        self.assertIn("Python", skill_names)

    @patch("analyzer.anthropic.Anthropic")
    def test_calls_correct_model(self, mock_cls):
        from analyzer import ANTHROPIC_MODEL
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=MOCK_LLM_RESPONSE_GOOD)]
        mock_client.messages.create.return_value = mock_msg

        from analyzer import analyze_with_anthropic
        run(analyze_with_anthropic("resume", "job"))

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["model"], ANTHROPIC_MODEL)


class TestAnalyzeWithOllama(unittest.TestCase):
    @patch("analyzer.httpx.AsyncClient")
    def test_returns_parsed_result_with_provider(self, mock_client_cls):
        from analyzer import analyze_with_ollama, _ollama_model
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": MOCK_LLM_RESPONSE_POOR}}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = run(analyze_with_ollama(MOCK_RESUME_DEVSECOPS, MOCK_JOB_MARKETING))

        self.assertEqual(result["score"], 1)
        self.assertEqual(result["llm_provider"], "ollama")
        self.assertEqual(result["llm_model"], _ollama_model())

    @patch("analyzer.httpx.AsyncClient")
    def test_ollama_connection_error(self, mock_client_cls):
        import httpx
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from analyzer import analyze_with_ollama
        with self.assertRaises(ValueError) as ctx:
            run(analyze_with_ollama("resume", "job"))
        self.assertIn("Cannot connect to Ollama", str(ctx.exception))


class TestAnalyzeMatchDispatch(unittest.TestCase):
    @patch("analyzer._call_anthropic_once", new_callable=AsyncMock)
    @patch("analyzer._call_ollama_once", new_callable=AsyncMock)
    def test_routes_to_anthropic_by_default(self, mock_ollama, mock_anthropic):
        mock_anthropic.return_value = {
            "score": 4, "adjusted_score": 4, "llm_provider": "anthropic",
            "llm_model": "claude-opus-4-5",
            "matched_skills": [{"skill": "Python", "match_type": "exact",
                                 "jd_snippet": "Python required",
                                 "resume_snippet": "Python dev", "category": "backend"}],
            "missing_skills": [], "reasoning": "Good match",
            "penalty_breakdown": {}, "suggestions": [],
        }
        from analyzer import analyze_match
        run(analyze_match("resume", "job"))
        mock_anthropic.assert_called_once()
        mock_ollama.assert_not_called()

    @patch("analyzer._call_anthropic_once", new_callable=AsyncMock)
    @patch("analyzer._call_ollama_once", new_callable=AsyncMock)
    def test_routes_to_ollama_when_specified(self, mock_ollama, mock_anthropic):
        mock_ollama.return_value = {
            "score": 3, "adjusted_score": 3, "llm_provider": "ollama",
            "llm_model": "llama3.1",
            "matched_skills": [{"skill": "Docker", "match_type": "exact",
                                 "jd_snippet": "Docker required",
                                 "resume_snippet": "Used Docker", "category": "devops"}],
            "missing_skills": [], "reasoning": "Partial match",
            "penalty_breakdown": {}, "suggestions": [],
        }
        from analyzer import analyze_match
        run(analyze_match("resume", "job", provider="ollama"))
        mock_ollama.assert_called_once()
        mock_anthropic.assert_not_called()


# =============================================================================
# 2. SCRAPER TESTS
# =============================================================================

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


# =============================================================================
# 3. DATABASE TESTS
# =============================================================================

class TestDatabase(unittest.TestCase):
    """
    Integration tests against a real temporary SQLite DB.

    Each test method wraps all async operations into a single coroutine and
    calls run() exactly once — this avoids any cross-loop connection issues.
    """

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        os.environ["DB_PATH"] = self.tmp.name

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_init_creates_all_tables(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ) as cur:
                    return {row[0] for row in await cur.fetchall()}

        tables = run(ops())
        self.assertIn("resumes", tables)
        self.assertIn("jobs", tables)
        self.assertIn("analyses", tables)
        self.assertIn("applications", tables)

    def test_init_is_idempotent(self):
        from database import init_db

        async def ops():
            await init_db()
            await init_db()   # Second call must not raise

        run(ops())

    def test_resume_insert_and_fetch(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                db.row_factory = aiosqlite.Row
                await db.execute(
                    "INSERT INTO resumes (label, content) VALUES (?, ?)",
                    ("DevSecOps v1", MOCK_RESUME_DEVSECOPS)
                )
                await db.commit()
                async with db.execute("SELECT * FROM resumes") as cur:
                    return [dict(r) for r in await cur.fetchall()]

        rows = run(ops())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "DevSecOps v1")
        self.assertIn("Security+", rows[0]["content"])

    def test_job_insert_and_fetch(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                db.row_factory = aiosqlite.Row
                await db.execute(
                    "INSERT INTO jobs (url, title, company, location, raw_description) "
                    "VALUES (?,?,?,?,?)",
                    ("https://example.com/job/1", "DevSecOps Engineer",
                     "Acme Federal", "Arlington, VA", MOCK_JOB_DEVSECOPS)
                )
                await db.commit()
                async with db.execute("SELECT * FROM jobs") as cur:
                    return [dict(r) for r in await cur.fetchall()]

        rows = run(ops())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "DevSecOps Engineer")
        self.assertEqual(rows[0]["company"], "Acme Federal")

    def test_duplicate_job_url_raises(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                await db.execute(
                    "INSERT INTO jobs (url, title, raw_description) VALUES (?,?,?)",
                    ("https://example.com/job/1", "Job A", "description")
                )
                await db.commit()
                # Same URL — UNIQUE constraint must fire
                await db.execute(
                    "INSERT INTO jobs (url, title, raw_description) VALUES (?,?,?)",
                    ("https://example.com/job/1", "Job B", "description")
                )
                await db.commit()

        with self.assertRaises(Exception):
            run(ops())

    def test_analysis_insert_and_cascade_delete(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA foreign_keys = ON")

                async with db.execute(
                    "INSERT INTO resumes (label, content) VALUES (?,?)",
                    ("v1", "resume text")
                ) as cur:
                    resume_id = cur.lastrowid

                async with db.execute(
                    "INSERT INTO jobs (url, raw_description) VALUES (?,?)",
                    ("https://example.com/job/1", "job desc")
                ) as cur:
                    job_id = cur.lastrowid

                await db.execute(
                    "INSERT INTO analyses "
                    "(job_id, resume_id, score, matched_skills, missing_skills, reasoning) "
                    "VALUES (?,?,?,?,?,?)",
                    (job_id, resume_id, 4, '["Python"]', '["K8s"]', "Good match")
                )
                await db.commit()

                # Delete job — analysis row must cascade-delete
                await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                await db.commit()

                async with db.execute(
                    "SELECT * FROM analyses WHERE job_id = ?", (job_id,)
                ) as cur:
                    return await cur.fetchall()

        remaining = run(ops())
        self.assertEqual(len(remaining), 0)

    def test_application_upsert(self):
        import aiosqlite
        from database import init_db

        async def ops():
            await init_db()
            async with aiosqlite.connect(self.tmp.name) as db:
                db.row_factory = aiosqlite.Row

                async with db.execute(
                    "INSERT INTO jobs (url, raw_description) VALUES (?,?)",
                    ("https://example.com/job/1", "desc")
                ) as cur:
                    job_id = cur.lastrowid

                await db.execute(
                    "INSERT INTO applications (job_id, status, recruiter_name) VALUES (?,?,?)",
                    (job_id, "applied", "Jane Smith")
                )
                await db.commit()

                # Upsert — update status only
                await db.execute(
                    """INSERT INTO applications (job_id, status, recruiter_name)
                       VALUES (?,?,?)
                       ON CONFLICT(job_id) DO UPDATE SET status=excluded.status""",
                    (job_id, "interviewing", "Jane Smith")
                )
                await db.commit()

                async with db.execute(
                    "SELECT * FROM applications WHERE job_id = ?", (job_id,)
                ) as cur:
                    return [dict(r) for r in await cur.fetchall()]

        rows = run(ops())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "interviewing")


# =============================================================================
# 4. API ENDPOINT TESTS
# =============================================================================

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

        # Add resume + job directly
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

        # Add an analysis
        with patch("main.analyze_match") as mock_analyze:
            mock_analyze.return_value = {
                "score": 3, "adjusted_score": 3,
                "penalty_breakdown": {"blockers": 0, "majors": 0, "minors": 0, "blocker_penalty": 0, "major_penalty": 0, "minor_penalty": 0, "count_penalty": 0, "total_penalty": 0},
                "matched_skills": ["Python"], "missing_skills": [],
                "reasoning": "Ok.", "llm_provider": "anthropic", "llm_model": "claude-opus-4-5",
            }
            await self.client.post(f"/api/jobs/{jid}/analyze", data={
                "resume_id": rid, "provider": "anthropic"
            })

        # Get the analysis id
        async with aiosqlite.connect(self.tmp.name) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT id FROM analyses WHERE job_id=?", (jid,)) as cur:
                analysis_id = (await cur.fetchone())["id"]

        # Delete it
        resp = await self.client.delete(f"/api/analyses/{analysis_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

        # Job should still exist
        job_resp = await self.client.get(f"/job/{jid}")
        self.assertEqual(job_resp.status_code, 200)

        # Analysis should be gone
        async with aiosqlite.connect(self.tmp.name) as db:
            async with db.execute("SELECT id FROM analyses WHERE id=?", (analysis_id,)) as cur:
                self.assertIsNone(await cur.fetchone())

    async def test_delete_analysis_404_for_missing(self):
        resp = await self.client.delete("/api/analyses/9999")
        self.assertEqual(resp.status_code, 404)

    # ── Resume endpoints ──────────────────────────────────────────────────────

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

    # ── Manual paste endpoint ─────────────────────────────────────────────────

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
        """Manual job should be analyzable just like a scraped one."""
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
                "score": 4,
                "adjusted_score": 2,
                "penalty_breakdown": {"blockers": 1, "majors": 0, "minors": 0, "blocker_penalty": 2, "major_penalty": 0, "minor_penalty": 0, "count_penalty": 0, "total_penalty": 2},
                "matched_skills": ["Python", "Docker"],
                "missing_skills": [{"skill": "Clearance", "severity": "blocker"}],
                "reasoning": "Good match.",
                "llm_provider": "anthropic",
                "llm_model": "claude-opus-4-5",
            }
            resp = await self.client.post(f"/api/jobs/{jid}/analyze", data={
                "resume_id": rid, "provider": "anthropic"
            })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["score"], 4)

    async def test_analyze_stores_llm_model(self):
        """llm_model should be persisted to the analyses table."""
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
                "score": 4,
                "adjusted_score": 4,
                "penalty_breakdown": {"blockers": 0, "majors": 0, "minors": 0, "blocker_penalty": 0, "major_penalty": 0, "minor_penalty": 0, "count_penalty": 0, "total_penalty": 0},
                "matched_skills": ["Python"],
                "missing_skills": [],
                "reasoning": "Good.",
                "llm_provider": "ollama",
                "llm_model": "llama3.1:8b",
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

    # ── Job endpoints ─────────────────────────────────────────────────────────

    @patch("main.scrape_job")
    async def test_add_job_success(self, mock_scrape):
        mock_scrape.return_value = {
            "title": "DevSecOps Engineer",
            "company": "Acme Federal",
            "location": "Arlington, VA",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        resp = await self.client.post("/api/jobs/add", data={
            "url": "https://example.com/job/1"
        })
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

    # ── Analysis endpoint ─────────────────────────────────────────────────────

    @patch("main.scrape_job")
    @patch("main.analyze_match")
    async def test_analyze_job(self, mock_analyze, mock_scrape):
        mock_scrape.return_value = {
            "title": "DevSecOps Engineer", "company": "Acme", "location": "VA",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        mock_analyze.return_value = {
            "score": 5,
            "adjusted_score": 5,
            "penalty_breakdown": {"blockers": 0, "majors": 0, "minors": 0, "blocker_penalty": 0, "major_penalty": 0, "minor_penalty": 0, "count_penalty": 0, "total_penalty": 0},
            "matched_skills": ["Python", "Docker", "AWS"],
            "missing_skills": [{"skill": "Secret Clearance", "severity": "blocker"}],
            "reasoning": "Great match.",
            "llm_provider": "anthropic",
            "llm_model": "claude-opus-4-5",
        }

        r = await self.client.post("/api/resumes/add", data={
            "label": "DevSecOps v1", "content": MOCK_RESUME_DEVSECOPS
        })
        rid = r.json()["resume_id"]

        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/analyze"})
        jid = j.json()["job_id"]

        resp = await self.client.post(f"/api/jobs/{jid}/analyze", data={
            "resume_id": rid,
            "provider": "anthropic"
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
        # Insert a job directly — uses the same temp DB the app reads from
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

    # ── Application endpoint ──────────────────────────────────────────────────

    @patch("main.scrape_job")
    async def test_save_and_update_application(self, mock_scrape):
        mock_scrape.return_value = {
            "title": "Job", "company": "Co", "location": "DC",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        j = await self.client.post("/api/jobs/add", data={"url": "https://example.com/job/app"})
        jid = j.json()["job_id"]

        resp = await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "applied",
            "recruiter_name": "Jane Smith",
            "recruiter_email": "jane@company.com",
            "recruiter_phone": "555-1234",
            "notes": "Great conversation on LinkedIn"
        })
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

        resp2 = await self.client.post(f"/api/jobs/{jid}/application", data={
            "status": "interviewing",
            "recruiter_name": "Jane Smith",
            "recruiter_email": "jane@company.com",
            "recruiter_phone": "555-1234",
            "notes": "Phone screen scheduled"
        })
        self.assertEqual(resp2.status_code, 200)

    # ── Page routes ───────────────────────────────────────────────────────────

    async def test_index_page_renders(self):
        resp = await self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Job Tracker", resp.content)

    async def test_resumes_page_renders(self):
        resp = await self.client.get("/resumes")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Resume Versions", resp.content)

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


# =============================================================================
# 5. END-TO-END SMOKE TEST
# =============================================================================

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
            "title": "DevSecOps Engineer",
            "company": "Acme Federal",
            "location": "Arlington, VA",
            "raw_description": MOCK_JOB_DEVSECOPS
        }
        mock_analyze.return_value = {
            "score": 5,
            "adjusted_score": 3,
            "penalty_breakdown": {"blockers": 1, "majors": 0, "minors": 0, "blocker_penalty": 2, "major_penalty": 0, "minor_penalty": 0, "count_penalty": 0, "total_penalty": 2},
            "matched_skills": ["Python", "Docker", "AWS", "Security+", "Splunk"],
            "missing_skills": [{"skill": "Active Secret Clearance", "severity": "blocker"}],
            "reasoning": "Excellent match for this DevSecOps federal role.",
            "llm_provider": "anthropic",
            "llm_model": "claude-opus-4-5",
        }

        # Step 1: add resume
        r = await self.client.post("/api/resumes/add", data={
            "label": "DevSecOps v1",
            "content": MOCK_RESUME_DEVSECOPS
        })
        self.assertEqual(r.status_code, 200)
        resume_id = r.json()["resume_id"]

        # Step 2: add job
        j = await self.client.post("/api/jobs/add", data={
            "url": "https://clearancejobs.com/jobs/devsecops-engineer"
        })
        self.assertEqual(j.status_code, 200)
        job_id = j.json()["job_id"]
        self.assertEqual(j.json()["title"], "DevSecOps Engineer")

        # Step 3: analyze
        a = await self.client.post(f"/api/jobs/{job_id}/analyze", data={
            "resume_id": resume_id,
            "provider": "anthropic"
        })
        self.assertEqual(a.status_code, 200)
        self.assertEqual(a.json()["score"], 5)
        self.assertIn("Python", a.json()["matched_skills"])

        # Step 4: save application
        app_resp = await self.client.post(f"/api/jobs/{job_id}/application", data={
            "status": "applied",
            "recruiter_name": "Sumpter",
            "recruiter_email": "sumpter@acme.gov",
            "recruiter_phone": "571-555-0100",
            "notes": "Responded to LinkedIn outreach. Strong interest in the role."
        })
        self.assertEqual(app_resp.status_code, 200)

        # Step 5: verify job detail page renders
        page = await self.client.get(f"/job/{job_id}")
        self.assertEqual(page.status_code, 200)
        content = page.content.decode()
        self.assertIn("DevSecOps Engineer", content)
        self.assertIn("Acme Federal", content)

        # Step 6: verify DB state directly
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


# =============================================================================
# Custom test runner
# =============================================================================

class VerboseResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)
        self.stream.write(f"  {Colors.GREEN}✓{Colors.RESET} {test._testMethodName}\n")
        self.stream.flush()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.stream.write(f"  {Colors.RED}✗{Colors.RESET} {test._testMethodName}\n")
        self.stream.flush()

    def addError(self, test, err):
        super().addError(test, err)
        self.stream.write(f"  {Colors.YELLOW}!{Colors.RESET} {test._testMethodName} (ERROR)\n")
        self.stream.flush()


class VerboseRunner(unittest.TextTestRunner):
    resultclass = VerboseResult

    def run(self, test):
        suites = list(test)
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}  Job Matcher - Unit Test Suite  (Python {sys.version[:6]}){Colors.RESET}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")

        groups = {}
        for suite in suites:
            for case in suite:
                cls = type(case).__name__
                groups.setdefault(cls, []).append(case)

        result = self._makeResult()
        result.failfast = self.failfast

        for cls_name, cases in groups.items():
            print(f"{Colors.BOLD}{cls_name}{Colors.RESET}")
            for case in cases:
                case.run(result)
            print()

        total  = result.testsRun
        passed = len(result.successes)
        failed = len(result.failures)
        errors = len(result.errors)

        print(f"{Colors.CYAN}{'-' * 60}{Colors.RESET}")
        status = Colors.GREEN if (failed + errors) == 0 else Colors.RED
        print(
            f"{status}{Colors.BOLD}  {passed}/{total} passed{Colors.RESET}  "
            f"| {Colors.RED}{failed} failed{Colors.RESET} "
            f"| {Colors.YELLOW}{errors} errors{Colors.RESET}\n"
        )

        if result.failures:
            print(f"{Colors.RED}{Colors.BOLD}FAILURES:{Colors.RESET}")
            for t, tb in result.failures:
                print(f"\n  {Colors.RED}{t}{Colors.RESET}")
                last = [ln for ln in tb.strip().splitlines() if ln.strip()][-1]
                print(f"  {last}")

        if result.errors:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}ERRORS:{Colors.RESET}")
            for t, tb in result.errors:
                print(f"\n  {Colors.YELLOW}{t}{Colors.RESET}")
                last = [ln for ln in tb.strip().splitlines() if ln.strip()][-1]
                print(f"  {last}")

        return result


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = VerboseRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)
    _LOOP.close()
    sys.exit(0 if result.wasSuccessful() else 1)
