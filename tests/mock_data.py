"""
tests/mock_data.py — Shared mock data, fixtures, and helpers for all test files.
"""

import asyncio
import json
import sys
import unittest


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


# ── Resume mock data ──────────────────────────────────────────────────────────

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


# ── Job mock data ─────────────────────────────────────────────────────────────

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

MOCK_JOB_NO_SALARY = """
Senior Python Developer — Remote

We are looking for an experienced Python developer to join our team.
Requirements:
- 5+ years Python experience
- FastAPI or Django
- PostgreSQL
- Docker and Kubernetes
- REST API design
""".strip()

MOCK_JOB_WITH_SALARY = """
Senior Python Developer — Remote
Salary: $130,000 – $160,000 per year

We are looking for an experienced Python developer to join our team.
Requirements:
- 5+ years Python experience
- FastAPI or Django
""".strip()


# ── HTML mock data ────────────────────────────────────────────────────────────

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


# ── LLM response mock data ────────────────────────────────────────────────────

MOCK_LLM_RESPONSE_GOOD = json.dumps({
    "score": 5,
    "matched_skills": [
        {"skill": "Python",    "match_type": "exact", "jd_snippet": "Python required",     "resume_snippet": "Python developer"},
        {"skill": "Docker",    "match_type": "exact", "jd_snippet": "Docker containers",   "resume_snippet": "Docker experience"},
        {"skill": "AWS",       "match_type": "exact", "jd_snippet": "AWS cloud",           "resume_snippet": "AWS Secrets Manager"},
        {"skill": "CI/CD",     "match_type": "exact", "jd_snippet": "CI/CD pipelines",     "resume_snippet": "Jenkins CI/CD"},
        {"skill": "Splunk",    "match_type": "exact", "jd_snippet": "Splunk SIEM",         "resume_snippet": "Splunk dashboards"},
        {"skill": "Security+", "match_type": "exact", "jd_snippet": "Security+ preferred", "resume_snippet": "CompTIA Security+"},
        {"skill": "Linux",     "match_type": "exact", "jd_snippet": "Linux systems",       "resume_snippet": "Linux administration"},
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

MOCK_LLM_RESPONSE_WITH_FENCES    = f"```json\n{MOCK_LLM_RESPONSE_GOOD}\n```"
MOCK_LLM_RESPONSE_INVALID_SCORE  = json.dumps({"score": 9, "matched_skills": [], "missing_skills": [], "reasoning": "Bad score"})
MOCK_LLM_RESPONSE_NO_JSON        = "I think this is a great match! Definitely apply."
MOCK_SALARY_RESPONSE             = '{"min": 120000, "max": 150000, "currency": "USD", "period": "annual", "confidence": "medium", "signals": ["senior level", "Python", "remote"]}'


# ── Terminal colors ───────────────────────────────────────────────────────────

class Colors:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


# ── Custom test runner ────────────────────────────────────────────────────────

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
