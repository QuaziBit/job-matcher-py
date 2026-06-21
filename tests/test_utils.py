"""
tests/test_utils.py — Unit tests for utils.py shared helpers.
Covers: URL validation, description truncation, resume comparison helpers.
"""

import unittest


class TestIsValidUrl(unittest.TestCase):
    """Tests for utils.is_valid_url — used across main.py to validate
    company_url, glassdoor_url, indeed_url, bbb_url, linkedin_url, and
    job source URLs before saving."""

    def test_https_url_is_valid(self):
        from utils import is_valid_url
        self.assertTrue(is_valid_url("https://example.com"))

    def test_http_url_is_valid(self):
        from utils import is_valid_url
        self.assertTrue(is_valid_url("http://example.com"))

    def test_empty_string_is_invalid(self):
        from utils import is_valid_url
        self.assertFalse(is_valid_url(""))

    def test_ftp_scheme_is_invalid(self):
        from utils import is_valid_url
        self.assertFalse(is_valid_url("ftp://example.com"))

    def test_bare_domain_without_scheme_is_invalid(self):
        from utils import is_valid_url
        self.assertFalse(is_valid_url("example.com"))

    def test_scheme_lookalike_is_invalid(self):
        # Guards against a looser `startswith("http")` check accepting
        # something like "httpfoo://evil.com" — must match the exact
        # "http://" or "https://" prefix.
        from utils import is_valid_url
        self.assertFalse(is_valid_url("httpfoo://evil.com"))

    def test_manual_scheme_is_invalid(self):
        from utils import is_valid_url
        self.assertFalse(is_valid_url("manual://abc123"))

    def test_whitespace_only_is_invalid(self):
        from utils import is_valid_url
        self.assertFalse(is_valid_url("   "))


class TestErrorResponse(unittest.TestCase):
    """Tests for utils.error_response — the shared {"error": ...} JSONResponse
    builder used throughout main.py route handlers (86 call sites)."""

    def test_default_status_code(self):
        from utils import error_response
        resp = error_response("Something went wrong")
        self.assertEqual(resp.status_code, 400)

    def test_explicit_status_code(self):
        from utils import error_response
        resp = error_response("Database error", 500)
        self.assertEqual(resp.status_code, 500)

    def test_body_contains_error_message(self):
        import json
        from utils import error_response
        resp = error_response("Database error", 500)
        body = json.loads(resp.body)
        self.assertEqual(body, {"error": "Database error"})

    def test_extra_kwargs_included_in_body(self):
        import json
        from utils import error_response
        resp = error_response("This URL has already been added.", 409, job_id=42)
        body = json.loads(resp.body)
        self.assertEqual(body, {"error": "This URL has already been added.", "job_id": 42})

    def test_extra_kwargs_do_not_override_error_key(self):
        # "error" must always come from the message argument, even if a
        # caller accidentally passes error= as a kwarg.
        import json
        from utils import error_response
        resp = error_response("Real error message", 422, job_id=1)
        body = json.loads(resp.body)
        self.assertEqual(body["error"], "Real error message")
        self.assertEqual(body["job_id"], 1)

    def test_returns_jsonresponse_instance(self):
        from fastapi.responses import JSONResponse
        from utils import error_response
        resp = error_response("oops", 422)
        self.assertIsInstance(resp, JSONResponse)


class TestTruncateDescription(unittest.TestCase):
    """Tests for utils.truncate_description — used before sending job
    descriptions to the LLM analysis pipeline."""

    def test_short_text_unchanged(self):
        from utils import truncate_description
        text = "short job description"
        self.assertEqual(truncate_description(text), text)

    def test_text_at_exact_limit_unchanged(self):
        from utils import truncate_description
        text = "x" * 8000
        self.assertEqual(truncate_description(text), text)

    def test_long_text_truncated_with_notice(self):
        from utils import truncate_description
        text = "x" * 9000
        result = truncate_description(text)
        self.assertTrue(result.startswith("x" * 8000))
        self.assertIn("[...truncated for analysis]", result)

    def test_truncated_length_is_capped(self):
        from utils import truncate_description
        text = "x" * 50000
        result = truncate_description(text)
        # Result should be max_chars plus the notice text, not the full input
        self.assertLess(len(result), len(text))

    def test_custom_max_chars(self):
        from utils import truncate_description
        text = "x" * 100
        result = truncate_description(text, max_chars=50)
        self.assertTrue(result.startswith("x" * 50))
        self.assertIn("[...truncated for analysis]", result)

    def test_empty_string(self):
        from utils import truncate_description
        self.assertEqual(truncate_description(""), "")


class TestHasBlocker(unittest.TestCase):
    def test_detects_blocker_severity(self):
        from utils import has_blocker
        skills = [{"skill": "Clearance", "severity": "blocker"}]
        self.assertTrue(has_blocker(skills))

    def test_no_blocker_returns_false(self):
        from utils import has_blocker
        skills = [{"skill": "AWS", "severity": "minor"}]
        self.assertFalse(has_blocker(skills))

    def test_empty_list_returns_false(self):
        from utils import has_blocker
        self.assertFalse(has_blocker([]))

    def test_ignores_non_dict_entries(self):
        from utils import has_blocker
        skills = ["not a dict", {"skill": "AWS", "severity": "major"}]
        self.assertFalse(has_blocker(skills))


class TestDetermineBetterFit(unittest.TestCase):
    def test_prefers_resume_without_blocker(self):
        from utils import determine_better_fit
        a = {"resume_label": "v1", "adjusted_score": 3,
             "missing_skills": [{"severity": "blocker"}]}
        b = {"resume_label": "v2", "adjusted_score": 2,
             "missing_skills": []}
        better, reason = determine_better_fit(a, b)
        self.assertEqual(better, "v2")
        self.assertIn("blockers", reason)

    def test_prefers_higher_adjusted_score_when_no_blockers(self):
        from utils import determine_better_fit
        a = {"resume_label": "v1", "adjusted_score": 4, "missing_skills": []}
        b = {"resume_label": "v2", "adjusted_score": 2, "missing_skills": []}
        better, reason = determine_better_fit(a, b)
        self.assertEqual(better, "v1")
        self.assertIn("Higher adjusted score", reason)

    def test_tie_when_scores_equal(self):
        from utils import determine_better_fit
        a = {"resume_label": "v1", "adjusted_score": 3, "missing_skills": []}
        b = {"resume_label": "v2", "adjusted_score": 3, "missing_skills": []}
        better, _ = determine_better_fit(a, b)
        self.assertEqual(better, "Tie")


class TestBuildComparison(unittest.TestCase):
    def test_returns_none_with_fewer_than_two_resumes(self):
        from utils import build_comparison
        analyses = [{"resume_id": 1, "resume_label": "v1",
                     "adjusted_score": 3, "missing_skills": []}]
        self.assertIsNone(build_comparison(analyses))

    def test_returns_none_with_empty_list(self):
        from utils import build_comparison
        self.assertIsNone(build_comparison([]))

    def test_builds_comparison_for_two_distinct_resumes(self):
        from utils import build_comparison
        analyses = [
            {"resume_id": 1, "resume_label": "v1", "adjusted_score": 4, "missing_skills": []},
            {"resume_id": 2, "resume_label": "v2", "adjusted_score": 2, "missing_skills": []},
        ]
        result = build_comparison(analyses)
        self.assertIsNotNone(result)
        self.assertEqual(result["better_fit"], "v1")

    def test_uses_only_first_two_distinct_resume_ids(self):
        from utils import build_comparison
        analyses = [
            {"resume_id": 1, "resume_label": "v1", "adjusted_score": 4, "missing_skills": []},
            {"resume_id": 1, "resume_label": "v1", "adjusted_score": 1, "missing_skills": []},
            {"resume_id": 2, "resume_label": "v2", "adjusted_score": 2, "missing_skills": []},
        ]
        result = build_comparison(analyses)
        self.assertIsNotNone(result)
        # Should use the first analysis seen for resume_id=1 (score 4), not the second (score 1)
        self.assertEqual(result["resume_a"]["adjusted_score"], 4)


if __name__ == "__main__":
    unittest.main()
