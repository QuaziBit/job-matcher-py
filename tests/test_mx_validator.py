"""
tests/test_mx_validator.py — Tests for email domain MX validation.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from asyncio import run

# ── Helpers ───────────────────────────────────────────────────────────────────

class TestExtractDomain(unittest.TestCase):
    def test_valid_email(self):
        from mx_validator import extract_domain
        self.assertEqual(extract_domain("alice@example.com"), "example.com")

    def test_uppercase_normalized(self):
        from mx_validator import extract_domain
        self.assertEqual(extract_domain("Alice@EXAMPLE.COM"), "example.com")

    def test_no_at_sign_returns_none(self):
        from mx_validator import extract_domain
        self.assertIsNone(extract_domain("notanemail"))

    def test_empty_string_returns_none(self):
        from mx_validator import extract_domain
        self.assertIsNone(extract_domain(""))

    def test_none_like_returns_none(self):
        from mx_validator import extract_domain
        self.assertIsNone(extract_domain(None))

    def test_domain_no_dot_returns_none(self):
        from mx_validator import extract_domain
        self.assertIsNone(extract_domain("user@localhost"))

    def test_subdomains_preserved(self):
        from mx_validator import extract_domain
        self.assertEqual(extract_domain("hr@mail.company.co.uk"), "mail.company.co.uk")

    def test_whitespace_stripped(self):
        from mx_validator import extract_domain
        self.assertEqual(extract_domain("  user@example.com  "), "example.com")


# ── nslookup parsing ──────────────────────────────────────────────────────────

class TestNslookupMX(unittest.TestCase):
    def test_parses_mx_record(self):
        from mx_validator import _nslookup_mx
        # Linux format
        output = "mail exchanger = 10 mx1.example.com\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=output, stderr="")
            result = _nslookup_mx("example.com")
        self.assertIn("mx1.example.com", result)

    def test_parses_mx_record_windows_format(self):
        from mx_validator import _nslookup_mx
        # Windows format: "domain  MX preference = 10, mail exchanger = mx1.example.com"
        output = "example.com\tMX preference = 10, mail exchanger = mx1.example.com\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=output, stderr="")
            result = _nslookup_mx("example.com")
        self.assertIn("mx1.example.com", result)

    def test_parses_multiple_mx_records(self):
        from mx_validator import _nslookup_mx
        output = (
            "mail exchanger = 10 mx1.example.com\n"
            "mail exchanger = 20 mx2.example.com\n"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=output, stderr="")
            result = _nslookup_mx("example.com")
        self.assertEqual(len(result), 2)

    def test_empty_output_returns_empty_list(self):
        from mx_validator import _nslookup_mx
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="")
            result = _nslookup_mx("nodomain.invalid")
        self.assertEqual(result, [])

    def test_trailing_dot_stripped(self):
        from mx_validator import _nslookup_mx
        output = "mail exchanger = 10 mx1.example.com.\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=output, stderr="")
            result = _nslookup_mx("example.com")
        self.assertEqual(result[0], "mx1.example.com")

    def test_nslookup_not_found_returns_empty(self):
        from mx_validator import _nslookup_mx
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _nslookup_mx("example.com")
        self.assertEqual(result, [])

    def test_timeout_returns_empty(self):
        import subprocess
        from mx_validator import _nslookup_mx
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nslookup", 5)):
            result = _nslookup_mx("example.com")
        self.assertEqual(result, [])


# ── validate_email_domain ─────────────────────────────────────────────────────

def _make_db_mock(cached_row=None):
    """
    Return a mock aiosqlite connection that handles both:
      - async with db.execute(...) as cur:   (used in get_cached_mx)
      - await db.execute(...)                (used in upsert_mx_cache)
    _FakeCtx is both awaitable (returns itself) and an async context manager.
    """
    cur = MagicMock()
    cur.fetchone = AsyncMock(return_value=cached_row)

    class _FakeCtx:
        """Supports both `await db.execute(...)` and `async with db.execute(...) as c`."""
        def __await__(self_):
            async def _noop():
                return self_
            return _noop().__await__()

        async def __aenter__(self_):
            return cur

        async def __aexit__(self_, *a):
            return False

    db = MagicMock()
    db.execute = MagicMock(return_value=_FakeCtx())
    db.commit = AsyncMock()
    return db


class TestValidateEmailDomain(unittest.TestCase):
    def test_invalid_email_returns_error(self):
        from mx_validator import validate_email_domain
        db = _make_db_mock()
        result = run(validate_email_domain("notanemail", db))
        self.assertFalse(result["valid"])
        self.assertIsNotNone(result["error"])

    def test_cached_hit_has_mx(self):
        from mx_validator import validate_email_domain
        # Simulate a cached row: (domain, has_mx=1, mx_records, checked_at)
        fake_row = {"domain": "example.com", "has_mx": 1, "mx_records": "mx1.example.com", "checked_at": "2099-01-01"}
        db = _make_db_mock(cached_row=fake_row)
        result = run(validate_email_domain("user@example.com", db))
        self.assertTrue(result["has_mx"])
        self.assertTrue(result["cached"])
        self.assertEqual(result["domain"], "example.com")

    def test_cached_hit_no_mx(self):
        from mx_validator import validate_email_domain
        fake_row = {"domain": "fake.xyz", "has_mx": 0, "mx_records": "", "checked_at": "2099-01-01"}
        db = _make_db_mock(cached_row=fake_row)
        result = run(validate_email_domain("hr@fake.xyz", db))
        self.assertFalse(result["has_mx"])
        self.assertFalse(result["valid"])
        self.assertTrue(result["cached"])

    def test_live_lookup_with_mx(self):
        from mx_validator import validate_email_domain
        db = _make_db_mock(cached_row=None)
        with patch("mx_validator.lookup_mx", new=AsyncMock(return_value=["mx1.example.com"])):
            result = run(validate_email_domain("user@example.com", db))
        self.assertTrue(result["has_mx"])
        self.assertTrue(result["valid"])
        self.assertFalse(result["cached"])
        self.assertIn("mx1.example.com", result["mx_records"])

    def test_live_lookup_no_mx(self):
        from mx_validator import validate_email_domain
        db = _make_db_mock(cached_row=None)
        with patch("mx_validator.lookup_mx", new=AsyncMock(return_value=[])):
            result = run(validate_email_domain("hr@nodomain.invalid", db))
        self.assertFalse(result["has_mx"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["mx_records"], [])

    def test_result_always_has_required_keys(self):
        from mx_validator import validate_email_domain
        db = _make_db_mock(cached_row=None)
        with patch("mx_validator.lookup_mx", new=AsyncMock(return_value=[])):
            result = run(validate_email_domain("x@example.com", db))
        for key in ("email", "domain", "valid", "has_mx", "mx_records", "cached", "error"):
            self.assertIn(key, result)


# ── API endpoint ──────────────────────────────────────────────────────────────

class TestValidateDomainEndpoint(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import tempfile, os
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        os.environ["DB_PATH"] = self._tmp.name
        from database import init_db
        await init_db()
        from httpx import AsyncClient, ASGITransport
        from main import app
        self._client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await self._client.aclose()
        import os
        try:
            os.unlink(self._tmp.name)
        except Exception:
            pass

    async def test_missing_email_returns_422(self):
        resp = await self._client.post("/api/email/validate-domain", data={})
        self.assertEqual(resp.status_code, 422)

    async def test_valid_domain_with_mx(self):
        with patch("main.validate_email_domain", new=AsyncMock(return_value={
            "email": "hr@example.com", "domain": "example.com",
            "valid": True, "has_mx": True,
            "mx_records": ["mx1.example.com"], "cached": False, "error": None,
        })):
            resp = await self._client.post(
                "/api/email/validate-domain", data={"email": "hr@example.com"}
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["has_mx"])
        self.assertEqual(data["domain"], "example.com")

    async def test_no_mx_domain(self):
        with patch("main.validate_email_domain", new=AsyncMock(return_value={
            "email": "hr@fake.xyz", "domain": "fake.xyz",
            "valid": False, "has_mx": False,
            "mx_records": [], "cached": False, "error": None,
        })):
            resp = await self._client.post(
                "/api/email/validate-domain", data={"email": "hr@fake.xyz"}
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["has_mx"])
        self.assertFalse(data["valid"])


if __name__ == "__main__":
    unittest.main()
