"""
tests/test_gemini.py — Unit tests for Google Gemini provider integration.
Covers: config helpers, LLM caller, salary caller, analyze_match routing,
        health checks, launcher form parsing, and .env persistence.
All external API calls are mocked — no real keys or network required.
"""

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.mock_data import (
    run,
    MOCK_RESUME_DEVSECOPS,
    MOCK_JOB_DEVSECOPS,
    MOCK_LLM_RESPONSE_GOOD,
    MOCK_SALARY_RESPONSE,
    MOCK_JOB_NO_SALARY,
    MOCK_JOB_WITH_SALARY,
)


# ── Config helpers ────────────────────────────────────────────────────────────

class TestGeminiConfig(unittest.TestCase):
    """Tests for gemini_model() env helper and CLOUD_PROVIDERS."""

    def test_gemini_model_default(self):
        from analyzer.config import gemini_model, GEMINI_MODEL
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_MODEL", None)
            self.assertEqual(gemini_model(), GEMINI_MODEL)

    def test_gemini_model_from_env(self):
        from analyzer.config import gemini_model
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-flash"}):
            self.assertEqual(gemini_model(), "gemini-2.5-flash")

    def test_cloud_providers_includes_gemini(self):
        from analyzer.config import CLOUD_PROVIDERS
        self.assertIn("gemini", CLOUD_PROVIDERS)

    def test_get_model_for_provider_gemini(self):
        from analyzer.llm import _get_model_for_provider
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-flash"}):
            self.assertEqual(_get_model_for_provider("gemini"), "gemini-2.5-flash")


# ── call_gemini_once ──────────────────────────────────────────────────────────

class TestCallGeminiOnce(unittest.TestCase):
    """Tests for call_gemini_once — mocks google.genai SDK."""

    def _make_gemini_response(self, text: str):
        resp = MagicMock()
        resp.text = text
        resp.usage_metadata = MagicMock(prompt_token_count=100, candidates_token_count=50)
        return resp

    def _make_genai_mock(self, text: str):
        """Build a mock for the google.genai SDK.
        Pattern: client = genai.Client(api_key=...)
                 response = await client.aio.models.generate_content(...)
        """
        mock_resp = self._make_gemini_response(text)
        mock_aio_models = MagicMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_resp)
        mock_aio = MagicMock()
        mock_aio.models = mock_aio_models
        mock_client = MagicMock()
        mock_client.aio = mock_aio
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_genai.types = MagicMock()
        return mock_genai, mock_client

    @patch.dict(os.environ, {"GEMINI_API_KEY": "AIza-test-key", "GEMINI_MODEL": "gemini-2.5-flash"})
    def test_returns_valid_result(self):
        from analyzer.llm import call_gemini_once
        mock_genai, _ = self._make_genai_mock(MOCK_LLM_RESPONSE_GOOD)
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_genai.types}):
            result = run(call_gemini_once(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertEqual(result["llm_provider"], "gemini")
        self.assertEqual(result["llm_model"], "gemini-2.5-flash")
        self.assertIn("score", result)
        self.assertIn("matched_skills", result)

    @patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_MODEL": "gemini-2.5-flash"})
    def test_raises_on_missing_key(self):
        from analyzer.llm import call_gemini_once
        with self.assertRaises(ValueError) as ctx:
            run(call_gemini_once(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertIn("Gemini API key", str(ctx.exception))

    @patch.dict(os.environ, {"GEMINI_API_KEY": "AIza-test-key"})
    def test_uses_env_model(self):
        from analyzer.llm import call_gemini_once
        mock_genai, _ = self._make_genai_mock(MOCK_LLM_RESPONSE_GOOD)
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-pro"}):
            with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_genai.types}):
                result = run(call_gemini_once(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertEqual(result["llm_model"], "gemini-2.5-pro")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "AIza-test-key", "GEMINI_MODEL": "gemini-2.5-flash"})
    def test_result_has_analysis_mode(self):
        from analyzer.llm import call_gemini_once
        mock_genai, _ = self._make_genai_mock(MOCK_LLM_RESPONSE_GOOD)
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_genai.types}):
            result = run(call_gemini_once(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertIn("analysis_mode", result)


# ── analyze_match routing (Gemini) ────────────────────────────────────────────

class TestAnalyzeMatchGemini(unittest.TestCase):
    """Tests for analyze_match() routing to Gemini."""

    def _good_result(self, provider, model):
        from analyzer.parsers import parse_response
        from analyzer.config import get_mode_config
        cfg = get_mode_config()
        base = parse_response(MOCK_LLM_RESPONSE_GOOD, MOCK_JOB_DEVSECOPS, cfg)
        base.update({
            "llm_provider": provider, "llm_model": model,
            "analysis_mode": "standard", "retry_count": 0,
            "used_fallback": False, "validation_errors": "",
        })
        return base

    def test_routes_gemini(self):
        from analyzer.llm import analyze_match
        expected = self._good_result("gemini", "gemini-2.5-flash")
        with patch("analyzer.llm.call_gemini_once", new=AsyncMock(return_value=expected)):
            result = run(analyze_match(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS, "gemini"))
        self.assertEqual(result["llm_provider"], "gemini")

    def test_gemini_fallback_on_all_retries_failed(self):
        from analyzer.llm import analyze_match
        with patch("analyzer.llm.call_gemini_once", new=AsyncMock(side_effect=ValueError("API error"))):
            result = run(analyze_match(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS, "gemini"))
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["llm_provider"], "gemini")

    def test_fallback_model_name_gemini(self):
        from analyzer.llm import analyze_match
        with patch("analyzer.llm.call_gemini_once", new=AsyncMock(side_effect=ValueError("fail"))):
            with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-flash"}):
                result = run(analyze_match(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS, "gemini"))
        self.assertEqual(result["llm_model"], "gemini-2.5-flash")


# ── Salary — Gemini ───────────────────────────────────────────────────────────

class TestCallSalaryLLMGemini(unittest.TestCase):
    """Tests for _call_salary_llm with Gemini provider."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "AIza-test", "GEMINI_MODEL": "gemini-2.5-flash"})
    def test_gemini_salary_call(self):
        from analyzer.salary import _call_salary_llm
        mock_resp = MagicMock()
        mock_resp.text = MOCK_SALARY_RESPONSE
        mock_resp.usage_metadata = MagicMock(prompt_token_count=50, candidates_token_count=20)
        mock_aio_models = MagicMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_resp)
        mock_aio = MagicMock()
        mock_aio.models = mock_aio_models
        mock_client = MagicMock()
        mock_client.aio = mock_aio
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_genai.types = MagicMock()
        with patch.dict("sys.modules", {"google.genai": mock_genai, "google.genai.types": mock_genai.types}):
            raw, ctx = run(_call_salary_llm("test prompt", "gemini"))
        self.assertEqual(raw, MOCK_SALARY_RESPONSE)

    @patch.dict(os.environ, {"GEMINI_API_KEY": ""})
    def test_gemini_missing_key_raises(self):
        from analyzer.salary import _call_salary_llm
        with self.assertRaises(ValueError) as ctx:
            run(_call_salary_llm("test prompt", "gemini"))
        self.assertIn("Gemini API key", str(ctx.exception))


class TestEstimateSalaryGemini(unittest.TestCase):
    """Tests for estimate_salary() and extract_salary() with Gemini."""

    def _mock_salary_call(self):
        return patch(
            "analyzer.salary._call_salary_llm",
            new=AsyncMock(return_value=(MOCK_SALARY_RESPONSE, None))
        )

    def test_estimate_salary_gemini(self):
        from analyzer.salary import estimate_salary
        with self._mock_salary_call():
            with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-flash"}):
                result = run(estimate_salary("Dev", "Acme", "Remote", MOCK_JOB_NO_SALARY, "gemini"))
        self.assertEqual(result["llm_provider"], "gemini")
        self.assertGreater(result["min"], 0)

    def test_extract_salary_gemini(self):
        from analyzer.salary import extract_salary
        with self._mock_salary_call():
            with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-flash"}):
                result = run(extract_salary("Dev", "Acme", "Remote", MOCK_JOB_WITH_SALARY, "gemini"))
        self.assertEqual(result["llm_provider"], "gemini")
        self.assertEqual(result["source"], "posted")

    def test_get_salary_model_name_gemini(self):
        from analyzer.salary import _get_salary_model_name
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-flash"}):
            self.assertEqual(_get_salary_model_name("gemini"), "gemini-2.5-flash")


# ── Health checks — Gemini ────────────────────────────────────────────────────

class TestHealthChecksGemini(unittest.TestCase):
    """Tests for check_gemini() in launcher.py."""

    def test_ok_with_valid_key(self):
        from launcher import check_gemini
        result = check_gemini("AIzaSyTestKey1234")
        self.assertEqual(result["status"], "ok")
        self.assertIn("Key present", result["message"])

    def test_warn_on_empty_key(self):
        from launcher import check_gemini
        result = check_gemini("")
        self.assertEqual(result["status"], "warn")
        self.assertIn("No key set", result["message"])

    def test_masked_key_in_message(self):
        from launcher import check_gemini
        result = check_gemini("AIzaSyTestKey1234")
        self.assertIn("...", result["message"])
        self.assertNotIn("AIzaSyTestKey1234", result["message"])


# ── Launcher config parsing ───────────────────────────────────────────────────

class TestLauncherGeminiConfig(unittest.TestCase):
    """Tests for Gemini key handling in launcher form parsing and .env save."""

    def _make_handler(self, form_data: dict):
        from launcher import LauncherHandler, Launcher
        cfg = {
            "port": 8000, "host": "127.0.0.1", "db_path": "job_matcher.db",
            "anthropic_api_key": "", "openai_api_key": "", "gemini_api_key": "",
            "ollama_base_url": "http://localhost:11434", "ollama_model": "llama3.1:8b",
            "ollama_timeout": 600, "analysis_mode": "standard",
        }
        launcher = MagicMock(spec=Launcher)
        launcher.cfg = cfg
        handler = LauncherHandler.__new__(LauncherHandler)
        handler.launcher = launcher
        handler.read_form = MagicMock(return_value=form_data)
        return handler

    def test_parses_gemini_key(self):
        from launcher import LauncherHandler
        handler = self._make_handler({"gemini_api_key": "AIzaMyGeminiKey"})
        cfg = handler._parse_config_from_form(handler.read_form())
        self.assertEqual(cfg["gemini_api_key"], "AIzaMyGeminiKey")

    def test_both_keys_parsed_together(self):
        from launcher import LauncherHandler
        handler = self._make_handler({
            "openai_api_key": "sk-openai123",
            "gemini_api_key": "AIza-gemini456",
        })
        cfg = handler._parse_config_from_form(handler.read_form())
        self.assertEqual(cfg["openai_api_key"], "sk-openai123")
        self.assertEqual(cfg["gemini_api_key"], "AIza-gemini456")

    def test_gemini_key_written_to_env(self):
        import tempfile, os as _os
        from launcher import Launcher
        launcher = Launcher.__new__(Launcher)
        launcher._lock = __import__("threading").Lock()
        launcher.cfg = {}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False, newline="\n") as f:
            f.write("ANTHROPIC_API_KEY=sk-ant-old\n")
            tmp_path = f.name
        try:
            with patch("launcher.os.path") as mock_path:
                mock_path.exists.return_value = True
                original_open = open
                def patched_open(path, *args, **kwargs):
                    if path == ".env":
                        return original_open(tmp_path, *args, **kwargs)
                    return original_open(path, *args, **kwargs)
                with patch("builtins.open", side_effect=patched_open):
                    launcher._save_to_env({
                        "anthropic_api_key": "sk-ant-old",
                        "openai_api_key":    "",
                        "gemini_api_key":    "AIza-new",
                        "ollama_base_url":   "http://localhost:11434",
                        "ollama_model":      "llama3.1:8b",
                        "ollama_timeout":    600,
                        "port":              8000,
                        "host":              "127.0.0.1",
                        "analysis_mode":     "standard",
                    })
            with open(tmp_path) as f:
                written = f.read()
            self.assertIn("GEMINI_API_KEY=AIza-new", written)
        finally:
            try:
                _os.unlink(tmp_path)
            except Exception:
                pass


# ── Combined health check report ──────────────────────────────────────────────

class TestRunHealthChecksProviders(unittest.TestCase):
    """Tests for run_health_checks() combined report."""

    def test_report_includes_openai_and_gemini(self):
        from launcher import run_health_checks
        with patch("launcher.check_sqlite", return_value={"status": "ok", "message": "ok"}):
            with patch("launcher.check_ollama", return_value=({"status": "warn", "message": "not running"}, [])):
                report = run_health_checks(
                    db_path="test.db",
                    ollama_url="http://localhost:11434",
                    api_key="sk-ant-test",
                    openai_key="sk-openai-test",
                    gemini_key="AIza-test",
                )
        self.assertIn("openai", report)
        self.assertIn("gemini", report)
        self.assertIn("sqlite", report)
        self.assertIn("anthropic", report)

    def test_openai_ok_in_report(self):
        from launcher import run_health_checks
        with patch("launcher.check_sqlite", return_value={"status": "ok", "message": "ok"}):
            with patch("launcher.check_ollama", return_value=({"status": "warn", "message": "down"}, [])):
                report = run_health_checks("db", "url", "", openai_key="sk-test1234")
        self.assertEqual(report["openai"]["status"], "ok")

    def test_gemini_warn_when_no_key(self):
        from launcher import run_health_checks
        with patch("launcher.check_sqlite", return_value={"status": "ok", "message": "ok"}):
            with patch("launcher.check_ollama", return_value=({"status": "warn", "message": "down"}, [])):
                report = run_health_checks("db", "url", "", gemini_key="")
        self.assertEqual(report["gemini"]["status"], "warn")


if __name__ == "__main__":
    unittest.main()
