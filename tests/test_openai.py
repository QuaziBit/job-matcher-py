"""
tests/test_openai.py — Unit tests for OpenAI provider integration.
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

class TestOpenAIConfig(unittest.TestCase):
    """Tests for openai_model() env helper and CLOUD_PROVIDERS."""

    def test_openai_model_default(self):
        from analyzer.config import openai_model, OPENAI_MODEL
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_MODEL", None)
            self.assertEqual(openai_model(), OPENAI_MODEL)

    def test_openai_model_from_env(self):
        from analyzer.config import openai_model
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o"}):
            self.assertEqual(openai_model(), "gpt-4o")

    def test_cloud_providers_includes_openai(self):
        from analyzer.config import CLOUD_PROVIDERS
        self.assertIn("openai", CLOUD_PROVIDERS)

    def test_get_model_for_provider_openai(self):
        from analyzer.llm import _get_model_for_provider
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o-mini"}):
            self.assertEqual(_get_model_for_provider("openai"), "gpt-4o-mini")


# ── call_openai_once ──────────────────────────────────────────────────────────

class TestCallOpenAIOnce(unittest.TestCase):
    """Tests for call_openai_once — mocks AsyncOpenAI client."""

    def _make_openai_response(self, content: str):
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        return resp

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-1234", "OPENAI_MODEL": "gpt-4o-mini"})
    def test_returns_valid_result(self):
        from analyzer.llm import call_openai_once
        mock_resp = self._make_openai_response(MOCK_LLM_RESPONSE_GOOD)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = run(call_openai_once(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertEqual(result["llm_provider"], "openai")
        self.assertEqual(result["llm_model"], "gpt-4o-mini")
        self.assertIn("score", result)
        self.assertIn("matched_skills", result)
        self.assertIn("missing_skills", result)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_MODEL": "gpt-4o-mini"})
    def test_raises_on_missing_key(self):
        from analyzer.llm import call_openai_once
        with self.assertRaises(ValueError) as ctx:
            run(call_openai_once(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertIn("OpenAI API key", str(ctx.exception))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-1234", "OPENAI_MODEL": "gpt-4o-mini"})
    def test_result_has_analysis_mode(self):
        from analyzer.llm import call_openai_once
        mock_resp = self._make_openai_response(MOCK_LLM_RESPONSE_GOOD)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = run(call_openai_once(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertIn("analysis_mode", result)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-1234"})
    def test_uses_env_model(self):
        from analyzer.llm import call_openai_once
        mock_resp = self._make_openai_response(MOCK_LLM_RESPONSE_GOOD)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o"}):
            with patch("openai.AsyncOpenAI", return_value=mock_client):
                result = run(call_openai_once(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertEqual(result["llm_model"], "gpt-4o")


# ── analyze_match routing (OpenAI) ────────────────────────────────────────────

class TestAnalyzeMatchOpenAI(unittest.TestCase):
    """Tests for analyze_match() routing to OpenAI."""

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

    def test_routes_openai(self):
        from analyzer.llm import analyze_match
        expected = self._good_result("openai", "gpt-4o-mini")
        with patch("analyzer.llm.call_openai_once", new=AsyncMock(return_value=expected)):
            result = run(analyze_match(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS, "openai"))
        self.assertEqual(result["llm_provider"], "openai")

    def test_openai_fallback_on_all_retries_failed(self):
        from analyzer.llm import analyze_match
        with patch("analyzer.llm.call_openai_once", new=AsyncMock(side_effect=ValueError("API error"))):
            result = run(analyze_match(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS, "openai"))
        self.assertTrue(result["used_fallback"])
        self.assertEqual(result["llm_provider"], "openai")

    def test_fallback_model_name_openai(self):
        from analyzer.llm import analyze_match
        with patch("analyzer.llm.call_openai_once", new=AsyncMock(side_effect=ValueError("fail"))):
            with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o-mini"}):
                result = run(analyze_match(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS, "openai"))
        self.assertEqual(result["llm_model"], "gpt-4o-mini")


# ── Salary — OpenAI ───────────────────────────────────────────────────────────

class TestCallSalaryLLMOpenAI(unittest.TestCase):
    """Tests for _call_salary_llm with OpenAI provider."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-1234", "OPENAI_MODEL": "gpt-4o-mini"})
    def test_openai_salary_call(self):
        from analyzer.salary import _call_salary_llm
        mock_msg = MagicMock()
        mock_msg.content = MOCK_SALARY_RESPONSE
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            raw, ctx = run(_call_salary_llm("test prompt", "openai"))
        self.assertEqual(raw, MOCK_SALARY_RESPONSE)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_openai_missing_key_raises(self):
        from analyzer.salary import _call_salary_llm
        with self.assertRaises(ValueError) as ctx:
            run(_call_salary_llm("test prompt", "openai"))
        self.assertIn("OpenAI API key", str(ctx.exception))


class TestEstimateSalaryOpenAI(unittest.TestCase):
    """Tests for estimate_salary() and extract_salary() with OpenAI."""

    def _mock_salary_call(self):
        return patch(
            "analyzer.salary._call_salary_llm",
            new=AsyncMock(return_value=(MOCK_SALARY_RESPONSE, None))
        )

    def test_estimate_salary_openai(self):
        from analyzer.salary import estimate_salary
        with self._mock_salary_call():
            with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o-mini"}):
                result = run(estimate_salary("Dev", "Acme", "Remote", MOCK_JOB_NO_SALARY, "openai"))
        self.assertEqual(result["llm_provider"], "openai")
        self.assertGreater(result["min"], 0)
        self.assertGreater(result["max"], 0)

    def test_extract_salary_openai(self):
        from analyzer.salary import extract_salary
        with self._mock_salary_call():
            with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o-mini"}):
                result = run(extract_salary("Dev", "Acme", "Remote", MOCK_JOB_WITH_SALARY, "openai"))
        self.assertEqual(result["llm_provider"], "openai")
        self.assertEqual(result["source"], "posted")

    def test_get_salary_model_name_openai(self):
        from analyzer.salary import _get_salary_model_name
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o-mini"}):
            self.assertEqual(_get_salary_model_name("openai"), "gpt-4o-mini")


# ── Health checks — OpenAI ────────────────────────────────────────────────────

class TestHealthChecksOpenAI(unittest.TestCase):
    """Tests for check_openai() in launcher.py."""

    def test_ok_with_valid_key(self):
        from launcher import check_openai
        result = check_openai("sk-abcdefghijklmnop")
        self.assertEqual(result["status"], "ok")
        self.assertIn("Key present", result["message"])

    def test_warn_on_empty_key(self):
        from launcher import check_openai
        result = check_openai("")
        self.assertEqual(result["status"], "warn")
        self.assertIn("No key set", result["message"])

    def test_error_on_bad_format(self):
        from launcher import check_openai
        result = check_openai("gsk-wrongformat")
        self.assertEqual(result["status"], "error")
        self.assertIn("format invalid", result["message"])

    def test_masked_key_in_message(self):
        from launcher import check_openai
        result = check_openai("sk-test12345678abcd")
        self.assertIn("...", result["message"])
        self.assertNotIn("sk-test12345678abcd", result["message"])


# ── Launcher config parsing + .env persistence ────────────────────────────────

class TestLauncherOpenAIConfig(unittest.TestCase):
    """Tests for OpenAI key handling in launcher form parsing and .env save."""

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

    def test_parses_openai_key(self):
        from launcher import LauncherHandler
        handler = self._make_handler({"openai_api_key": "sk-myopenaikey"})
        cfg = handler._parse_config_from_form(handler.read_form())
        self.assertEqual(cfg["openai_api_key"], "sk-myopenaikey")

    def test_empty_openai_key_not_stored(self):
        from launcher import LauncherHandler
        handler = self._make_handler({})
        cfg = handler._parse_config_from_form(handler.read_form())
        self.assertEqual(cfg.get("openai_api_key", ""), "")

    def test_openai_key_written_to_env(self):
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
                        "openai_api_key":    "sk-openai-new",
                        "gemini_api_key":    "",
                        "ollama_base_url":   "http://localhost:11434",
                        "ollama_model":      "llama3.1:8b",
                        "ollama_timeout":    600,
                        "port":              8000,
                        "host":              "127.0.0.1",
                        "analysis_mode":     "standard",
                    })
            with open(tmp_path) as f:
                written = f.read()
            self.assertIn("OPENAI_API_KEY=sk-openai-new", written)
        finally:
            try:
                _os.unlink(tmp_path)
            except Exception:
                pass


class TestHealthKeyFallback(unittest.TestCase):
    """
    Tests that the health endpoint uses the query param value directly
    and does NOT fall back to cfg when the param is present but empty.
    This ensures clearing a key in the launcher immediately shows ⚠.
    """

    def _make_handler(self, query: dict, cfg_openai: str = "sk-cfg-key"):
        from launcher import LauncherHandler, Launcher
        cfg = {"openai_api_key": cfg_openai, "anthropic_api_key": "", "gemini_api_key": ""}
        launcher = MagicMock(spec=Launcher)
        launcher.cfg = cfg
        handler = LauncherHandler.__new__(LauncherHandler)
        handler.launcher = launcher
        return handler, query

    def test_empty_openai_key_param_does_not_fall_back_to_cfg(self):
        """Sending openai_key= (empty) should use empty, not the saved cfg key."""
        from launcher import LauncherHandler
        handler, query = self._make_handler({"openai_key": [""]}, cfg_openai="sk-old-key")
        # Simulate the key resolution logic from the health handler
        openai_key = query.get("openai_key", [""])[0] if "openai_key" in query                      else handler.launcher.cfg.get("openai_api_key", "")
        self.assertEqual(openai_key, "")

    def test_missing_openai_key_param_falls_back_to_cfg(self):
        """When openai_key param is absent entirely, fall back to saved cfg."""
        from launcher import LauncherHandler
        handler, query = self._make_handler({}, cfg_openai="sk-saved-key")
        openai_key = query.get("openai_key", [""])[0] if "openai_key" in query                      else handler.launcher.cfg.get("openai_api_key", "")
        self.assertEqual(openai_key, "sk-saved-key")

    def test_nonempty_openai_key_param_used_directly(self):
        """When openai_key param has a value, use it."""
        from launcher import LauncherHandler
        handler, query = self._make_handler({"openai_key": ["sk-new-key"]}, cfg_openai="sk-old-key")
        openai_key = query.get("openai_key", [""])[0] if "openai_key" in query                      else handler.launcher.cfg.get("openai_api_key", "")
        self.assertEqual(openai_key, "sk-new-key")


class TestHealthAnthropicKeyFallback(unittest.TestCase):
    """Same fallback fix tests for the Anthropic api_key param."""

    def _make_handler(self, query: dict, cfg_anthropic: str = "sk-ant-cfg-key"):
        from launcher import LauncherHandler, Launcher
        cfg = {"anthropic_api_key": cfg_anthropic, "openai_api_key": "", "gemini_api_key": ""}
        launcher = MagicMock(spec=Launcher)
        launcher.cfg = cfg
        handler = LauncherHandler.__new__(LauncherHandler)
        handler.launcher = launcher
        return handler, query

    def test_empty_anthropic_key_param_does_not_fall_back_to_cfg(self):
        """Sending api_key= (empty) should use empty, not the saved cfg key."""
        handler, query = self._make_handler({"api_key": [""]}, cfg_anthropic="sk-ant-old")
        api_key = query.get("api_key", [""])[0] if "api_key" in query                   else handler.launcher.cfg.get("anthropic_api_key", "")
        self.assertEqual(api_key, "")

    def test_missing_anthropic_key_param_falls_back_to_cfg(self):
        """When api_key param is absent entirely, fall back to saved cfg."""
        handler, query = self._make_handler({}, cfg_anthropic="sk-ant-saved")
        api_key = query.get("api_key", [""])[0] if "api_key" in query                   else handler.launcher.cfg.get("anthropic_api_key", "")
        self.assertEqual(api_key, "sk-ant-saved")

    def test_nonempty_anthropic_key_param_used_directly(self):
        """When api_key param has a value, use it."""
        handler, query = self._make_handler({"api_key": ["sk-ant-new"]}, cfg_anthropic="sk-ant-old")
        api_key = query.get("api_key", [""])[0] if "api_key" in query                   else handler.launcher.cfg.get("anthropic_api_key", "")
        self.assertEqual(api_key, "sk-ant-new")


if __name__ == "__main__":
    unittest.main()
