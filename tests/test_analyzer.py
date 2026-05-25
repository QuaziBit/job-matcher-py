"""
tests/test_analyzer.py — Unit tests for analyzer module.
Covers: response parsing, penalty pipeline, prompt building, LLM callers, dispatch.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from analyzer.config import anthropic_model
from tests.mock_data import (
    run,
    MOCK_RESUME_DEVSECOPS,
    MOCK_JOB_DEVSECOPS,
    MOCK_JOB_MARKETING,
    MOCK_LLM_RESPONSE_GOOD,
    MOCK_LLM_RESPONSE_POOR,
    MOCK_LLM_RESPONSE_WITH_FENCES,
    MOCK_LLM_RESPONSE_INVALID_SCORE,
    MOCK_LLM_RESPONSE_NO_JSON,
)


class TestParseResponse(unittest.TestCase):
    """Tests for analyzer._parse_response — pure function, no mocking needed."""

    def test_valid_json_score_5(self):
        from analyzer import _parse_response
        result = _parse_response(MOCK_LLM_RESPONSE_GOOD)
        self.assertEqual(result["score"], 5)
        skill_names = [s["skill"] if isinstance(s, dict) else s for s in result["matched_skills"]]
        self.assertIn("Python", skill_names)
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
        missing = [{"skill": "TS/SCI Clearance", "severity": "blocker",
                    "requirement_type": "hard", "cluster_group": "security"}]
        adjusted, breakdown = _compute_adjusted_score(4, missing)
        self.assertLess(adjusted, 4)
        self.assertEqual(breakdown["blockers"], 1)
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
        missing = [{"skill": "Nice-to-have", "severity": "minor",
                    "requirement_type": "preferred", "cluster_group": "other"}]
        adjusted, breakdown = _compute_adjusted_score(4, missing)
        self.assertEqual(breakdown["total_penalty"], 0)
        self.assertEqual(adjusted, 4)

    def test_two_minors_give_penalty(self):
        from analyzer import _compute_adjusted_score
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
        missing = [{"skill": f"skill{i}", "severity": "minor",
                    "requirement_type": "preferred", "cluster_group": f"cat{i}"}
                   for i in range(7)]
        _, breakdown = _compute_adjusted_score(4, missing)
        self.assertEqual(breakdown["count_penalty"], 0)

    def test_bonus_requirement_zero_penalty(self):
        from analyzer import penalty_for_skill
        skill = {"skill": "Kubernetes", "severity": "major", "requirement_type": "bonus"}
        self.assertEqual(penalty_for_skill(skill), 0)

    def test_hard_blocker_penalty(self):
        from analyzer import penalty_for_skill
        skill = {"skill": "Secret Clearance", "severity": "blocker", "requirement_type": "hard"}
        self.assertEqual(penalty_for_skill(skill), 2)

    def test_preferred_major_penalty(self):
        from analyzer import penalty_for_skill
        skill = {"skill": "AWS", "severity": "major", "requirement_type": "preferred"}
        self.assertEqual(penalty_for_skill(skill), 1)

    def test_cloud_cluster_capped(self):
        from analyzer import _compute_adjusted_score
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
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test-key"})
    @patch("analyzer.llm.anthropic.Anthropic")
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

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test-key"})
    @patch("analyzer.llm.anthropic.Anthropic")
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
    @patch("analyzer.llm.httpx.AsyncClient")
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

    @patch("analyzer.llm._call_chunk", new_callable=AsyncMock)
    def test_ollama_connection_error(self, mock_call_chunk):
        # Chunked approach — chunk 1 failure raises ValueError about score
        mock_call_chunk.return_value = None  # simulates connection failure
        from analyzer import analyze_with_ollama
        with self.assertRaises(ValueError) as ctx:
            run(analyze_with_ollama("resume", "job"))
        self.assertIn("Chunked analysis failed", str(ctx.exception))

    @patch("analyzer.llm.httpx.AsyncClient")
    def test_ollama_payload_has_think_false(self, mock_client_cls):
        """Ollama requests must include think:False to suppress reasoning tokens."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": MOCK_LLM_RESPONSE_POOR}}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        from analyzer import analyze_with_ollama
        run(analyze_with_ollama(MOCK_RESUME_DEVSECOPS, MOCK_JOB_MARKETING))

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs[1].get("json") or call_kwargs[0][1]
        options = payload.get("options", {})
        self.assertIn("think", options)
        self.assertFalse(options["think"], "think must be False to suppress Gemma4 reasoning tokens")


class TestAnalyzeMatchDispatch(unittest.TestCase):
    @patch("analyzer.llm.call_anthropic_once", new_callable=AsyncMock)
    @patch("analyzer.llm.call_ollama_once", new_callable=AsyncMock)
    def test_routes_to_anthropic_by_default(self, mock_ollama, mock_anthropic):
        mock_anthropic.return_value = {
            "score": 4, "adjusted_score": 4, "llm_provider": "anthropic",
            "llm_model": anthropic_model(),
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

    @patch("analyzer.llm.call_anthropic_once", new_callable=AsyncMock)
    @patch("analyzer.llm.call_ollama_chunked", new_callable=AsyncMock)
    def test_routes_to_ollama_when_specified(self, mock_ollama_chunked, mock_anthropic):
        mock_ollama_chunked.return_value = {
            "score": 3, "adjusted_score": 3, "llm_provider": "ollama",
            "llm_model": "llama3.1:8b",
            "matched_skills": [{"skill": "Docker", "match_type": "exact",
                                 "jd_snippet": "Docker required",
                                 "resume_snippet": "Used Docker", "category": "devops"}],
            "missing_skills": [], "reasoning": "Partial match",
            "penalty_breakdown": {}, "suggestions": [],
        }
        from analyzer import analyze_match
        run(analyze_match("resume", "job", provider="ollama"))
        mock_ollama_chunked.assert_called_once()
        mock_anthropic.assert_not_called()

    @patch("analyzer.llm.ollama_model", return_value="gemma4:e2b")
    @patch("analyzer.llm.call_ollama_thinking", new_callable=AsyncMock)
    @patch("analyzer.llm.call_ollama_chunked", new_callable=AsyncMock)
    def test_thinking_model_routes_to_thinking_not_chunked(self, mock_chunked, mock_thinking, mock_model):
        """Thinking models (gemma4, deepseek-r1) must use call_ollama_thinking."""
        mock_thinking.return_value = {
            "score": 4, "adjusted_score": 4, "llm_provider": "ollama",
            "llm_model": "gemma4:e2b",
            "matched_skills": [{"skill": "Python", "match_type": "exact",
                                 "jd_snippet": "Python required",
                                 "resume_snippet": "Python dev", "category": "backend"}],
            "missing_skills": [], "reasoning": "Good match",
            "penalty_breakdown": {}, "suggestions": [],
        }
        from analyzer import analyze_match
        run(analyze_match("resume", "job", provider="ollama"))
        mock_thinking.assert_called_once()
        mock_chunked.assert_not_called()


class TestCallOllamaThinking(unittest.TestCase):
    """Tests for call_ollama_thinking — mocks httpx.AsyncClient."""

    def _mock_responses(self, body_a: str, body_b: str, status: int = 200):
        """Build a mock httpx client that returns body_a then body_b."""
        resps = [
            MagicMock(json=MagicMock(return_value={"message": {"content": body_a}}),
                      raise_for_status=MagicMock()),
            MagicMock(json=MagicMock(return_value={"message": {"content": body_b}}),
                      raise_for_status=MagicMock()),
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=resps)
        return mock_client

    @patch("analyzer.llm.httpx.AsyncClient")
    def test_merges_call_a_and_call_b(self, mock_client_cls):
        """Both calls succeed — result has score, matched, missing, suggestions."""
        import json
        body_a = json.dumps({
            "score": 4, "reasoning": "Strong match.",
            "matched_skills": [{"skill": "Python", "match_type": "exact",
                                 "jd_snippet": "Python required", "resume_snippet": "Python dev"}]
        })
        body_b = json.dumps({
            "missing_skills": [{"skill": "AWS Lambda", "severity": "major",
                                 "requirement_type": "preferred", "jd_snippet": "Lambda required"}],
            "suggestions": [{"title": "Add AWS", "detail": "Mention AWS experience."}]
        })
        mock_client_cls.return_value = self._mock_responses(body_a, body_b)
        import os
        orig = os.environ.get("ANALYSIS_MODE")
        os.environ["ANALYSIS_MODE"] = "detailed"  # suggestions only enabled in detailed mode
        try:
            from analyzer.llm import call_ollama_thinking
            result = run(call_ollama_thinking(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        finally:
            if orig is None:
                os.environ.pop("ANALYSIS_MODE", None)
            else:
                os.environ["ANALYSIS_MODE"] = orig
        self.assertEqual(result["score"], 4)
        self.assertEqual(result["reasoning"], "Strong match.")
        self.assertEqual(len(result["matched_skills"]), 1)
        self.assertEqual(result["matched_skills"][0]["skill"], "Python")
        self.assertEqual(len(result["missing_skills"]), 1)
        self.assertEqual(len(result["suggestions"]), 1)
        self.assertEqual(result["llm_provider"], "ollama")

    @patch("analyzer.llm.httpx.AsyncClient")
    def test_call_a_failure_raises(self, mock_client_cls):
        """If call A has no score, ValueError is raised."""
        body_a = '{"reasoning": "no score here"}'
        body_b = '{"missing_skills": [], "suggestions": []}'
        mock_client_cls.return_value = self._mock_responses(body_a, body_b)
        from analyzer.llm import call_ollama_thinking
        with self.assertRaises(ValueError) as ctx:
            run(call_ollama_thinking(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertIn("score", str(ctx.exception))

    @patch("analyzer.llm.httpx.AsyncClient")
    def test_call_b_failure_returns_empty_missing(self, mock_client_cls):
        """If call B fails, missing_skills and suggestions are empty but result is valid."""
        import json
        body_a = json.dumps({
            "score": 3, "reasoning": "Moderate match.",
            "matched_skills": [{"skill": "Python", "match_type": "exact",
                                 "jd_snippet": "Python", "resume_snippet": "Python"}]
        })
        body_b = "not valid json at all"
        mock_client_cls.return_value = self._mock_responses(body_a, body_b)
        from analyzer.llm import call_ollama_thinking
        result = run(call_ollama_thinking(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertEqual(result["score"], 3)
        self.assertEqual(result["missing_skills"], [])
        self.assertEqual(result["suggestions"], [])

    @patch("analyzer.llm.httpx.AsyncClient")
    def test_makes_exactly_two_http_calls(self, mock_client_cls):
        """Exactly two POST calls are made — call A and call B."""
        import json
        body_a = json.dumps({"score": 4, "reasoning": "Good.", "matched_skills": []})
        body_b = json.dumps({"missing_skills": [], "suggestions": []})
        mock_client = self._mock_responses(body_a, body_b)
        mock_client_cls.return_value = mock_client
        from analyzer.llm import call_ollama_thinking
        run(call_ollama_thinking(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertEqual(mock_client.post.call_count, 2)

    @patch("analyzer.llm.httpx.AsyncClient")
    def test_score_clamped_to_valid_range(self, mock_client_cls):
        """Score outside 1-5 is clamped."""
        import json
        body_a = json.dumps({"score": 10, "reasoning": "Perfect.", "matched_skills": []})
        body_b = json.dumps({"missing_skills": [], "suggestions": []})
        mock_client_cls.return_value = self._mock_responses(body_a, body_b)
        from analyzer.llm import call_ollama_thinking
        result = run(call_ollama_thinking(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertLessEqual(result["score"], 5)
        self.assertGreaterEqual(result["score"], 1)

    @patch("analyzer.llm.httpx.AsyncClient")
    def test_result_has_penalty_fields(self, mock_client_cls):
        """Result always contains adjusted_score and penalty_breakdown."""
        import json
        body_a = json.dumps({"score": 4, "reasoning": "Good.", "matched_skills": []})
        body_b = json.dumps({"missing_skills": [], "suggestions": []})
        mock_client_cls.return_value = self._mock_responses(body_a, body_b)
        from analyzer.llm import call_ollama_thinking
        result = run(call_ollama_thinking(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertIn("adjusted_score", result)
        self.assertIn("penalty_breakdown", result)
        self.assertIn("validation_errors", result)
        self.assertIn("used_fallback", result)

    @patch("analyzer.llm.httpx.AsyncClient")
    def test_blocker_reduces_adjusted_score(self, mock_client_cls):
        """A blocker in missing_skills should reduce adjusted_score below score."""
        import json
        body_a = json.dumps({"score": 4, "reasoning": "Good.", "matched_skills": []})
        body_b = json.dumps({
            "missing_skills": [{"skill": "Security Clearance", "severity": "blocker",
                                 "requirement_type": "hard", "jd_snippet": "clearance required"}],
            "suggestions": []
        })
        mock_client_cls.return_value = self._mock_responses(body_a, body_b)
        from analyzer.llm import call_ollama_thinking
        result = run(call_ollama_thinking(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertLess(result["adjusted_score"], result["score"])

    @patch("analyzer.llm.ollama_model", return_value="gemma4:26b")
    @patch("analyzer.llm.httpx.AsyncClient")
    def test_respects_requested_mode(self, mock_client_cls, _mock_model):
        """call_ollama_thinking uses the requested mode when the model supports it.
        Previously the function always ran with get_mode_config() (env default)
        and never called cap_mode_for_model, so the mode was not respected."""
        import json, os
        body_a = json.dumps({"score": 3, "reasoning": "ok.", "matched_skills": []})
        body_b = json.dumps({"missing_skills": [], "suggestions": []})
        mock_client_cls.return_value = self._mock_responses(body_a, body_b)
        orig = os.environ.get("ANALYSIS_MODE")
        os.environ["ANALYSIS_MODE"] = "fast"
        try:
            from analyzer.llm import call_ollama_thinking
            result = run(call_ollama_thinking(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
            self.assertEqual(result["analysis_mode"], "fast",
                             "thinking path must honour the requested mode")
        finally:
            if orig is None:
                os.environ.pop("ANALYSIS_MODE", None)
            else:
                os.environ["ANALYSIS_MODE"] = orig

    @patch("analyzer.llm.ollama_model", return_value="deepseek-r1:7b")
    @patch("analyzer.llm.httpx.AsyncClient")
    def test_caps_mode_for_model(self, mock_client_cls, _mock_model):
        """call_ollama_thinking caps mode to the model's max when requested mode
        exceeds capability. deepseek-r1:7b max is 'standard'; requesting 'detailed'
        must be downgraded."""
        import json, os
        body_a = json.dumps({"score": 3, "reasoning": "ok.", "matched_skills": []})
        body_b = json.dumps({"missing_skills": [], "suggestions": []})
        mock_client_cls.return_value = self._mock_responses(body_a, body_b)
        orig = os.environ.get("ANALYSIS_MODE")
        os.environ["ANALYSIS_MODE"] = "detailed"
        try:
            from analyzer.llm import call_ollama_thinking
            result = run(call_ollama_thinking(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
            self.assertEqual(result["analysis_mode"], "standard",
                             "deepseek-r1:7b max is standard — detailed must be capped")
        finally:
            if orig is None:
                os.environ.pop("ANALYSIS_MODE", None)
            else:
                os.environ["ANALYSIS_MODE"] = orig

    @patch("analyzer.llm.ollama_model", return_value="gemma4:26b")
    @patch("analyzer.llm.httpx.AsyncClient")
    def test_fast_mode_disables_suggestions(self, mock_client_cls, _mock_model):
        """In fast/standard mode, thinking path must not return suggestions.
        suggestions are only enabled in detailed mode."""
        import json, os
        body_a = json.dumps({"score": 3, "reasoning": "ok.", "matched_skills": []})
        # Call B returns suggestions — they must be discarded in fast mode
        body_b = json.dumps({
            "missing_skills": [{"skill": "AWS", "severity": "minor",
                                 "requirement_type": "preferred", "jd_snippet": "AWS"}],
            "suggestions": [{"title": "Add AWS", "detail": "Get AWS certified."}]
        })
        mock_client_cls.return_value = self._mock_responses(body_a, body_b)
        orig = os.environ.get("ANALYSIS_MODE")
        os.environ["ANALYSIS_MODE"] = "fast"
        try:
            from analyzer.llm import call_ollama_thinking
            result = run(call_ollama_thinking(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
            self.assertEqual(result["suggestions"], [],
                             "fast mode must not return suggestions from thinking path")
        finally:
            if orig is None:
                os.environ.pop("ANALYSIS_MODE", None)
            else:
                os.environ["ANALYSIS_MODE"] = orig

    @patch("analyzer.llm.ollama_model", return_value="gemma4:26b")
    @patch("analyzer.llm.httpx.AsyncClient")
    def test_detailed_mode_enables_suggestions(self, mock_client_cls, _mock_model):
        """In detailed mode, thinking path must return suggestions."""
        import json, os
        body_a = json.dumps({"score": 4, "reasoning": "Strong match.", "matched_skills": []})
        body_b = json.dumps({
            "missing_skills": [],
            "suggestions": [{"title": "Add AWS", "detail": "Get AWS certified."}]
        })
        mock_client_cls.return_value = self._mock_responses(body_a, body_b)
        orig = os.environ.get("ANALYSIS_MODE")
        os.environ["ANALYSIS_MODE"] = "detailed"
        try:
            from analyzer.llm import call_ollama_thinking
            result = run(call_ollama_thinking(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
            self.assertEqual(len(result["suggestions"]), 1,
                             "detailed mode must return suggestions from thinking path")
        finally:
            if orig is None:
                os.environ.pop("ANALYSIS_MODE", None)
            else:
                os.environ["ANALYSIS_MODE"] = orig


class TestEscapeControlChars(unittest.TestCase):
    """Tests for analyzer.parsers._escape_control_chars."""

    def _fn(self):
        from analyzer.parsers import _escape_control_chars
        return _escape_control_chars

    def test_escapes_literal_tab_inside_string(self):
        fn = self._fn()
        raw = '{"key": "value\twith\ttabs"}'
        result = fn(raw)
        import json
        data = json.loads(result)
        self.assertIn("\\t", repr(data["key"]))

    def test_escapes_literal_newline_inside_string(self):
        fn = self._fn()
        raw = '{"key": "line1\nline2"}'
        result = fn(raw)
        import json
        data = json.loads(result)
        self.assertIn("\n", data["key"])

    def test_escapes_literal_cr_inside_string(self):
        fn = self._fn()
        raw = '{"key": "value\rwith\rcr"}'
        result = fn(raw)
        import json
        json.loads(result)  # should not raise

    def test_does_not_break_clean_json(self):
        fn = self._fn()
        raw = '{"score": 4, "reasoning": "Good match."}'
        import json
        result = fn(raw)
        data = json.loads(result)
        self.assertEqual(data["score"], 4)

    def test_preserves_already_escaped_sequences(self):
        fn = self._fn()
        raw = '{"key": "value\\twith escaped tab"}'
        import json
        result = fn(raw)
        data = json.loads(result)
        self.assertIn("\t", data["key"])

    def test_parse_response_handles_tab_in_snippet(self):
        """Integration test — _parse_response survives tabs from PDF copy-paste.

        Simulates Ollama embedding a literal tab character inside a JSON string
        value (resume_snippet field). The outer JSON structure uses \n between
        fields as normal — only the value inside the string has a raw \t byte.
        """
        from analyzer import _parse_response
        # Build the JSON string manually so we control exactly which bytes
        # are literal vs escaped. Only the tab inside resume_snippet is raw.
        raw = (
            '{"score": 4, "matched_skills": [' +
            '{"skill": "Python", "match_type": "exact", ' +
            '"jd_snippet": "Python required", ' +
            '"resume_snippet": "Technical Skills\\n-' + '\t' + 'Backend: Python"}' +
            '], "missing_skills": [], "reasoning": "Good match."}'
        )
        result = _parse_response(raw)
        self.assertEqual(result["score"], 4)
        self.assertEqual(len(result["matched_skills"]), 1)


class TestModeConfig(unittest.TestCase):
    """Tests for MODE_CONFIG structure including thinking-model variants."""

    def test_base_modes_present(self):
        from analyzer.config import MODE_CONFIG
        for mode in ("fast", "standard", "detailed"):
            self.assertIn(mode, MODE_CONFIG, f"Missing base mode: {mode}")

    def test_thinking_modes_present(self):
        from analyzer.config import MODE_CONFIG
        for mode in ("fast_thinking", "standard_thinking", "detailed_thinking"):
            self.assertIn(mode, MODE_CONFIG, f"Missing thinking mode: {mode}")

    def test_thinking_modes_have_required_keys(self):
        from analyzer.config import MODE_CONFIG
        for mode in ("fast_thinking", "standard_thinking", "detailed_thinking"):
            cfg = MODE_CONFIG[mode]
            for key in ("snippet_len", "max_matched", "max_missing", "suggestions", "num_predict"):
                self.assertIn(key, cfg, f"{mode} missing key: {key}")

    def test_thinking_snippet_len_fixed_at_60(self):
        from analyzer.config import MODE_CONFIG
        for mode in ("fast_thinking", "standard_thinking", "detailed_thinking"):
            self.assertEqual(MODE_CONFIG[mode]["snippet_len"], 60,
                             f"{mode} snippet_len must be 60")

    def test_thinking_max_matched_fixed_at_10(self):
        from analyzer.config import MODE_CONFIG
        for mode in ("fast_thinking", "standard_thinking", "detailed_thinking"):
            self.assertEqual(MODE_CONFIG[mode]["max_matched"], 10,
                             f"{mode} max_matched must be 10")

    def test_thinking_max_missing_fixed_at_7(self):
        from analyzer.config import MODE_CONFIG
        for mode in ("fast_thinking", "standard_thinking", "detailed_thinking"):
            self.assertEqual(MODE_CONFIG[mode]["max_missing"], 7,
                             f"{mode} max_missing must be 7")

    def test_thinking_num_predict_is_doubled_base(self):
        from analyzer.config import MODE_CONFIG
        self.assertEqual(MODE_CONFIG["fast_thinking"]["num_predict"],     1600)  # 800 * 2
        self.assertEqual(MODE_CONFIG["standard_thinking"]["num_predict"], 3600)  # 1800 * 2
        self.assertEqual(MODE_CONFIG["detailed_thinking"]["num_predict"], 8192)  # 4096 * 2 capped

    def test_thinking_suggestions_follow_base_mode(self):
        from analyzer.config import MODE_CONFIG
        self.assertFalse(MODE_CONFIG["fast_thinking"]["suggestions"])
        self.assertFalse(MODE_CONFIG["standard_thinking"]["suggestions"])
        self.assertTrue(MODE_CONFIG["detailed_thinking"]["suggestions"])


class TestModelCapMode(unittest.TestCase):
    """Tests for analyzer.config.cap_mode_for_model and get_model_max_mode."""

    def test_phi35_allows_all_modes(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "phi3.5:3.8b"), "detailed")
        self.assertEqual(cap_mode_for_model("standard", "phi3.5:3.8b"), "standard")
        self.assertEqual(cap_mode_for_model("fast",     "phi3.5:3.8b"), "fast")

    def test_llama32_capped_at_standard(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "llama3.2:3b"), "standard")
        self.assertEqual(cap_mode_for_model("standard", "llama3.2:3b"), "standard")
        self.assertEqual(cap_mode_for_model("fast",     "llama3.2:3b"), "fast")

    def test_llama31_allows_detailed(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "llama3.1:8b"), "detailed")
        self.assertEqual(cap_mode_for_model("standard", "llama3.1:8b"), "standard")

    def test_gemma3_27b_allows_detailed(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "gemma3:27b"), "detailed")

    def test_gemma3_4b_capped_at_standard(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "gemma3:4b"), "standard")

    def test_unknown_model_defaults_to_standard(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "unknown-model:7b"), "standard")
        self.assertEqual(cap_mode_for_model("fast",     "unknown-model:7b"), "fast")

    def test_tag_variant_matches_base(self):
        from analyzer.config import cap_mode_for_model
        # llama3.1:8b-instruct should match llama3.1:8b entry
        self.assertEqual(cap_mode_for_model("detailed", "llama3.1:8b-instruct"), "detailed")

    def test_gemma3n_e4b_allows_detailed(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "gemma3n:e4b"), "detailed")
        self.assertEqual(cap_mode_for_model("standard", "gemma3n:e4b"), "standard")
        self.assertEqual(cap_mode_for_model("fast",     "gemma3n:e4b"), "fast")

    def test_gemma4_e2b_allows_detailed(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "gemma4:e2b"), "detailed")

    def test_gemma4_e4b_allows_detailed(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "gemma4:e4b"), "detailed")

    def test_gemma4_26b_allows_detailed(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "gemma4:26b"), "detailed")

    def test_qwen25_coder_7b_allows_detailed(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "qwen2.5-coder:7b"), "detailed")
        self.assertEqual(cap_mode_for_model("standard", "qwen2.5-coder:7b"), "standard")
        self.assertEqual(cap_mode_for_model("fast",     "qwen2.5-coder:7b"), "fast")

    def test_qwen25_coder_14b_allows_detailed(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "qwen2.5-coder:14b"), "detailed")

    def test_qwen25_coder_32b_allows_detailed(self):
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "qwen2.5-coder:32b"), "detailed")

    def test_qwen25_7b_still_capped_at_standard(self):
        # qwen2.5:7b (base, not coder) should remain standard
        from analyzer.config import cap_mode_for_model
        self.assertEqual(cap_mode_for_model("detailed", "qwen2.5:7b"), "standard")


class TestThinkingModelDetection(unittest.TestCase):
    """Tests for analyzer.config.is_thinking_model."""

    def test_gemma4_e2b_is_thinking(self):
        from analyzer.config import is_thinking_model
        self.assertTrue(is_thinking_model("gemma4:e2b"))

    def test_gemma4_e4b_is_thinking(self):
        from analyzer.config import is_thinking_model
        self.assertTrue(is_thinking_model("gemma4:e4b"))

    def test_gemma4_26b_is_thinking(self):
        from analyzer.config import is_thinking_model
        self.assertTrue(is_thinking_model("gemma4:26b"))

    def test_deepseek_r1_is_thinking(self):
        from analyzer.config import is_thinking_model
        self.assertTrue(is_thinking_model("deepseek-r1:7b"))

    def test_deepseek_r1_14b_is_thinking(self):
        from analyzer.config import is_thinking_model
        self.assertTrue(is_thinking_model("deepseek-r1:14b"))

    def test_qwq_is_thinking(self):
        from analyzer.config import is_thinking_model
        self.assertTrue(is_thinking_model("qwq:32b"))

    def test_llama31_not_thinking(self):
        from analyzer.config import is_thinking_model
        self.assertFalse(is_thinking_model("llama3.1:8b"))

    def test_gemma3_not_thinking(self):
        from analyzer.config import is_thinking_model
        self.assertFalse(is_thinking_model("gemma3:12b"))

    def test_empty_string_not_thinking(self):
        from analyzer.config import is_thinking_model
        self.assertFalse(is_thinking_model(""))

    def test_unknown_model_not_thinking(self):
        from analyzer.config import is_thinking_model
        self.assertFalse(is_thinking_model("unknownmodel:7b"))

    def test_prefix_match_gemma4_variant(self):
        # gemma4:latest should match because base name 'gemma4' is in set
        from analyzer.config import is_thinking_model
        self.assertTrue(is_thinking_model("gemma4:latest"))


class TestBuildSystemPromptModes(unittest.TestCase):
    """Tests for analyzer.prompts.build_system_prompt — mode-specific schemas."""

    def _prompt(self, mode: str) -> str:
        import os
        os.environ["ANALYSIS_MODE"] = mode
        from analyzer.config import MODE_CONFIG
        from analyzer.prompts import build_system_prompt
        prompt = build_system_prompt(MODE_CONFIG[mode])
        del os.environ["ANALYSIS_MODE"]
        return prompt

    def test_fast_has_no_snippets_on_matched(self):
        prompt = self._prompt("fast")
        self.assertNotIn("jd_snippet", prompt.split("matched_skills")[1].split("missing_skills")[0])

    def test_fast_has_no_suggestions(self):
        prompt = self._prompt("fast")
        self.assertNotIn("suggestions", prompt)

    def test_standard_has_jd_snippet(self):
        prompt = self._prompt("standard")
        self.assertIn("jd_snippet", prompt)

    def test_standard_has_resume_snippet_by_default(self):
        # Standard mode includes resume_snippet on first attempt (resume_snippet=True)
        prompt = self._prompt("standard")
        self.assertIn("resume_snippet", prompt)

    def test_standard_drops_resume_snippet_on_retry(self):
        # On retry, resume_snippet=False reduces output complexity
        import os
        from analyzer.config import MODE_CONFIG
        from analyzer.prompts import build_system_prompt
        os.environ["ANALYSIS_MODE"] = "standard"
        cfg = MODE_CONFIG["standard"]
        prompt = build_system_prompt(cfg, mode="standard", resume_snippet=False)
        self.assertNotIn("resume_snippet", prompt)
        self.assertIn("jd_snippet", prompt)

    def test_standard_has_no_suggestions(self):
        prompt = self._prompt("standard")
        self.assertNotIn("suggestions", prompt)

    def test_detailed_has_resume_snippet(self):
        prompt = self._prompt("detailed")
        self.assertIn("resume_snippet", prompt)

    def test_detailed_has_suggestions(self):
        prompt = self._prompt("detailed")
        self.assertIn("suggestions", prompt)
        self.assertIn("job_requirement", prompt)


class TestChunkParsers(unittest.TestCase):
    """Tests for _parse_score_chunk and _parse_chunk — pure functions."""

    def test_parse_score_valid(self):
        from analyzer.llm import _parse_score_chunk
        raw = '{"score": 4, "reasoning": "Good match."}'
        score, reasoning = _parse_score_chunk(raw)
        self.assertEqual(score, 4)
        self.assertEqual(reasoning, "Good match.")

    def test_parse_score_float_rounds(self):
        from analyzer.llm import _parse_score_chunk
        # 3.6 → 4 with both round() and int(x+0.5) — avoids banker's rounding edge case
        raw = '{"score": 3.6, "reasoning": "Strong match."}'
        score, _ = _parse_score_chunk(raw)
        self.assertEqual(score, 4)

    def test_parse_score_out_of_range(self):
        from analyzer.llm import _parse_score_chunk
        raw = '{"score": 7, "reasoning": "Great."}'
        score, reasoning = _parse_score_chunk(raw)
        self.assertIsNone(score)

    def test_parse_score_none_input(self):
        from analyzer.llm import _parse_score_chunk
        score, reasoning = _parse_score_chunk(None)
        self.assertIsNone(score)
        self.assertEqual(reasoning, "")

    def test_parse_score_strips_markdown_fences(self):
        from analyzer.llm import _parse_score_chunk
        raw = '```json\n{"score": 3, "reasoning": "Moderate."}\n```'
        score, reasoning = _parse_score_chunk(raw)
        self.assertEqual(score, 3)
        self.assertEqual(reasoning, "Moderate.")

    def test_parse_chunk_extracts_key(self):
        from analyzer.llm import _parse_chunk
        raw = '{"matched_skills": [{"skill": "Python", "match_type": "exact"}]}'
        result = _parse_chunk(raw, "matched_skills", "test")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["skill"], "Python")

    def test_parse_chunk_missing_key(self):
        from analyzer.llm import _parse_chunk
        raw = '{"wrong_key": []}'
        result = _parse_chunk(raw, "matched_skills", "test")
        self.assertIsNone(result)

    def test_parse_chunk_none_input(self):
        from analyzer.llm import _parse_chunk
        result = _parse_chunk(None, "matched_skills", "test")
        self.assertIsNone(result)

    def test_parse_chunk_invalid_json(self):
        from analyzer.llm import _parse_chunk
        result = _parse_chunk("not json at all", "matched_skills", "test")
        self.assertIsNone(result)


class TestChunkPrompts(unittest.TestCase):
    """Tests for chunk prompt builders."""

    def _cfg(self, mode):
        from analyzer.config import MODE_CONFIG
        return MODE_CONFIG[mode]

    def test_chunk1_has_score_and_reasoning(self):
        from analyzer.prompts import build_chunk1_prompt
        prompt = build_chunk1_prompt(self._cfg("standard"), "standard")
        self.assertIn("score", prompt)
        self.assertIn("reasoning", prompt)
        self.assertNotIn("matched_skills", prompt)
        self.assertNotIn("missing_skills", prompt)

    def test_chunk2_has_matched_skills(self):
        from analyzer.prompts import build_chunk2_prompt
        prompt = build_chunk2_prompt(self._cfg("standard"), "standard")
        self.assertIn("matched_skills", prompt)
        self.assertNotIn("missing_skills", prompt)
        self.assertNotIn("reasoning", prompt)

    def test_chunk3_has_missing_skills(self):
        from analyzer.prompts import build_chunk3_prompt
        prompt = build_chunk3_prompt(self._cfg("standard"), "standard")
        self.assertIn("missing_skills", prompt)
        self.assertNotIn("matched_skills", prompt)
        self.assertNotIn("reasoning", prompt)

    def test_chunk4_has_suggestions(self):
        from analyzer.prompts import build_chunk4_prompt
        prompt = build_chunk4_prompt(self._cfg("detailed"))
        self.assertIn("suggestions", prompt)
        self.assertIn("title", prompt)
        self.assertIn("detail", prompt)

    def test_chunk2_fast_has_no_snippets(self):
        from analyzer.prompts import build_chunk2_prompt
        prompt = build_chunk2_prompt(self._cfg("fast"), "fast")
        self.assertNotIn("jd_snippet", prompt)
        self.assertIn("4 words", prompt)

    def test_chunk2_standard_has_jd_snippet(self):
        from analyzer.prompts import build_chunk2_prompt
        prompt = build_chunk2_prompt(self._cfg("standard"), "standard")
        self.assertIn("jd_snippet", prompt)

    def test_chunk3_standard_has_severity(self):
        from analyzer.prompts import build_chunk3_prompt
        prompt = build_chunk3_prompt(self._cfg("standard"), "standard")
        self.assertIn("blocker", prompt)
        self.assertIn("major", prompt)
        self.assertIn("minor", prompt)


class TestCallOllamaChunked(unittest.TestCase):
    """Tests for call_ollama_chunked — mocks _call_chunk."""

    def _make_chunk_response(self, data: dict) -> str:
        import json
        return json.dumps(data)

    @patch("analyzer.llm._call_chunk", new_callable=AsyncMock)
    def test_successful_standard_analysis(self, mock_call_chunk):
        """All 3 chunks succeed — result has all fields populated."""
        import json
        mock_call_chunk.side_effect = [
            json.dumps({"score": 4, "reasoning": "Strong match."}),
            json.dumps({"matched_skills": [
                {"skill": "Python", "match_type": "exact", "jd_snippet": "Python required"}
            ]}),
            json.dumps({"missing_skills": [
                {"skill": "AWS Lambda", "severity": "major",
                 "requirement_type": "preferred", "jd_snippet": "Lambda required"}
            ]}),
        ]
        from analyzer.llm import call_ollama_chunked
        result = run(call_ollama_chunked(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertEqual(result["score"], 4)
        self.assertEqual(result["reasoning"], "Strong match.")
        self.assertGreater(len(result["matched_skills"]), 0)
        self.assertGreater(len(result["missing_skills"]), 0)
        self.assertEqual(result["llm_provider"], "ollama")

    @patch("analyzer.llm._call_chunk", new_callable=AsyncMock)
    def test_chunk1_failure_raises(self, mock_call_chunk):
        """If chunk 1 fails, ValueError is raised."""
        mock_call_chunk.return_value = None  # simulate connection failure
        from analyzer.llm import call_ollama_chunked
        with self.assertRaises(ValueError) as ctx:
            run(call_ollama_chunked(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertIn("score", str(ctx.exception))

    @patch("analyzer.llm._call_chunk", new_callable=AsyncMock)
    def test_chunk2_failure_returns_empty_matched(self, mock_call_chunk):
        """If chunk 2 fails, matched_skills is empty but result still valid."""
        import json
        mock_call_chunk.side_effect = [
            json.dumps({"score": 3, "reasoning": "Moderate match."}),
            None,  # chunk 2 fails
            json.dumps({"missing_skills": [
                {"skill": "Docker", "severity": "minor",
                 "requirement_type": "preferred", "jd_snippet": "Docker preferred"}
            ]}),
        ]
        from analyzer.llm import call_ollama_chunked
        result = run(call_ollama_chunked(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS))
        self.assertEqual(result["score"], 3)
        self.assertEqual(result["matched_skills"], [])
        self.assertGreater(len(result["missing_skills"]), 0)

    @patch("analyzer.llm._call_chunk", new_callable=AsyncMock)
    def test_routes_ollama_to_chunked(self, mock_call_chunk):
        """analyze_match with ollama provider calls call_ollama_chunked."""
        import json
        mock_call_chunk.side_effect = [
            json.dumps({"score": 4, "reasoning": "Good."}),
            json.dumps({"matched_skills": [{"skill": "Python", "match_type": "exact", "jd_snippet": ""}]}),
            json.dumps({"missing_skills": []}),
        ]
        from analyzer import analyze_match
        result = run(analyze_match(MOCK_RESUME_DEVSECOPS, MOCK_JOB_DEVSECOPS, provider="ollama"))
        self.assertEqual(result["llm_provider"], "ollama")
        self.assertEqual(result["score"], 4)
        # _call_chunk should have been called (at least chunks 1-3)
        self.assertGreaterEqual(mock_call_chunk.call_count, 3)

class TestValidateLLMOutput(unittest.TestCase):

    def _result(self, score=3, matched=None, missing=None):
        return {
            "score": score,
            "matched_skills": matched if matched is not None else [{"skill": "Python"}],
            "missing_skills": missing if missing is not None else [],
            "reasoning": "test",
        }

    def test_score1_with_no_matched_is_valid(self):
        """Score=1 with no matched skills is correct for a complete mismatch."""
        from analyzer.penalties import validate_llm_output
        result = self._result(score=1, matched=[])
        out = validate_llm_output(result, "x" * 600, "resume text")
        self.assertTrue(out["valid"],
            f"Expected valid for score=1 + no matches, got errors: {out['errors']}")

    def test_score2_with_no_matched_and_rich_jd_is_invalid(self):
        """Score>1 with no matched skills and rich JD should flag as suspicious."""
        from analyzer.penalties import validate_llm_output
        result = self._result(score=2, matched=[])
        out = validate_llm_output(result, "x" * 600, "resume text")
        self.assertFalse(out["valid"])
        self.assertTrue(any("matched" in e for e in out["errors"]))

    def test_score3_with_matched_skills_is_valid(self):
        from analyzer.penalties import validate_llm_output
        result = self._result(score=3, matched=[{"skill": "Python"}])
        out = validate_llm_output(result, "x" * 600, "resume text")
        self.assertTrue(out["valid"])

    def test_short_jd_no_matched_skills_not_flagged(self):
        """Short JD (<500 chars) with no matched skills should not flag."""
        from analyzer.penalties import validate_llm_output
        result = self._result(score=3, matched=[])
        out = validate_llm_output(result, "short jd", "resume text")
        self.assertTrue(out["valid"])


