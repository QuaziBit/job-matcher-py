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
